# 2.9 Triton 编程实战：Online Softmax 分块归约模拟
#
# 运行：
#   cd /data/ai_infra/02_CUDA编程与HPC高性能计算
#   /data/qwen35_env/bin/python 2.9_Triton编程实战/triton_softmax_demo.py
#
# ============================================================================
# 附：对应的完整 Triton kernel 代码（需要 GPU 环境 + pip install triton 运行，
#     本脚本正文只用 numpy，任何环境可跑）
#
#   import triton
#   import triton.language as tl
#
#   @triton.jit
#   def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
#       row = tl.program_id(0)
#       cols = tl.arange(0, BLOCK_SIZE)
#       mask = cols < n_cols
#       x = tl.load(in_ptr + row * n_cols + cols, mask=mask, other=float("-inf"))
#       row_max = tl.max(x, axis=0)
#       x = tl.exp(x - row_max)
#       row_sum = tl.sum(x, axis=0)
#       tl.store(out_ptr + row * n_cols + cols, x / row_sum, mask=mask)
#
#   # 当一行长度超过单个 tile 容量时，Triton/CUDA kernel 需要沿列方向循环，
#   # 用 online softmax 逐块维护 running max 与缩放后的 running sum，
#   # 本脚本模拟的正是这一分块归约逻辑。
# ============================================================================
import numpy as np


def naive_softmax(x):
    """朴素两遍 softmax：先全局 max 求 exp，再全局 sum 做归一化。"""
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def online_softmax_blocked(x, block_size, verbose=True):
    """Online softmax：分块扫描，维护 running max 与缩放后的 running sum。

    每读入一个 block：
      m_new = max(m_old, max(block))
      s_new = s_old * exp(m_old - m_new) + sum(exp(block - m_new))
    全部块扫描完后再统一归一化，整个过程中间状态只有 (m, s) 两个标量，
    显存占用 O(1)，输入只需读取一遍。
    """
    n = len(x)
    m = -np.inf  # running max
    s = 0.0      # running sum（相对于 m 缩放）
    if verbose:
        print(f"  [online softmax] 行长度 n={n}, BLOCK_SIZE={block_size}, "
              f"共 {(n + block_size - 1) // block_size} 个分块")
        print(f"  {'块id':>4} | {'块内 max':>10} | {'更新后 m':>10} | "
              f"{'缩放因子 exp(m_old-m)':>18} | {'更新后 s':>12}")
        print("  " + "-" * 66)
    n_blocks = (n + block_size - 1) // block_size
    for bid in range(n_blocks):
        lo, hi = bid * block_size, min((bid + 1) * block_size, n)
        blk = x[lo:hi]
        blk_max = np.max(blk)
        m_new = max(m, blk_max)
        scale = np.exp(m - m_new) if np.isfinite(m) else 0.0
        s = s * scale + np.sum(np.exp(blk - m_new))
        if verbose:
            scale_str = f"{scale:.6f}" if bid > 0 else "-"
            print(f"  {bid:>4} | {blk_max:>10.4f} | {m_new:>10.4f} | "
                  f"{scale_str:>18} | {s:>12.4f}")
        m = m_new
    # 第二遍只做归一化（FlashAttention 中这一步被融合进对 V 的累加）
    return np.exp(x - m) / s


def demo_single_row(n=1037, block_size=256, seed=0):
    print("=" * 72)
    print(f"示例 1：单行 softmax，naive vs online（行长 {n} 不是块长 {block_size} 的整数倍）")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) * 4.0  # 放大幅度，制造数值差异更明显的场景

    ref = naive_softmax(x)
    out = online_softmax_blocked(x, block_size)

    print(f"\n  数值一致性检查:")
    print(f"    max |online - naive| = {np.max(np.abs(out - ref)):.3e}")
    print(f"    allclose(atol=1e-12)  = {np.allclose(out, ref, atol=1e-12)}")
    print(f"    softmax 输出之和      = {np.sum(out):.10f}（应为 1）")
    return out, ref


def demo_memory_traffic(n=1_000_000):
    """对比 naive 与 online/fused softmax 的显存读写次数（教学估算）。"""
    print()
    print("=" * 72)
    print(f"示例 2：显存访问量对比（行长 n={n:,}，FP32）")
    print("=" * 72)
    bytes_per_elem = 4
    mb = n * bytes_per_elem / 1e6

    # naive：读 x 求 max（1 次读）+ 读 x 求 exp 和（1 次读）+ 读 exp 写结果（1 读 1 写）
    naive_read = 3 * mb
    naive_write = 1 * mb
    # fused/online：读 x 一遍维护 (m, s)，归一化阶段读 x 写结果（或二次读已缓存数据）
    fused_read = 2 * mb
    fused_write = 1 * mb

    print(f"  {'方案':<22} | {'全局内存读':>12} | {'全局内存写':>12} | {'总流量':>12}")
    print("  " + "-" * 66)
    print(f"  {'naive（三遍扫描）':<20} | {naive_read:>10.1f} MB | {naive_write:>10.1f} MB | "
          f"{naive_read + naive_write:>10.1f} MB")
    print(f"  {'online/fused（Triton）':<18} | {fused_read:>10.1f} MB | {fused_write:>10.1f} MB | "
          f"{fused_read + fused_write:>10.1f} MB")
    ratio = (naive_read + naive_write) / (fused_read + fused_write)
    print(f"\n  总流量比值 naive/fused = {ratio:.2f}x")
    print()
    print("  说明：softmax 是 memory-bound 算子，显存流量直接决定耗时。")
    print("        online softmax 用 O(1) 的 (m, s) 状态换掉一遍全局扫描，")
    print("        这正是 FlashAttention 把 O(N^2) 注意力矩阵永不落地的同一思想。")


def demo_matrix(rows=4, n=777, block_size=256, seed=1):
    print()
    print("=" * 72)
    print(f"示例 3：批量行（{rows} x {n}）逐行 online softmax，与 numpy 参考对比")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, n)) * 3.0
    ok = True
    for i in range(rows):
        ref = naive_softmax(X[i])
        out = online_softmax_blocked(X[i], block_size, verbose=False)
        ok &= np.allclose(out, ref, atol=1e-12)
    print(f"  全部 {rows} 行 allclose(atol=1e-12) = {ok}")


def main():
    demo_single_row()
    demo_memory_traffic()
    demo_matrix()


if __name__ == "__main__":
    main()
