# 2.10 Hopper 与 Blackwell 新特性：Warp Specialization 流水线离散事件模拟
#
# 运行：
#   cd /data/ai_infra/02_CUDA编程与HPC高性能计算
#   /data/qwen35_env/bin/python 2.10_Hopper与Blackwell新特性/warp_specialization_sim.py
#
# 说明：本脚本用简化的离散事件模型，对比一个 GEMM tile 流水线（沿 K 维
# 逐块加载 -> 计算）在两种写法下的 SM 计算单元利用率：
#   1) 传统交替：同一批 warp 串行执行 "搬 K-tile -> 算 K-tile"
#   2) warp specialization：生产者 warp 专职搬运（多级缓冲），
#      消费者 warp 专职计算，搬运与计算通过流水重叠
# 仅依赖 stdlib。
import unicodedata


def pad(s, width):
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, width - w)


def simulate_traditional(n_tiles, t_load, t_compute):
    """传统交替写法：每个 tile 先搬后算，串行，搬运时计算单元空转。"""
    time = 0.0
    busy = 0.0
    for _ in range(n_tiles):
        time += t_load      # 搬运，计算单元空转
        time += t_compute   # 计算
        busy += t_compute
    return time, busy


def simulate_warp_spec(n_tiles, t_load, t_compute, n_buffers=3):
    """Warp specialization 离散事件模拟。

    生产者：按 t_load 节奏产出 tile 写入共享缓冲；缓冲满（在途 tile 数
    达到 n_buffers）则停工等待消费者释放。
    消费者：取一个已就绪的 tile 计算 t_compute；无就绪 tile 则等待。
    返回 (总时长, 计算单元忙时累计)。
    """
    produced = 0
    consumed = 0
    producer_free_at = 0.0
    consumer_free_at = 0.0
    tile_ready_at = []    # 每个已产出 tile 的就绪时刻
    busy = 0.0
    while consumed < n_tiles:
        can_produce = (produced < n_tiles
                       and produced - consumed < n_buffers
                       and producer_free_at <= consumer_free_at)
        if can_produce:
            # 生产事件
            producer_free_at += t_load
            tile_ready_at.append(producer_free_at)
            produced += 1
        else:
            # 消费事件（等待必要的 tile 就绪）
            start = max(consumer_free_at, tile_ready_at[consumed])
            consumer_free_at = start + t_compute
            busy += t_compute
            consumed += 1
    return consumer_free_at, busy


def compare(n_tiles, t_load, t_compute):
    t1, b1 = simulate_traditional(n_tiles, t_load, t_compute)
    t2, b2 = simulate_warp_spec(n_tiles, t_load, t_compute)
    return (t1, b1 / t1 * 100), (t2, b2 / t2 * 100)


def main():
    print("=" * 78)
    print("GEMM K-tile 流水线：传统交替 vs Warp Specialization（离散事件简模）")
    print("=" * 78)
    print()
    print("模型设定：一个 block 沿 K 维顺序处理 64 个 tile，每 tile 搬运耗时")
    print("t_load、计算耗时 t_compute。传统写法同一批 warp 串行'搬->算'；")
    print("warp specialization 用生产者/消费者分工 + 3 级共享缓冲重叠两者。")
    print()

    scenarios = [
        ("搬运=计算（均衡）", 10.0, 10.0),
        ("搬运>计算（访存偏重）", 14.0, 8.0),
        ("搬运<计算（计算偏重）", 6.0, 12.0),
        ("搬运>>计算（HBM 延迟高）", 20.0, 8.0),
    ]
    n_tiles = 64

    header = ["场景", "t_load", "t_compute",
              "传统:总时长", "传统:利用率", "warp-spec:总时长", "warp-spec:利用率", "加速比"]
    widths = [22, 8, 10, 13, 12, 17, 17, 8]
    print("  " + " | ".join(pad(h, w) for h, w in zip(header, widths)))
    print("  " + "-" * 112)
    for name, tl_, tc in scenarios:
        (t1, u1), (t2, u2) = compare(n_tiles, tl_, tc)
        row = [name, f"{tl_:.0f}", f"{tc:.0f}",
               f"{t1:.0f}", f"{u1:.1f}%", f"{t2:.0f}", f"{u2:.1f}%", f"{t1 / t2:.2f}x"]
        print("  " + " | ".join(pad(v, w) for v, w in zip(row, widths)))
    print()
    print("解读：")
    print("  1. 传统写法的利用率 = t_compute / (t_load + t_compute)，")
    print("     搬运期间计算单元必然空转。")
    print("  2. warp specialization 稳态吞吐由 max(t_load, t_compute) 决定：")
    print("     只要生产者喂得上，消费者（Tensor Core）就能接近满负荷。")
    print("  3. 搬运与计算接近均衡时分工收益最大（接近 2x）；当搬运远重于")
    print("     计算时，瓶颈在搬运本身，分工只能消除等待、不能让生产者变快——")
    print("     这时需要先靠 TMA 这类硬件把 t_load 压下去，分工才能发挥作用。")
    print("  4. FA3 在 H100 上从约 35% 提到约 75% 的峰值利用率，本质就是把")
    print("     '空转时间'用这种流水重叠填掉（还叠加了 GEMM-softmax 交错）。")
    print()
    print("局限说明：本模型是教学简模，未建模 L2 命中、mbarrier 同步开销、")
    print("多 block 调度等；真实 kernel 分析请使用 Nsight Compute。")


if __name__ == "__main__":
    main()
