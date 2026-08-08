# tensor_core_int8.py
import torch
import time


def check_tensor_core_support():
    """检查当前 GPU 是否支持 INT8 Tensor Core"""
    if not torch.cuda.is_available():
        print("CUDA not available, cannot test Tensor Core INT8")
        return False

    device = torch.device('cuda')
    prop = torch.cuda.get_device_properties(device)
    print(f"GPU: {prop.name}")
    print(f"Compute Capability: {prop.major}.{prop.minor}")

    # Turing 及以上架构支持 INT8 Tensor Core
    if prop.major >= 7:
        print("INT8 Tensor Core: supported")
        return True
    else:
        print("INT8 Tensor Core: NOT supported")
        return False


def demo_int8_matmul():
    """使用 PyTorch 的 int8 矩阵乘"""
    torch.manual_seed(42)
    M, K, N = 256, 512, 256

    if hasattr(torch, '_int_mm') and torch.cuda.is_available():
        a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
        b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')
        c = torch._int_mm(a, b)
        print(f"Using torch._int_mm on CUDA")
    else:
        # PyTorch 1.10 没有 torch._int_mm，退到 CPU 用 int32 累加演示
        a = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        b = torch.randint(-128, 127, (K, N), dtype=torch.int8)
        c = torch.matmul(a.to(torch.int32), b.to(torch.int32))
        print(f"torch._int_mm not available, using CPU int32 matmul fallback")

    print(f"INT8 x INT8 -> INT32 result shape: {c.shape}")
    print(f"Result dtype: {c.dtype}")
    print(f"Sample values: {c[0, :5]}")


def benchmark_int8_vs_fp16(M, K, N, iterations=100):
    """粗略对比 INT8 与 FP16 矩阵乘耗时"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark")
        return

    if not hasattr(torch, '_int_mm'):
        print("torch._int_mm not available in this PyTorch version, skipping INT8 speedup benchmark")
        print("  (In production, use TensorRT / CUTLASS / newer PyTorch for INT8 Tensor Core GEMM)")
        return

    a_i8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
    b_i8 = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')
    a_f16 = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b_f16 = torch.randn(K, N, dtype=torch.float16, device='cuda')

    # 预热
    for _ in range(10):
        torch._int_mm(a_i8, b_i8)
        torch.matmul(a_f16, b_f16)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        torch._int_mm(a_i8, b_i8)
    torch.cuda.synchronize()
    int8_time = time.time() - start

    start = time.time()
    for _ in range(iterations):
        torch.matmul(a_f16, b_f16)
    torch.cuda.synchronize()
    fp16_time = time.time() - start

    print(f"M={M}, K={K}, N={N}")
    print(f"  INT8 time: {int8_time * 1000:.2f} ms")
    print(f"  FP16 time: {fp16_time * 1000:.2f} ms")
    print(f"  Speedup: {fp16_time / int8_time:.2f}x")


if __name__ == "__main__":
    check_tensor_core_support()
    demo_int8_matmul()
    benchmark_int8_vs_fp16(512, 1024, 512)
