# 2.4 PyTorch：autograd / 显存 / profiler 演示
#
# 运行：
#   /data/liyangyang/qwen35_env/bin/python 1.8_PyTorch框架/pytorch_infra_demo.py

import torch
import torch.nn as nn


def autograd_demo():
    x = torch.randn(4, 4, requires_grad=True)
    y = (x * x).sum()
    y.backward()
    ok = torch.allclose(x.grad, 2 * x)
    print(f"autograd: dy/dx == 2x ? {ok}")


def memory_demo(device):
    if not torch.cuda.is_available():
        print("memory: CUDA 不可用，跳过")
        return
    torch.cuda.reset_peak_memory_stats(device)
    m = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True), 4).to(device)
    x = torch.randn(8, 128, 512, device=device)
    out = m(x)
    loss = out.sum()
    loss.backward()
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"memory: 训练一步峰值显存 {peak:.2f} GB（激活 + 梯度 + 参数）")


def profiler_demo():
    model = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024))
    x = torch.randn(64, 1024)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        for _ in range(10):
            model(x)
    print("profiler: top CPU ops")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=4))


def main():
    autograd_demo()
    memory_demo("cuda")
    profiler_demo()


if __name__ == "__main__":
    main()
