# mixed_precision.py
import torch
import torch.nn as nn


def test_amp_training():
    """演示 PyTorch AMP 自动混合精度训练/推理"""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    x = torch.randn(32, 128, device=device)
    target = torch.randn(32, 10, device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            y = model(x)
            loss = nn.MSELoss()(y, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"AMP FP16 training loss: {loss.item():.6f}")
    else:
        y = model(x)
        loss = nn.MSELoss()(y, target)
        loss.backward()
        optimizer.step()
        print(f"FP32 training loss: {loss.item():.6f}")


def test_dtype_conversion():
    """手动演示 FP32 -> FP16 / BF16 / FP8 转换"""
    torch.manual_seed(42)
    x = torch.randn(4, 4)

    x_fp16 = x.half()
    x_bf16 = x.bfloat16() if torch.cuda.is_available() else None

    print("FP32:\n", x)
    print("FP16:\n", x_fp16)
    if x_bf16 is not None:
        print("BF16:\n", x_bf16)

    # FP8 模拟（PyTorch 部分版本支持 torch.float8_e4m3fn）
    if hasattr(torch, "float8_e4m3fn"):
        x_fp8 = x.to(torch.float8_e4m3fn)
        print("FP8 E4M3:\n", x_fp8)
    else:
        print("当前 PyTorch 版本不支持 FP8 张量类型")


if __name__ == "__main__":
    test_dtype_conversion()
    print()
    test_amp_training()
