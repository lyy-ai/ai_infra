# pytorch_ptq_demo.py
import torch
import torch.quantization
from torchvision import models


def demo_pytorch_ptq():
    # 兼容不同 torchvision 版本：旧版用 pretrained，新版用 weights
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.eval()

    # 1. 设置 qconfig
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    # 2. 准备模型
    model_prepared = torch.quantization.prepare(model)

    # 3. 用校准数据前向（收集统计信息）
    with torch.no_grad():
        for _ in range(10):
            dummy = torch.randn(1, 3, 224, 224)
            model_prepared(dummy)

    # 4. 转换
    model_quantized = torch.quantization.convert(model_prepared)

    print(model_quantized)
    print("\n量化完成，可用 torch.jit.save 导出。")


if __name__ == "__main__":
    demo_pytorch_ptq()
