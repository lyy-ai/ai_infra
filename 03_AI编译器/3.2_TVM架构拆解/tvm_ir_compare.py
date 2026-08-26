#!/usr/bin/env python3
"""4.2 TVM 架构拆解：对比静态 shape 与动态 shape 的 Relax IR

运行：
    cd /data/ai_infra/03_AI编译器
    /data/qwen35_env/bin/python 3.2_TVM架构拆解/tvm_ir_compare.py

依赖：tvm（已在 qwen35_env 中安装：apache-tvm 0.25.0，仅含 Relax）
"""
import tvm
from tvm import relax
from tvm.relax.frontend import nn


def static_shape_demo():
    print("=" * 40)
    print("静态 shape 示例：Linear 模块")
    print("=" * 40)

    class Linear(nn.Module):
        def __init__(self):
            self.w = nn.Parameter((784, 128), "float32")

        def forward(self, x: nn.Tensor):
            return nn.matmul(x, self.w)

    model = Linear()
    mod, _ = model.export_tvm(spec={"forward": {"x": nn.spec.Tensor([1, 784], "float32")}})
    print(mod)


def dynamic_shape_demo():
    print("\n" + "=" * 40)
    print("动态 shape 示例：batch 为符号变量")
    print("=" * 40)

    class LinearDynamic(nn.Module):
        def __init__(self):
            self.w = nn.Parameter((784, 128), "float32")

        def forward(self, x: nn.Tensor):
            return nn.matmul(x, self.w)

    model = LinearDynamic()
    mod, _ = model.export_tvm(spec={"forward": {"x": nn.spec.Tensor(["batch", 784], "float32")}})
    print(mod)


def lower_and_build_demo():
    print("\n" + "=" * 40)
    print("编译到后端：LegalizeOps + FoldConstant + build")
    print("=" * 40)

    class Tiny(nn.Module):
        def forward(self, x: nn.Tensor):
            return nn.relu(x)

    model = Tiny()
    mod, _ = model.export_tvm(spec={"forward": {"x": nn.spec.Tensor([1, 4], "float32")}})
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.FoldConstant()(mod)

    ex = relax.build(mod, target="llvm")
    print("Build success，产物类型:", type(ex))


def main():
    static_shape_demo()
    dynamic_shape_demo()
    lower_and_build_demo()


if __name__ == "__main__":
    main()
