#!/usr/bin/env python3
"""4.5 编译后端：LLVM / CUDA 后端配置对比（Relax 版）

运行：
    cd /data/ai_infra/03_AI编译器
    /data/qwen35_env/bin/python 3.5_编译后端/backend_compare.py

依赖：tvm（已在 qwen35_env 中安装。当前 pip wheel 未链接 CUDA runtime，
      故 CUDA 后端仅做 build 配置演示，真实运行需源码编译 TVM 并启用 CUDA runtime。）
"""
from tvm import relax
from tvm.relax.frontend import nn


class Tiny(nn.Module):
    def forward(self, x: nn.Tensor):
        return nn.relu(x)


def build_for_target(mod, target: str, output: str):
    print(f"\n--- Building for target: {target} ---")
    try:
        ex = relax.build(mod, target=target)
        ex.export_library(output)
        print(f"  成功：{output}")
    except Exception as e:
        print(f"  失败：{e}")


def main():
    model = Tiny()
    mod, _ = model.export_tvm(spec={"forward": {"x": nn.spec.Tensor([1, 4], "float32")}})
    mod = relax.transform.LegalizeOps()(mod)

    # 当前环境可用后端：llvm
    build_for_target(mod, "llvm", "3.5_编译后端/model_cpu.so")

    # CUDA 配置演示：当前 pip wheel 可以 build 出 cuda 目标，但加载运行需 CUDA runtime
    build_for_target(mod, "cuda", "3.5_编译后端/model_cuda.so")


if __name__ == "__main__":
    main()
