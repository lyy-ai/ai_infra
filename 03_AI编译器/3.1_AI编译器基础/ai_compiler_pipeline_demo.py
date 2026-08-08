#!/usr/bin/env python3
"""4.1 AI 编译器基础：三段式编译 Pipeline 示例（Relax 版）

运行：
    cd /data/liyangyang/ai_infra/03_AI编译器/3.1_AI编译器基础
    /data/liyangyang/qwen35_env/bin/python ai_compiler_pipeline_demo.py

依赖：tvm, onnx（已在 qwen35_env 中安装：apache-tvm 0.25.0，仅含 Relax）
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ONNX_PATH = os.path.join(_BASE_DIR, "sample.onnx")
_SO_PATH = os.path.join(_BASE_DIR, "sample.so")


def create_sample_onnx() -> onnx.ModelProto:
    """创建一个最小 ONNX 模型：input -> relu -> output"""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph([node], "relu", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, _ONNX_PATH)
    return model


def main():
    # 1. 前端：加载 ONNX 模型并解析为 Relax IR
    model = onnx.load(_ONNX_PATH) if False else create_sample_onnx()
    mod = from_onnx(model, shape_dict={"input": [1, 4]}, dtype_dict="float32")

    print("=== Frontend (Relax IR) ===")
    print(mod)

    # 2. 中端：图优化 Pass
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.DeadCodeElimination()(mod)
    mod = relax.transform.FoldConstant()(mod)
    mod = relax.transform.FuseOps()(mod)

    print("\n=== MiddleEnd (after passes) ===")
    print(mod)

    # 3. 后端：编译为 LLVM CPU 可执行文件（当前 pip wheel 未链接 CUDA runtime，故用 llvm）
    ex = relax.build(mod, target="llvm")
    ex.export_library(_SO_PATH)
    print("\n编译产物：", _SO_PATH)


if __name__ == "__main__":
    main()
