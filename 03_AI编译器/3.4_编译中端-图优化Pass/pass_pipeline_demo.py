#!/usr/bin/env python3
"""4.4 编译中端 - 图优化 Pass：Relax Pass Pipeline 示例

运行：
    cd /data/liyangyang/ai_infra/03_AI编译器
    /data/liyangyang/qwen35_env/bin/python 3.4_编译中端-图优化Pass/pass_pipeline_demo.py

依赖：tvm, onnx（已在 qwen35_env 中安装）
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx


def create_sample_onnx() -> onnx.ModelProto:
    """构造一个带冗余的 ONNX 图：input -> relu -> relu -> output"""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    n1 = helper.make_node("Relu", ["input"], ["r1"])
    n2 = helper.make_node("Relu", ["r1"], ["output"])

    graph = helper.make_graph([n1, n2], "double_relu", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def main():
    model = create_sample_onnx()
    mod = from_onnx(model, shape_dict={"input": [1, 4]}, dtype_dict="float32")

    print("=== Before Passes ===")
    print(mod)

    # Relax 常见 Pass pipeline
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.FoldConstant()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.DeadCodeElimination()(mod)
    mod = relax.transform.RemoveUnusedOutputs()(mod)

    print("\n=== After Passes ===")
    print(mod)

    print("\n常用 Relax Pass 顺序：LegalizeOps -> FoldConstant -> FuseOps -> DCE -> RemoveUnusedOutputs")


if __name__ == "__main__":
    main()
