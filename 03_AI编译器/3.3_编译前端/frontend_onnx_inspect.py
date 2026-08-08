#!/usr/bin/env python3
"""4.3 编译前端：ONNX 解析与 IR 查看（Relax 版）

运行：
    cd /data/liyangyang/ai_infra/03_AI编译器
    /data/liyangyang/qwen35_env/bin/python 3.3_编译前端/frontend_onnx_inspect.py

依赖：tvm, onnx（已在 qwen35_env 中安装）
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
from tvm.relax.frontend.onnx import from_onnx


def create_sample_onnx(path: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph([node], "relu", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return model


def main():
    model_path = "3.3_编译前端/sample.onnx"
    model = create_sample_onnx(model_path)

    mod = from_onnx(model, shape_dict={"input": [1, 4]}, dtype_dict="float32")

    print("=" * 40)
    print(f"模型：{model_path}")
    print("=" * 40)
    print(mod)


if __name__ == "__main__":
    main()
