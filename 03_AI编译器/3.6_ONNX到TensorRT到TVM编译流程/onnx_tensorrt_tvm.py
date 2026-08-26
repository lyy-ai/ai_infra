#!/usr/bin/env python3
"""4.6 ONNX 到 TVM 编译流程：端到端示例（Relax 版）

运行：
    cd /data/ai_infra/03_AI编译器
    /data/qwen35_env/bin/python 3.6_ONNX到TensorRT到TVM编译流程/onnx_tensorrt_tvm.py

依赖：tvm, onnx（已在 qwen35_env 中安装。TensorRT 后端需额外安装 tensorrt 包与 NVIDIA 环境。）
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx


def create_sample_onnx() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    w = np.ones((4, 4), dtype=np.float32)
    w_init = helper.make_tensor("W", TensorProto.FLOAT, [4, 4], w.flatten().tolist())
    n1 = helper.make_node("MatMul", ["input", "W"], ["mm"])
    n2 = helper.make_node("Relu", ["mm"], ["output"])

    graph = helper.make_graph([n1, n2], "matmul_relu", [x], [y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def main():
    # 1. 加载 ONNX
    model = create_sample_onnx()

    # 2. 前端解析为 Relax
    mod = from_onnx(model, shape_dict={"input": [1, 4]}, dtype_dict="float32")

    # 3. 中端优化
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.FoldConstant()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.DeadCodeElimination()(mod)

    print("=== 优化后的 Relax IR ===")
    print(mod)

    # 4. 后端编译（LLVM，若环境支持 CUDA 可改为 target='cuda'）
    ex = relax.build(mod, target="llvm")
    ex.export_library("3.6_ONNX到TensorRT到TVM编译流程/model.so")
    print("\n编译产物：3.6_ONNX到TensorRT到TVM编译流程/model.so")

    # TensorRT 接入需要安装 NVIDIA TensorRT 并配置 relax 后端，此处为注释说明：
    # from tvm.relax.backend import tensorrt
    # mod = tensorrt.partition_for_tensorrt(mod)


if __name__ == "__main__":
    main()
