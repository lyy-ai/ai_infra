# 9.4 自研芯片算子适配：算子接口模板（Python 伪代码）
#
# 运行：
#   cd /data/liyangyang/ai_infra/08_多平台适配
#   /data/liyangyang/qwen35_env/bin/python 8.4_自研芯片算子适配/op_interface_template.py

import numpy as np
from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    FP32 = "float32"
    FP16 = "float16"
    INT8 = "int8"


class OpStatus(Enum):
    OK = 0
    INVALID_SHAPE = 1
    UNSUPPORTED_DTYPE = 2


@dataclass
class Tensor:
    data: np.ndarray
    dtype: DataType


def matmul_ppu(a: Tensor, b: Tensor, trans_a=False, trans_b=False) -> tuple[OpStatus, Tensor]:
    """
    自研芯片 MatMul 算子接口模板：
    1. 校验 shape/dtype
    2. 调用底层 kernel（此处用 numpy 模拟）
    3. 返回状态与结果
    """
    if a.dtype != b.dtype:
        return OpStatus.UNSUPPORTED_DTYPE, Tensor(None, None)
    A = a.data.T if trans_a else a.data
    B = b.data.T if trans_b else b.data
    if A.shape[-1] != B.shape[-2]:
        return OpStatus.INVALID_SHAPE, Tensor(None, None)
    out = np.matmul(A, B)
    return OpStatus.OK, Tensor(out, a.dtype)


def main():
    a = Tensor(np.random.randn(2, 3, 4).astype(np.float32), DataType.FP32)
    b = Tensor(np.random.randn(2, 4, 5).astype(np.float32), DataType.FP32)
    status, out = matmul_ppu(a, b)
    print(f"MatMul status: {status.name}")
    print(f"Output shape: {out.data.shape}")


if __name__ == "__main__":
    main()
