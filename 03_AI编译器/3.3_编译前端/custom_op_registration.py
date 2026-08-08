#!/usr/bin/env python3
"""4.3 编译前端：自定义算子注册示例（Relax 版）

运行：
    cd /data/liyangyang/ai_infra/03_AI编译器
    /data/liyangyang/qwen35_env/bin/python 3.3_编译前端/custom_op_registration.py

依赖：tvm（已在 qwen35_env 中安装）
"""
import tvm
from tvm import relax
from tvm.relax.frontend import nn


def register_custom_relu():
    """注册一个自定义的 'my_relu' 算子，底层通过 call_pure_packed 调用外部函数。"""

    def my_relu(x: nn.Tensor) -> nn.Tensor:
        return relax.call_pure_packed(
            "tvm.contrib.my_relu",
            x,
            sinfo_args=relax.TensorStructInfo(x.shape, x.dtype),
        )

    nn.add = my_relu  # 仅为演示：在 nn 模块上挂载，实际项目应走正式注册路径
    print("custom_op 'my_relu' registered via call_pure_packed")


def main():
    register_custom_relu()

    class Tiny(nn.Module):
        def forward(self, x: nn.Tensor):
            return nn.relu(x)

    model = Tiny()
    mod, _ = model.export_tvm(spec={"forward": {"x": nn.spec.Tensor([1, 4], "float32")}})
    print("\n导出的 Relax IR（自定义算子可在此阶段被替换为 call_pure_packed）:")
    print(mod)


if __name__ == "__main__":
    main()
