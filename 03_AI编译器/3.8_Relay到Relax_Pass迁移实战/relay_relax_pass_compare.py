#!/usr/bin/env python3
"""4.8 Relay 到 Relax Pass 迁移实战：Relax Function Pass 注册示例

运行：
    cd /data/liyangyang/ai_infra/03_AI编译器
    /data/liyangyang/qwen35_env/bin/python 3.8_Relay到Relax_Pass迁移实战/relay_relax_pass_compare.py

依赖：tvm（已在 qwen35_env 中安装：apache-tvm 0.25.0，仅含 Relax）
"""
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax import transform as relax_T


@relax_T.function_pass(opt_level=1)
class PrintOpsPass:
    """示例 Relax Function Pass：打印函数内所有算子调用。"""

    def __init__(self):
        pass

    def transform_function(self, func, mod, ctx):
        print("\n处理 Relax 函数")
        # 简单遍历函数体，统计 Call 节点数量
        self._count = 0
        self._visit(func.body)
        print(f"  算子调用数量: {self._count}")
        return func

    def _visit(self, expr):
        """仅遍历 Relax 高层表达式，不进入 TIR。"""
        # 只处理 Relax Expr / Binding / Block
        if not isinstance(expr, (relax.Expr, relax.Binding, relax.BindingBlock, relax.DataflowBlock)):
            return

        if isinstance(expr, relax.Call):
            self._count += 1
            op_name = getattr(expr.op, "name", str(expr.op))
            print(f"    - {op_name}")
            for arg in expr.args:
                self._visit(arg)
            return

        if isinstance(expr, relax.SeqExpr):
            for block in expr.blocks:
                self._visit(block)
            self._visit(expr.body)
            return

        if isinstance(expr, (relax.DataflowBlock, relax.BindingBlock)):
            for binding in expr.bindings:
                self._visit(binding)
            return

        if isinstance(expr, relax.VarBinding):
            self._visit(expr.value)
            return

        if isinstance(expr, relax.Tuple):
            for field in expr.fields:
                self._visit(field)
            return

        if isinstance(expr, relax.TupleGetItem):
            self._visit(expr.tuple_value)
            return

        if isinstance(expr, relax.If):
            self._visit(expr.cond)
            self._visit(expr.true_branch)
            self._visit(expr.false_branch)
            return

        # 其他 leaf 表达式不递归


def main():
    class Tiny(nn.Module):
        def __init__(self):
            self.w = nn.Parameter((4, 4), "float32")

        def forward(self, x: nn.Tensor):
            return nn.relu(nn.matmul(x, self.w))

    model = Tiny()
    mod, _ = model.export_tvm(spec={"forward": {"x": nn.spec.Tensor([1, 4], "float32")}})
    mod = relax.transform.LegalizeOps()(mod)

    print("=" * 40)
    print("Relax Function Pass 注册示例")
    print("=" * 40)
    mod = PrintOpsPass()(mod)

    print("\n应用后的模块:")
    print(mod)


if __name__ == "__main__":
    main()
