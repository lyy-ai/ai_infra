#!/usr/bin/env python3
"""4.7 Laser 编译框架：HPC Plugin 机制示例（Mock 版）

运行：
    cd /data/liyangyang/ai_infra/03_AI编译器
    /data/liyangyang/qwen35_env/bin/python 3.7_Laser编译框架与自动化编译/laser_hpc_plugin.py

说明：
    - 真实项目中的 `laser.hpc` 为内部包，这里用 mock 实现演示 plugin 注册与匹配逻辑。
    - qwen35_env 中未安装真实 laser 包，本脚本不依赖外部包即可运行。
"""
from typing import Any


class HpcNode:
    def __init__(self, op: str, dtype: str):
        self.op = op
        self.dtype = dtype

    def __repr__(self):
        return f"HpcNode(op={self.op}, dtype={self.dtype})"


class HpcBackend:
    def __init__(self, name: str):
        self.name = name

    def match(self, node: HpcNode) -> bool:
        raise NotImplementedError

    def codegen(self, node: HpcNode) -> str:
        raise NotImplementedError


class MyCutlassPlugin(HpcBackend):
    def __init__(self):
        super().__init__("cutlass")

    def match(self, node: HpcNode) -> bool:
        return node.op == "nn.matmul" and node.dtype == "float16"

    def codegen(self, node: HpcNode) -> str:
        return f"cutlass_gemm({node})"


PLUGINS: dict[str, HpcBackend] = {}


def register_hpc_plugin(name: str, plugin: HpcBackend):
    PLUGINS[name] = plugin


def dispatch(node: HpcNode) -> str | None:
    for name, plugin in PLUGINS.items():
        if plugin.match(node):
            return plugin.codegen(node)
    return None


def main():
    register_hpc_plugin("my_cutlass", MyCutlassPlugin())

    nodes = [
        HpcNode("nn.matmul", "float16"),
        HpcNode("nn.conv2d", "float16"),
        HpcNode("nn.matmul", "float32"),
    ]

    for node in nodes:
        code = dispatch(node)
        print(f"{node} -> {code if code else 'fallback to default codegen'}")


if __name__ == "__main__":
    main()
