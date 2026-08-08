# 2.7 工程工具链：环境体检
#
# 运行：
#   /data/liyangyang/qwen35_env/bin/python 1.11_工程工具链/toolchain_check.py

import shutil


TOOLS = [
    ("gdb", "C++ 崩溃/core dump 调试"),
    ("perf", "CPU 热点分析"),
    ("nsys", "Nsight Systems：系统级 timeline"),
    ("ncu", "Nsight Compute：单 kernel 分析"),
    ("git", "版本协作 / bisect 回归定位"),
    ("nvidia-smi", "GPU 状态监控"),
    ("nvcc", "CUDA 编译器"),
]


def main():
    print("=== 工程工具链体检 ===\n")
    print(f"{'工具':<12} | {'状态':<6} | 用途")
    print("-" * 60)
    for name, desc in TOOLS:
        ok = shutil.which(name) is not None
        print(f"{name:<12} | {'OK' if ok else '缺失':<6} | {desc}")
    print()
    print("调试决策树：整体慢→nsys；单 kernel 慢→ncu；CPU 慢→perf；崩溃→gdb(+CUDA_LAUNCH_BLOCKING=1)")


if __name__ == "__main__":
    main()
