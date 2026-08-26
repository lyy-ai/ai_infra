# 10.2 性能回归平台：git bisect 辅助脚本
#
# 运行：
#   cd /data/ai_infra/09_企业级工程体系
#   /data/qwen35_env/bin/python 9.2_性能回归平台/git_bisect_helper.py


def print_bisect_command(good_commit, bad_commit, script_path="scripts/check_regression.py"):
    print("=== git bisect 性能回归定位命令 ===")
    print(f"git bisect start {bad_commit} {good_commit}")
    print(f"git bisect run python {script_path} --metric throughput --threshold 0.05")
    print("# 找到退化 commit 后：")
    print("git bisect reset")
    print("# 然后对退化 commit 前后做 profiling 对比")


def main():
    print_bisect_command(good_commit="v2.1.0", bad_commit="HEAD")


if __name__ == "__main__":
    main()
