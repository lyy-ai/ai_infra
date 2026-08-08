# 10.1 CI/CD 与自动化：版本名称生成器
#
# 运行：
#   cd /data/liyangyang/ai_infra/09_企业级工程体系
#   /data/liyangyang/qwen35_env/bin/python 9.1_CI_CD与自动化/version_manager.py


def make_version(model_version, backend, precision, tp, pp, commit_short="a1b2c3d"):
    return f"model-v{model_version}-{backend}-{precision}-tp{tp}-pp{pp}-{commit_short}"


def main():
    versions = [
        make_version("2.1.0", "trt", "fp16", 1, 1),
        make_version("2.1.0", "trt", "fp16", 4, 1),
        make_version("2.2.0", "cann", "fp16", 8, 1),
        make_version("2.2.0", "custom", "int8", 2, 2),
    ]
    print("=== Generated versions ===")
    for v in versions:
        print(v)


if __name__ == "__main__":
    main()
