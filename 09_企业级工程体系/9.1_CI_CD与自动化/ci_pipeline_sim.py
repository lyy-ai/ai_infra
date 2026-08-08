# 10.1 CI/CD 与自动化：流水线阶段模拟
#
# 运行：
#   cd /data/liyangyang/ai_infra/09_企业级工程体系
#   /data/liyangyang/qwen35_env/bin/python 9.1_CI_CD与自动化/ci_pipeline_sim.py

import random


def run_stage(name, failure_rate=0.0):
    ok = random.random() >= failure_rate
    print(f"[{name:12s}] {'PASS' if ok else 'FAIL'}")
    return ok


def run_pipeline():
    random.seed(42)
    stages = [
        ("lint", 0.0),
        ("unit_test", 0.0),
        ("build_engine", 0.05),
        ("accuracy_check", 0.05),
        ("benchmark", 0.05),
        ("e2e_test", 0.05),
        ("package", 0.0),
    ]
    print("=== CI/CD Pipeline ===")
    results = []
    for name, rate in stages:
        ok = run_stage(name, rate)
        results.append((name, ok))
        if not ok:
            print("Pipeline stopped due to failure.")
            break
    if all(ok for _, ok in results):
        print("All stages passed. Ready for release.")


if __name__ == "__main__":
    run_pipeline()
