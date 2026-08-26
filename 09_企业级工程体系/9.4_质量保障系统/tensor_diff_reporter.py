# 10.4 质量保障系统：Tensor Diff 报告
#
# 运行：
#   cd /data/ai_infra/09_企业级工程体系
#   /data/qwen35_env/bin/python 9.4_质量保障系统/tensor_diff_reporter.py

import numpy as np


def tensor_diff_report(ref, target, layer_name="output", fp16_tol=1e-3):
    diff = np.abs(ref - target)
    rel = diff / np.maximum(np.abs(ref), fp16_tol)
    metrics = {
        "layer": layer_name,
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "max_relative_diff": float(np.max(rel)),
        "cosine_similarity": float(np.dot(ref.flatten(), target.flatten()) /
                                    (np.linalg.norm(ref) * np.linalg.norm(target) + 1e-12)),
    }
    metrics["passed"] = metrics["max_abs_diff"] <= fp16_tol and metrics["max_relative_diff"] <= fp16_tol
    return metrics


def main():
    np.random.seed(42)
    ref = np.random.randn(1000).astype(np.float32)
    # 模拟量化后微小误差（远小于 FP16 容差）
    target = ref + np.random.randn(1000).astype(np.float32) * 1e-6
    report = tensor_diff_report(ref, target)
    print("=== Tensor Diff Report ===")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.6e}")
        else:
            print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
