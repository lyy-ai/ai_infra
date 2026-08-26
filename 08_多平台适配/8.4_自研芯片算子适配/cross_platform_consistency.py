# 9.4 自研芯片算子适配：跨平台一致性验证
#
# 运行：
#   cd /data/ai_infra/08_多平台适配
#   /data/qwen35_env/bin/python 8.4_自研芯片算子适配/cross_platform_consistency.py

import numpy as np


def compute_consistency(ref: np.ndarray, target: np.ndarray, fp16_tol=1e-3, fp32_tol=1e-5):
    diff = np.abs(ref - target)
    # 相对误差分母用 max(|ref|, tol) 避免接近 0 时数值爆炸
    rel = diff / np.maximum(np.abs(ref), fp16_tol)
    metrics = {
        "max_abs_diff": float(np.max(diff)),
        "max_relative_diff": float(np.max(rel)),
        "mean_abs_diff": float(np.mean(diff)),
        "cosine_similarity": float(np.dot(ref.flatten(), target.flatten()) /
                                   (np.linalg.norm(ref) * np.linalg.norm(target) + 1e-12)),
    }
    return metrics


def judge(metrics, tol_abs, tol_rel):
    return metrics["max_abs_diff"] <= tol_abs and metrics["max_relative_diff"] <= tol_rel


def main():
    np.random.seed(42)
    ref = np.random.randn(1000).astype(np.float32)
    # 模拟自研芯片引入的微小误差（远小于 FP16 容差）
    target = ref + np.random.randn(1000).astype(np.float32) * 1e-6

    metrics = compute_consistency(ref, target)
    print("=== Cross-platform consistency ===")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6e}")
    passed = judge(metrics, tol_abs=1e-3, tol_rel=1e-3)
    print(f"\nFP16 tolerance check: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
