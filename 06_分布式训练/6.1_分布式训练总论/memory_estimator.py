# 7.1 分布式训练总论：训练显存占用估算
#
# 运行：
#   cd /data/ai_infra/06_分布式训练
#   /data/qwen35_env/bin/python 6.1_分布式训练总论/memory_estimator.py


def estimate_training_memory(params_b, batch_size, seq_len, hidden,
                             optimizer="adam", precision="fp16"):
    """估算训练显存占用（单位 GB）"""
    P = params_b * 1e9
    bytes_per_param = 4 if precision == "fp32" else 2

    # 参数 + 梯度 + 优化器状态
    param_mem = P * bytes_per_param / 1e9
    grad_mem = P * 4 / 1e9  # 梯度通常 FP32
    opt_mem = P * (8 if optimizer == "adam" else 4) / 1e9

    # 激活值估算：每层大约 2 * batch * seq * hidden * layers
    # 简化为根据参数规模反推层数
    layers = max(1, int((P / (hidden * hidden * 12)) ** 0.5))
    act_mem = 2 * batch_size * seq_len * hidden * layers * 4 / 1e9

    total = param_mem + grad_mem + opt_mem + act_mem
    return {
        "params": param_mem,
        "gradients": grad_mem,
        "optimizer": opt_mem,
        "activations": act_mem,
        "total": total,
    }


def main():
    for params_b in [1, 7, 70, 175]:
        mem = estimate_training_memory(
            params_b=params_b,
            batch_size=1,
            seq_len=2048,
            hidden=4096,
            optimizer="adam",
            precision="fp16",
        )
        print(f"\nModel: {params_b}B parameters")
        for k, v in mem.items():
            print(f"  {k:12s}: {v:8.2f} GB")


if __name__ == "__main__":
    main()
