# 7.3 ZeRO 系列：各级 ZeRO 显存占用估算
#
# 运行：
#   cd /data/ai_infra/06_分布式训练
#   /data/qwen35_env/bin/python 6.3_ZeRO系列/zero_memory_estimator.py


def estimate_zero_memory(params_b, world_size, hidden, seq_len, batch_size, num_layers):
    P = params_b * 1e9 * 4 / 1e9  # FP32 params in GB
    G = P  # gradients FP32
    OS = 2 * P  # Adam m + v

    # activation: roughly 2 * batch * seq * hidden * layers * 4B (simplified)
    act = 2 * batch_size * seq_len * hidden * num_layers * 4 / 1e9

    ddp = P + G + OS + act

    z1 = P + G + OS / world_size + act
    z2 = P + G / world_size + OS / world_size + act
    z3 = P / world_size + G / world_size + OS / world_size + act

    return {
        "DDP": ddp,
        "ZeRO-1": z1,
        "ZeRO-2": z2,
        "ZeRO-3": z3,
        "activation": act,
    }


def main():
    params_b = 7
    world_size = 8
    hidden = 4096
    seq_len = 2048
    batch_size = 1
    num_layers = 32

    mem = estimate_zero_memory(params_b, world_size, hidden, seq_len, batch_size, num_layers)
    print(f"Model: {params_b}B, world_size: {world_size}")
    for k, v in mem.items():
        print(f"  {k:12s}: {v:8.2f} GB")


if __name__ == "__main__":
    main()
