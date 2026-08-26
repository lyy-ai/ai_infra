# 7.7 训练框架实战：PyTorch FSDP 示例
#
# 运行（需要多卡环境）：
#   cd /data/ai_infra/06_分布式训练
#   python -m torchrun --nproc_per_node=2 6.7_训练框架实战/fsdp_example.py
import os
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools


def get_simple_transformer():
    layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=4)


def main():
    torch.distributed.init_process_group("gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model = get_simple_transformer()
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={nn.TransformerEncoderLayer},
        ),
        device_id=torch.device("cpu"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    input = torch.randn(2, 128, 256)
    output = model(input)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    print(f"rank {torch.distributed.get_rank()} done")


if __name__ == "__main__":
    main()
