# 7.2 数据并行：DDP 多进程训练示例
#
# 运行：
#   cd /data/ai_infra/06_分布式训练
#   python -m torchrun --nproc_per_node=2 6.2_数据并行/train_ddp.py
#
# 注：若只有单卡或 CPU 环境，可用 spawn 模拟多进程（不实际使用多 GPU）
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size):
    setup(rank, world_size)

    model = torch.nn.Linear(10, 1)
    model = DDP(model)

    dataset = TensorDataset(
        torch.randn(128, 10),
        torch.randn(128, 1),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(2):
        sampler.set_epoch(epoch)
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"rank {rank}, epoch {epoch}, loss {loss.item():.4f}")

    cleanup()


def main():
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
