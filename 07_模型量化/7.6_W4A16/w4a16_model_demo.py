# w4a16_model_demo.py
import torch
import torch.nn as nn


class W4A16Layer(nn.Module):
    """简化的 W4A16 线性层，演示模型级集成"""

    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = (in_features + group_size - 1) // group_size
        padded = self.num_groups * group_size
        
        # 量化权重：用 uint8 存 int4，每 byte 存两个 4bit
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, padded // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            "scales",
            torch.ones(out_features, self.num_groups, dtype=torch.float16)
        )

    def forward(self, x):
        # 从 uint8 解出两个 int4
        q_low = self.qweight & 0xF
        q_high = (self.qweight >> 4) & 0xF
        q = torch.stack([q_low, q_high], dim=-1).reshape(
            self.out_features, self.num_groups, self.group_size
        )
        q = q[:, :, :self.in_features]
        
        # 反量化
        w_deq = (self.scales.unsqueeze(-1) * (q.to(torch.float16) - 7.0)).float()
        w_deq = w_deq.reshape(self.out_features, -1)[:, :self.in_features]
        
        return torch.matmul(x, w_deq.t())


class TinyW4A16Model(nn.Module):
    """一个微型 W4A16 MLP 模型"""

    def __init__(self, hidden_size=512, intermediate_size=1024, group_size=128):
        super().__init__()
        self.gate_proj = W4A16Layer(hidden_size, intermediate_size, group_size)
        self.up_proj = W4A16Layer(hidden_size, intermediate_size, group_size)
        self.down_proj = W4A16Layer(intermediate_size, hidden_size, group_size)
        self.act = nn.SiLU()

    def forward(self, x):
        # 在 CPU 上 SiLU 可能不支持 fp16，先转 fp32
        x_f = x.float()
        gate = self.act(self.gate_proj(x_f))
        up = self.up_proj(x_f)
        out = self.down_proj(gate * up)
        return out.half()


def demo_model():
    torch.manual_seed(0)
    model = TinyW4A16Model(hidden_size=256, intermediate_size=512, group_size=32).half()
    x = torch.randn(2, 10, 256, dtype=torch.float16)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (logical): {total_params}")


if __name__ == "__main__":
    demo_model()
