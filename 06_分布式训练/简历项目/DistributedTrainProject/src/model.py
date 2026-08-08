# 小 GPT 模型 + 合成数据：单卡/DDP 共用
import torch
import torch.nn as nn


class SmallGPT(nn.Module):
    def __init__(self, vocab=16000, hidden=768, layers=12, heads=12, seq=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(seq, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, idx):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.emb(idx) + self.pos(pos)
        x = self.blocks(x)
        return self.head(self.ln(x))


def num_params(model):
    return sum(p.numel() for p in model.parameters())


def make_batch(batch, seq, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randint(0, vocab, (batch, seq), device=device)
    y = torch.randint(0, vocab, (batch, seq), device=device)
    return x, y
