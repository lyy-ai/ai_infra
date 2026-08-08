# qat_training_flow.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class FakeQuantize(nn.Module):
    """简化版伪量化节点"""
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = 2 ** (num_bits - 1) - 1

    def forward(self, x):
        scale = x.abs().max() / (2 ** (self.num_bits - 1) - 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        x_int = torch.round(x / scale).clamp(self.qmin, self.qmax)
        x_deq = x_int * scale
        return x + (x_deq - x).detach()


class QATLinear(nn.Module):
    """带伪量化的线性层"""
    def __init__(self, in_features, out_features, num_bits=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.weight_fake_quant = FakeQuantize(num_bits)
        self.act_fake_quant = FakeQuantize(num_bits)

    def forward(self, x):
        w_q = self.weight_fake_quant(self.weight)
        x_q = self.act_fake_quant(x)
        return torch.nn.functional.linear(x_q, w_q, self.bias)


class TinyNet(nn.Module):
    def __init__(self, num_bits=8):
        super().__init__()
        self.fc1 = QATLinear(784, 128, num_bits)
        self.relu = nn.ReLU()
        self.fc2 = QATLinear(128, 10, num_bits)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_dataloader(batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('/tmp/mnist', train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx >= 30:  # 每个 epoch 只跑 30 batch，演示用
            break
    return total_loss / (batch_idx + 1)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            if total >= 1000:
                break
    return correct / total


def demo_qat_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = TinyNet(num_bits=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)

    print("Before QAT fine-tuning:")
    acc_before = evaluate(model, test_loader, device)
    print(f"  Test accuracy: {acc_before:.4f}")

    for epoch in range(3):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, test_acc={acc:.4f}")

    print("\nQAT training completed. Model is now adapted to 8-bit quantization.")


if __name__ == "__main__":
    demo_qat_training()
