import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==============================
# Prunable Conv Layer
# ==============================
class PrunableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.gate_scores = nn.Parameter(torch.ones_like(self.conv.weight))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.conv2d(x, self.conv.weight * gates, self.conv.bias, padding=1)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

# ==============================
# Prunable Linear Layer
# ==============================
class PrunableLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.gate_scores = nn.Parameter(torch.ones(out_f, in_f))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

# ==============================
# CNN Model (FIXED SHAPES)
# ==============================
class PrunableCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PrunableConv2d(3, 32, 3)
        self.conv2 = PrunableConv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        # after 2 poolings: 32x32 → 16x16 → 8x8
        self.fc1 = PrunableLinear(64 * 8 * 8, 128)
        self.fc2 = PrunableLinear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def prunable_layers(self):
        return [self.conv1, self.conv2, self.fc1, self.fc2]

# ==============================
# Sparsity Loss (TRUE L1)
# ==============================
def sparsity_loss(model):
    total = 0
    for layer in model.prunable_layers():
        total += layer.get_gates().abs().sum()
    return total

# ==============================
# Metrics
# ==============================
def compute_sparsity(model, threshold=1e-2):
    gates = torch.cat([l.get_gates().flatten() for l in model.prunable_layers()])
    return (gates < threshold).float().mean().item() * 100

def avg_gate(model):
    gates = torch.cat([l.get_gates().flatten() for l in model.prunable_layers()])
    return gates.mean().item()

# ==============================
# Train (Warmup + Annealing)
# ==============================
def train(model, loader, optimizer, epoch, target_lambda, total_epochs):
    model.train()

    warmup = 3
    if epoch <= warmup:
        current_lambda = 0.0
    else:
        progress = (epoch - warmup) / (total_epochs - warmup)
        current_lambda = target_lambda * progress

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        ce = F.cross_entropy(out, y)
        sp = sparsity_loss(model)

        loss = ce + current_lambda * sp

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

# ==============================
# Evaluation
# ==============================
def evaluate(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()

    return 100 * correct / len(loader.dataset)

# ==============================
# Data
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# ==============================
# Experiments
# ==============================
lambdas = {'Low': 2e-6, 'Medium': 8e-6}
epochs = 10

for name, lam in lambdas.items():
    print(f"\n🔥 {name} λ={lam}")

    model = PrunableCNN().to(device)

    gate_params = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    other_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': gate_params, 'lr': 1e-2}
    ])

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, lam, epochs)

        acc = evaluate(model, test_loader)
        sp = compute_sparsity(model)
        avg = avg_gate(model)

        print(f"Epoch {epoch} | Acc: {acc:.2f}% | Sparsity: {sp:.2f}% | AvgGate: {avg:.4f}")

# ==============================
# Gate Distribution Plot
# ==============================
model.eval()
gates = torch.cat([l.get_gates().flatten().detach().cpu() for l in model.prunable_layers()])

plt.hist(gates.numpy(), bins=50)
plt.title("CNN Gate Distribution")
plt.show()