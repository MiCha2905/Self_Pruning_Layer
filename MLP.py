

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==============================
# Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==============================
# Prunable Linear Layer (SOFT GATING)
# ==============================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))  # start open

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)  # SOFT gates only
        return F.linear(x, self.weight * gates, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

# ==============================
# MLP Model
# ==============================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def prunable_layers(self):
        return [self.fc1, self.fc2, self.fc3, self.fc4]

# ==============================
# TRUE L1 SPARSITY LOSS
# ==============================
def sparsity_loss(model):
    total = 0
    for layer in model.prunable_layers():
        total += layer.get_gates().abs().sum()
    return total

# ==============================
# METRICS
# ==============================
def compute_sparsity(model, threshold=1e-2):
    gates = torch.cat([l.get_gates().flatten() for l in model.prunable_layers()])
    return (gates < threshold).float().mean().item() * 100

def avg_gate(model):
    gates = torch.cat([l.get_gates().flatten() for l in model.prunable_layers()])
    return gates.mean().item()

# ==============================
# TRAIN FUNCTION (WARMUP + ANNEALING)
# ==============================
def train(model, loader, optimizer, epoch, target_lambda, total_epochs):
    model.train()

    warmup_epochs = 3
    if epoch <= warmup_epochs:
        current_lambda = 0.0
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        current_lambda = target_lambda * progress

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        ce_loss = F.cross_entropy(out, y)
        sp_loss = sparsity_loss(model)

        loss = ce_loss + current_lambda * sp_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

# ==============================
# EVALUATION
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
# DATA
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
# EXPERIMENTS
# ==============================
lambdas = {
    'Low': 1e-6,
    'Medium': 5e-6,
    'High': 2e-5
}
epochs = 10
results = {}

for name, target_lambda in lambdas.items():
    print(f"\n🔥 {name} Sparsity (λ={target_lambda})")

    model = Net().to(device)

    # Separate learning rates (IMPORTANT)
    gate_params = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    other_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': gate_params, 'lr': 1e-2}
    ])

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, target_lambda, epochs)

        acc = evaluate(model, test_loader)
        spar = compute_sparsity(model)
        avg = avg_gate(model)

        print(f"Epoch {epoch} | Acc: {acc:.2f}% | Sparsity: {spar:.2f}% | AvgGate: {avg:.4f}")

    results[name] = (acc, spar)

# ==============================
# FINAL RESULTS
# ==============================
print("\n📊 FINAL RESULTS")
for name, (acc, spar) in results.items():
    print(f"{name} λ={lambdas[name]} → Acc: {acc:.2f}% | Sparsity: {spar:.2f}%")

# ==============================
# PLOT GATE DISTRIBUTION
# ==============================
model.eval()
gates = torch.cat([l.get_gates().flatten().detach().cpu() for l in model.prunable_layers()])

plt.hist(gates.numpy(), bins=50)
plt.title("Gate Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.show()
