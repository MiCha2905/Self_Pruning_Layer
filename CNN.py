# =========================================
# 📦 IMPORTS
# =========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================================
# ⚙️ DEVICE
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================
# 🧠 PRUNABLE CONV LAYER
# =========================================
class PrunableConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.gate_scores = nn.Parameter(torch.ones_like(self.weight))

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.conv2d(x, self.weight * gates, self.bias,
                        stride=self.stride, padding=self.padding)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

# =========================================
# 🧠 PRUNABLE LINEAR
# =========================================
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

# =========================================
# 🧠 CNN MODEL
# =========================================
class PrunableCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PrunableConv2d(3, 32, 3, padding=1)
        self.conv2 = PrunableConv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # After 2 pools: 32x32 → 8x8
        self.fc1 = PrunableLinear(64 * 8 * 8, 128)
        self.fc2 = PrunableLinear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def prunable_layers(self):
        return [self.conv1, self.conv2, self.fc1, self.fc2]

# =========================================
# 📉 SPARSITY LOSS
# =========================================
def sparsity_loss(model):
    return sum(layer.get_gates().abs().sum() for layer in model.prunable_layers())

# =========================================
# 📊 METRICS
# =========================================
def compute_sparsity(model, threshold=1e-2):
    gates = torch.cat([l.get_gates().flatten() for l in model.prunable_layers()])
    return (gates < threshold).float().mean().item() * 100

def avg_gate(model):
    gates = torch.cat([l.get_gates().flatten() for l in model.prunable_layers()])
    return gates.mean().item()

# =========================================
# 🏋️ TRAIN FUNCTION
# =========================================
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

# =========================================
# 🧪 EVALUATE
# =========================================
def evaluate(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()

    return 100 * correct / len(loader.dataset)

# =========================================
# 📦 DATA
# =========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# =========================================
# 🚀 EXPERIMENTS
# =========================================
lambdas = {'Low': 2e-6, 'Medium': 8e-6}
epochs = 10

history = {'acc': {}, 'sparsity': {}}
results = {}
trained_models = {}

for name, lam in lambdas.items():
    print(f"\n🔥 {name} λ={lam}")

    model = PrunableCNN().to(device)

    history['acc'][name] = []
    history['sparsity'][name] = []

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

        history['acc'][name].append(acc)
        history['sparsity'][name].append(sp)

        print(f"Epoch {epoch} | Acc: {acc:.2f}% | Sparsity: {sp:.2f}% | AvgGate: {avg:.4f}")

    results[name] = (acc, sp)
    trained_models[name] = model

# =========================================
# 📊 1. TRAINING DYNAMICS
# =========================================
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
for name in history['acc']:
    plt.plot(epochs_range, history['acc'][name], label=f"λ={lambdas[name]}")
plt.title("CNN Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
for name in history['sparsity']:
    plt.plot(epochs_range, history['sparsity'][name], label=f"λ={lambdas[name]}")
plt.title("CNN Sparsity Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Sparsity (%)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# =========================================
# 📊 2. GATE DISTRIBUTION
# =========================================
plt.figure(figsize=(12,4))

for i, name in enumerate(trained_models):
    model = trained_models[name]

    gates = torch.cat([
        l.get_gates().flatten().detach().cpu()
        for l in model.prunable_layers()
    ])

    plt.subplot(1,2,i+1)
    plt.hist(gates.numpy(), bins=50)
    plt.axvline(x=0.01, linestyle='--')

    acc, sp = results[name]
    plt.title(f"λ={lambdas[name]}\nAcc={acc:.1f}% | Spar={sp:.1f}%")
    plt.xlabel("Gate Value")

plt.tight_layout()
plt.show()

# =========================================
# 📊 3. ACCURACY vs SPARSITY
# =========================================
plt.figure(figsize=(6,5))

final_acc = []
final_sp = []

for name in results:
    acc, sp = results[name]
    final_acc.append(acc)
    final_sp.append(sp)

plt.scatter(final_sp, final_acc, s=120)

for i, name in enumerate(results):
    plt.text(final_sp[i], final_acc[i], f"λ={lambdas[name]}")

plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("CNN Accuracy vs Sparsity Trade-off")
plt.grid()

plt.show()
