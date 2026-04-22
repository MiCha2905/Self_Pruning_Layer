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
# Prunable Linear Layer
# ==============================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
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
# Sparsity Loss
# ==============================
def sparsity_loss(model):
    return sum(layer.get_gates().abs().sum() for layer in model.prunable_layers())

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
# Train
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
# Evaluate
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
lambdas = {
    'Low': 1e-6,
    'Medium': 5e-6,
    'High': 2e-5
}

epochs = 10

history = {'acc': {}, 'sparsity': {}}
results = {}
trained_models = {}

# ==============================
# Training Loop
# ==============================
for name, lam in lambdas.items():
    print(f"\n🔥 {name} λ={lam}")

    model = Net().to(device)

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

# ==============================
# 📊 1. TRAINING DYNAMICS
# ==============================
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
for name in history['acc']:
    plt.plot(epochs_range, history['acc'][name], label=f"λ={lambdas[name]}")
plt.axhline(y=54, linestyle='--', color='gray', label='Baseline')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
for name in history['sparsity']:
    plt.plot(epochs_range, history['sparsity'][name], label=f"λ={lambdas[name]}")
plt.title("Sparsity Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Sparsity (%)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# ==============================
# 📊 2. GATE DISTRIBUTION
# ==============================
plt.figure(figsize=(15,4))

for i, name in enumerate(trained_models):
    model = trained_models[name]

    gates = torch.cat([
        l.get_gates().flatten().detach().cpu()
        for l in model.prunable_layers()
    ])

    plt.subplot(1,3,i+1)
    plt.hist(gates.numpy(), bins=50)
    plt.axvline(x=0.01, linestyle='--')

    acc, sp = results[name]
    plt.title(f"λ={lambdas[name]}\nAcc={acc:.1f}% | Spar={sp:.1f}%")
    plt.xlabel("Gate Value")

plt.tight_layout()
plt.show()

# ==============================
# 📊 3. ACCURACY vs SPARSITY
# ==============================
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
plt.title("Accuracy vs Sparsity Trade-off")
plt.grid()

plt.show()
