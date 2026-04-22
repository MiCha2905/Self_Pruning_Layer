# 🧠 Self-Pruning Neural Network

**Author:** Sonali Gupta  
**Date:** April 2026  
**Project:** Dynamic Architecture Optimization via Learnable Gating  

---

## 🚀 Project Overview

Modern neural networks are highly overparameterized, making them inefficient for real-world deployment.

This project implements a **Self-Pruning Neural Network** that learns to remove its own unnecessary connections during training using **learnable gates and L1 regularization**.

Instead of pruning after training, the network:
- Starts dense  
- Learns important connections  
- Gradually becomes sparse  

---

## ⚙️ Technical Implementation

### 🔹 Prunable Layer

Each weight \( W \) is paired with a learnable gate:

\[
g = \sigma(S)
\]

\[
W_{pruned} = W \cdot g
\]

- \( g \in (0,1) \)
- \( g \to 0 \) → connection removed  
- \( g \to 1 \) → connection preserved  

---

### 🔹 Loss Function

\[
\text{Total Loss} = \text{CrossEntropy} + \lambda \cdot \text{Sparsity Loss}
\]

---

### 🔹 Sparsity Loss

\[
\text{Sparsity Loss} = \sum g_i
\]

- L1 penalty encourages gates to move **toward zero**
- A threshold \(10^{-2}\) is used to define pruned connections

---

### 🔹 Training Strategy

- **Warm-up (first 3 epochs):** no pruning  
- **Annealing:** gradually increase λ  
- **Separate learning rates:**
  - Weights → 0.001  
  - Gates → 0.01  

---

## 📊 Results (MLP on CIFAR-10)

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|-----------|------------------|-------------|
| 1e-6      | 55.99            | 48.71       |
| 5e-6      | **56.88**        | 76.90       |
| 2e-5      | 56.09            | **91.06**   |

---

### 🔍 Observations

- Increasing λ → higher sparsity  
- Moderate λ gives best performance  
- ~90% weights can be removed with minimal accuracy loss  

---

## ✅ Validation of Learned Sparsity

To verify pruning is meaningful:

- **Hard pruning:** accuracy ≈ 56% (no loss)  
- **Random pruning:** accuracy ≈ 10%  

👉 This confirms:
> The model learned **structured and meaningful sparsity**, not random weight removal.

---

## 🔬 Extension: Prunable CNN

The same approach was applied to a CNN.

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
|-----------|-------------|-------------|
| 2e-6      | 72.90       | 49.73       |
| 8e-6      | 71.39       | 66.61       |

---

### 🧠 Key Insight

- CNN → higher accuracy (~73%)  
- MLP → higher sparsity (~90%)  

👉 CNNs are more efficient due to **spatial feature sharing**, leading to less redundancy.

---

## 📈 Gate Distribution

The learned gate values show:

- Large spike near **0** → pruned weights  
- Cluster away from 0 → important weights  

👉 This confirms successful separation of useful vs redundant connections.

---

## 🏆 Key Takeaways

- Neural networks contain significant redundancy  
- L1 regularization effectively induces sparsity  
- Moderate pruning can improve generalization  
- CNNs are more robust but less aggressively prunable  

---

## 📌 Conclusion

This project demonstrates that:

- Networks can **self-prune during training**  
- Up to **90% of weights can be removed**  
- Performance can be maintained  

---

## ⚡ Final Statement

> Neural networks can learn not only how to perform tasks, but also how to optimize their own structure by eliminating unnecessary parameters.
