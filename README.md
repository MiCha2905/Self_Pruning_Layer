# 🧠 Self-Pruning Neural Network

**Author:** Sonali Gupta  
**Date:** April 2026  
**Project:** Dynamic Architecture Optimization via Learnable Gating  

---

## 🚀 Project Overview

Modern neural networks are highly overparameterized, meaning they contain far more weights than necessary. This leads to:

- High memory usage 📦  
- Slow inference ⚡  
- Difficult deployment on edge devices 📱  

---

### 💡 Proposed Solution

This project implements a **Self-Pruning Neural Network**, where the model learns during training:

> Which weights are important and which can be removed.

Instead of pruning after training, the network:
- Starts dense  
- Learns meaningful features  
- Gradually removes unnecessary connections  

---

## 🧩 Core Idea

Each weight is paired with a learnable **gate**:

\[
g = \sigma(S)
\]

\[
W_{pruned} = W \cdot g
\]

- \( g \approx 1 \) → important connection  
- \( g \approx 0 \) → pruned connection  

---

## ⚙️ Technical Implementation

### 🔹 Prunable Layer

Custom layers (`PrunableLinear`, `PrunableConv2d`) were implemented where:

- Each weight has a corresponding gate  
- Gates are learned using backpropagation  
- Forward pass multiplies weights with gates  

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

- L1 penalty encourages gates to move **towards zero**  
- A threshold (e.g., \(10^{-2}\)) is used to define pruned weights  

---
## 📌 Required Analysis

### 🔹 Why L1 Penalty Encourages Sparsity

We optimize:

\[
\text{Loss} = \text{CrossEntropy} + \lambda \sum_i g_i
\]

- The term \( \sum g_i \) penalizes **all active gates**
- Minimizing it pushes gates **toward zero**
- However, reducing all gates harms accuracy

👉 The model learns a balance:

- Keep **important gates high**
- Push **unimportant gates toward zero**

**Key detail:**  
Sigmoid outputs lie in (0,1), so gates don’t reach exact zero.  
We apply a threshold (e.g., \(10^{-2}\)) to define sparsity.

---

### 🔹 Training Strategy

- **Warm-up phase:** No pruning initially  
- **Lambda annealing:** Gradually increases sparsity pressure  
- **Separate learning rates:**
  - Weights → 0.001  
  - Gates → 0.01  

---

## 📊 Results (MLP on CIFAR-10)

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
|-----------|-------------|-------------|
| 1e-6      | 55.99       | 48.71       |
| 5e-6      | **56.88**   | 76.90       |
| 2e-5      | 56.09       | **91.06**   |

---

## 🔬 CNN Extension

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
|-----------|-------------|-------------|
| 2e-6      | 72.90       | 49.73       |
| 8e-6      | 71.39       | 66.61       |

---

## 📊 Visualizations & Analysis

To understand pruning behavior, we analyze both **MLP and CNN models**.

---

# 🔹 MLP Visualizations

## 📈 Accuracy Over Epochs

<img width="752" height="517" alt="MLP_AccuracyOverEpochs" src="https://github.com/user-attachments/assets/e9f28aef-1e2b-4826-8785-81752466929e" />

- Accuracy steadily improves  
- Converges around ~56%  
- Pruning does not harm performance  



## 📉 Sparsity Over Epochs

<img width="750" height="522" alt="MLP_SparsityOverEpochs" src="https://github.com/user-attachments/assets/0a61be8e-fe3c-48a2-b76d-b9e4a62da246" />

- No pruning during warm-up  
- Sharp sparsity increase later  
- Higher λ → stronger pruning  

---

## 📊 Gate Value Distribution

<img width="1067" height="292" alt="MLP_Compare_Visualzation" src="https://github.com/user-attachments/assets/9b873095-4672-4ddf-82c3-8ca28c98c87a" />


- Spike near **0** → pruned weights  
- Small cluster → important weights  

👉 Confirms effective pruning  

---

# 🔹 CNN Visualizations

## 📈 Accuracy & Sparsity Over Epochs

<img width="996" height="347" alt="CNN_Sparsity AccuracyOverEpochs" src="https://github.com/user-attachments/assets/8a261dc5-bd1a-4b1f-8105-a1201619496e" />


- Faster convergence than MLP  
- Accuracy stabilizes around **72–73%**  
- Gradual sparsity growth  

---

## 📊 Gate Value Distribution

<img width="842" height="278" alt="CNN_compare" src="https://github.com/user-attachments/assets/c65769d4-6114-41c0-83cc-f4bbf5ceddf1" />



- Fewer weights near zero  
- Indicates lower redundancy  

---

# 🧠 Key Insights

### 🔸 Warm-up Phase
- Model learns before pruning begins  

---

### 🔸 Effect of λ
- Higher λ → more sparsity  
- Trade-off between compression and accuracy  

---

### 🔸 MLP vs CNN

| Model | Accuracy | Sparsity | Insight |
|------|---------|---------|--------|
| MLP  | ~56% | up to ~90% | Highly redundant |
| CNN  | ~73% | ~65% | More efficient |

---

### 🔸 Gate Distribution

- Bimodal pattern:
  - Near 0 → pruned  
  - Away from 0 → important  

---

## ✅ Validation of Sparsity

- Hard pruning → accuracy preserved (~56%)  
- Random pruning → accuracy drops (~10%)  

👉 Confirms meaningful pruning  

---

## 🏆 Conclusion

This project shows that:

- Neural networks can **self-prune during training**  
- Up to **90% of parameters can be removed**  
- Performance can be maintained  

---

## ⚡ Final Thought

> Neural networks can learn not only how to predict, but also how to optimize their own structure by removing unnecessary parameters.
