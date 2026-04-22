# 🧠 Self-Pruning Neural Network

**Author:** Sonali Gupta  
**Date:** April 2026  
**Project:** Dynamic Architecture Optimization via Learnable Gating  

---

## 🚀 What is this project about?

Modern neural networks are **very large and overparameterized**, meaning they contain far more weights than actually needed.

This creates problems:
- High memory usage 📦  
- Slow inference ⚡  
- Difficult deployment on edge devices 📱  

---

### 💡 Solution: Let the network prune itself

Instead of manually removing weights **after training**, we design a system where:

> The network **learns during training which connections are unnecessary** and removes them automatically.

This is called a **Self-Pruning Neural Network**.

---

## 🧩 Core Idea (Intuition First)

Every connection (weight) in the network gets a **gate**:

- Gate ≈ 1 → keep the connection  
- Gate ≈ 0 → remove the connection  

So the model learns:
> “Which weights actually matter?”

---

## ⚙️ Technical Implementation

---

### 🔹 1. Prunable Layer

Each weight \( W \) is paired with a learnable parameter \( S \):

\[
g = \sigma(S)
\]

\[
W_{pruned} = W \cdot g
\]

Where:
- \( \sigma \) = sigmoid function  
- \( g \in (0,1) \)

---

### 🧠 Why use sigmoid?

- Smooth and differentiable  
- Allows gradient-based learning  
- Keeps values between 0 and 1  

---

### 🔹 2. Loss Function (Very Important)

We modify the standard loss:

\[
\text{Total Loss} = \text{Classification Loss} + \lambda \cdot \text{Sparsity Loss}
\]

---

### 🔹 3. Classification Loss

- Standard **CrossEntropy Loss**
- Ensures the model learns the task (image classification)

---

### 🔹 4. Sparsity Loss

\[
\text{Sparsity Loss} = \sum g_i
\]

---

### 🧠 Why this works

- This is an **L1 penalty on gates**
- It pushes gate values **towards zero**

---

### ⚠️ Important detail

Sigmoid never gives exact zero.

So:
- During training → values approach 0  
- During evaluation → we apply threshold (e.g., \(10^{-2}\))  

👉 This gives **effective sparsity**

---

## 🔄 Training Strategy

To avoid destroying learning early, we use:

---

### 🔹 Warm-up Phase

- First few epochs → **no sparsity penalty**
- Model learns basic features

---

### 🔹 Lambda Annealing

We gradually increase λ:

- Early training → focus on accuracy  
- Later training → focus on pruning  

---

### 🔹 Different Learning Rates

- Weights → slow learning (0.001)  
- Gates → fast learning (0.01)  

👉 This helps gates quickly adapt

---

## 📊 Results (MLP on CIFAR-10)

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|-----------|------------------|-------------|
| 1e-6      | 55.99            | 48.71       |
| 5e-6      | **56.88**        | 76.90       |
| 2e-5      | 56.09            | **91.06**   |

---

## 🔍 What do these results mean?

- Increasing λ → more pruning  
- Moderate λ → best balance  
- Very high sparsity → still works well  

👉 The model removes **~90% weights** without losing accuracy

---

## ✅ Validation: Is pruning actually meaningful?

We tested two scenarios:

### 🔹 Hard Pruning
- Removed all low-gate weights
- Accuracy stayed ~56%

---

### 🔹 Random Pruning
- Removed same number of weights randomly
- Accuracy dropped to ~10%

---

### 🧠 Conclusion

> The model learned **which weights matter**, not just removing randomly.

---

## 🔬 Extension: CNN Version

We applied the same idea to a **Convolutional Neural Network (CNN)**.

---

### 📊 CNN Results

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
|-----------|-------------|-------------|
| 2e-6      | 72.90       | 49.73       |
| 8e-6      | 71.39       | 66.61       |

---

## 🧠 Key Insight

- CNN → higher accuracy  
- MLP → higher sparsity  

---

### 💡 Why?

CNNs:
- reuse weights across space  
- already efficient  

MLPs:
- fully connected  
- more redundancy  

---

## 📈 Gate Distribution

After training:

- Many gates near **0** → pruned  
- Some gates near **1** → important  

👉 This creates a **bimodal distribution**

---

## 🏆 Key Takeaways

- Neural networks are highly redundant  
- L1 regularization effectively induces sparsity  
- Moderate pruning improves generalization  
- CNNs are more efficient but less prunable  

---

## 📌 Final Conclusion

This project shows that:

- Networks can **self-optimize their structure**
- Up to **90% of parameters can be removed**
- Performance can still be maintained  

---

## ⚡ Final Thought

> Neural networks can learn not just *how to predict*, but also *how to simplify themselves*.
