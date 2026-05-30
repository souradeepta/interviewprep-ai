# Interview Formulas Quick Reference

All formulas a candidate needs to be able to write on a whiteboard.

---

## Loss Functions

### Regression Losses

| Loss | Formula | Use When |
|------|---------|---------|
| MSE (Mean Squared Error) | L = (1/n) Σ(y_i - ŷ_i)² | Symmetric errors, outliers acceptable |
| MAE (Mean Absolute Error) | L = (1/n) Σ\|y_i - ŷ_i\| | Robust to outliers, median regression |
| Huber Loss | L = { (1/2)(y-ŷ)² if \|y-ŷ\| ≤ δ ; δ(\|y-ŷ\| - δ/2) otherwise } | Best of both: smooth near zero, linear for outliers |

### Classification Losses

| Loss | Formula | Use When |
|------|---------|---------|
| Binary Cross-Entropy (BCE) | L = -(y log(p) + (1-y) log(1-p)) | Binary classification, logistic output |
| Categorical Cross-Entropy | L = -Σ_c y_c log(p_c) | Multi-class, softmax output |
| Focal Loss | L = -(1-p_t)^γ log(p_t) | Severe class imbalance (object detection) |

**Focal loss** adds (1-p_t)^γ weighting so easy examples (high p_t) contribute less; γ=2 is standard.

---

## Gradient Descent

### Basic Update Rule
```
θ ← θ - α · ∇_θ L(θ)
```
Where α is the learning rate and ∇_θ L is the gradient of loss w.r.t. parameters.

### SGD with Momentum
```
v_t = β · v_{t-1} + (1 - β) · ∇_θ L
θ ← θ - α · v_t
```
β = 0.9 typical; momentum smooths gradient noise.

### Adam Update Rule
```
m_t = β_1 · m_{t-1} + (1 - β_1) · g_t          # 1st moment (mean)
v_t = β_2 · v_{t-1} + (1 - β_2) · g_t²          # 2nd moment (variance)

m̂_t = m_t / (1 - β_1^t)                         # bias-corrected mean
v̂_t = v_t / (1 - β_2^t)                         # bias-corrected variance

θ ← θ - α · m̂_t / (√v̂_t + ε)
```
Defaults: β_1=0.9, β_2=0.999, ε=1e-8, α=1e-3

**AdamW** adds decoupled weight decay: θ ← θ(1 - αλ) - α · m̂_t / (√v̂_t + ε)
where λ is the weight decay coefficient (not applied through gradient).

---

## Attention Mechanism

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```
- Q ∈ R^{n×d_k}: queries
- K ∈ R^{m×d_k}: keys
- V ∈ R^{m×d_v}: values
- √d_k scaling prevents dot products from growing large (which saturates softmax)

### Multi-Head Attention
```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
```
h heads allow attending to different representation subspaces simultaneously.

---

## Evaluation Metrics

### Classification

| Metric | Formula | Notes |
|--------|---------|-------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Misleading for imbalanced classes |
| Precision | TP / (TP + FP) | Of predicted positives, how many are correct |
| Recall | TP / (TP + FN) | Of actual positives, how many found |
| F1 | 2 · (P · R) / (P + R) | Harmonic mean; use when P and R both matter |
| AUC-ROC | Area under TPR vs FPR curve | Threshold-invariant; 0.5 = random |

### Ranking Metrics

**NDCG@K (Normalized Discounted Cumulative Gain)**
```
DCG@K = Σ_{i=1}^{K} rel_i / log_2(i + 1)

IDCG@K = DCG@K for ideal ranking

NDCG@K = DCG@K / IDCG@K
```
Range [0, 1]; 1.0 is perfect ranking. Discount factor log_2(i+1) penalizes relevant items ranked low.

**MRR (Mean Reciprocal Rank)**
```
MRR = (1/|Q|) Σ_{q} 1/rank_q
```
Where rank_q is the position of the first relevant result for query q.

**MAP (Mean Average Precision)**
```
AP = (1/R) Σ_{k=1}^{n} P(k) · rel(k)
MAP = (1/|Q|) Σ_q AP(q)
```

---

## Statistics

### Confidence Interval (mean)
```
CI = x̄ ± z_{α/2} · (σ / √n)
```
z_{0.025} = 1.96 for 95% CI; replace σ with s (sample std) for t-distribution.

### Sample Size (for proportion)
```
n = z_{α/2}² · p(1-p) / e²
```
e = margin of error, p = expected proportion (use 0.5 for maximum conservatism).

### Cohen's d (effect size)
```
d = (μ_1 - μ_2) / σ_pooled
```
Small: d=0.2, Medium: d=0.5, Large: d=0.8

### Pooled Standard Deviation
```
σ_pooled = √( ((n_1-1)σ_1² + (n_2-1)σ_2²) / (n_1 + n_2 - 2) )
```

---

## Reinforcement Learning

### Bellman Equation (V-function)
```
V(s) = E[R(s, a) + γ · V(s')]
```
γ ∈ [0,1] is discount factor; s' is the next state.

### Q-Learning Update
```
Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') - Q(s, a)]
```
TD error = r + γ · max Q(s', a') - Q(s, a)

### Policy Gradient (REINFORCE)
```
∇_θ J(θ) = E_π [ ∇_θ log π_θ(a|s) · G_t ]
```
G_t = cumulative discounted return from time t.

### PPO Clip Objective
```
L^{CLIP}(θ) = E_t [ min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t) ]
```
r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)  (probability ratio)
A_t = advantage estimate; ε = 0.1 or 0.2 typical.

---

## Information Theory

### Entropy
```
H(X) = -Σ_x p(x) log_2 p(x)
```
Measures uncertainty; max for uniform distribution.

### Cross-Entropy
```
H(p, q) = -Σ_x p(x) log q(x)
```
Used as loss: p = true labels, q = model predictions.

### KL Divergence (relative entropy)
```
KL(p || q) = Σ_x p(x) log(p(x) / q(x))
```
Note: KL(p||q) ≠ KL(q||p); not a true distance; KL >= 0 always.

Relationship: H(p, q) = H(p) + KL(p || q)

---

## LoRA (Low-Rank Adaptation)

### Core Formula
```
W' = W_0 + ΔW = W_0 + A · B
```
- W_0 ∈ R^{d×k}: frozen pretrained weights
- A ∈ R^{d×r}: trainable low-rank matrix (init random)
- B ∈ R^{r×k}: trainable low-rank matrix (init zero, so ΔW=0 at start)
- r << d and r << k (rank is typically 4–32)

### Parameter Savings
```
Trainable params: r(d + k)  vs  full fine-tune: d · k
Savings ratio: 1 - r(d+k)/(dk)  ≈  1 - r/min(d,k)  for r << d,k
```
For d=k=4096 and r=8: 8·8192 = 65536 vs 16.8M params (99.6% reduction).

### Scaling Factor
```
ΔW_scaled = (α / r) · A · B
```
α is a hyperparameter (often set equal to r for unit scaling).
