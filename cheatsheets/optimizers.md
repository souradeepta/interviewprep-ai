# Optimizers Quick Reference

## Optimizer Comparison Table

| Optimizer | Update Rule (summary) | Best For | Typical LR | Key Hyperparams |
|-----------|----------------------|---------|------------|-----------------|
| SGD | θ ← θ - α · g_t | Simple models, linear regression baseline | 0.01–0.1 | lr |
| SGD + Momentum | v_t = β·v_{t-1} + g_t; θ ← θ - α·v_t | CNNs with LR schedule, ResNets | 0.01–0.1 | lr, momentum β (0.9) |
| AdaGrad | θ ← θ - α/√(G_t + ε) · g_t | Sparse data (NLP bag-of-words, embeddings) | 0.01 | lr, epsilon |
| RMSProp | v_t = β·v_{t-1} + (1-β)·g_t²; θ ← θ - α/√(v_t+ε)·g_t | RNNs, non-stationary objectives | 1e-3 | lr, decay β (0.9) |
| Adam | Bias-corrected m and v; θ ← θ - α·m̂/(√v̂+ε) | Default for transformers, most deep learning | 1e-3 to 1e-4 | lr, β_1 (0.9), β_2 (0.999) |
| AdamW | Adam + decoupled weight decay | Fine-tuning pretrained models (BERT, GPT) | 1e-4 to 5e-5 | lr, weight_decay (0.01–0.1) |
| LAMB | Layer-wise adaptive LR + trust ratio | Large-batch training (batch 32K+), BERT pretraining | 1e-3 to 6e-3 | lr, β_1, β_2, weight_decay |

---

## Full Update Rules

### SGD (Stochastic Gradient Descent)
```
θ_t+1 = θ_t - α · ∇_θ L(θ_t)
```
- Noisy due to mini-batches; noise actually helps escape sharp minima
- Converges to flatter minima than Adam in many experiments

### SGD with Momentum
```
v_t = β · v_{t-1} + (1 - β) · ∇_θ L
θ_t+1 = θ_t - α · v_t
```
β = 0.9 standard; v acts as exponential moving average of gradients, damping oscillations.

### Nesterov Momentum (NAG)
```
θ_lookahead = θ_t - α · β · v_{t-1}
g_t = ∇_θ L(θ_lookahead)
v_t = β · v_{t-1} + (1 - β) · g_t
θ_t+1 = θ_t - α · v_t
```
Computes gradient at lookahead position; slightly faster convergence than standard momentum.

### AdaGrad
```
G_t = G_{t-1} + g_t²         (accumulated squared gradients)
θ_t+1 = θ_t - (α / √(G_t + ε)) · g_t
```
LR adapts per-parameter; parameters with large gradients get smaller updates. Problem: LR monotonically decreases to zero — use RMSProp to fix.

### RMSProp
```
v_t = β · v_{t-1} + (1 - β) · g_t²    (exponential moving average)
θ_t+1 = θ_t - (α / √(v_t + ε)) · g_t
```
β = 0.9, ε = 1e-8 typical. Fixes AdaGrad's shrinking LR by using EMA instead of cumulative sum.

### Adam
```
m_t = β_1 · m_{t-1} + (1 - β_1) · g_t       # mean (1st moment)
v_t = β_2 · v_{t-1} + (1 - β_2) · g_t²      # variance (2nd moment)

m̂_t = m_t / (1 - β_1^t)                     # bias correction
v̂_t = v_t / (1 - β_2^t)                     # bias correction

θ_t+1 = θ_t - α · m̂_t / (√v̂_t + ε)
```
Defaults: β_1=0.9, β_2=0.999, ε=1e-8, α=1e-3

Bias correction matters early in training when m and v are initialized at zero (cold start).

### AdamW (Adam with Decoupled Weight Decay)
```
# Weight decay applied BEFORE gradient update (decoupled)
θ_t+1 = θ_t · (1 - α · λ) - α · m̂_t / (√v̂_t + ε)
```
λ = weight decay coefficient (0.01–0.1); decoupling from gradient prevents L2 and Adam interacting incorrectly.
**Use AdamW over Adam for fine-tuning pretrained transformers.**

### LAMB (Layer-wise Adaptive Moments)
```
# Compute per-layer trust ratio
r_t = ||θ_t|| / ||m̂_t / (√v̂_t + ε) + λθ_t||

θ_t+1 = θ_t - α · r_t · (m̂_t / (√v̂_t + ε) + λθ_t)
```
Scales LR per-layer by trust ratio; enables large-batch training (32K+) with linear LR scaling.

---

## Learning Rate Schedule Comparison

| Schedule | Formula | Shape | Best For |
|----------|---------|-------|---------|
| Constant | α_t = α_0 | Flat | Baseline, debug runs |
| Step decay | α_t = α_0 · γ^(t // k) | Staircase drops | CNNs trained from scratch |
| Cosine decay | α_t = α_min + (α_0 - α_min) · (1 + cos(πt/T)) / 2 | Smooth S-curve to zero | Standard for transformers, fine-tuning |
| Warmup + Cosine | Linear warmup 0 → α_0 for W steps, then cosine | Ramp then decay | BERT pretraining, GPT fine-tuning |
| Cyclic LR | Oscillates between min and max LR | Triangle/saw waves | Can find flat minima faster, exploration |
| 1-Cycle | Linear ramp up, cosine down, brief anneal | Single cycle | FastAI's super-convergence, fast training |

### Warmup + Cosine Decay (Detailed)
```
For step t:
  if t < warmup_steps:
      α_t = α_max * (t / warmup_steps)
  else:
      progress = (t - warmup_steps) / (total_steps - warmup_steps)
      α_t = α_min + 0.5 * (α_max - α_min) * (1 + cos(π * progress))
```
Typical values: warmup_steps = 500–2000, α_min = 0 or 1e-6.

---

## Gradient Clipping

### Why: Prevents exploding gradients, especially in RNNs and transformers.

### Norm clipping (most common)
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
If ||∇|| > max_norm: scale gradients so ||∇|| = max_norm

### Value clipping
```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```
Clips each gradient value to [-clip_value, +clip_value].

### When to clip:
- Always in RNN/LSTM training (max_norm=1.0–5.0)
- Transformer training: max_norm=1.0 standard
- Skip when training small MLPs with stable losses

### Signs you need clipping:
- Loss suddenly spikes (NaN or very large value)
- Gradient norms logged and intermittently 100x normal

---

## Quick Optimizer Selection Guide

```
Starting a new project?
  -> Adam with lr=1e-3 (safe default for most architectures)

Fine-tuning a pretrained transformer?
  -> AdamW with lr=1e-4 to 5e-5, weight_decay=0.01

Training a ResNet-style CNN from scratch?
  -> SGD + Momentum (β=0.9) + Cosine schedule + warmup
  -> Often converges to better generalization than Adam

Training an RNN / LSTM?
  -> RMSProp or Adam, clip gradients at max_norm=1.0

Very large batch (>4K)?
  -> LAMB with linear LR scaling: lr = lr_base * batch_size / base_batch

Sparse model (embeddings, recommendation)?
  -> AdaGrad or SparseAdam (updates only non-zero embeddings)
```
