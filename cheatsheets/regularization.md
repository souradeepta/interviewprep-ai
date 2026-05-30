# Regularization Quick Reference

---

## Method Comparison Table

| Method | Formula / Mechanism | Effect on Weights | When to Use | Key Hyperparams |
|--------|--------------------|--------------------|-------------|-----------------|
| L1 (Lasso) | Loss + λΣ\|w_i\| | Pushes many weights to exactly 0 (sparse) | Feature selection; high-d with few relevant features | λ (alpha) |
| L2 (Ridge) | Loss + λΣw_i² | Shrinks all weights toward 0, none exactly 0 | Default for most models; correlated features | λ (alpha) |
| Elastic Net | Loss + λ_1Σ\|w_i\| + λ_2Σw_i² | Sparse + shrinkage; compromise | Highly correlated features + sparsity desired | λ_1, λ_2 (or λ, l1_ratio) |
| Dropout | Randomly zero out fraction p of activations | Prevents co-adaptation; implicit ensemble | Dense layers in deep nets; p=0.1–0.5 | p (drop probability) |
| DropConnect | Randomly zero out fraction of weights (not activations) | More aggressive than Dropout | Rarely used; slight improvement over Dropout in some tasks | p |
| Batch Normalization | Normalize activations per mini-batch; learn γ,β | Reduces internal covariate shift; implicit regularizer | After linear/conv layers in deep nets; very common | momentum, eps |
| Layer Normalization | Normalize per sample across features | Same as BN but stable at batch size 1 | Transformers, RNNs, wherever batch stats unreliable | eps |
| Weight Decay | L2 penalty applied to optimizer update | Same as L2 for SGD; decoupled from gradient in AdamW | Default in all modern training; AdamW decouples correctly | weight_decay coeff |
| Label Smoothing | Replace hard targets y∈{0,1} with y∈{ε/(K-1), 1-ε} | No effect on weights; reduces overconfidence | Classification with K classes; prevents probability saturation | ε (smoothing factor) |
| Data Augmentation | Apply random transforms to training examples | Implicit regularization via distribution expansion | Vision (flip, crop, color jitter), NLP (back-translate, synonym) | Transform type, magnitude |

---

## L1, L2, and Elastic Net (Detailed)

### L1 Regularization (Lasso)
```
Objective = Loss(y, ŷ) + λ Σ_i |w_i|

Gradient of penalty: sign(w_i)  ->  subgradient, not differentiable at w=0
Effect: many weights become exactly zero (feature selection)
```
Use L1 when: you believe only a small subset of features are relevant (sparse signal assumption).

### L2 Regularization (Ridge)
```
Objective = Loss(y, ŷ) + λ Σ_i w_i²

Gradient of penalty: 2λ · w_i  ->  proportional shrinkage
Effect: weights shrink toward zero but rarely reach exactly zero
```
Use L2 when: all features may contribute; prevents large weights; stabilizes learning.

### Elastic Net
```
Objective = Loss(y, ŷ) + λ_1 Σ|w_i| + λ_2 Σw_i²

In sklearn: alpha controls overall strength, l1_ratio ∈ [0,1] controls L1/L2 mix
```
Use when: correlated features (L1 picks one arbitrarily; Elastic Net keeps group).

### Weight Decay vs L2 — The Important Distinction
```
L2 regularization (vanilla): gradient of penalty is added to gradient before optimizer step
  g_t = ∇Loss + 2λw  ->  then Adam applies adaptive scaling to BOTH

Weight decay (decoupled, as in AdamW): w_t+1 = w_t(1 - α·λ) - α · Adam_update(g_t)
  Decay applied SEPARATELY from gradient; unaffected by adaptive LR scaling
```
For SGD: L2 == weight decay (mathematically equivalent).
For Adam: L2 != weight decay; use AdamW for correct behavior.

---

## Dropout and Variants

### Standard Dropout
```python
# During training: zero out each activation with probability p
h = activation(Wх + b)
h_dropped = h * Bernoulli(1-p) / (1-p)   # inverted dropout (scaling at train time)

# During inference: use all activations (no dropout)
h_inference = activation(Wх + b)
```
- p=0.1: light regularization (attention in transformers)
- p=0.2–0.3: standard (MLP hidden layers)
- p=0.5: aggressive (fully connected image classifiers)

### Spatial Dropout (for CNNs)
Drops entire feature maps (channels) rather than individual activations. Better for convolutional features with spatial correlation.

### Monte Carlo Dropout (at inference)
Run model K times with dropout ON at inference; mean = prediction, variance = uncertainty estimate. Used for Bayesian approximation.

---

## Normalization Methods

### Batch Normalization
```
mu_B = (1/m) Σ x_i                   (batch mean)
sigma_B² = (1/m) Σ (x_i - mu_B)²    (batch variance)
x_hat = (x - mu_B) / sqrt(sigma_B² + eps)
y = gamma * x_hat + beta              (learnable scale and shift)
```
- Placed after linear/conv layer, before activation (or after, depending on convention)
- Uses running mean/variance at inference
- Problem: unstable with batch size < 8; doesn't work for RNNs/transformers

### Layer Normalization
```
mu_l = (1/d) Σ x_j                   (mean across feature dim)
sigma_l² = (1/d) Σ (x_j - mu_l)²
x_hat = (x - mu_l) / sqrt(sigma_l² + eps)
y = gamma * x_hat + beta
```
- Normalizes across features (not batch); stable at batch size 1
- Standard in Transformers, RNNs, NLP

### When to Use Which
| Scenario | Normalization |
|----------|-------------|
| CNN (large batch) | Batch Norm |
| Transformer / NLP | Layer Norm |
| RNN / LSTM | Layer Norm |
| Graph NN | Instance Norm or Group Norm |
| Small batch (<8) | Group Norm or Layer Norm |
| Generative (GAN) | Instance Norm, Spectral Norm |

---

## Label Smoothing
```
Soft target for class c:
  y_smooth(c) = (1 - ε)  if c == true_class
                ε / (K-1) if c != true_class

Cross-entropy loss: L = -Σ_c y_smooth(c) · log p(c)
```
- ε = 0.0: hard targets (standard cross-entropy)
- ε = 0.1: standard label smoothing (ImageNet training, transformer training)
- Effect: prevents model from being overconfident; equivalent to adding KL divergence from uniform prior

---

## Data Augmentation as Regularization

| Domain | Common Augmentations | Libraries |
|--------|---------------------|-----------|
| Image | Random crop, horizontal flip, color jitter, rotation, mixup, cutout, CutMix | torchvision.transforms, albumentations |
| Text | Back-translation, synonym replacement, random deletion, EDA | nlpaug, TextAttack |
| Tabular | Gaussian noise injection, SMOTE (oversampling), feature dropout | imbalanced-learn |
| Audio | Time stretch, pitch shift, additive noise, SpecAugment | librosa, audiomentations |

### Mixup
```
x_mix = lambda * x_i + (1 - lambda) * x_j     (lambda ~ Beta(alpha, alpha))
y_mix = lambda * y_i + (1 - lambda) * y_j
```
Trains on convex combinations of training pairs; significantly improves generalization.

### CutMix
```
x_mix = M * x_i + (1 - M) * x_j               (M is a binary spatial mask)
y_mix = lambda * y_i + (1 - lambda) * y_j      (lambda proportional to mask area)
```
Cuts and pastes rectangular regions between images; stronger than Mixup for vision.

---

## Regularization Selection Guide

```
Overfitting with many features, want interpretability?
  -> L1 or Elastic Net

General purpose regularization for weights?
  -> L2 / Weight Decay (λ = 0.01 for AdamW, 1e-4 for SGD)

Deep network with large fully-connected layers?
  -> Dropout (p = 0.1–0.3) + Layer/Batch Norm

Transformer model?
  -> Dropout (p = 0.1) + Layer Norm + Label Smoothing (ε = 0.1)

Image classification?
  -> Data augmentation + Batch Norm + Dropout before final FC

Small dataset, complex model?
  -> Aggressive augmentation + L2 + Early stopping

Model outputs overconfident probabilities?
  -> Label smoothing (ε = 0.1) + Temperature scaling at inference
```
