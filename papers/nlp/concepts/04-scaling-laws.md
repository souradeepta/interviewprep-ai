---
title: "Scaling Laws for Neural Language Models"
authors: "Kaplan, McCandlish, Henighan, et al."
year: 2020
venue: "arXiv"
doi: "https://doi.org/10.48550/arXiv.2001.08361"
arxiv: "https://arxiv.org/abs/2001.08361"
domain: "nlp"
difficulty: "intermediate"
interview_frequency: "high"
related_concepts:
  - llm/concepts/01-transformers
  - ml/concepts/30-learning-curves
  - ml/concepts/45-model-selection
---

# Scaling Laws for Neural Language Models

## Paper Overview

**Title:** Scaling Laws for Neural Language Models

**Authors:** Jared Kaplan, Sam McCandlish, Tom Henighan, et al. (Johns Hopkins, OpenAI)

**Published:** arXiv 2020 | [arXiv](https://arxiv.org/abs/2001.08361)

**Citation:** 5,000+ (fundamental for AI resource planning)

This paper discovered that model performance follows **power laws** with respect to model size, training data size, and compute budget. Loss scales predictably: Loss ∝ N^(-α) where N is parameters/data/compute and α ≈ 0.07-0.08. This finding is crucial because: (1) you can predict model performance before training (save compute), (2) you can allocate compute optimally (what's the best ratio of parameters:data?), (3) you can estimate how much compute you need for a target loss. The paper showed there's a compute-optimal ratio: roughly equal compute for parameters and data—don't overfitting by fixing one while scaling the other wildly. This informed the design of GPT-3, PaLM, Chinchilla, and every large model since.

**Why this matters for interviews:** You need to understand scaling laws to reason about model design. "Why is this model 7B parameters?" → because scaling laws show it's close to compute-optimal. Questions like "How much data/compute do you need for X accuracy?" require scaling law intuition.

---

## Core Contribution

### The Problem: Unpredictable Scaling

Before this paper, ML lacked principled ways to:
- Predict loss without training (required expensive experiments)
- Allocate compute optimally (was empirical trial-and-error)
- Estimate data/compute needs for a target performance

### The Solution: Empirical Power Laws

For large language models:

$$L(N) = a \cdot N^{-\alpha}$$

where:
- L = loss
- N = parameters (or data size, or compute)
- a = scaling constant (task-dependent)
- α ≈ 0.07-0.08 (surprisingly consistent)

**Interpretation:** 10x bigger model → ~1.58x better (46% improvement). Small but consistent gains at massive scale.

### Key Insight: Compute-Optimal Allocation

Training compute C should be allocated:
- ~20% to parameters
- ~80% to data (more epochs)

Or equivalently: roughly equal FLOPs in parameters and data (compute = 6ND, so D ≈ C/(6N)).

---

## Key Ideas & Algorithm

### Power Law Scaling for Different Factors

**Model Size Scaling:**
$$L(N) = a_N \cdot N^{-0.07}$$

Example: GPT-2 (1.5B) vs. GPT-3 (175B):
- 175B / 1.5B = 116× larger
- Expected improvement: 116^(0.07) = 1.76×
- Observed: ~1.7× empirically

**Data Scaling:**
$$L(D) = a_D \cdot D^{-0.09}$$

More data helps but slower than parameters.

**Compute Scaling (most relevant for budget allocation):**
$$L(C) = a_C \cdot C^{-0.08}$$

This relates to both parameters and data: compute C ≈ 6ND (6 FLOPs per multiply-add).

### Compute-Optimal Allocation Strategy

Given compute budget C, allocate:

**Option 1: Compute-optimal ratio (from paper)**
- Effective batch size: B ≈ 2000
- Parameters: N ≈ C / (6000 * B) = C / (12M)
- Tokens: D ≈ C / (6 * N) = 20 * N
- Data size: C / 6N

Example: C = 1 exaflop (10^18 FLOPs)
- N ≈ 83B parameters
- D ≈ 1.6T tokens

**Option 2: Chinchilla Scaling (refined version)**
- Equal FLOPs in model and data
- More aggressive data than Option 1
- Results in larger N relative to D

### Extrapolation and Prediction

Once you measure loss at a few scales, fit a power law and extrapolate:

```
Measure: L(10B) = 2.0, L(30B) = 1.8, L(100B) = 1.6
Fit: L ∝ N^(-0.08)
Predict: L(1T) = ?
Formula: L(1T) / L(100B) = (100B / 1T)^(0.08) ≈ 1.15
→ L(1T) ≈ 1.6 / 1.15 ≈ 1.39
```

---

## Architecture & Trade-offs

### Scaling Dimensions

| Dimension | Role | Trade-off |
|-----------|------|-----------|
| Model size (N) | Capacity | Larger → slower inference |
| Data (D) | Coverage | More → slower training |
| Compute (C) | Training speed | Limited by budget |
| Batch size (B) | Gradient noise | Smaller → noisier but less memory |

### Compute-Optimal vs. Other Strategies

| Strategy | Example | Pros | Cons |
|----------|---------|------|------|
| Compute-optimal | N∝D | Balanced, predictable | Medium model + medium data |
| Parameter-heavy | Larger N, less D | Few-shot improves | Underfitting on task |
| Data-heavy | Small N, more D | Lower final loss | Limited ceiling (model too small) |

### When to Deviate from Scaling Laws

Scaling laws assume:
- Standard transformer architecture
- Natural language tasks
- Standard training setup

Deviations:
- **Specialized models:** MoE (sparse models scale differently)
- **Downstream tasks:** Few-shot vs. fine-tuning scaling differ
- **Inference:** Inference-optimal allocation differs from training-optimal
- **Very small models:** Power laws break down (<100M parameters)

---

## Interview Q&A

**Q: What is the scaling law for language models? How do you use it to predict performance?**

A: Loss follows power law: L ∝ N^(-α) where α ≈ 0.07-0.08. Meaning: 10x bigger model → ~1.6x better (46% improvement). To predict: (1) measure loss at 2-3 scales (e.g., 10B, 30B, 100B), (2) fit power law: log(L) = log(a) - α*log(N), (3) extrapolate. Example: if L(100B)=1.6 and α=0.08, then L(1T) ≈ 1.6 * (100B/1T)^(0.08) ≈ 1.39. This saves compute: you can predict GPT-3's loss before training it, costing only ~1% of the full training.

**Q: How do you allocate compute optimally between model and data?**

A: Compute C ≈ 6ND, so allocate roughly equal FLOPs to parameters and data. Rule of thumb: if your model is N parameters, train on ~20N tokens (versus 2N tokens standard). Formally: N_opt ∝ C^(1/1.63) and D_opt ∝ C^(0.63/1.63), which gives D_opt ≈ 20*N_opt. Consequence: most models are undertrained! GPT-3 (175B) trained on 300B tokens, but scaling laws suggest 3.5T tokens optimal. Why was it undertrained? They prioritized few-shot learning (larger model helps more) over final loss.

**Q: What's the difference between compute-optimal, parameter-optimal, and data-optimal?**

A: (1) **Compute-optimal:** Minimize loss for fixed compute budget. Allocates roughly equally to N and D. (2) **Parameter-optimal:** Minimum parameters needed for a target loss. Often smaller N than compute-optimal. (3) **Data-optimal:** Minimum data needed for a target loss. Often more D than compute-optimal. In practice: compute-optimal is best if you have the compute budget. If you're constrained on parameters (inference cost), smaller N is okay but you'll need more data to hit the same loss.

**Q: Can you use scaling laws to predict few-shot performance or transfer learning?**

A: Scaling laws for pre-training loss don't directly predict downstream task performance. But empirically: models that are better on pre-training loss also do better few-shot (strong correlation). Why? Better pre-training → more general patterns learned → better few-shot. However, few-shot performance improves faster with scale than pre-training loss (cubed relationship empirically). So: use pre-training loss scaling laws to allocate compute, then expect bonus on few-shot and fine-tuning.

**Q: What happens if you scale only model size but not data?**

A: You overfit. Training loss improves, but test loss plateaus or diverges. The model memorizes training data. Scaling laws show: to maintain the same generalization gap, you need roughly equal scaling of N and D. If you double N without scaling D, your generalization gap worsens. In practice: compute-optimal allocations (N ≈ D) are empirically best.

---

## Best Practices

- **Estimate compute needed:** Use scaling laws to plan: want loss L_target? Estimate C_needed from power law L_target ∝ C^(-0.08). Then allocate N_opt and D_opt from that C.

- **Measure at multiple scales:** Don't trust a single data point. Measure loss at 3-5 scales, fit power law, extrapolate.

- **Account for overhead:** Training isn't perfectly efficient. Add 10-20% overhead for convergence, hyperparameter search, validation.

- **Allocate compute fairly:** Don't overweight parameters if you have the compute budget. Roughly equal allocation (N ≈ D) maximizes performance.

- **Re-evaluate for downstream tasks:** Scaling laws for pre-training loss are useful but don't predict all downstream tasks. Few-shot and fine-tuning scaling differ. Measure on your task.

- **Consider inference cost:** Compute-optimal training might not be inference-optimal. For deployment, smaller N (even if undertrained) might be preferable to save inference cost.

---

## Common Pitfalls

- **Mistake: Assuming linear scaling.** Scaling laws are power laws (diminishing returns), not linear. 2x parameters ≠ 2x better. Impact: over-optimistic estimates.
  → Fix: Use power law formula L ∝ N^(-0.08), not linear approximation.

- **Mistake: Extrapolating beyond data range.** Power laws fit in observed range (100M-175B) but might break down at extreme scales (10T). Impact: predictions outside training range are unreliable.
  → Fix: Only extrapolate 2-5x within training range. For larger extrapolation, add uncertainty bounds.

- **Mistake: Ignoring architecture effects.** Different architectures (RNN, CNN, Transformer, MoE) might have different scaling constants. Scaling laws derived from transformers.
  → Fix: Measure scaling on your specific architecture.

- **Mistake: Confusing training loss and task performance.** Pre-training loss follows power laws, but downstream task loss doesn't always (depends on task complexity). Impact: allocate N,D to minimize pre-training loss, but task performance plateau.
  → Fix: Measure on downstream task to validate. Use scaling laws for guidance, not gospel.

- **Mistake: Not accounting for convergence.** Reaching the scaling law requires good training setup (learning rate schedule, warmup, etc.). Poor training doesn't follow power laws.
  → Fix: Ensure training is properly tuned before extrapolating.

---

## Code Examples

### Example 1: Fit Power Law from Measurements

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Measured losses at different model sizes
model_sizes = np.array([125e6, 350e6, 1.3e9, 6.7e9, 13e9, 175e9])  # Parameters
losses = np.array([4.2, 3.9, 3.5, 2.9, 2.5, 2.1])  # Validation loss

# Power law: L = a * N^(-alpha)
def power_law(N, a, alpha):
    return a * np.power(N, -alpha)

# Fit power law
params, cov = curve_fit(power_law, model_sizes, losses, p0=[10, 0.08])
a, alpha = params
print(f"Power law: L = {a:.2f} * N^(-{alpha:.3f})")

# Predict for larger models
future_sizes = np.array([100e9, 500e9, 1e12])
predicted_losses = power_law(future_sizes, a, alpha)

for size, loss in zip(future_sizes, predicted_losses):
    print(f"Model {size/1e9:.0f}B params → predicted loss {loss:.2f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(model_sizes, losses, 'o-', label='Measured', markersize=10)
ax.loglog(future_sizes, predicted_losses, 's--', label='Predicted', markersize=8)
ax.set_xlabel('Model Size (parameters)')
ax.set_ylabel('Loss')
ax.set_title('Scaling Laws for Language Models')
ax.legend()
ax.grid()
plt.show()
```

### Example 2: Compute-Optimal Allocation

```python
def compute_optimal_allocation(compute_budget):
    """
    Given compute budget (FLOPs), allocate to parameters and data optimally.
    Formula: C = 6*N*D (6 FLOPs per token per parameter)
    Optimal: N ≈ D (equal allocation)
    """
    # Solve C = 6*N*D with N ≈ D
    # C = 6*N^2 → N = sqrt(C/6)
    
    N_optimal = np.sqrt(compute_budget / 6)
    D_optimal = compute_budget / (6 * N_optimal)
    
    return N_optimal, D_optimal

# Example: 1 exaflop compute budget
C = 1e18  # 1 exaflop = 10^18 FLOPs
N, D = compute_optimal_allocation(C)

print(f"Compute budget: {C:.2e} FLOPs")
print(f"Optimal model size: {N/1e9:.1f}B parameters")
print(f"Optimal tokens: {D/1e12:.1f}T tokens")
print(f"Verify: 6*N*D = {6*N*D:.2e} (should match compute budget)")
```

### Example 3: Scaling Law Prediction

```python
def predict_loss_at_scale(measured_loss, measured_size, target_size, alpha=0.08):
    """
    Given measured loss at one scale, predict at another scale.
    L(target) / L(measured) = (measured_size / target_size)^alpha
    """
    ratio = (measured_size / target_size) ** alpha
    predicted_loss = measured_loss * ratio
    return predicted_loss

# GPT-2 (1.5B) has loss ~3.5
# Predict GPT-3 (175B) loss
gpt2_loss = 3.5
gpt2_size = 1.5e9
gpt3_size = 175e9

gpt3_predicted_loss = predict_loss_at_scale(gpt2_loss, gpt2_size, gpt3_size, alpha=0.08)
gpt3_actual_loss = 2.1

print(f"GPT-2 (1.5B) loss: {gpt2_loss}")
print(f"GPT-3 (175B) predicted loss: {gpt3_predicted_loss:.2f}")
print(f"GPT-3 (175B) actual loss: {gpt3_actual_loss}")
print(f"Error: {abs(gpt3_predicted_loss - gpt3_actual_loss) / gpt3_actual_loss * 100:.1f}%")
```

---

## Related Concepts

- [Attention Is All You Need](./01-attention-is-all-you-need.md) — Architecture being scaled
- [Language Models are Few-Shot Learners (GPT-3)](./03-gpt3.md) — Application of scaling laws
- [LoRA: Low-Rank Adaptation](./05-lora.md) — Efficient scaling for fine-tuning
- [modern-ai/concepts/06-model-selection](../../modern-ai/concepts/06-model-selection.md) — Choosing model size and data

