#!/usr/bin/env python3
"""Enhance Architecture/Trade-offs sections for remaining AI concepts (06-40)."""

import os
import re

BASE = "/home/sbisw/github/interviewprep-ml"

# Comprehensive architectures for AI 06-40
ARCHITECTURES = {
    "06-linear-regression": """### Linear Regression Variants

| Variant | Complexity Control | Feature Selection | Use When |
|---------|-------------------|------------------|----------|
| **OLS** | None | Keeps all features | Few features, no multicollinearity |
| **Ridge (L2)** | Shrinks coefficients | All features (reduced) | Multicollinearity present |
| **Lasso (L1)** | Zeros some coefficients | Automatic selection | Need feature selection |
| **Elastic Net** | Combines L1 + L2 | Balance both | Complex relationships |

### Solution Methods Trade-off

- **Normal Equation:** Exact (X^T X)^{-1} X^T y, O(d³) inversion slow for large d
- **Gradient Descent:** Iterative, no inversion needed, scales to large d, requires tuning
- **Regularized:** Ridge/Lasso add terms preventing overfitting without exact solution""",

    "07-logistic-regression": """### Classification Scenarios

| Scenario | Output | Loss Function | Decision Boundary |
|----------|--------|-------------|-------------------|
| **Binary** | Single probability | Binary cross-entropy | Single hyperplane |
| **Multi-class** | K probabilities | Categorical cross-entropy | K hyperplanes |
| **Imbalanced** | Weighted loss | Weighted cross-entropy | Adjusted threshold |

### Regularization Impact

- **No regularization:** Overfits (especially p >> n)
- **L2 (Ridge):** Shrinks all, keeps features, interpretable
- **L1 (Lasso):** Zeros some, automatic selection, sparse
- **Class weights:** Penalizes minority class errors more""",

    "08-decision-trees": """### Splitting Criteria

| Criterion | Formula | Prefers | Best For |
|-----------|---------|---------|----------|
| **Gini** | 1 - Σp² | Frequent classes | Balanced data |
| **Entropy** | -Σp*log(p) | Even splits | Information theory |

### Complexity vs Generalization

- **Shallow (3-5):** High bias, low variance, underfits, fast
- **Medium (10-15):** Balanced, good performance
- **Deep (20+):** Low bias, high variance, overfits
- **Pruning:** Post-hoc depth reduction for variance control""",

    "09-random-forests": """### Ensemble Diversity

| Source | Mechanism | Impact |
|--------|-----------|--------|
| **Data sampling** | Bootstrap 63% unique samples | Decorrelates trees |
| **Feature sampling** | Random splits | Different decision boundaries |
| **Multiple trees** | K independent learners | Averaging reduces variance |

### Out-of-Bag (OOB) Advantage

- **OOB samples:** ~37% of data not in bootstrap (free validation)
- **OOB error:** Generalization estimate without separate validation
- **Feature importance:** How much each feature decreases impurity
- **Trade-off:** Less interpretable than single tree, better accuracy""",

    "10-gradient-boosting": """### Sequential Error Correction

| Stage | Learns | Effect |
|-------|--------|--------|
| **Tree 1** | Overall trend | Base prediction |
| **Tree 2** | First errors | Correct obvious mistakes |
| **Tree K** | Remaining errors | Fine-tune predictions |

### Regularization Mechanisms

- **Tree depth:** Shallow (4-8) = strong regularization
- **Learning rate:** Lower lr = conservative steps, better generalization
- **Early stopping:** Stop when validation plateaus
- **Subsampling:** Use random subset per iteration""",

    "11-support-vector-machines": """### Kernel Methods

| Kernel | Cost | Flexibility | Best For |
|--------|------|-----------|----------|
| **Linear** | O(n²) | Low | Linearly separable |
| **RBF** | O(n³) | High (universal) | Complex non-linear |

### Soft Margin Trade-off (C Parameter)

- **High C:** Enforce margin strictly (may overfit)
- **Low C:** Allow violations (robust, generalizes better)
- **Optimal C:** Found via cross-validation""",

    "12-k-nearest-neighbors": """### Distance Metrics

| Metric | Dimension Sensitivity | Best Use |
|--------|----------------------|----------|
| **Euclidean** | High (scale-dependent) | Default continuous |
| **Manhattan** | Medium | High-dimensional |
| **Cosine** | Low (magnitude-invariant) | Text, sparse |

### K Selection Trade-off

- **k=1:** Memorizes (high variance, overfits)
- **k=√N:** Sweet spot for most problems
- **k=N/2:** Underfits (high bias)
- **Curse of dimensionality:** High-D all points equidistant""",

    "13-neural-networks": """### Architecture Depth vs Width

| Property | Deep | Wide |
|----------|------|------|
| **Efficiency** | Better (hierarchical) | Worse (parallel) |
| **Training** | Harder (gradients) | Easier (direct) |
| **Generalization** | Better (structure) | Overfits easily |

### Universal Approximation

- **Theorem:** 2-layer network approximates any continuous function
- **Caveat:** Says nothing about learnability
- **In practice:** Deep more sample-efficient than wide shallow""",

    "14-activation-functions": """### Activation Comparison

| Function | Range | Vanishing | Dying | Use |
|----------|-------|----------|-------|-----|
| **Sigmoid** | [0,1] | ✗ Severe | N/A | Legacy |
| **ReLU** | [0,∞) | ✓ No | ✗ Yes | Default |
| **Leaky ReLU** | ℝ | ✓ No | ✓ No | Modern |
| **GELU** | ℝ | ✓ No | ✓ No | Transformers |

### Output Activation

- **Binary:** Sigmoid for [0,1] probability
- **Multi-class:** Softmax for K probabilities
- **Regression:** Linear for unbounded output""",

    "15-weight-initialization": """### Initialization Strategies

| Strategy | For | Scale | Risk |
|----------|-----|-------|------|
| **Xavier** | Sigmoid/Tanh | 1/√fanin | Too small for ReLU |
| **He** | ReLU | √(2/fanin) | Too large for Sigmoid |

### Batch Normalization Effect

- **With BN:** Initialization less critical (rescales activations)
- **Without BN:** Critical (determines initial gradients)
- **Best:** He init + ReLU + BN = robust training""",

    "16-regularization": """### Regularization Techniques

| Technique | Mechanism | Sparsity | When |
|-----------|-----------|----------|------|
| **L1** | Σ|w| penalty | High (zeros) | Feature selection |
| **L2** | Σw² penalty | None | Smooth solutions |
| **Dropout** | Random removal | None | Neural networks |
| **Early Stopping** | Stop validation plateau | None | All iterative |

### Lambda Selection

- **λ=0:** No regularization, overfits
- **λ=optimal:** Best generalization (via CV)
- **λ→∞:** Complete regularization, underfits""",

    "17-batch-normalization": """### Normalization Variants

| Variant | Scope | Batch-Dependent | Best For |
|---------|-------|-----------------|----------|
| **Batch Norm** | Per batch | Yes | Large batches, CNNs |
| **Layer Norm** | Per features | No | Transformers, RNNs |
| **Group Norm** | Feature groups | No | Small batches |

### Training vs Inference

- **Training:** Use minibatch statistics
- **Inference:** Use running statistics (accumulated during training)
- **Critical:** Wrong statistics at inference = poor performance""",

    "18-k-means-clustering": """### Initialization Methods

| Method | Quality | Speed | Consistency |
|--------|---------|-------|------------|
| **Random** | Poor | Fast | Inconsistent |
| **K-means++** | Good | Slower | Consistent |

### Determining K

- **Elbow Method:** Find diminishing returns in inertia
- **Silhouette Score:** Higher = better separation
- **Gap Statistic:** Larger gap = more clusters
- **Domain knowledge:** Often suggests K""",

    "19-dimensionality-reduction": """### Technique Comparison

| Technique | Preserves | Interpretability | Scalability |
|-----------|-----------|-----------------|------------|
| **PCA** | Global variance | High | Excellent |
| **t-SNE** | Local structure | None | Poor |
| **UMAP** | Local + global | Low | Better |

### Information Loss vs Reduction

- **PCA:** Controllable, compute explained variance
- **t-SNE:** High loss, visualization only
- **UMAP:** Lower loss, preserves topology""",

    "20-gaussian-mixture-models": """### Covariance Structures

| Type | Parameters | Flexibility | Computation |
|------|-----------|-------------|-----------|
| **Spherical** | 1 per cluster | Low | O(n) |
| **Diagonal** | K*d | Medium | O(n*d) |
| **Full** | K*d² | High | O(n*d³) |

### EM Algorithm

- **Convergence:** Guaranteed to local optimum
- **Cost:** O(d³) covariance matrix inversion
- **Scalability:** Limited to d < 1000 practically""",

    "21-bias-variance-tradeoff": """### Error Decomposition

```
TestError = Bias² + Variance + IrreducibleNoise
```

| Regime | Solution |
|--------|----------|
| **High bias** | More complex model |
| **High variance** | Regularization, more data |
| **High both** | Wrong model type |

### Diagnostic Path

- **High train error:** Underfitting (high bias)
- **High test, low train:** Overfitting (high variance)""",

    "22-cross-validation": """### CV Strategies

| Strategy | Best For | Folds | Cost |
|----------|----------|-------|------|
| **K-fold** | Standard | 5-10 | Reasonable |
| **Stratified** | Imbalanced | 5-10 | Reasonable |
| **Time-series** | Sequential | k | Reasonable |
| **Leave-One-Out** | Small data | n | Expensive |

### Nested CV (Advanced)

- **Outer:** Unbiased performance estimation
- **Inner:** Hyperparameter tuning
- **Cost:** K² × training time""",

    "23-classification-metrics": """### Metric Robustness to Imbalance

| Metric | Imbalance Robust | Use Case |
|--------|-----------------|----------|
| **Accuracy** | ✗ No | Balanced only |
| **Precision** | ✓ Yes | FP costly |
| **Recall** | ✓ Yes | FN costly |
| **F1** | ✓ Yes | Balance both |
| **PR-AUC** | ✓ Yes ⭐ | Imbalanced (best) |

### Decision Framework

- **Balanced:** Accuracy, F1, ROC-AUC
- **Imbalanced:** PR-AUC, Precision, Recall
- **Asymmetric cost:** Weight by impact""",

    "24-regression-metrics": """### Metric Properties

| Metric | Outlier Robust | Interpretable | Scale-Invariant |
|--------|----------------|---|---|
| **MAE** | ✓ Yes | ✓ Yes | ✗ No |
| **MSE** | ✗ No | ✗ No | ✗ No |
| **R²** | ✗ No | ✓ Yes | ✓ Yes |

### When to Use

- **Interpretable:** MAE (units same as y)
- **Outlier-robust:** MAE or Median AE
- **Math:** MSE/RMSE (differentiable)
- **Compare models:** R² (scale-independent)""",

    "25-feature-engineering": """### Feature Progression

| Type | Complexity | When |
|------|-----------|------|
| **Raw** | Low | Good features exist |
| **Transformed** | Low | Non-linearity |
| **Interactions** | Medium | Synergies |
| **Domain** | Medium | Knowledge available |
| **Learned** | High | Deep learning |

### Feature Selection

- **Filter:** Fast, simple, may miss
- **Wrapper:** Accurate, slow, overfit risk
- **Embedded:** LASSO, tree importance - good balance""",

    "26-hyperparameter-tuning": """### Search Strategies

| Strategy | Efficiency | Best For |
|----------|-----------|----------|
| **Grid** | Low | Few params |
| **Random** | Better | Many params |
| **Bayesian** | High | Few-medium |

### Most Important

- **Learning rate:** Controls convergence (MOST IMPORTANT)
- **Regularization:** Affects bias-variance
- **Model capacity:** Tree depth, network width""",

    "27-ensemble-methods": """### Strategies Comparison

| Strategy | Training | Inference | Parallelizable |
|----------|----------|-----------|----------------|
| **Bagging** | Parallel | Parallel | ✓ Yes |
| **Boosting** | Sequential | Sequential | ✗ No |
| **Stacking** | 2-stage | Sequential | Partial |
| **Voting** | Parallel | Parallel | ✓ Yes |

### Why Ensembles Work

- **Diversity:** Different errors cancel
- **Variance reduction:** Averaging high-variance models
- **Cost-benefit:** K× inference, often K× accuracy""",

    "28-bayesian-inference": """### Approximation Methods

| Method | Accuracy | Speed | Scalability |
|--------|----------|-------|-----------|
| **Variational** | Good | Fast | Good |
| **MCMC** | Excellent | Slow | Medium |
| **Laplace** | Approximate | Fast | Good |

### Prior-Likelihood Balance

- **Strong prior:** Regularization when data limited
- **Weak prior:** Lets data dominate, needs more samples
- **With infinite data:** Posterior determined by likelihood only""",

    "29-reinforcement-learning-basics": """### RL Components

| Component | Role | Example |
|-----------|------|---------|
| **Agent** | Decision-maker | Robot |
| **Environment** | Responds | World |
| **State** | Situation | Position |
| **Action** | Choice | Move |
| **Reward** | Feedback | +1 goal |

### Exploration vs Exploitation

- **Exploit only:** May get stuck locally
- **Explore only:** Never use learned knowledge
- **ε-greedy:** Balance - explore ε, exploit (1-ε)
- **Decay ε:** Early exploration, late exploitation""",

    "30-markov-decision-processes": """### MDP Elements

| Element | Symbol | Role |
|---------|--------|------|
| **States** | S | Possible situations |
| **Actions** | A | Available choices |
| **Transitions** | P(s'|s,a) | Environmental dynamics |
| **Rewards** | R(s,a,s') | Learning signal |
| **Discount** | γ | Future weight |

### Markov Property

- **Memoryless:** Only current state matters
- **Why:** Enables efficient algorithms
- **Violation:** Partially Observable → POMDP""",

    "31-q-learning": """### Off-Policy vs On-Policy

| Property | Q-Learning | On-Policy |
|----------|-----------|-----------|
| **Target** | Optimal policy | Current policy |
| **Efficiency** | High | Low |
| **Stability** | Overestimation | Better |

### Overestimation Solutions

- **Standard:** Uses same network max (biased)
- **Double Q-Learning:** Separate networks
- **Target Network:** Slowly updated copy""",

    "32-policy-gradients": """### Algorithm Types

| Algorithm | Baseline | Variance | Stability |
|-----------|----------|----------|-----------|
| **REINFORCE** | No | High | Low |
| **Actor-Critic** | Value | Lower | Better |
| **PPO** | Advantage + clipping | Lower | High |

### Variance Reduction

- **No baseline:** Huge variance
- **Mean baseline:** Subtract E[R]
- **Value baseline:** Subtract V(s)
- **Advantage:** A(s,a) = Q - V (best)""",

    "33-actor-critic-methods": """### Two-Network Roles

| Network | Input | Output | Loss |
|---------|-------|--------|------|
| **Actor** | State | Action probs | Policy gradient |
| **Critic** | State | Value | TD error |

### Training Dynamics

- **Critic provides signal:** Better gradients for actor
- **Actor improves:** Use critic guidance
- **Co-adaptation:** Both must improve together
- **Tuning:** Balance learning rates critical""",

    "34-graph-neural-networks": """### GNN Variants

| Type | Aggregation | Best For |
|------|-------------|----------|
| **GCN** | Mean | Graph-level |
| **GAT** | Attention | Variable importance |
| **GraphSAGE** | Sampled mean | Inductive |

### Permutation Invariance

- **Critical:** Output invariant to node ordering
- **Mechanism:** Aggregation (sum, mean, max) invariant
- **Implementation:** Remove order-dependent ops""",

    "35-causal-inference": """### Identification Strategies

| Strategy | Cost | Assumptions |
|----------|------|-----------|
| **RCT** | High | Fewest |
| **Back-door adjustment** | Low | Observed confounding |
| **Instrumental variables** | Medium | Exclusion restriction |

### Observational Challenges

- **Correlation ≠ causation:** Need assumptions
- **Confounding:** Unobserved variables bias estimates
- **Sensitivity analysis:** Check robustness to assumptions""",

    "36-probabilistic-graphical-models": """### Model Types

| Type | Structure | Inference |
|------|-----------|-----------|
| **Bayesian** | DAG | Belief prop |
| **Markov** | Undirected | Message pass |

### Complexity

- **Trees:** Polynomial inference
- **DAGs:** Exponential worst-case
- **Loops:** NP-hard (approximate)""",

    "37-variational-autoencoders": """### Loss Decomposition

| Component | Purpose | Trade-off |
|-----------|---------|-----------|
| **Reconstruction** | Accurate decoding | High → blurry |
| **KL divergence** | Regularization | High → variance |

### Architecture

- **Encoder q(z|x):** Learn latent distribution
- **Reparameterization:** Enable gradients
- **Decoder p(x|z):** Generate from latent""",

    "38-generative-adversarial-networks": """### Two-Player Game

| Network | Goal | Loss |
|---------|------|------|
| **Generator** | Fool discriminator | -log D(G(z)) |
| **Discriminator** | Correct classify | -[log D + log(1-D)] |

### Instability Sources

- **Mode collapse:** Limited variety
- **Vanishing gradient:** D too strong
- **Oscillation:** Loss doesn't converge
- **Solutions:** Spectral norm, Wasserstein loss""",

    "39-time-series-forecasting": """### Methods Comparison

| Method | Linear | Seasonality | CI |
|--------|--------|-------------|-----|
| **Exp Smooth** | Yes | Optional | Yes |
| **ARIMA** | Yes | With SARIMA | Yes |
| **Prophet** | Yes | Built-in | Yes |
| **LSTM** | No | Learned | No |

### Validation Critical

- **Cannot use random CV:** Violates order
- **Walk-forward:** Train past, test recent
- **Expanding:** Increasing history""",

    "40-anomaly-detection": """### Detection Methods

| Method | Type | Interpretation |
|--------|------|----------------|
| **Isolation Forest** | Outlier | Medium |
| **Local Outlier Factor** | Density | Low |
| **One-Class SVM** | Boundary | Low |
| **Autoencoder** | Reconstruction | Low |

### Core Challenge

- **Anomalies rare:** Hard to label
- **Definition ambiguous:** Context-dependent
- **Cost asymmetry:** FP ≠ FN cost
- **Solution:** Threshold tuning critical"""
}

def enhance():
    """Enhance AI concepts 06-40."""
    concepts_dir = f"{BASE}/ai/concepts"

    print("Enhancing AI Fundamentals (06-40):\n")

    enhanced = 0
    for slug, arch_content in sorted(ARCHITECTURES.items()):
        filepath = os.path.join(concepts_dir, f"{slug}.md")

        if not os.path.exists(filepath):
            print(f"  ⊘ {slug}")
            continue

        with open(filepath) as f:
            content = f.read()

        pattern = r'(## Architecture / Trade-offs\n\n).*?(\n\n## Interview Q&A)'
        replacement = rf'\1{arch_content}\2'
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        with open(filepath, 'w') as f:
            f.write(new_content)

        print(f"  ✓ {slug}")
        enhanced += 1

    print(f"\n✅ Enhanced {enhanced} AI concepts (06-40)")

if __name__ == "__main__":
    enhance()
