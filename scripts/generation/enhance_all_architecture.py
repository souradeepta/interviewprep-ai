#!/usr/bin/env python3
"""Enhance Architecture/Trade-offs sections across all 148 concepts."""

import os
import re

BASE = "/home/sbisw/github/interviewprep-ml"

# Architecture enhancements for AI concepts 01-40
AI_ARCHITECTURES = {
    "01-gradient-descent": {
        "subsections": [
            ("Gradient Descent Variants", """| Variant | Data Per Step | Update Frequency | Stability | Speed | Memory |
|---------|---------------|------------------|-----------|-------|--------|
| **Batch GD** | All N samples | Once per epoch | Very stable | Slow | High |
| **SGD** | 1 sample | N times per epoch | Noisy/unstable | Fast | Low |
| **Mini-batch** | 32-256 samples | N/batch times | Balanced | Fast | Medium |"""),
            ("Learning Rate Trade-offs", """- **High (0.1+):** Fast progress, oscillates, may diverge
- **Low (1e-5):** Stable, very slow convergence
- **Adaptive:** Per-parameter rates, better convergence
- **Scheduled:** Start high, decay over time (most practical)"""),
            ("Batch Size Effects", """- **Size=1:** Noisy gradients, escapes local minima, slow per-step
- **Size=N:** Clean gradient, computational efficiency issues, may get stuck
- **Optimal=32-256:** Balances gradient quality and hardware efficiency""")
        ],
        "diagram": """```mermaid
graph TD
    A["Training Data"] -->|Batch Split| B["Mini-batch"]
    B --> C["Compute Loss"]
    C --> D["Compute Gradient"]
    D --> E["Update Weights<br/>θ = θ - α*∇L"]
    E --> F{All batches<br/>processed?}
    F -->|No| B
    F -->|Yes| G{Converged?}
    G -->|No| A
    G -->|Yes| H["Final Weights"]
    style E fill:#fff3e0
```"""
    },

    "02-backpropagation": {
        "subsections": [
            ("Forward vs Backward Complexity", """| Direction | Operations | Memory | Cost |
|-----------|-----------|--------|------|
| **Forward** | Matrix multiplications | Store activations | O(n_params) |
| **Backward** | Chain rule gradients | Reuse activations | ~2x forward |
| **Total** | Forward + Backward | High (all activations) | 3x forward |"""),
            ("Gradient Flow Problems", """- **Vanishing Gradient:** Sigmoid layers cause exponentially small gradients
  - Solution: ReLU, skip connections, batch normalization
- **Exploding Gradient:** Large weights cause exponentially large gradients
  - Solution: Gradient clipping, careful initialization
- **Dead Neurons:** ReLU outputs 0 permanently
  - Solution: Leaky ReLU, ELU, proper initialization"""),
        ],
        "diagram": """```mermaid
graph LR
    A["Input x"] -->|Forward| B["Layer 1"]
    B -->|Forward| C["Layer 2"]
    C -->|Forward| D["Output ŷ"]
    D -->|Loss| E["L(ŷ,y)"]
    E -->|Backward| F["∇w2"]
    F -->|Backward| G["∇w1"]
    style A fill:#e1f5ff
    style D fill:#fff3e0
    style E fill:#ffebee
```"""
    },

    "03-loss-functions": {
        "subsections": [
            ("Loss Function Comparison", """| Loss | Use Case | Properties | Outlier Sensitivity |
|------|----------|-----------|---------------------|
| **MSE** | Regression | Differentiable, smooth | High (quadratic) |
| **MAE** | Robust regression | Piecewise linear | Low (linear) |
| **Cross-Entropy** | Classification | Info-theoretic | Medium |
| **Hinge** | SVM | Margin-based | Low |
| **Focal** | Imbalanced classification | Down-weights easy samples | Flexible |"""),
            ("Mathematical Properties", """- **Convexity:** MSE convex, Cross-entropy convex
- **Differentiability:** All common losses differentiable (mostly)
- **Gradient Magnitude:** MSE has larger gradients for large errors
- **Interpretability:** MAE direct in original units"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Regression Task"] -->|Use| B["MSE or MAE"]
    C["Binary Classification"] -->|Use| D["Cross-Entropy"]
    E["Multi-class"] -->|Use| F["Cross-Entropy + Softmax"]
    G["Imbalanced Data"] -->|Use| H["Focal Loss"]
    I["Margin-based"] -->|Use| J["Hinge Loss"]
```"""
    },

    "04-optimization-algorithms": {
        "subsections": [
            ("Optimizer Comparison", """| Optimizer | Learning Rate Sensitivity | Memory | Speed | Best For |
|-----------|--------------------------|--------|-------|----------|
| **SGD** | High (needs tuning) | Low | Medium | Well-behaved losses |
| **Momentum** | Medium (helps smooth) | Low | Fast | Accelerating convergence |
| **Adam** | Low (adaptive per param) | Medium | Fast | Most problems (default) |
| **RMSprop** | Low | Low | Fast | Sparse gradients |
| **AdamW** | Low | Medium | Fast | Deep learning + regularization |"""),
            ("Convergence Trade-offs", """- **SGD:** Simple, robust, needs careful tuning
- **Momentum:** Faster through plateaus, better generalization
- **Adaptive (Adam):** Less tuning, sometimes worse generalization
- **AdamW:** Adam + weight decay (modern best practice)"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Optimizer Choice"] -->|Simple| B["SGD"]
    A -->|Fast convergence| C["Momentum"]
    A -->|Low tuning| D["Adam/AdamW"]
    A -->|Sparse gradients| E["RMSprop"]
    B --> F["Requires LR scheduling"]
    C --> G["Good for deep nets"]
    D --> H["Default for most"]
```"""
    },

    "05-learning-rate-scheduling": {
        "subsections": [
            ("Schedule Types", """| Schedule | Formula | Best For | Trade-off |
|----------|---------|----------|-----------|
| **Constant** | lr = fixed | Baseline | May oscillate/plateau |
| **Step Decay** | lr *= gamma @ epoch N | Simple to implement | Discontinuous jumps |
| **Exponential** | lr *= exp(-k*epoch) | Smooth decay | Hard to tune k |
| **Cosine** | lr = 0.5*base*(1+cos(π*t/T)) | Smooth + principled | More computation |
| **Warmup** | lr linearly increase → decrease | Prevents instability | Extra hyperparameter |"""),
            ("When to Use Each", """- **Linear decay:** Fast early progress, slow fine-tuning
- **Step decay:** Sudden drops encourage escape from plateaus
- **Cosine annealing:** Mathematically principled, smooth transitions
- **Warmup:** Essential for Transformers (prevent gradient explosion)
- **Cyclical:** Multiple restarts help find diverse solutions"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Start High LR"] -->|Warmup Phase| B["Peak LR"]
    B -->|Training| C["Decay Phase"]
    C -->|End of epoch| D["Near-zero LR"]
    A -->|Goal| E["Fast initial progress<br/>Fine-tuning at end"]
```"""
    },

    "06-linear-regression": {
        "subsections": [
            ("Linear Regression Variants", """| Variant | Complexity Penalty | Feature Selection | Use When |
|---------|-------------------|------------------|----------|
| **OLS** | None | Keeps all | Few features, no overfitting |
| **Ridge (L2)** | λ*||w||² | Shrinks all | Multicollinearity present |
| **Lasso (L1)** | λ*||w||₁ | Zeros some | Need feature selection |
| **Elastic Net** | λ₁*||w||₁ + λ₂*||w||² | Shrink + select | Balance of both |"""),
            ("Closed-form vs Iterative", """- **Normal Equation:** (X^T X)^(-1) X^T y, requires matrix inversion (slow for large d)
- **Gradient Descent:** Iterative, no inversion needed, scales better
- **Trade-off:** Closed-form exact but numerically unstable; GD approximate but stable"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Linear Regression Problem"] -->|Few features| B["OLS"]
    A -->|Multicollinearity| C["Ridge"]
    A -->|Feature selection needed| D["Lasso"]
    A -->|Balance both| E["Elastic Net"]
    B -->|Solution| F["Normal Equation"]
    C -->|Solution| G["Regularized Normal Eq"]
    D -->|Solution| H["Coordinate Descent"]
```"""
    },

    "07-logistic-regression": {
        "subsections": [
            ("Binary vs Multi-class", """| Aspect | Binary Classification | Multi-class |
|--------|----------------------|--------------|
| **Output** | Single probability | K probabilities (softmax) |
| **Loss** | Binary cross-entropy | Categorical cross-entropy |
| **Decision Boundary** | Single hyperplane | K hyperplanes |
| **Interpretability** | High (coefficients = log-odds) | Medium (per-class coefficients) |"""),
            ("Regularization Effects", """- **No regularization:** May overfit, especially when p >> n
- **L2 (Ridge):** Shrinks all coefficients, keeps all features
- **L1 (Lasso):** Zeros some coefficients, automatic feature selection
- **Class imbalance:** Use class weights to penalize minority class errors more"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Binary Problem"] -->|Linear| B["Logistic Regression"]
    C["Multi-class Problem"] -->|Linear| D["Multinomial Logistic"]
    B -->|Sigmoid| E["σ(w·x+b)"]
    D -->|Softmax| F["exp(w·x)/Σexp"]
    E --> G["Probability p ∈ [0,1]"]
    F --> H["K probabilities"]
```"""
    },

    "08-decision-trees": {
        "subsections": [
            ("Splitting Criteria Comparison", """| Criterion | Formula | Tree Property | Best For |
|-----------|---------|----------------|----------|
| **Gini** | 1 - Σp²ᵢ | Prefers frequent classes | Balanced data |
| **Entropy** | -Σpᵢlog(pᵢ) | Information gain | Any distribution |
| **Chi-Square** | Σ(O-E)²/E | Statistical test | Mixed categorical |"""),
            ("Depth vs Bias-Variance", """- **Shallow (depth=3-5):** High bias, low variance, underfits, fast
- **Medium (depth=10-15):** Balanced, often best single tree
- **Deep (depth=20+):** Low bias, high variance, overfits, slow
- **Pruning:** Reduces depth post-hoc to reduce variance"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Root: Feature A > 0.5?"] -->|Yes| B["Feature B > 0.3?"]
    A -->|No| C["Leaf: Class 1"]
    B -->|Yes| D["Leaf: Class 2"]
    B -->|No| E["Leaf: Class 1"]
    style C fill:#e8f5e9
    style D fill:#e8f5e9
    style E fill:#e8f5e9
```"""
    },

    "09-random-forests": {
        "subsections": [
            ("Ensemble Properties", """| Property | Effect | Mechanism |
|----------|--------|-----------|
| **Diversity** | Reduces error | Different samples → different trees |
| **Independence** | Better averaging | Bootstrap sampling decorrelates |
| **Out-of-Bag (OOB)** | Free validation | Samples not in bootstrap ~37% |
| **Feature importance** | Interpretability | Sum impurity decrease per feature |"""),
            ("Bootstrap Aggregating (Bagging)", """- **Bootstrap:** Sample N with replacement from N samples (~63% unique)
- **Independence:** Different random subsets → diverse trees
- **Voting:** Average predictions (regression) or majority vote (classification)
- **Variance reduction:** Uncorrelated errors cancel out when averaged"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Original Data<br/>(N samples)"] -->|Bootstrap| B["Sample 1"]
    A -->|Bootstrap| C["Sample 2"]
    A -->|Bootstrap| D["... Sample K"]
    B -->|Train| E["Tree 1"]
    C -->|Train| F["Tree 2"]
    D -->|Train| G["Tree K"]
    E -->|Average| H["Ensemble Prediction"]
    F -->|Average| H
    G -->|Average| H
```"""
    },

    "10-gradient-boosting": {
        "subsections": [
            ("Sequential Learning", """| Stage | Input | Output | Goal |
|-------|-------|--------|------|
| **1** | Target y | Tree 1 | Fit target |
| **2** | Residuals y-ŷ₁ | Tree 2 | Fit errors |
| **3** | Residuals y-ŷ₂ | Tree 3 | Fit remaining errors |
| **K** | Residuals y-ŷₖ₋₁ | Tree K | Final corrections |"""),
            ("Regularization Strategies", """- **Tree depth:** Shallow trees (4-8) = regularization
- **Learning rate:** Lower lr = more conservative steps, more trees needed
- **Early stopping:** Stop when validation error plateaus
- **Subsampling:** Train on random subset of samples/features"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Initial Prediction<br/>F₀ = mean(y)"] --> B["Residuals<br/>r = y - F₀"]
    B --> C["Fit Tree 1<br/>to r"]
    C --> D["Update<br/>F₁ = F₀ + lr*Tree₁"]
    D --> E["New Residuals<br/>r = y - F₁"]
    E --> F["Fit Tree 2"]
    F --> G["F₂ = F₁ + lr*Tree₂"]
    G -->|Repeat| E
```"""
    },

    "11-support-vector-machines": {
        "subsections": [
            ("Kernel Methods", """| Kernel | Formula | Capacity | Best For |
|--------|---------|----------|----------|
| **Linear** | k(x,z) = x·z | Low | Linearly separable |
| **Polynomial** | k(x,z) = (x·z + c)ᵈ | Medium | Polynomial boundaries |
| **RBF** | k(x,z) = exp(-γ||x-z||²) | High | Non-linear, complex |
| **Sigmoid** | k(x,z) = tanh(αx·z + c) | Medium | Similar to neural net |"""),
            ("Soft Margin Trade-off", """- **C (penalty parameter):**
  - High C: Enforce margin strictly, may overfit
  - Low C: Allow violations, more robust
- **Margin vs Misclassification Trade-off:**
  - Large margin = safe but may allow more errors
  - Perfect classification = small margin, risky"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Linear Data"] -->|Linear Kernel| B["Linear SVM"]
    C["Non-linear Data"] -->|RBF Kernel| D["Non-linear SVM"]
    B -->|High C| E["Tight Margin<br/>Overfit Risk"]
    B -->|Low C| F["Loose Margin<br/>Generalize Better"]
```"""
    },

    "12-k-nearest-neighbors": {
        "subsections": [
            ("Distance Metrics", """| Metric | Formula | Use Case | Sensitivity |
|--------|---------|----------|-------------|
| **Euclidean** | √(Σ(xᵢ-zᵢ)²) | Default, continuous | Scale-dependent |
| **Manhattan** | Σ|xᵢ-zᵢ| | High-dimensional | Robust to outliers |
| **Cosine** | 1 - (x·z)/(||x||||z||) | Text, sparse | Magnitude-invariant |
| **Hamming** | # different positions | Categorical | Exact matches |"""),
            ("K Selection Trade-off", """- **k=1:** Low bias, high variance, overfits
- **k=N/2:** High bias, low variance, underfits
- **k=√N:** Sweet spot, needs tuning per problem
- **Curse of dimensionality:** In high dimensions, all points are far apart"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Query Point"] -->|k=1| B["1 Nearest<br/>High Variance"]
    A -->|k=5| C["5 Nearest<br/>Balanced"]
    A -->|k=N| D["All Points<br/>High Bias"]
    B --> E["Sensitive to<br/>outliers"]
    C --> F["Usually best<br/>performance"]
    D --> G["Predicts mean<br/>Always"]
```"""
    },

    "13-neural-networks": {
        "subsections": [
            ("Depth vs Width Trade-off", """| Aspect | Deep Networks | Wide Networks |
|--------|---------------|----------------|
| **Parameters** | Fewer (sharing) | More (independent) |
| **Expressivity** | Hierarchical features | Parallel features |
| **Trainability** | Harder (vanishing gradients) | Easier (direct gradients) |
| **Generalization** | Better (useful hierarchy) | May overfit |
| **Modern practice** | ResNets (skip connections) | Wide Transformers |"""),
            ("Universal Approximation", """- **Theorem:** 2-layer network with enough units can approximate any function
- **Catch:** Says nothing about learnability or efficiency
- **In practice:** Deep networks more parameter-efficient than wide shallow networks
- **Implication:** Depth enables learning rich hierarchical representations"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Input x"] -->|Layer 1| B["Hidden 1"]
    B -->|Layer 2| C["Hidden 2"]
    C -->|Layer 3| D["Output ŷ"]
    E["Shallow:<br/>Many parameters"] -.->|Inefficient| F["Function f"]
    G["Deep:<br/>Fewer parameters"] -.->|Efficient| F
```"""
    },

    "14-activation-functions": {
        "subsections": [
            ("Activation Comparison", """| Activation | Range | Derivative | Vanishing Gradient | Dying ReLU | Modern Use |
|------------|-------|-----------|-------------------|-----------|-----------|
| **Sigmoid** | [0,1] | max 0.25 | ✗ Yes (severe) | N/A | Legacy |
| **Tanh** | [-1,1] | max 1.0 | ✗ Yes | N/A | RNNs |
| **ReLU** | [0,∞) | 0 or 1 | ✓ No | ✗ Yes | Default |
| **Leaky ReLU** | ℝ | 0.01 or 1 | ✓ No | ✓ Fixed | Modern |
| **GELU** | ℝ | Smooth | ✓ No | ✓ No | Transformers |"""),
            ("Gradient Flow", """- **Sigmoid/Tanh:** Derivative saturates (→0) for extreme values
- **ReLU:** Derivative is constant (1 or 0), preserves gradient magnitude
- **Output activation:** Must match task (sigmoid for [0,1], softmax for probabilities, linear for ℝ)"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Choose Activation"] -->|Hidden layers| B["ReLU or Leaky ReLU"]
    A -->|Output: Binary| C["Sigmoid"]
    A -->|Output: Multi-class| D["Softmax"]
    A -->|Output: Regression| E["Linear"]
    A -->|Output: ∈ [-1,1]| F["Tanh"]
    B --> G["Fast, avoids vanishing"]
```"""
    },

    "15-weight-initialization": {
        "subsections": [
            ("Initialization Strategies", """| Strategy | Scale | When to Use | Risk |
|----------|-------|-----------|------|
| **Xavier/Glorot** | 1/√(fanᵢₙ) | Sigmoid/Tanh | Vanishing with ReLU |
| **He Initialization** | √(2/fanᵢₙ) | ReLU networks | Too large for Sigmoid |
| **Zero Initialization** | w=0 | Biases only | Neurons symmetric, can't learn |
| **Orthogonal** | Orthogonal matrix | Deep networks | Computationally expensive |
| **Small Random** | ~0.01 | Legacy | Slow learning |"""),
            ("Batch Normalization Effect", """- **With BN:** Initialization less critical (BN rescales activations)
- **Without BN:** Initialization critical (determines gradient flow)
- **Modern practice:** He init + ReLU + optional BN = robust training"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Initialize Weights"] -->|Sigmoid/Tanh| B["Xavier: 1/√fanin"]
    A -->|ReLU| C["He: √(2/fanin)"]
    B --> D["Mean 0, Var ~1"]
    C --> E["Larger scale<br/>for ReLU"]
    F["Batch Norm"] -->|Rescales| G["Initialization<br/>less critical"]
```"""
    },

    "16-regularization": {
        "subsections": [
            ("Regularization Techniques", """| Technique | Strength | Mechanism | Best For |
|-----------|----------|-----------|----------|
| **L1 (Lasso)** | λΣ\|w\| | Zeros weights | Feature selection |
| **L2 (Ridge)** | λΣw² | Shrinks weights | Smooth solutions |
| **Dropout** | p (fraction) | Random neuron removal | Neural networks |
| **Early Stopping** | patience epochs | Stop when validation plateaus | All models |
| **Data Augmentation** | Synthetic samples | Expand training set | Deep learning |"""),
            ("Lambda (Regularization Strength)", """- **λ=0:** No regularization, may overfit
- **λ=optimal:** Best generalization (found via CV)
- **λ=∞:** Complete regularization, underfits
- **Trade-off:** λ controls bias-variance trade-off directly"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Training Loss<br/>(Low)"] -->|Overfit| B["Bad<br/>Generalization"]
    C["Training Loss<br/>+ λ*Complexity"] -->|Balance| D["Good<br/>Generalization"]
    E["Increasing λ"] -->|More regularization| F["Lower variance,<br/>higher bias"]
```"""
    },

    "17-batch-normalization": {
        "subsections": [
            ("Normalization Variants", """| Variant | Normalization Axis | Batch Dependency | Best For |
|---------|-------------------|------------------|----------|
| **Batch Norm** | Across batch | Yes (during training) | Large batches, CNNs |
| **Layer Norm** | Across features | No | Transformers, RNNs |
| **Group Norm** | Across feature groups | No | Small batches, RNNs |
| **Instance Norm** | Per sample | No | Style transfer |"""),
            ("Training vs Inference", """- **Training:** Use minibatch statistics for normalization
- **Inference:** Use running statistics (computed during training)
- **Running statistics:** Exponential moving average to smooth estimates
- **Critical:** Wrong statistics at inference = poor performance"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Batch of Activations"] -->|Training| B["Normalize per<br/>Minibatch"]
    B -->|Compute| C["Running Statistics<br/>(exponential moving avg)"]
    A -->|Inference| D["Normalize per<br/>Running Statistics"]
    C -->|Used at| D
```"""
    },

    "18-k-means-clustering": {
        "subsections": [
            ("Initialization Strategies", """| Strategy | Quality | Speed | Convergence |
|----------|---------|-------|-------------|
| **Random** | Poor | Fast | Many iterations |
| **K-means++** | Good | Slightly slower | Fewer iterations |
| **Forgy** | Medium | Fast | Medium iterations |
| **Uniform** | Poor | Fast | Many iterations |"""),
            ("Choosing K", """- **Elbow method:** Plot inertia vs K, find 'elbow' point
- **Silhouette score:** Ranges [-1,1], higher = better separation
- **Gap statistic:** Compare to random data, larger gap = more clusters
- **Domain knowledge:** Often K is given or guided by problem context"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Inertia vs K"] -->|Elbow| B["Optimal K"]
    C["Silhouette Score"] -->|Peak| B
    D["Gap Statistic"] -->|Large Gap| B
    B -->|Gives| E["K Clusters"]
```"""
    },

    "19-dimensionality-reduction": {
        "subsections": [
            ("Reduction Techniques", """| Technique | Linear | Preserves | Best For |
|-----------|--------|-----------|----------|
| **PCA** | Yes | Global variance | Linear patterns, visualization |
| **t-SNE** | No | Local structure | Visualization only |
| **UMAP** | No | Local + global | Balance of both |
| **Autoencoders** | No (non-linear) | Feature relationships | Complex structures |"""),
            ("Information Loss", """- **PCA:** Controllable, can compute explained variance
- **t-SNE:** High loss, cannot extrapolate beyond training data
- **UMAP:** Lower loss than t-SNE, preserves more structure
- **Trade-off:** More reduction = more information loss"""),
        ],
        "diagram": """```mermaid
graph TD
    A["High-dimensional Data"] -->|Linear| B["PCA"]
    A -->|Non-linear | visualization| C["t-SNE"]
    A -->|Non-linear | balance| D["UMAP"]
    B -->|Interpretable| E["Components = feature combinations"]
    C -->|Visual only| F["No prediction on new data"]
    D -->|Balanced| G["Useful for exploration"]
```"""
    },

    "20-gaussian-mixture-models": {
        "subsections": [
            ("Covariance Assumptions", """| Type | Parameters | Flexibility | Complexity |
|------|-----------|-------------|-----------|
| **Spherical** | σ² per cluster | Low | Minimal |
| **Diagonal** | σ²ⱼ per feature per cluster | Medium | Low |
| **Full** | Σ (full covariance) | High | High |"""),
            ("EM Algorithm Trade-offs", """- **E-step:** Assign soft responsibilities (flexible)
- **M-step:** Update parameters (complex, may be intractable)
- **Convergence:** Guaranteed to local optimum, not global
- **Computational cost:** Scales with d³ (covariance matrix inversion)"""),
        ],
        "diagram": """```mermaid
graph TD
    A["E-step:<br/>Compute responsibilities<br/>P(cluster|data)"] -->|Soft assignment| B["Update Estimate"]
    B -->|M-step:<br/>Maximize likelihood<br/>with responsibilities| C["New Parameters"]
    C -->|Repeat until<br/>convergence| A
```"""
    },

    "21-bias-variance-tradeoff": {
        "subsections": [
            ("Error Decomposition", """```
Test Error = Bias² + Variance + Irreducible Noise

Bias = E[ŷ] - y  (systematic error)
Variance = E[(ŷ - E[ŷ])²]  (sensitivity to training data)
Noise = E[(y - E[y])²]  (inherent data randomness)
```

| Model Complexity | Bias | Variance | Total Error |
|-----------------|------|----------|-------------|
| **Underfitting** | High | Low | High |
| **Optimal** | Medium | Medium | Low ⭐ |
| **Overfitting** | Low | High | High |"""),
            ("Fixing High Error", """- **High training error:** Underfitting (high bias)
  - Solution: More complex model, better features, longer training
- **High test error, low training:** Overfitting (high variance)
  - Solution: Regularization, more data, simpler model
- **High both:** Noise ceiling or wrong model type"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Model Complexity"] -->|Low| B["High Bias<br/>Low Variance"]
    A -->|Medium| C["Balanced ⭐"]
    A -->|High| D["Low Bias<br/>High Variance"]
    B -->|Underfitting| E["Poor generalization"]
    C -->|Good| F["Good generalization"]
    D -->|Overfitting| E
```"""
    },

    "22-cross-validation": {
        "subsections": [
            ("CV Strategies", """| Strategy | Folds | Use When | Trade-off |
|----------|-------|----------|-----------|
| **K-fold** | K (~5-10) | Standard | Computational cost |
| **Stratified** | K with class balance | Imbalanced classes | Must group by class |
| **Time-series** | Sequential splits | Time-dependent data | No random shuffling |
| **GroupKFold** | K per group | Grouped samples | Respects dependencies |
| **Leave-One-Out** | N (all samples) | Very small data | Expensive (N trainings) |"""),
            ("Nested CV", """- **Outer CV:** For unbiased performance estimate
- **Inner CV:** For hyperparameter tuning
- **Benefit:** Prevents optimistic bias from hyperparameter overfitting
- **Cost:** K² × training time"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Data"] -->|Split K ways| B["Fold 1-K"]
    B -->|Train K-1| C["Train Set"]
    B -->|Test 1| D["Validation Set"]
    C -->|Average K scores| E["Performance Estimate"]
    E -->|Unbiased| F["Generalization"]
```"""
    },

    "23-classification-metrics": {
        "subsections": [
            ("Metrics Sensitivity", """| Metric | Class Imbalance Sensitive | Use Case | Range |
|--------|---------------------------|----------|-------|
| **Accuracy** | ✗ Yes (bad) | Balanced classes | [0,1] |
| **Precision** | ✓ Robust | False positive cost | [0,1] |
| **Recall** | ✓ Robust | False negative cost | [0,1] |
| **F1** | ✓ Robust | Balance both | [0,1] |
| **ROC-AUC** | ✗ Somewhat | Ranking quality | [0,1] |
| **PR-AUC** | ✓ Yes | Imbalanced data | [0,1] |"""),
            ("Choosing Metrics", """- **Balanced classes:** Accuracy, F1, ROC-AUC
- **Imbalanced classes:** Precision, Recall, F1, PR-AUC
- **High false positive cost:** Precision
- **High false negative cost:** Recall
- **Threshold tuning:** ROC-AUC or PR-curve"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Classification Problem"] -->|Balanced| B["Use Accuracy or F1"]
    A -->|Imbalanced| C["Use Precision-Recall"]
    A -->|Cost asymmetry| D["Weight metrics<br/>by business cost"]
    B --> E["Standard metrics"]
    C --> F["Imbalance-robust"]
    D --> G["Custom thresholds"]
```"""
    },

    "24-regression-metrics": {
        "subsections": [
            ("Metrics Trade-offs", """| Metric | Outlier Sensitive | Interpretation | Scale |
|--------|------------------|-----------------|-------|
| **MAE** | ✓ Robust | Avg absolute error | Same as y |
| **MSE** | ✗ Sensitive | Avg squared error | y² units |
| **RMSE** | ✗ Sensitive | Squared root of MSE | Same as y |
| **R²** | ✗ Sensitive | Fraction variance explained | [0,1] |
| **MAPE** | Medium | Percent error | % |"""),
            ("When to Use Each", """- **Interpretability:** MAE (same units as target)
- **Outlier robustness:** MAE, median AE
- **Mathematical convenience:** MSE, RMSE (differentiable)
- **Comparing datasets:** R² (scale-independent)
- **Relative error:** MAPE (when % error matters)"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Regression Task"] -->|Interpretability| B["Use MAE"]
    A -->|Outliers present| C["Use MAE or MedianAE"]
    A -->|Standard practice| D["Report MAE + RMSE"]
    A -->|Compare models| E["Use R²"]
    A -->|Relative errors| F["Use MAPE"]
```"""
    },

    "25-feature-engineering": {
        "subsections": [
            ("Feature Types", """| Type | Examples | Complexity | Use When |
|------|----------|-----------|----------|
| **Raw** | Original features | Low | Few, good features |
| **Transformed** | Logarithm, sqrt, Box-Cox | Low-Medium | Non-linear relationships |
| **Interactions** | x₁*x₂, x₁²x₂ | Medium | Synergies between features |
| **Domain** | Lag features, rolling stats | Medium-High | Domain knowledge critical |
| **Learned** | Neural network embeddings | High | Deep learning models |"""),
            ("Feature Selection", """- **Filter methods:** Independent of model (fast, simple)
- **Wrapper methods:** Use model performance (slow, accurate)
- **Embedded methods:** Built into model training (LASSO, tree importance)
- **Trade-off:** More features = more capacity but overfitting risk"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Raw Features"] -->|Linear| B["Use raw features"]
    A -->|Non-linear| C["Transform (log, sqrt)"]
    A -->|Synergies| D["Create interactions"]
    C -->|Select important| E["Remove low-variance"]
    D -->|Avoid multicollinearity| E
```"""
    },

    "26-hyperparameter-tuning": {
        "subsections": [
            ("Search Strategies", """| Strategy | Search Space | Efficiency | Cost |
|----------|--------------|-----------|------|
| **Grid Search** | Cartesian product | Low (exhaustive) | High |
| **Random Search** | Uniform random | Better | Medium |
| **Bayesian Optimization** | Model-guided | High (smart) | Medium |
| **Population-based** | Evolutionary | Medium-High | High |
| **Successive Halving** | Bracket elimination | High | Medium |"""),
            ("Important Hyperparameters", """- **Learning rate:** Most important, affects convergence speed
- **Regularization:** Controls overfitting, affects generalization
- **Model capacity:** Tree depth, network width, etc.
- **Data sampling:** Batch size, validation split, class weights
- **Order of importance:** Depends on algorithm type"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Define search space"] -->|Few params| B["Grid Search"]
    A -->|Many params| C["Random Search"]
    A -->|Complex| D["Bayesian Optimization"]
    B -->|Try all| E["Best hyperparameters"]
    C -->|Sample smart| E
    D -->|Model performance| E
```"""
    },

    "27-ensemble-methods": {
        "subsections": [
            ("Ensemble Strategies", """| Strategy | Independence | Diversity | Training |
|----------|--------------|-----------|----------|
| **Bagging** | Parallel | Data sampling | Parallel |
| **Boosting** | Sequential | Error focus | Sequential |
| **Stacking** | Parallel | Model diversity | 2-stage |
| **Voting** | Parallel | Any models | Parallel |
| **Blending** | Parallel | Holdout validation | 2-stage |"""),
            ("Why Ensembles Work", """- **Diversity:** Different errors cancel when averaged
- **Bias-variance:** Combining high-variance models reduces variance
- **Parallel:** Bagging/Voting easily parallelizable
- **Sequential:** Boosting focuses on hard examples
- **Cost:** K× inference, usually K× accuracy improvement"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Ensemble"] -->|Parallel| B["Bagging/Voting"]
    A -->|Sequential| C["Boosting"]
    A -->|Meta-learning| D["Stacking"]
    B -->|Reduces| E["Variance"]
    C -->|Reduces| F["Bias"]
    D -->|Learns| G["Which model best"]
```"""
    },

    "28-bayesian-inference": {
        "subsections": [
            ("Approximation Methods", """| Method | Accuracy | Speed | Scalability |
|--------|----------|-------|-------------|
| **Exact** | Perfect | Intractable | Impossible |
| **Variational** | Good | Fast | Good |
| **MCMC** | Excellent | Slow | Medium |
| **Laplace** | Medium | Fast | Good |
| **ABC** | Approximate | Medium | Medium |"""),
            ("Prior vs Likelihood vs Posterior", """- **Prior P(θ):** Domain knowledge (weak or strong)
- **Likelihood P(data|θ):** Data evidence
- **Posterior P(θ|data):** Combines both
- **Trade-off:** Strong prior reduces flexibility, weak prior needs more data"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Prior<br/>P(θ)"] -->|×| B["Likelihood<br/>P(data|θ)"]
    B -->|∝| C["Posterior<br/>P(θ|data)"]
    D["Strong Prior"] -->|↑ Influence| C
    E["More Data"] -->|↑ Influence| C
```"""
    },

    "29-reinforcement-learning-basics": {
        "subsections": [
            ("RL Components", """| Component | Role | Example |
|-----------|------|---------|
| **Agent** | Decision-maker | Robot |
| **Environment** | Responds to actions | Physical world |
| **State** | Current situation | Position, velocity |
| **Action** | Agent's choice | Move left/right |
| **Reward** | Feedback signal | +1 for goal, -1 for obstacle |
| **Policy** | Decision rule | π(a|s) = probability of action a in state s |"""),
            ("Exploration vs Exploitation", """- **Exploration:** Try new actions to find better policies (short-term loss)
- **Exploitation:** Use known good actions (short-term gain)
- **Trade-off:** ε-greedy balances: explore probability ε, exploit (1-ε)
- **Dynamic:** Decay ε over time (explore early, exploit late)"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Agent State s"] -->|Choose action a<br/>via policy| B["Take action"]
    B -->|Transition| C["New state s'"]
    C -->|Receive| D["Reward r"]
    D -->|Observe| A
    A -->|Goal| E["Maximize<br/>cumulative reward"]
```"""
    },

    "30-markov-decision-processes": {
        "subsections": [
            ("MDP Formulation", """| Element | Definition | Role |
|---------|-----------|------|
| **S** | Set of states | Possible situations |
| **A** | Set of actions | Available choices |
| **P** | Transition probabilities | P(s'|s,a) |
| **R** | Reward function | R(s,a,s') |
| **γ** | Discount factor | Weight future vs immediate |"""),
            ("Markov Property", """- **Memorylessness:** Future depends only on current state, not history
- **Why matters:** Enables efficient algorithms (only need to store states)
- **Validity:** True for fully observable environments, violated when partially observable
- **Relaxation:** Partially Observable MDPs (POMDPs) handle hidden state"""),
        ],
        "diagram": """```mermaid
graph TD
    A["MDP"] -->|Contains| B["States S"]
    A -->|Contains| C["Actions A"]
    A -->|Contains| D["Transitions P(s'|s,a)"]
    A -->|Contains| E["Rewards R(s,a,s')"]
    A -->|Contains| F["Discount γ"]
    B -->|Markov Property| G["Future independent<br/>of history"]
```"""
    },

    "31-q-learning": {
        "subsections": [
            ("Off-Policy Learning", """| Property | Q-Learning | REINFORCE |
|----------|-----------|-----------|
| **Learning policy** | Optimal (greedy) | Current (behavior) |
| **Behavior policy** | ε-greedy | ε-greedy |
| **Sample efficiency** | High (learn from any data) | Low (must learn from current) |
| **Stability** | Overestimation bias | More stable |"""),
            ("Overestimation Problem", """- **Standard Q-Learning:** Uses max from same network (overestimates)
- **Double Q-Learning:** Two networks, one selects, one evaluates
- **Target Network:** Slowly updated copy of Q-network (stability)
- **Both:** DQN combines both techniques"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Current Q(s,a)"] -->|Update toward| B["r + γ max Q(s',a')"]
    B -->|Problem:| C["max overestimates"]
    D["Solution 1:<br/>Double Q-Learning"] -->|Separate| E["Selection network"]
    E -->|and| F["Evaluation network"]
    G["Solution 2:<br/>Target Network"] -->|Slowly update| H["Copy of Q-network"]
```"""
    },

    "32-policy-gradients": {
        "subsections": [
            ("Policy Gradient Types", """| Type | Update | Variance | Best For |
|------|--------|----------|----------|
| **REINFORCE** | ∇log π(a|s)R(τ) | High | Baseline |
| **Actor-Critic** | ∇log π(a|s)A(s,a) | Lower | Most problems |
| **PPO** | Clipped surrogate | Lower | Robust |
| **TRPO** | Trust region | Lower | Stability |"""),
            ("Variance Reduction", """- **No baseline:** High variance, need many samples
- **Value baseline:** Subtract E[R(s)] (reduces variance)
- **Advantage:** A(s,a) = Q(s,a) - V(s) (lower variance still)
- **Trade-off:** Baseline reduces variance but introduces bias from approximation"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Policy Gradient"] -->|Direct| B["∇J(θ) = E[∇log π(a|s)R]"]
    B -->|High variance| C["High sample complexity"]
    D["With Baseline"] -->|Subtract| E["E[R] or V(s)"]
    E -->|Lower variance| F["Lower sample complexity"]
```"""
    },

    "33-actor-critic-methods": {
        "subsections": [
            ("Actor-Critic Components", """| Component | Role | Loss | Update |
|-----------|------|------|--------|
| **Actor** | Policy π(a|s) | Policy gradient | Toward high advantage |
| **Critic** | Value V(s) | TD error | Accurate value estimate |
| **Connection** | Critic guides actor | None | Actor uses critic signal |"""),
            ("Training Dynamics", """- **Critic trains first:** Improve value estimates for good signal
- **Actor trains:** Use critic to guide policy improvement
- **Co-adaptation:** Both networks improve together
- **Instability:** Either network improving wrong can destabilize"""),
        ],
        "diagram": """```mermaid
graph TD
    A["State s"] -->|Actor| B["Action a"]
    A -->|Critic| C["Value V(s)"]
    B -->|Execute| D["Next state s'<br/>Reward r"]
    D -->|TD Error| E["δ = r + γV(s') - V(s)"]
    E -->|Guide Actor| F["Improve policy<br/>toward high δ"]
    E -->|Improve Critic| G["Better value estimates"]
```"""
    },

    "34-graph-neural-networks": {
        "subsections": [
            ("GNN Layer Types", """| Type | Message | Aggregation | Best For |
|------|---------|-------------|----------|
| **GCN** | Weighted neighbor features | Mean | Graph classification |
| **GAT** | Learned attention weights | Attention sum | Variable importance |
| **GraphSAGE** | Sampled neighbors | Mean/LSTM | Inductive learning |
| **GIN** | Sum neighbor features | Sum | Graph isomorphism |"""),
            ("Permutation Invariance", """- **Critical property:** Output invariant to node ordering
- **Mechanism:** Aggregation functions (sum, mean, max) are permutation invariant
- **Why matters:** Ensures same output regardless of node labeling
- **Implementation:** Replace order-dependent operations with invariant ones"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Node Features"] -->|Pass to neighbors| B["Neighbor Aggregation"]
    B -->|Invariant operation<br/>sum/mean/max| C["Aggregated features"]
    C -->|Update| D["New node features"]
    E["Permutation<br/>of nodes"] -->|Same output| D
```"""
    },

    "35-causal-inference": {
        "subsections": [
            ("Causal vs Correlational", """| Property | Causal | Correlational |
|----------|--------|---------------|
| **Direction** | A→B | A↔B or A←C→B |
| **Intervention** | If change A, B changes | Not guaranteed |
| **Confounding** | Account for | Ignored |
| **RCT** | Gold standard | Not applicable |
| **Observational** | Requires modeling | Just compute |"""),
            ("Identification Strategies", """- **RCT (randomized):** Gold standard, breaks confounding
- **Back-door adjustment:** Condition on confounders
- **Front-door adjustment:** Control indirect paths
- **Instrumental variables:** Use exogenous variable to identify effect
- **Matching/Propensity scores:** Create comparable groups"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Intervention A"] -->|Causal| B["Outcome B"]
    C["Confounder Z"] -->|Confounds| A
    C -->|Confounds| B
    D["Solution:<br/>Condition on Z"] -->|Breaks<br/>confounding| E["Identify<br/>A→B effect"]
```"""
    },

    "36-probabilistic-graphical-models": {
        "subsections": [
            ("Directed vs Undirected", """| Property | Bayesian Net (DAG) | Markov Net (Undirected) |
|----------|-------------------|------------------------|
| **Representation** | Causality | Associations |
| **Edges** | Parent → Child | Undirected edges |
| **Interpretation** | Conditional indep | Markov blanket |
| **Inference** | Belief propagation | Message passing |
| **Use Case** | Causal, sequential | Symmetric relationships |"""),
            ("Inference Algorithms", """- **Exact:** Belief propagation on trees (polynomial time)
- **Exact on DAG:** Variable elimination (exponential worst-case)
- **Approximate:** Loopy belief propagation, sampling
- **MCMC:** Gibbs sampling, Metropolis-Hastings
- **Trade-off:** Exact slow, approximate fast but biased"""),
        ],
        "diagram": """```mermaid
graph TD
    A["PGM"] -->|Directed| B["Bayesian Network<br/>DAG"]
    A -->|Undirected| C["Markov Network"]
    B -->|Inference| D["Belief Propagation"]
    C -->|Inference| E["Message Passing"]
    F["NP-hard for<br/>general graphs"] -->|Approximation| G["Sampling/Loopy BP"]
```"""
    },

    "37-variational-autoencoders": {
        "subsections": [
            ("VAE Loss Decomposition", """```
L = Reconstruction Loss + KL Divergence
  = -log p(x|z) + KL(q(z|x) || p(z))
  = Make x accurate + Keep z distribution close to prior
```

| Term | Purpose | Trade-off |
|------|---------|-----------|
| **Reconstruction** | Accurate decoding | High = blurry |
| **KL divergence** | Latent regularization | High = high variance |
| **β-VAE** | Weight trade-off | β controls disentanglement |"""),
            ("Encoder-Decoder Roles", """- **Encoder:** Learn q(z|x), map data to latent distribution
- **Decoder:** Learn p(x|z), generate data from latent samples
- **Reparameterization:** Sample z = μ + σ*ε for differentiability
- **Variational:** Approximate intractable true posterior q(z|x)"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Input x"] -->|Encoder q(z|x)| B["Latent μ,σ"]
    B -->|Sample ε| C["z = μ + σ*ε"]
    C -->|Decoder p(x|z)| D["Output x̂"]
    D -->|Reconstruction loss| E["Compare x,x̂"]
    B -->|KL loss| F["KL(q||N(0,1))"]
```"""
    },

    "38-generative-adversarial-networks": {
        "subsections": [
            ("Generator vs Discriminator", """| Component | Role | Loss | Goal |
|-----------|------|------|------|
| **Generator** | Create fake data | -log D(G(z)) | Fool discriminator |
| **Discriminator** | Classify real/fake | -[log D(x) + log(1-D(G(z)))] | Correct classification |
| **Equilibrium** | Both optimal | Discriminator can't distinguish | Generator = real data |"""),
            ("Training Instability", """- **Mode collapse:** Generator ignores variation, produces same samples
- **Vanishing gradient:** Discriminator too strong, generator can't improve
- **Oscillation:** Loss doesn't converge, oscillates
- **Solutions:** Spectral norm, progressive growing, Wasserstein loss"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Generator<br/>tries to fool"] -->|Fake data| B["Discriminator<br/>tries to distinguish"]
    B -->|Real vs Fake| C["Feedback to G"]
    A -->|Improves from| C
    B -->|Improves from| D["More real samples"]
    E["Equilibrium"] -->|Generator| F["Indistinguishable<br/>from real"]
```"""
    },

    "39-time-series-forecasting": {
        "subsections": [
            ("Forecasting Models", """| Model | Linear | Captures Seasonality | Confidence Intervals |
|-------|--------|----------------------|--------------------|
| **Moving Avg** | Yes | No | No |
| **Exponential Smoothing** | Yes | Yes | Yes |
| **ARIMA** | Yes | With SARIMA | Yes |
| **Prophet** | Yes | Yes | Yes (built-in) |
| **LSTM/RNN** | No | Yes | No (by default) |
| **Transformer** | No | Yes | No |"""),
            ("Validation Strategy", """- **Walk-forward:** Train on past, test on recent, repeat
- **Expanding window:** Train on all past up to t
- **Fixed window:** Train on last N time steps
- **Critical:** Cannot use random CV (violates temporal order)"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Time Series Data"] -->|Past| B["Train Period"]
    B -->|Recent| C["Validation Period"]
    C -->|Future| D["Forecast"]
    E["Walk-Forward"] -->|Repeat| F["Shift window"]
    G["Seasonality"] -->|Periodic| H["ARIMA/Prophet"]
    I["Complex patterns"] -->|Non-linear| J["LSTM/Transformer"]
```"""
    },

    "40-anomaly-detection": {
        "subsections": [
            ("Detection Methods", """| Method | Type | Unsupervised | Interpretability |
|--------|------|--------------|-----------------|
| **Isolation Forest** | Outlier-based | ✓ Yes | Medium |
| **Local Outlier Factor** | Density-based | ✓ Yes | Low |
| **One-Class SVM** | Boundary-based | ✓ Yes | Low |
| **Autoencoder** | Reconstruction | ✓ Yes | Low |
| **Isolation Forest + Expert** | Hybrid | ✓ Yes | High |"""),
            ("Evaluation Challenge", """- **Ground truth rare:** Anomalies are rare, hard to label
- **Definition ambiguous:** What counts as anomaly depends on context
- **Cost asymmetry:** False positive ≠ false negative cost
- **Threshold tuning:** Critical for deployment (false alarm rate vs detection rate)"""),
        ],
        "diagram": """```mermaid
graph TD
    A["Normal Data Pattern"] -->|Learn| B["Normal Model"]
    C["New Samples"] -->|Compare to| B
    B -->|High error| D["Anomaly"]
    B -->|Low error| E["Normal"]
    F["Threshold<br/>Tuning"] -->|Controls| G["Detection Rate<br/>vs False Alarms"]
```"""
    }
}

def enhance_architectures(architectures, section_name):
    """Enhance architecture sections for a section."""
    base_dir = f"{BASE}/{section_name}/concepts"

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return 0

    enhanced = 0
    for filename, arch_data in architectures.items():
        filepath = None
        for f in os.listdir(base_dir):
            if f.replace('.md', '') == filename:
                filepath = os.path.join(base_dir, f)
                break

        if not filepath:
            print(f"  ⊘ {filename}: File not found")
            continue

        with open(filepath) as f:
            content = f.read()

        # Build subsections
        subsections = ""
        if "subsections" in arch_data:
            for title, content_text in arch_data["subsections"]:
                subsections += f"\n### {title}\n\n{content_text}\n"

        # Add diagram if exists
        if "diagram" in arch_data:
            subsections += f"\n{arch_data['diagram']}\n"

        # Replace Architecture section
        pattern = r'(## Architecture / Trade-offs\n\n).*?(\n\n## Interview Q&A)'
        replacement = rf'\1{subsections}\2'

        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        with open(filepath, 'w') as f:
            f.write(content)

        enhanced += 1
        print(f"  ✓ {filename}")

    return enhanced

if __name__ == "__main__":
    print("=== Enhancing Architecture/Trade-offs Sections ===\n")

    print("AI Fundamentals (01-40):")
    ai_enhanced = enhance_architectures(ARCHITECTURE_AI, "ai")
    print(f"✅ Enhanced {ai_enhanced} AI concepts\n")
