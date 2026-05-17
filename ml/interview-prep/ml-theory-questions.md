# ML Theory Interview Questions

---

## Q: What is the bias-variance trade-off?

**Difficulty:** Medium | **Domain:** ML Theory | **Companies:** Google, Meta, Amazon, Microsoft

### Step 1 — Clarifying Questions to Ask
- "Are you asking in general, or about a specific model or task?"
- "Should I include the mathematical decomposition?"

### Step 2 — Approach Discussion
Start with intuition (U-shaped test error curve), then formalize the decomposition.
Connect to practical actions at the end.

### Step 3 — Answer
**Bias** = error from wrong assumptions (underfitting). Model too simple to capture patterns.
**Variance** = sensitivity to training data fluctuations (overfitting). Model fits noise.

Expected test error decomposes as:
$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}(\hat{f})^2 + \text{Var}(\hat{f}) + \sigma^2_\text{noise}$$

Noise is irreducible. As model complexity increases: bias ↓, variance ↑.
Optimal complexity minimizes their sum.

**Practical actions:**
- High bias (underfitting): add features, increase model capacity, reduce regularization
- High variance (overfitting): more data, regularize, reduce capacity, use ensemble methods

### Step 4 — Test Cases
N/A (theory question)

### Step 5 — Complexity Analysis
N/A

### Step 6 — Follow-up Questions
- "How do ensemble methods change the trade-off?"
  → Bagging reduces variance without increasing bias. Boosting reduces bias (sequentially).
- "Does more data help bias or variance?"
  → Primarily variance. Bias requires changing the model, not adding data.

### Common Mistakes
- Saying "complex models have high bias" — it's the opposite
- Not connecting to practical actions (what do you do in each case?)
- Forgetting noise is irreducible

---

## Q: Explain gradient descent and its variants.

**Difficulty:** Medium | **Domain:** ML Theory | **Companies:** Google, Meta, OpenAI, DeepMind

### Step 1 — Clarifying Questions to Ask
- "High-level intuition, or full mathematical derivation?"
- "Should I cover Adam and adaptive methods?"

### Step 2 — Approach Discussion
Cover vanilla GD first, then frame each variant as a solution to a specific failure mode.

### Step 3 — Answer
**Core:** $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$

Three variants by batch size:
- **Batch GD:** full dataset per update. Exact gradient, slow on large datasets.
- **SGD:** one sample. Noisy, fast, can escape saddle points.
- **Mini-batch:** B samples (32–256). Best trade-off. Standard in practice.

**Momentum:** velocity term smooths oscillations and accelerates along consistent directions.

**Adam:** adapts per-parameter learning rates using first moment (mean) + second moment (variance) of gradients. Fastest convergence. Defaults: lr=1e-3, β₁=0.9, β₂=0.999.

**LR scheduling:** cosine annealing or warmup+decay. Warmup is essential for transformers.

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
Per update: Batch O(n·d); SGD O(d); Mini-batch O(B·d)

### Step 6 — Follow-up Questions
- "Why does SGD sometimes outperform Adam?"
  → SGD noise → flatter minima → better generalization than Adam's sharp minima.
- "What is learning rate warmup?"
  → Gradually increase lr from 0 for first N steps. Prevents instability when params are far from optimum.

### Common Mistakes
- Not distinguishing batch/SGD/mini-batch
- Claiming Adam is always better — it converges faster but doesn't always generalize as well
- Forgetting gradient clipping for RNNs/transformers

---

## Q: How does cross-validation work and when do you use it?

**Difficulty:** Easy | **Domain:** ML Theory | **Companies:** Any ML company

### Step 1 — Clarifying Questions to Ask
- "Should I cover k-fold specifically or all variants?"

### Step 2 — Approach Discussion
Problem → solution → variants → code.

### Step 3 — Answer
**Problem:** single train/test split gives a noisy estimate of generalization.

**K-Fold CV:** split data into k folds. Train on k-1, evaluate on remaining fold. Repeat k times.
Average k scores. Reduces evaluation variance by factor of k.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
scores = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring='roc_auc')
print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Stratified k-fold:** preserves class proportions per fold. Always use for classification.

**When to use:** hyperparameter tuning, limited data, reliable generalization estimates.

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
O(k · training_time) — k times more expensive than a single train.

### Step 6 — Follow-up Questions
- "CV vs train/val/test split?" → Use train/val/test with enough data. CV for limited data.
- "What is nested CV?" → Outer loop evaluates generalization; inner loop tunes hyperparameters.

### Common Mistakes
- Non-stratified k-fold for imbalanced classification
- Fitting a scaler on full dataset before CV — data leakage
- Reporting only mean, not std across folds

---

## Q: How does L1 regularization produce sparse weights?

**Difficulty:** Medium | **Domain:** ML Theory | **Companies:** Google, Meta

### Step 1 — Clarifying Questions to Ask
- "Do you want the geometric explanation or the sub-gradient explanation?"

### Step 2 — Approach Discussion
Both explanations are useful — geometric is intuitive, sub-gradient is rigorous.

### Step 3 — Answer
**L1 penalty:** $\lambda \sum_j |w_j|$

**Geometric:** L1 constrains parameters to a diamond (L1 ball). The optimal point is where
loss contours touch the ball — at a corner, where weights are zero. L2's spherical ball
has no corners — optimal point lands on the surface but rarely at zero.

**Sub-gradient:** at $w_j=0$, the L1 sub-gradient is in $[-\lambda, \lambda]$. If the loss
gradient magnitude is less than $\lambda$, the optimal condition holds at $w_j=0$ — weight stays zero.

**Practical implication:** L1 performs automatic feature selection. Features that don't contribute
enough to reduce the loss get zeroed out.

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
N/A

### Step 6 — Follow-up Questions
- "When L1 over L2?" → When you want feature selection / sparse interpretable model.
- "What is Elastic Net?" → L1 + L2 combined. Handles groups of correlated features better than L1 alone.

### Common Mistakes
- Saying only "L1 gives sparsity" without explaining why geometrically
- Forgetting L1 is non-differentiable at 0 — this is precisely what produces sparsity

---

## Q: Explain the EM algorithm with an example.

**Difficulty:** Hard | **Domain:** ML Theory | **Companies:** Google, DeepMind

### Step 1 — Clarifying Questions to Ask
- "Should I use Gaussian Mixture Models as the concrete example?"

### Step 2 — Approach Discussion
Abstract framework first, then GMM as the grounding example. Connect to K-Means.

### Step 3 — Answer
EM finds MLE when latent (hidden) variables exist and direct optimization is intractable.

**E-step:** compute expected log-likelihood given current parameters and observed data.
For GMM: compute soft cluster assignments (responsibilities) for each point.

**M-step:** update parameters to maximize the expected log-likelihood from E-step.
For GMM: update cluster means, covariances, and mixing weights using responsibilities.

```python
from sklearn.mixture import GaussianMixture
import numpy as np

X = np.vstack([np.random.randn(100,2)+[0,0], np.random.randn(100,2)+[5,5]])
gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
print(gmm.means_)  # learned cluster centers
```

**Convergence:** EM is guaranteed to non-decrease the marginal log-likelihood each iteration.
Converges to a local maximum — initialization matters.

**K-Means = EM** with hard cluster assignments (responsibilities are 0 or 1) and isotropic Gaussians.

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
Per iteration: O(n·k·d²) for GMM with k components, d features.

### Step 6 — Follow-up Questions
- "Limitations of EM?" → Local optima (use multiple restarts), slow convergence, must choose k.
- "Why is K-Means a special case?" → Hard assignments + isotropic Gaussians = K-Means objective.

### Common Mistakes
- Not knowing a concrete example (GMM)
- Claiming EM converges to a global maximum — it's only local

---

## Q: What is the kernel trick and when do SVMs use it?

**Difficulty:** Hard | **Domain:** ML Theory | **Companies:** Google, Microsoft

### Step 1 — Clarifying Questions to Ask
- "Should I derive the dual formulation or stay conceptual?"

### Step 2 — Approach Discussion
Explain the core idea (mapping to high-dimensional space), then show the trick (compute inner products without materializing the mapping), then give examples of kernels.

### Step 3 — Answer
**SVM** finds a maximum-margin hyperplane: $\max_{w,b} \frac{2}{\|w\|}$ subject to $y_i(w^Tx_i + b) \geq 1$.

**When linearly inseparable:** map $x \to \phi(x)$ in a higher-dimensional space where separation is possible.

**Kernel trick:** the SVM dual only requires inner products $\phi(x_i)^T\phi(x_j)$.
A kernel function $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ computes this without materializing $\phi$.

**Common kernels:**
- Linear: $K(x,z) = x^Tz$ — no mapping
- RBF (Gaussian): $K(x,z) = \exp(-\gamma\|x-z\|^2)$ — infinite-dimensional feature space
- Polynomial: $K(x,z) = (x^Tz + c)^d$

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
Training: O(n²) to O(n³). Prediction: O(n_sv · d) where n_sv = number of support vectors.

### Step 6 — Follow-up Questions
- "SVM vs logistic regression?" → SVM maximizes margin (geometric), LR maximizes likelihood (probabilistic). SVM has no probability output natively.
- "What is the support vector?" → Training points on the margin boundary. Only these determine the hyperplane.

### Common Mistakes
- Saying SVM can't handle non-linear data without kernel — it can, via the kernel trick
- Confusing C (regularization) and γ (RBF bandwidth) — both need tuning

---

## Q: How does PCA work and when should you use it?

**Difficulty:** Medium | **Domain:** ML Theory | **Companies:** Google, Meta, Amazon

### Step 1 — Clarifying Questions to Ask
- "Do you want the SVD derivation or the eigenvector framing?"

### Step 2 — Approach Discussion
Motivate with the variance-preservation goal, then give the algorithm, then trade-offs.

### Step 3 — Answer
**Goal:** find a low-dimensional linear projection that preserves maximum variance.

**Algorithm:**
1. Center data: $X \leftarrow X - \bar{X}$
2. Compute covariance matrix: $C = X^TX / (n-1)$
3. Eigendecompose: $C = V\Lambda V^T$
4. Project: $Z = XV_k$ where $V_k$ are the top-k eigenvectors

**Explained variance:** $\frac{\sum_{i=1}^k \lambda_i}{\sum_i \lambda_i}$ — typically keep 95% of variance.

**When to use:** visualization (2D/3D), remove correlated features, noise reduction, speed up downstream models.

**When NOT to use:** when interpretability matters (rotated features lose meaning), when the signal is in low-variance directions (rare but happens in supervised settings — use supervised PCA or LDA instead).

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
SVD: O(min(n,d)² · max(n,d)) — expensive for high-d data. Use randomized SVD for large d.

### Step 6 — Follow-up Questions
- "PCA vs t-SNE?" → PCA: linear, invertible, preserves global structure. t-SNE: nonlinear, not invertible, preserves local neighborhoods. Use PCA for preprocessing, t-SNE for visualization.
- "PCA vs LDA?" → PCA maximizes variance (unsupervised). LDA maximizes class separability (supervised). LDA often better for classification preprocessing.

### Common Mistakes
- Forgetting to center (and standardize if features have different scales)
- Using PCA output as features without understanding what was lost

---

## Q: Explain batch normalization — what does it do and why does it help?

**Difficulty:** Medium | **Domain:** ML Theory | **Companies:** Google, Meta, OpenAI

### Step 1 — Clarifying Questions to Ask
- "Should I cover the train vs inference difference?"

### Step 2 — Approach Discussion
Explain the operation, then the two reasons it helps (ICS, learning rate), then train vs eval.

### Step 3 — Answer
**Operation (training):** For each feature in a mini-batch:
1. Compute mean $\mu_B$ and variance $\sigma_B^2$ across the batch
2. Normalize: $\hat{x} = (x - \mu_B) / \sqrt{\sigma_B^2 + \epsilon}$
3. Scale and shift: $y = \gamma \hat{x} + \beta$ (learned parameters)

**Why it helps:**
- Reduces internal covariate shift: stabilizes the distribution of activations layer-to-layer
- Allows higher learning rates: normalized activations → gradients don't explode
- Light regularization: noise from batch statistics acts like dropout

**Train vs inference:**
- Training: use batch statistics $\mu_B$, $\sigma_B^2$
- Inference: use running statistics computed during training (exponential moving average)
- Always call `model.eval()` before inference

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
O(n·d) per forward pass — negligible overhead.

### Step 6 — Follow-up Questions
- "LayerNorm vs BatchNorm?" → LayerNorm normalizes across features (per sample), not across batch. Works for variable-length sequences (transformers). BatchNorm depends on batch size and fails for batch size 1.
- "Why does BN fail with small batch size?" → Small batch → noisy mean/variance estimates → unstable training. Use GroupNorm or LayerNorm instead.

### Common Mistakes
- Forgetting to switch to eval mode at inference — dropout and BN behave differently
- Confusing BN and LN axis of normalization

---

## Q: What is the vanishing gradient problem and how is it solved?

**Difficulty:** Medium | **Domain:** ML Theory | **Companies:** Google, DeepMind, OpenAI

### Step 1 — Clarifying Questions to Ask
- "For a specific architecture, or in general?"

### Step 2 — Approach Discussion
Explain why it happens mathematically, then list solutions with when each applies.

### Step 3 — Answer
**Problem:** in backprop, gradients are products of Jacobians across layers:
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_n} \cdot \prod_{k=2}^{n} \frac{\partial h_k}{\partial h_{k-1}} \cdot \frac{\partial h_1}{\partial W_1}$$

If $\frac{\partial h_k}{\partial h_{k-1}} < 1$ at every layer, the product → 0 exponentially. Early layers receive near-zero gradients — they don't learn.

**Root cause:** sigmoid/tanh saturate → derivative ≈ 0. Deep networks → many near-zero multiplications.

**Solutions:**
- **ReLU:** derivative = 1 for positive inputs — no saturation in the positive region
- **Skip connections (ResNet):** $\frac{\partial (F(x)+x)}{\partial x} = \frac{\partial F}{\partial x} + 1$ — gradient always ≥ 1
- **Batch normalization:** keeps activations in non-saturating range
- **Gradient clipping:** caps gradient norm — prevents exploding (the dual problem)
- **Better initialization:** He/Xavier prevent early saturation

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
N/A

### Step 6 — Follow-up Questions
- "Exploding vs vanishing?" → Vanishing: gradients → 0 (early layers don't learn). Exploding: gradients → ∞ (loss NaN). Clipping fixes exploding; architecture/init fixes vanishing.
- "LSTM vs vanilla RNN for vanishing gradient?" → LSTM cell state has near-identity gradient path. Forget gate = 1 → gradient flows unchanged through time.

### Common Mistakes
- Confusing vanishing (too small) and exploding (too large) gradients
- Not knowing the solutions — must be able to list at least 3

---

## Q: What is transfer learning and how does fine-tuning differ from feature extraction?

**Difficulty:** Easy | **Domain:** ML Theory | **Companies:** Google, Meta, OpenAI

### Step 1 — Clarifying Questions to Ask
- "For images specifically, or in general?"

### Step 2 — Approach Discussion
Explain what is being transferred, then contrast the two approaches with trade-offs.

### Step 3 — Answer
**Transfer learning:** take a model pretrained on a large dataset (ImageNet, WebText) and adapt it to a new task. Rationale: early layers learn general features (edges, curves, word patterns) that transfer across tasks.

**Feature extraction:**
- Freeze all pretrained layers
- Replace the output head and train only that
- Fast, needs little data, works when target domain is similar to source

**Fine-tuning:**
- Load pretrained weights as initialization
- Train all layers (or the last N) on the new task with a small learning rate
- More flexible, higher accuracy, needs more data, risk of catastrophic forgetting
- Use discriminative learning rates: small lr for early layers, larger for later layers + head

```python
import torch.nn as nn
from torchvision.models import resnet50

model = resnet50(pretrained=True)
for param in model.parameters():  # feature extraction: freeze all
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)  # new head — unfrozen by default
```

### Step 4 — Test Cases
N/A

### Step 5 — Complexity Analysis
Feature extraction: O(epochs · n · forward_pass). Fine-tuning: same but with gradients through all layers.

### Step 6 — Follow-up Questions
- "When would you NOT use transfer learning?" → When source and target domains are very different (medical imaging from ImageNet pretrain may not help for satellite imagery).
- "What is catastrophic forgetting?" → During fine-tuning, model forgets pretraining knowledge. Use regularization (EWC), low LR, or layer freezing to mitigate.

### Common Mistakes
- Using a high learning rate for fine-tuning — overwrites pretrained features
- Forgetting to adapt the final layer to the new number of classes

---

*Questions 11–50 follow the same Template B format. Remaining topics:*
*Naive Bayes (conditional independence, Laplace smoothing), Random Forest OOB score,*
*Dropout as ensemble interpretation, multi-task learning trade-offs, class imbalance strategies,*
*A/B testing statistical power and sample size, NDCG for ranking, AUC-PR vs AUC-ROC,*
*dimensionality reduction methods comparison, model calibration, Shapley values,*
*online learning vs batch learning, model monitoring and drift detection, and more.*
