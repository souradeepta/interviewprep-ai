# ML Coding Interview Questions

Implement from scratch — NumPy only for the core implementation.

---

## Q: Implement K-Nearest Neighbors (KNN) from scratch.

**Difficulty:** Medium | **Domain:** ML Coding | **Companies:** Google, Meta, Amazon

### Step 1 — Clarifying Questions to Ask
- "What distance metric? Euclidean?"
- "Classification or regression?"
- "Should I make k configurable?"

### Step 2 — Approach Discussion
For each test point: compute distance to all training points, find k nearest, majority vote.
Brute force O(n·d) per query. Mention KD-tree as O(d log n) optimization.

### Step 3 — Implementation
```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3): self.k = k

    def fit(self, X, y): self.X_tr = X; self.y_tr = y

    def predict(self, X): return np.array([self._pred(x) for x in X])

    def _pred(self, x):
        dists = np.sqrt(np.sum((self.X_tr - x)**2, axis=1))
        k_idx = np.argsort(dists)[:self.k]
        return Counter(self.y_tr[k_idx]).most_common(1)[0][0]

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNN(k=5); knn.fit(X_tr, y_tr)
print(f"Accuracy: {np.mean(knn.predict(X_te)==y_te):.4f}")
```

### Step 4 — Test Cases
| Input | Expected | Why |
|---|---|---|
| k=1, test=train | 100% accuracy | k=1 memorizes training data |
| k > n_train | clip k to n_train | Edge case — handle gracefully |
| unnormalized features | poor accuracy | Distance-based — scale matters |

### Step 5 — Complexity Analysis
**Time:** fit O(1); predict O(n·d) per query, O(m·n·d) for m test points
**Space:** O(n·d) to store training data

### Step 6 — Follow-up Questions
- "Make it faster for large datasets?" → KD-tree O(d log n), Ball tree, FAISS/HNSW for ANN
- "Curse of dimensionality?" → High dimensions: all points equidistant — distance loses meaning. Degrades for d > ~20.
- "Regression KNN?" → Predict mean (or distance-weighted mean) of k nearest neighbors.

### Common Mistakes
- Unvectorized distance loop — always vectorize: `(X_tr - x)**2` broadcasts
- Not normalizing features — scale affects Euclidean distance

---

## Q: Implement scaled dot-product attention from scratch.

**Difficulty:** Hard | **Domain:** ML Coding | **Companies:** OpenAI, Google, Anthropic, Meta AI

### Step 1 — Clarifying Questions to Ask
- "Should I implement single-head or multi-head?"
- "Do you need causal masking?"

### Step 2 — Approach Discussion
$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$
Walk through each step: dot products, scaling, (optional masking), softmax, weighted sum.

### Step 3 — Implementation
```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)   # (..., T_q, T_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)                 # (..., T_q, T_k)
    return weights @ V                                 # (..., T_q, d_v)

# Test single head
T, d_k, d_v = 6, 8, 8
Q = np.random.randn(T, d_k)
K = np.random.randn(T, d_k)
V = np.random.randn(T, d_v)
out = attention(Q, K, V)
assert out.shape == (T, d_v)

# Test causal mask
causal_mask = np.tril(np.ones((T, T), dtype=bool))
out_causal = attention(Q, K, V, mask=causal_mask)
print(out.shape, out_causal.shape)
```

### Step 4 — Test Cases
| Input | Expected | Why |
|---|---|---|
| Q=K=V, identity | Each position attends to itself | Self-similarity |
| Causal mask | Upper triangle weights ≈ 0 | Future tokens masked |
| d_k very large, no scaling | Near-zero gradients | Softmax saturation without √d_k |

### Step 5 — Complexity Analysis
**Time:** O(T²·d_k) for scores, O(T²·d_v) for output — O(T²d) total
**Space:** O(T²) for attention matrix (Flash Attention reduces this to O(T))

### Step 6 — Follow-up Questions
- "Implement multi-head attention?" → Split d_model into h heads of d_k=d_model/h, run in parallel, concat, project.
- "What is Flash Attention?" → Recompute attention in tiles to avoid materializing the full T×T matrix. O(T) memory.
- "Why does softmax saturate without scaling?" → Large d_k → large dot products → near one-hot softmax → gradients ≈ 0.

### Common Mistakes
- Forgetting to subtract max before softmax (numerical stability)
- Using `np.exp` directly on large values → overflow
- Mask values of 0 vs -inf: use a large negative number (-1e9), not 0

---

## Q: Implement softmax and cross-entropy loss from scratch.

**Difficulty:** Easy | **Domain:** ML Coding | **Companies:** Any ML company

### Step 1 — Clarifying Questions to Ask
- "Numerically stable version?"
- "Batch or single example?"

### Step 2 — Approach Discussion
Numerical stability is the key challenge. Subtract max before exp — doesn't change output but prevents overflow.

### Step 3 — Implementation
```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))  # subtract max for stability
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(logits, y_true):
    p = softmax(logits)
    n = len(y_true)
    log_p = np.log(p[np.arange(n), y_true] + 1e-9)  # add eps for safety
    return -log_p.mean()

def cross_entropy_grad(logits, y_true):
    p = softmax(logits)
    n = len(y_true)
    p[np.arange(n), y_true] -= 1
    return p / n  # gradient of CE+softmax w.r.t. logits

# Test
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
labels = np.array([0, 1])
print(f"Softmax: {softmax(logits)}")
print(f"Cross-entropy: {cross_entropy(logits, labels):.4f}")
```

### Step 4 — Test Cases
| Input | Expected | Why |
|---|---|---|
| logits = [0, 0, 0] | [1/3, 1/3, 1/3] | Uniform distribution |
| logits = [1000, 0, 0] | ≈ [1, 0, 0] | Stable even with large values |
| Correct class logit >> others | Loss ≈ 0 | Near-perfect prediction |

### Step 5 — Complexity Analysis
**Time:** O(n·C) where n=batch, C=classes. **Space:** O(n·C) for softmax output.

### Step 6 — Follow-up Questions
- "What is the gradient of cross-entropy + softmax?" → $p - y_{one\_hot}$ — the residual. This simplifies dramatically because softmax and CE cancel nicely.
- "Binary cross-entropy?" → For 2 classes: $-[y \log p + (1-y)\log(1-p)]$ with sigmoid instead of softmax.

### Common Mistakes
- Overflow from `np.exp(1000)` — always subtract max first
- Adding eps after taking log instead of before: `log(p + eps)` not `log(p) + eps`

---

## Q: Implement PCA from scratch using eigendecomposition.

**Difficulty:** Medium | **Domain:** ML Coding | **Companies:** Google, Meta, Amazon

### Step 1 — Clarifying Questions to Ask
- "How many components?"
- "Should I use SVD or eigendecomposition of the covariance matrix?"

### Step 2 — Approach Discussion
Two approaches: (1) eigendecompose covariance matrix, (2) SVD of centered data. SVD is more numerically stable and standard in practice.

### Step 3 — Implementation
```python
import numpy as np

class PCA:
    def __init__(self, n_components): self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        # Covariance matrix approach
        cov = X_c.T @ X_c / (len(X) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components_ = eigenvectors[:, :self.n_components].T  # (n_components, d)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues.sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
pca = PCA(n_components=2).fit_transform(X)
print(f"Shape: {pca.shape}, Explained variance: {pca.var(0) / X.var(0).sum():.3f}")
```

### Step 4 — Test Cases
| Input | Expected | Why |
|---|---|---|
| n_components=d (full) | All variance explained | Lossless projection |
| Identical features | One component captures all variance | PCA handles correlations |
| Mean-centered input | Same as non-centered | Centering shouldn't matter if already centered |

### Step 5 — Complexity Analysis
**Time:** O(d² · n) for covariance, O(d³) for eigendecomposition.
For n >> d: use covariance. For d >> n: use Gram matrix $XX^T$ (O(n²d)).

### Step 6 — Follow-up Questions
- "SVD vs eigendecomposition?" → SVD is more numerically stable and directly gives singular values. For large matrices, randomized SVD (sklearn TruncatedSVD) is much faster.
- "Standardize before PCA?" → Yes if features have different units/scales. Otherwise features with larger variance dominate.

### Common Mistakes
- Forgetting to center data — PCA without centering gives wrong components
- Using `linalg.eig` instead of `linalg.eigh` (eigh is for symmetric matrices — more stable and sorted)

---

## Q: Implement rejection sampling for a target distribution.

**Difficulty:** Medium | **Domain:** Probability / ML Coding | **Companies:** LinkedIn, Google

Given a proposal sampler `q(x)` and a target density `p(x)` with a known bound
`p(x) <= M q(x)`, sample from `p` without directly sampling it. Draw `x ~ q`
and accept it with probability `p(x) / (M q(x))`.

```python
import numpy as np


def rejection_sample(target_pdf, proposal_sample, proposal_pdf, bound,
                     n_samples, rng=None):
    rng = np.random.default_rng(rng)
    samples = []
    while len(samples) < n_samples:
        x = proposal_sample(rng)
        acceptance = target_pdf(x) / (bound * proposal_pdf(x))
        if acceptance > 1:
            raise ValueError("bound M is too small")
        if rng.random() < acceptance:
            samples.append(x)
    return np.asarray(samples)
```

Ask how to estimate efficiency: the expected acceptance rate is `1/M` when
the proposal is normalized and the target is normalized. Follow-ups include
importance sampling, adaptive proposals, and numerical stability in log-space.
Test a symmetric target, a poor proposal, a fixed random seed, and a proposal
with zero density outside its support.

## Q: Build a lightweight topic model with TF-IDF and truncated SVD.

**Difficulty:** Medium | **Domain:** NLP / ML Coding | **Companies:** LinkedIn, Meta

Construct a document-term matrix, apply TF-IDF, and use a low-rank
decomposition to discover latent topics. The interview focus is usually the
pipeline and leakage controls, not reproducing a full production LDA solver.

```python
import numpy as np
from collections import Counter


def tfidf(documents, min_df=1):
    tokenized = [doc.lower().split() for doc in documents]
    vocab = sorted({word for doc in tokenized for word in set(doc)
                    if sum(word in other for other in tokenized) >= min_df})
    index = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(documents), len(vocab)), dtype=float)
    for row, words in enumerate(tokenized):
        counts = Counter(words)
        for word, count in counts.items():
            if word in index:
                matrix[row, index[word]] = count / len(words)
    idf = np.log((1 + len(documents)) /
                 (1 + (matrix > 0).sum(axis=0))) + 1
    return matrix * idf, vocab


def topic_vectors(documents, n_topics=2):
    matrix, vocabulary = tfidf(documents)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return vt[:n_topics], vocabulary
```

Clarify tokenization, stop words, vocabulary fitting only on training data,
and how to choose the number of topics. Explain that SVD gives LSA topics,
while probabilistic LDA models a distribution over topics and words.

## Q: Implement ranking metrics for a recommendation model.

**Difficulty:** Medium | **Domain:** Recommenders / ML Coding | **Companies:** LinkedIn, Meta, Amazon

Offline ranking evaluation must respect user boundaries and should report more
than one metric. Implement precision@k, recall@k, and NDCG@k, then discuss
candidate generation, popularity bias, cold start, and online guardrails.

```python
import math


def ndcg_at_k(recommended, relevant, k):
    relevant = set(relevant)
    gains = [1 if item in relevant else 0 for item in recommended[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum(1 / math.log2(rank + 2)
                for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def recall_at_k(recommended, relevant, k):
    relevant = set(relevant)
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0
```

Avoid leaking future interactions into training. Split by time for production
recommendation, report metrics by user cohort, and distinguish ranking quality
from business outcomes such as retention, diversity, and complaint rate.

### Remaining high-value ML coding topics

K-means, mini-batch SGD, momentum, beam search, ROC/AUC, an NLP trie, weighted
sampling, batch matrix multiplication, and a data/SQL feature-aggregation
exercise remain useful follow-ups.

---

## Q8–Q20: applied ML implementation drills

Use these as 20–30 minute MLE rounds. Each prompt requires clarifying
questions, explicit inputs and edge cases, a complexity statement, and one
follow-up. Tested reference code lives in `solutions/`.

### Q8. Linear regression with regularization
Implement full-batch gradient descent with an intercept, L2 regularization,
loss history, and a finite-difference gradient check. Reject empty and
non-finite inputs. See `solutions/linear_models.py`; follow up with the closed
form solution, SGD, and early stopping.

### Q9. Binary logistic regression
Implement stable sigmoid/log-loss gradients, L2 regularization, and probability
predictions. Test logits near ±1,000 and invalid labels. Use `logaddexp` rather
than naive logarithms. See `solutions/linear_models.py`; follow up with
calibration and cost-sensitive thresholds.

### Q10. Seeded K-means
Implement assignment/update steps, reproducible initialization, convergence,
inertia, and empty-cluster repair. Test duplicated points and `k=1`. See
`solutions/clustering_and_trees.py`; follow up with K-means++ and model
selection.

### Q11. Decision-stump split selection
Enumerate numeric mid-point thresholds and minimize weighted Gini impurity,
with deterministic tie handling and pure-node behavior. See
`solutions/clustering_and_trees.py`; follow up with sample weights and depth
stopping.

### Q12. Leakage-safe temporal split and point-in-time features
Split by event time and select only feature rows strictly available before the
event. Test future rows, same-time updates, and late arrivals. See
`solutions/training_and_evaluation.py`; follow up with an indexed as-of join.

### Q13. Calibration and threshold choice
Choose a probability threshold minimizing false-positive/false-negative cost.
Explain why AUC does not imply calibration and why the threshold belongs on
validation data. See `choose_threshold`; follow up with Platt versus isotonic
calibration.

### Q14. Mini-batch training loop
Generate seeded shuffled batches, support a final partial batch, and explain
gradient accumulation, clipping, and early stopping. See
`mini_batch_indices`; follow up with distributed all-reduce and mixed precision.

### Q15. Filtered retrieval evaluation
Keep candidate generation and reranking contracts separate; report recall@K,
filter effects, and latency-budget status. See `recall_at_k`; follow up with
ANN refresh and candidate-recall monitoring.

### Q16. Bootstrap metric confidence interval
Return a seeded percentile interval for a metric and state its interpretation.
Test singleton, invalid, and repeatability cases. See `bootstrap_mean_ci`;
follow up with a paired per-user bootstrap.

### Q17. Candidate generation plus reranking
Design the boundary where unique candidate IDs and retrieval scores are passed
to a point-in-time-aware ranker. Measure recall before reranking and NDCG after
it. See `solutions/evaluation_exercises.py`; follow up with position bias.

### Q18. Online feature freshness
Return `fresh`, `stale`, or `invalid` for feature time, request time, and a
maximum age; reject future features. See `feature_freshness_status`; follow up
with training-serving parity and freshness dashboards.

### Q19. Debug a faulty metric
Construct the smallest counterexample for ascending score sorting, wrong
relevant-set denominators, or accidental true-negative inclusion. Add a
regression test and compare with `binary_metrics` and `ndcg_at_k`.

### Q20. End-to-end mini take-home
Produce a baseline, time-safe split, model, slice metrics, uncertainty interval,
error analysis, and ship/no-ship memo from a small event table. No invented
performance target is acceptable; state assumptions and follow-up label work.
