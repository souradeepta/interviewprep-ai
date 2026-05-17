# Unsupervised Learning

## TL;DR
Finding structure in unlabeled data. Key tasks: clustering (K-Means, DBSCAN), dimensionality
reduction (PCA, t-SNE, UMAP), and density estimation. Used for EDA, feature learning, and
anomaly detection.

## Core Intuition
No answer key. Like sorting foreign coins you've never seen — grouping by size, color, and
markings without knowing their value. The algorithm must find patterns on its own.

## How It Works

**K-Means:** assign points to nearest centroid, update centroids to cluster mean, repeat.
Minimizes inertia (within-cluster sum of squares). Requires choosing k.

**PCA:** finds orthogonal directions of maximum variance. Projects data to top-k components.
Linear — use t-SNE/UMAP for nonlinear visualization.

**DBSCAN:** density-based. Groups closely packed points, marks sparse points as outliers.
Doesn't require specifying k. Parameters: ε (neighborhood radius), min_samples.

## Key Properties / Trade-offs
- K-Means assumes spherical clusters; fails on complex shapes
- PCA is linear; t-SNE/UMAP better for visualization but not invertible
- DBSCAN handles noise naturally but ε is sensitive
- Evaluation is hard — no ground truth. Use silhouette score or domain validation.

## Common Mistakes / Gotchas
- Not normalizing before K-Means (Euclidean distance — scale matters)
- Choosing k arbitrarily — use elbow method and silhouette scores
- Confusing t-SNE distances as meaningful (only topology is preserved, not distances)

## Code Example
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

X = np.random.randn(200, 10)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
X_2d = PCA(n_components=2).fit_transform(X)
print(f"Explained variance: {PCA(2).fit(X).explained_variance_ratio_}")
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "How does K-Means work?" | Init k centroids, assign each point to nearest, update centroids to mean, repeat until convergence. |
| "What is PCA?" | Finds directions of maximum variance (principal components), projects data onto them. Reduces dimensionality while preserving variance. |
| "How do you choose k?" | Elbow method (plot inertia vs k, pick knee) + silhouette score (higher = better-separated clusters). |

## Related Topics
- [Feature Engineering](feature-engineering.md) — [Implementations: K-Means](../implementations/kmeans-from-scratch.ipynb)

## Resources
- [A Tutorial on PCA](https://arxiv.org/abs/1404.1100) — Shlens
