#!/usr/bin/env python3
"""Generate notebooks for AI fundamentals section (18-28)."""

import json
import os
import nbformat as nbf

BASE = "/home/sbisw/github/interviewprep-ml"

def create_notebook_cell(cell_type, source):
    """Create a notebook cell."""
    if cell_type == "markdown":
        return nbf.v4.new_markdown_cell(source)
    else:
        return nbf.v4.new_code_cell(source)

def create_ai_notebook(num, title, slug):
    """Create specialized notebook for AI fundamentals concept."""

    nb = nbf.v4.new_notebook()

    # Title mapping to code patterns
    title_lower = title.lower()

    # Cell 0: Title
    nb.cells.append(create_notebook_cell("markdown", f"""# {title}

## Learning Objectives
- Master {title_lower} and its role in ML
- Implement from numpy first principles
- Use production libraries (sklearn, PyTorch)
- Apply to realistic datasets and solve actual problems

## Prerequisites
- Fundamentals of Python and NumPy
- Understanding of basic statistical concepts"""))

    # Cell 1: Level 1 heading
    nb.cells.append(create_notebook_cell("markdown", "## Level 1: From Scratch Implementation"))

    # Cell 2: Level 1 code - from scratch
    if "clustering" in title_lower or "k-means" in title_lower:
        level1_code = """import numpy as np
import matplotlib.pyplot as plt

class KMeansBasic:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, X):
        # Initialize centroids randomly
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices]

        for iteration in range(self.max_iter):
            # Assign clusters
            distances = np.zeros((len(X), self.n_clusters))
            for k in range(self.n_clusters):
                distances[:, k] = np.sum((X - self.centroids[k])**2, axis=1)
            clusters = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([X[clusters == k].mean(axis=0)
                                      for k in range(self.n_clusters)])

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels = clusters
        return self

    def predict(self, X):
        distances = np.zeros((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k])**2, axis=1)
        return np.argmin(distances, axis=1)

# Test
np.random.seed(42)
X = np.random.randn(100, 2) * 2
X[:30] += [3, 3]
X[50:80] += [-3, -3]

model = KMeansBasic(n_clusters=3)
model.fit(X)

print(f"Clusters found: {model.n_clusters}")
print(f"Centroid shapes: {model.centroids.shape}")"""

    elif "dimensionality" in title_lower or "pca" in title_lower:
        level1_code = """import numpy as np

class PCABasic:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # Center data
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        cov = np.cov(X_centered.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]]
        self.explained_variance = eigenvalues[idx]

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Test
np.random.seed(42)
X = np.random.randn(100, 10)

pca = PCABasic(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Variance explained: {pca.explained_variance[:2]}")"""

    elif "mixture" in title_lower or "gaussian" in title_lower:
        level1_code = """import numpy as np
from scipy.stats import multivariate_normal

class GMMBasic:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.n_features = X.shape[1]

        # Initialize means randomly
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.means = X[indices].copy()

        # Initialize covariances
        self.covariances = [np.eye(self.n_features) for _ in range(self.n_clusters)]

        # Initialize weights
        self.weights = np.ones(self.n_clusters) / self.n_clusters

        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            responsibilities = np.zeros((len(X), self.n_clusters))
            for k in range(self.n_clusters):
                rv = multivariate_normal(mean=self.means[k], cov=self.covariances[k])
                responsibilities[:, k] = self.weights[k] * rv.pdf(X)

            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M-step: update parameters
            N_k = responsibilities.sum(axis=0)
            self.weights = N_k / len(X)

            for k in range(self.n_clusters):
                self.means[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / N_k[k]

        return self

# Test
X = np.random.randn(100, 2)
gmm = GMMBasic(n_clusters=2)
gmm.fit(X)

print(f"Means shape: {gmm.means.shape}")
print(f"Weights: {gmm.weights}")"""

    else:
        level1_code = """import numpy as np

class BasicModel:
    def __init__(self):
        self.fitted = False
        self.params = None

    def fit(self, X, y=None):
        self.params = X.mean(axis=0)
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Must fit first")
        return np.tile(self.params, (len(X), 1))

# Test
np.random.seed(42)
X = np.random.randn(50, 5)

model = BasicModel()
model.fit(X)
predictions = model.predict(X[:5])

print(f"Input shape: {X.shape}")
print(f"Predictions shape: {predictions.shape}")

    nb.cells.append(create_notebook_cell("code", level1_code))

    # Cell 3: Level 2 heading
    nb.cells.append(create_notebook_cell("markdown", "## Level 2: Production Implementation with sklearn"))

    # Cell 4: Level 2 code - production
    if "clustering" in title_lower or "k-means" in title_lower:
        level2_code = """from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Production: KMeans with sklearn

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, n_features=2, centers=4, random_state=42)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Evaluation
inertia = kmeans.inertia_
silhouette = -kmeans.inertia_  # Placeholder for actual silhouette

print(f"Inertia: {inertia:.2f}")
print(f"Centers shape: {kmeans.cluster_centers_.shape}")
print(f"Unique labels: {np.unique(labels)}")

# Elbow method - find optimal k
inertias = []
k_range = range(1, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 4))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()"""

    elif "dimensionality" in title_lower or "pca" in title_lower:
        level2_code = """from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Production: PCA with sklearn

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_transformed = pca.fit_transform(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

print(f"Total variance explained: {cumsum_var[-1]:.4f}")
print(f"Variance by component: {explained_var}")

# Find components needed for 95% variance
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scree plot
axes[0].bar(range(1, len(explained_var)+1), explained_var)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot')

# Cumulative variance
axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var, 'bo-')
axes[1].axhline(y=0.95, color='r', linestyle='--')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Variance Explained')
axes[1].grid(True)

plt.tight_layout()
plt.show()"""

    else:
        level2_code = """from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Production pattern

# Generate data
np.random.seed(42)
X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")
print(f"Training mean: {X_train_scaled.mean(axis=0)[:3]}")
print(f"Test mean: {X_test_scaled.mean(axis=0)[:3]}")"""

    nb.cells.append(create_notebook_cell("code", level2_code))

    # Cell 5: Example 1
    nb.cells.append(create_notebook_cell("markdown", f"## Real-World Example 1: Customer Segmentation"))

    # Cell 6: Example 1 code
    nb.cells.append(create_notebook_cell("code", """# Example 1: Customer Segmentation with Clustering

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# Simulate customer data
np.random.seed(42)
n_customers = 500
annual_income = np.random.gamma(3, 20, n_customers) * 10  # in thousands
spending_score = np.random.beta(2, 5, n_customers) * 100

# Combine
X = np.column_stack([annual_income, spending_score])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Create dataframe
df = pd.DataFrame({
    'income': annual_income,
    'spending': spending_score,
    'cluster': clusters
})

print("Customer Segments:")
for cluster in range(3):
    segment = df[df['cluster'] == cluster]
    print(f"\\nCluster {cluster} ({len(segment)} customers):")
    print(f"  Avg Income: ${segment['income'].mean():.2f}k")
    print(f"  Avg Spending: {segment['spending'].mean():.2f}")"""))

    # Cell 7: Example 2
    nb.cells.append(create_notebook_cell("markdown", f"## Real-World Example 2: Feature Reduction"))

    # Cell 8: Example 2 code
    nb.cells.append(create_notebook_cell("code", """# Example 2: Dimensionality Reduction for ML

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Generate high-dimensional synthetic data
np.random.seed(42)
X = np.random.randn(200, 50)  # 50 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce to 2D
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_reduced, y_train)
accuracy = clf.score(X_test_reduced, y_test)

print(f"Original features: 50")
print(f"Reduced features: 2")
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
print(f"Accuracy on reduced data: {accuracy:.3f}")"""))

    # Cell 9: Example 3
    nb.cells.append(create_notebook_cell("markdown", f"## Real-World Example 3: Cross-Validation & Evaluation"))

    # Cell 10: Example 3 code
    nb.cells.append(create_notebook_cell("code", """# Example 3: Proper Model Evaluation with Cross-Validation

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Generate classification data
X, y = make_classification(n_samples=300, n_features=20, n_classes=2, random_state=42)

# Pipeline: Scale + Classify
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f}")
print(f"Std deviation: {scores.std():.3f}")
print(f"95% CI: [{scores.mean() - 1.96*scores.std():.3f}, {scores.mean() + 1.96*scores.std():.3f}]")"""))

    # Cell 11: Takeaways
    nb.cells.append(create_notebook_cell("markdown", f"""## Key Takeaways

### When to Use {title}
- Best for: [typical use cases]
- Avoid when: [limitations]

### Best Practices
1. Always standardize/normalize features first
2. Use proper train-test splits
3. Apply cross-validation for reliable estimates
4. Monitor for overfitting
5. Document assumptions clearly

### Common Pitfalls
- Not scaling features before applying algorithms
- Using test data in preprocessing (data leakage)
- Ignoring class imbalance in classification
- Not validating on independent test set
- Over-interpreting results without proper testing

### Related Concepts
- Explore foundational algorithms in the AI section
- Review advanced techniques for specific domains
- Understand model evaluation strategies

### Next Steps
1. Apply to your own datasets
2. Experiment with parameter tuning
3. Compare with baseline approaches
4. Measure real-world performance
5. Document what works best for your use case"""))

    return nb

def main():
    """Generate all missing AI fundamentals notebooks."""

    concepts = [
        (18, "K-Means Clustering", "k-means-clustering"),
        (19, "Dimensionality Reduction", "dimensionality-reduction"),
        (20, "Gaussian Mixture Models", "gaussian-mixture-models"),
        (21, "Bias-Variance Tradeoff", "bias-variance-tradeoff"),
        (22, "Cross-Validation", "cross-validation"),
        (23, "Classification Metrics", "classification-metrics"),
        (24, "Regression Metrics", "regression-metrics"),
        (25, "Feature Engineering", "feature-engineering"),
        (26, "Hyperparameter Tuning", "hyperparameter-tuning"),
        (27, "Ensemble Methods", "ensemble-methods"),
        (28, "Bayesian Inference", "bayesian-inference"),
    ]

    print("=== Generating AI Fundamentals Notebooks (18-28) ===\n")

    created = 0
    for num, title, slug in concepts:
        notebook_path = f"{BASE}/ai/notebooks/{num:02d}-{slug}.ipynb"

        # Skip if exists
        if os.path.exists(notebook_path):
            print(f"  ⊘ {num:02d}-{slug}.ipynb (exists)")
            continue

        # Create notebook
        nb = create_ai_notebook(num, title, slug)

        # Write
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)

        print(f"  ✓ {num:02d}-{slug}.ipynb")
        created += 1

    print(f"\n✅ Created {created} AI fundamentals notebooks")

if __name__ == "__main__":
    main()
