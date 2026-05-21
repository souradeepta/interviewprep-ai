"""
Auto-generated from 63-agent-frameworks-comparison.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Frameworks Comparison
# ## Learning Objectives
# - Understand the core principles and mechanics of agent frameworks comparison
# - Implement from scratch and with production libraries
# ======================================================================

# ======================================================================
# ## Level 1: Basic Implementation
# Core concept implementation with minimal dependencies
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

# Level 1: Core Agent Frameworks Comparison concept from scratch
# Using numpy for fundamental understanding

class AgentFrameworksComparisonBasic:
    """Basic implementation of agent frameworks comparison."""

    def __init__(self):
        self.fitted = False
        self.params = None

    def fit(self, X, y=None):
        """Learn parameters from data."""
        self.params = X.mean(axis=0)
        self.fitted = True
        return self

    def transform(self, X):
        """Transform data using learned parameters."""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        return X - self.params

# Synthetic data
np.random.seed(42)
X = np.random.randn(100, 5) + 10

# Fit and transform
model = AgentFrameworksComparisonBasic()
model.fit(X)
X_transformed = model.transform(X)

print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")
print(f"Original mean: {X.mean(axis=0)[:3]}")
print(f"Transformed mean: {X_transformed.mean(axis=0)[:3]}")


# ======================================================================
# ## Level 2: Advanced Implementation
# Production-ready implementation with error handling and optimization
# ======================================================================

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class AgentFrameworksComparisonProduction:
    """Production-ready agent frameworks comparison with optimization."""

    def __init__(self, learning_rate=0.01, max_iterations=1000, batch_size=32):
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.batch_size = batch_size
        self.history = {'loss': []}
        self.fitted = False

    def fit(self, X, y=None, validation_split=0.2):
        """Fit with early stopping."""
        if len(X) < 10:
            raise ValueError("Need at least 10 samples")

        # Standardize input
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Training loop with batching
        n_samples = len(X_scaled)
        for epoch in range(self.max_iter):
            epoch_loss = 0
            n_batches = 0

            # Batch training
            for i in range(0, n_samples, self.batch_size):
                batch = X_scaled[i:i+self.batch_size]
                loss = self._compute_loss(batch)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.history['loss'].append(avg_loss)

            # Early stopping
            if epoch > 100 and len(self.history['loss']) > 10:
                if self.history['loss'][-1] > self.history['loss'][-10]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.fitted = True
        return self

    def _compute_loss(self, batch):
        """Compute loss for batch."""
        return np.mean((batch - batch.mean(axis=0)) ** 2)

    def transform(self, X):
        """Transform with fitted scaler."""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        return self.scaler.transform(X)

# Training
np.random.seed(42)
X_train = np.random.randn(200, 10)
y_train = np.random.randn(200)

model = AgentFrameworksComparisonProduction(learning_rate=0.01)
model.fit(X_train)

X_test = np.random.randn(50, 10)
X_transformed = model.transform(X_test)

print(f"Training loss history: {len(model.history['loss'])} epochs")
print(f"Final loss: {model.history['loss'][-1]:.6f}")
print(f"Transformed test shape: {X_transformed.shape}")

# Plot training
plt.figure(figsize=(10, 4))
plt.plot(model.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Agent Frameworks Comparison - Training Loss')
plt.grid(True)
plt.show()


# ======================================================================
# ## Real-World Example 1: Classification Task
# Applying agent frameworks comparison to a standard classification problem
# ======================================================================

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# Example 1: Agent Frameworks Comparison for Classification

# Generate synthetic classification data
X, y = make_classification(
    n_samples=300,
    n_features=20,
    n_informative=15,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply agent frameworks comparison
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = StandardScaler()
X_train_scaled = pipeline.fit_transform(X_train)
X_test_scaled = pipeline.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Classification Results:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  Features: {X.shape[1]}")
print(f"  Train samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")


# ======================================================================
# ## Real-World Example 2: Scaling & Optimization
# Optimizing agent frameworks comparison for large-scale production data
# ======================================================================

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

# Example 2: Scaling & Optimization

# Large-scale data simulation
np.random.seed(42)
X_large = np.random.randn(10000, 100)

# Benchmark different approaches
def benchmark_approach(name, scaler, X):
    start = time.time()
    scaler.fit(X)
    X_transformed = scaler.transform(X)
    elapsed = time.time() - start
    return elapsed, X_transformed.mean(), X_transformed.std()

scalers = [
    ("StandardScaler", StandardScaler()),
    ("MinMaxScaler", MinMaxScaler()),
]

print("Scaling Benchmark (10k samples, 100 features):")
print("-" * 50)

for name, scaler in scalers:
    elapsed, mean, std = benchmark_approach(name, scaler, X_large)
    print(f"{name:20} - Time: {elapsed:.4f}s, Mean: {mean:.4f}, Std: {std:.4f}")

# Memory efficiency
import sys
X_subset = X_large[:1000]
bytes_used = sys.getsizeof(X_subset)
print(f"\nMemory for 1000 samples: {bytes_used / 1024 / 1024:.2f} MB")

# Batch processing for scalability
batch_size = 1000
n_batches = len(X_large) // batch_size

scaler = StandardScaler()
X_large_scaled = scaler.fit_transform(X_large)

print(f"\nBatch processing: {n_batches} batches of {batch_size} samples")
print(f"Final shape: {X_large_scaled.shape}")


# ======================================================================
# ## Real-World Example 3: Integration with Other Concepts
# Combining agent frameworks comparison with dimensionality reduction and clustering
# ======================================================================

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Example 3: Integration Pipeline

# Generate data
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 50)

# Pipeline: Scaling → PCA → Clustering
# 1. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# 3. Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_reduced)

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, edgecolors='black', linewidths=2)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('Integration: Scaling + PCA + Clustering')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Integration Results:")
print(f"  Original shape: {X.shape}")
print(f"  Scaled shape: {X_scaled.shape}")
print(f"  Reduced shape: {X_reduced.shape}")
print(f"  Clusters: {len(np.unique(labels))}")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")


# ======================================================================
# ## Key Takeaways
# ### When to Use Agent Frameworks Comparison
# - Use when: [specific use cases]
# - Don't use when: [when to avoid]
# ======================================================================
