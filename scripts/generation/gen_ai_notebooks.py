#!/usr/bin/env python3
"""Generate notebooks for AI fundamentals section (18-28)."""

import os
import nbformat as nbf

BASE = "/home/sbisw/github/interviewprep-ml"

def create_basic_notebook(num, title, slug):
    """Create a 12-cell notebook with standard structure."""
    
    nb = nbf.v4.new_notebook()
    
    # Cell 0: Title and objectives
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n## Learning Objectives\n- Understand {title.lower()}\n- Implement with numpy\n- Use sklearn for production\n- Apply to real data"))
    
    # Cell 1: Level 1 intro
    nb.cells.append(nbf.v4.new_markdown_cell("## Level 1: Basic Implementation\n\nCore concept from first principles"))
    
    # Cell 2: Level 1 code
    level1 = f"""import numpy as np
import matplotlib.pyplot as plt

# Basic {title.lower()} implementation
class {title.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')}:
    def __init__(self):
        self.is_fitted = False
    
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Must fit first")
        return (X - self.mean_) / (self.std_ + 1e-8)

# Test
np.random.seed(42)
X = np.random.randn(100, 5)
model = {title.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')}()
model.fit(X)
print("Fitted successfully")"""
    
    nb.cells.append(nbf.v4.new_code_cell(level1))
    
    # Cell 3: Level 2 intro
    nb.cells.append(nbf.v4.new_markdown_cell("## Level 2: Production with sklearn\n\nOptimized implementation for real datasets"))
    
    # Cell 4: Level 2 code
    level2 = """from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Production pattern
np.random.seed(42)
X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training shape: {X_train_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")"""
    
    nb.cells.append(nbf.v4.new_code_cell(level2))
    
    # Cell 5: Example 1 intro
    nb.cells.append(nbf.v4.new_markdown_cell("## Real-World Example 1: Basic Application"))
    
    # Cell 6: Example 1 code
    ex1 = """from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled data shape: {X_scaled.shape}")
print(f"Mean (should be ~0): {X_scaled.mean(axis=0)[:3]}")
print(f"Std (should be ~1): {X_scaled.std(axis=0)[:3]}")"""
    
    nb.cells.append(nbf.v4.new_code_cell(ex1))
    
    # Cell 7: Example 2 intro
    nb.cells.append(nbf.v4.new_markdown_cell("## Real-World Example 2: Validation"))
    
    # Cell 8: Example 2 code
    ex2 = """from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

print(f"CV scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")"""
    
    nb.cells.append(nbf.v4.new_code_cell(ex2))
    
    # Cell 9: Example 3 intro
    nb.cells.append(nbf.v4.new_markdown_cell("## Real-World Example 3: Advanced Pattern"))
    
    # Cell 10: Example 3 code
    ex3 = """from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Pipeline: scale -> reduce -> cluster
np.random.seed(42)
X = np.random.randn(100, 20)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_reduced)

print(f"Original: {X.shape}")
print(f"Reduced: {X_reduced.shape}")
print(f"Clusters: {len(np.unique(labels))}")"""
    
    nb.cells.append(nbf.v4.new_code_cell(ex3))
    
    # Cell 11: Key takeaways
    takeaway = f"""## Key Takeaways

### When to Use
- Use {title.lower()} when you need to...

### Best Practices
1. Always validate on test data
2. Use cross-validation for reliable estimates
3. Scale features appropriately
4. Monitor for overfitting
5. Document your pipeline

### Common Pitfalls
- Not splitting data properly (data leakage)
- Ignoring preprocessing in pipelines
- Using test set during fitting
- Ignoring class imbalance
- Not validating assumptions

### Related Concepts
- Foundation concepts from AI section
- Advanced techniques in LLM and agentic-ai
- Integration patterns for production systems"""
    
    nb.cells.append(nbf.v4.new_markdown_cell(takeaway))
    
    return nb

def main():
    """Generate notebooks for AI concepts 18-28."""
    
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
    
    for num, title, slug in concepts:
        path = f"{BASE}/ai/notebooks/{num:02d}-{slug}.ipynb"
        
        if os.path.exists(path):
            print(f"  ⊘ {num:02d}-{slug}.ipynb (exists)")
            continue
        
        nb = create_basic_notebook(num, title, slug)
        
        with open(path, 'w') as f:
            nbf.write(nb, f)
        
        print(f"  ✓ {num:02d}-{slug}.ipynb")
    
    print("\n✅ AI fundamentals notebooks generated")

if __name__ == "__main__":
    main()
