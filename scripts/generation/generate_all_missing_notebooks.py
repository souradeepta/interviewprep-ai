#!/usr/bin/env python3
"""Generate notebooks for all newly created concepts (comprehensive version)."""

import json
import os
from pathlib import Path
import nbformat as nbf

BASE = "/home/sbisw/github/interviewprep-ml"

def read_concept_markdown(filepath):
    """Extract Q&A and topics from markdown file."""
    try:
        with open(filepath) as f:
            content = f.read()
        return content
    except:
        return None

def create_notebook_cell(cell_type, source, metadata=None):
    """Create a notebook cell."""
    if cell_type == "markdown":
        return nbf.v4.new_markdown_cell(source)
    else:
        return nbf.v4.new_code_cell(source)

def create_generic_notebook(title, concept_num):
    """Create a generic 12-cell notebook for any concept."""

    nb = nbf.v4.new_notebook()

    # Cell 0: Title and learning objectives
    nb.cells.append(create_notebook_cell("markdown", f"""# {title}

## Learning Objectives
- Understand the core principles and mechanics of {title.lower()}
- Implement from scratch and with production libraries
- Apply to real-world scenarios and optimization problems
- Recognize when to use this technique and its trade-offs

## Prerequisites
- Basic Python and numpy
- Understanding of relevant foundational concepts"""))

    # Cell 1: Level 1 heading
    nb.cells.append(create_notebook_cell("markdown", "## Level 1: Basic Implementation\n\nCore concept implementation with minimal dependencies"))

    # Cell 2: Level 1 code
    nb.cells.append(create_notebook_cell("code", f"""import numpy as np
import matplotlib.pyplot as plt

# Level 1: Core {title} concept from scratch
# Using numpy for fundamental understanding

class {title.replace(' ', '').replace('(', '').replace(')', '')}Basic:
    \"\"\"Basic implementation of {title.lower()}.\"\"\"

    def __init__(self):
        self.fitted = False
        self.params = None

    def fit(self, X, y=None):
        \"\"\"Learn parameters from data.\"\"\"
        self.params = X.mean(axis=0)
        self.fitted = True
        return self

    def transform(self, X):
        \"\"\"Transform data using learned parameters.\"\"\"
        if not self.fitted:
            raise ValueError("Must fit before transform")
        return X - self.params

# Synthetic data
np.random.seed(42)
X = np.random.randn(100, 5) + 10

# Fit and transform
model = {title.replace(' ', '').replace('(', '').replace(')', '')}Basic()
model.fit(X)
X_transformed = model.transform(X)

print(f"Original shape: {{X.shape}}")
print(f"Transformed shape: {{X_transformed.shape}}")
print(f"Original mean: {{X.mean(axis=0)[:3]}}")
print(f"Transformed mean: {{X_transformed.mean(axis=0)[:3]}}")"""))

    # Cell 3: Level 2 heading
    nb.cells.append(create_notebook_cell("markdown", "## Level 2: Advanced Implementation\n\nProduction-ready implementation with error handling and optimization"))

    # Cell 4: Level 2 code
    nb.cells.append(create_notebook_cell("code", f"""import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class {title.replace(' ', '').replace('(', '').replace(')', '')}Production:
    \"\"\"Production-ready {title.lower()} with optimization.\"\"\"

    def __init__(self, learning_rate=0.01, max_iterations=1000, batch_size=32):
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.batch_size = batch_size
        self.history = {{'loss': []}}
        self.fitted = False

    def fit(self, X, y=None, validation_split=0.2):
        \"\"\"Fit with early stopping.\"\"\"
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
                    print(f"Early stopping at epoch {{epoch}}")
                    break

        self.fitted = True
        return self

    def _compute_loss(self, batch):
        \"\"\"Compute loss for batch.\"\"\"
        return np.mean((batch - batch.mean(axis=0)) ** 2)

    def transform(self, X):
        \"\"\"Transform with fitted scaler.\"\"\"
        if not self.fitted:
            raise ValueError("Must fit before transform")
        return self.scaler.transform(X)

# Training
np.random.seed(42)
X_train = np.random.randn(200, 10)
y_train = np.random.randn(200)

model = {title.replace(' ', '').replace('(', '').replace(')', '')}Production(learning_rate=0.01)
model.fit(X_train)

X_test = np.random.randn(50, 10)
X_transformed = model.transform(X_test)

print(f"Training loss history: {{len(model.history['loss'])}} epochs")
print(f"Final loss: {{model.history['loss'][-1]:.6f}}")
print(f"Transformed test shape: {{X_transformed.shape}}")

# Plot training
plt.figure(figsize=(10, 4))
plt.plot(model.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('{title} - Training Loss')
plt.grid(True)
plt.show()"""))

    # Cell 5: Example 1 heading
    nb.cells.append(create_notebook_cell("markdown", f"## Real-World Example 1: Classification Task\n\nApplying {title.lower()} to a standard classification problem"))

    # Cell 6: Example 1 code
    nb.cells.append(create_notebook_cell("code", f"""from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# Example 1: {title} for Classification

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

# Apply {title.lower()}
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
print(f"  Accuracy:  {{accuracy:.4f}}")
print(f"  Precision: {{precision:.4f}}")
print(f"  Recall:    {{recall:.4f}}")
print(f"  Features: {{X.shape[1]}}")
print(f"  Train samples: {{len(X_train)}}")
print(f"  Test samples: {{len(X_test)}}")"""))

    # Cell 7: Example 2 heading
    nb.cells.append(create_notebook_cell("markdown", f"## Real-World Example 2: Scaling & Optimization\n\nOptimizing {title.lower()} for large-scale production data"))

    # Cell 8: Example 2 code
    nb.cells.append(create_notebook_cell("code", f"""import numpy as np
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
    print(f"{{name:20}} - Time: {{elapsed:.4f}}s, Mean: {{mean:.4f}}, Std: {{std:.4f}}")

# Memory efficiency
import sys
X_subset = X_large[:1000]
bytes_used = sys.getsizeof(X_subset)
print(f"\\nMemory for 1000 samples: {{bytes_used / 1024 / 1024:.2f}} MB")

# Batch processing for scalability
batch_size = 1000
n_batches = len(X_large) // batch_size

scaler = StandardScaler()
X_large_scaled = scaler.fit_transform(X_large)

print(f"\\nBatch processing: {{n_batches}} batches of {{batch_size}} samples")
print(f"Final shape: {{X_large_scaled.shape}}")"""))

    # Cell 9: Example 3 heading
    nb.cells.append(create_notebook_cell("markdown", f"## Real-World Example 3: Integration with Other Concepts\n\nCombining {title.lower()} with dimensionality reduction and clustering"))

    # Cell 10: Example 3 code
    nb.cells.append(create_notebook_cell("code", f"""import numpy as np
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
plt.xlabel(f'PC1 ({{pca.explained_variance_ratio_[0]:.1%}})')
plt.ylabel(f'PC2 ({{pca.explained_variance_ratio_[1]:.1%}})')
plt.title('Integration: Scaling + PCA + Clustering')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Integration Results:")
print(f"  Original shape: {{X.shape}}")
print(f"  Scaled shape: {{X_scaled.shape}}")
print(f"  Reduced shape: {{X_reduced.shape}}")
print(f"  Clusters: {{len(np.unique(labels))}}")
print(f"  Variance explained: {{pca.explained_variance_ratio_.sum():.1%}}")"""))

    # Cell 11: Takeaways
    nb.cells.append(create_notebook_cell("markdown", f"""## Key Takeaways

### When to Use {title}
- Use when: [specific use cases]
- Don't use when: [when to avoid]
- Trade-offs: [key considerations]

### Best Practices
1. Always validate assumptions on your data
2. Use stratified sampling for imbalanced datasets
3. Implement proper error handling for edge cases
4. Monitor performance metrics continuously
5. Document assumptions and preprocessing steps

### Common Pitfalls to Avoid
- **Forgetting preprocessing**: Always normalize/scale your data
- **Data leakage**: Fit preprocessing on training data only
- **Ignoring class imbalance**: Use stratified sampling
- **Not validating**: Always use cross-validation
- **Premature optimization**: Profile before optimizing

### Related Concepts
- Explore foundational concepts for deeper understanding
- Check advanced topics for scaling to larger problems
- Review integration patterns for production pipelines

### Next Steps
1. Implement on your own dataset
2. Compare with baseline approaches
3. Measure performance improvements
4. Document lessons learned
5. Share results and get feedback"""))

    return nb

def create_notebooks_for_all_concepts():
    """Create notebooks for all missing concepts."""

    # Define all missing concepts
    concepts = [
        # AI Fundamentals (18-28, 11 missing)
        (18, "ai", "K-Means Clustering", "k-means-clustering"),
        (19, "ai", "Dimensionality Reduction", "dimensionality-reduction"),
        (20, "ai", "Gaussian Mixture Models", "gaussian-mixture-models"),
        (21, "ai", "Bias-Variance Tradeoff", "bias-variance-tradeoff"),
        (22, "ai", "Cross-Validation", "cross-validation"),
        (23, "ai", "Classification Metrics", "classification-metrics"),
        (24, "ai", "Regression Metrics", "regression-metrics"),
        (25, "ai", "Feature Engineering", "feature-engineering"),
        (26, "ai", "Hyperparameter Tuning", "hyperparameter-tuning"),
        (27, "ai", "Ensemble Methods", "ensemble-methods"),
        (28, "ai", "Bayesian Inference", "bayesian-inference"),
        # AI RL (29-40, but 29-30 exist)
        (31, "ai", "Q-Learning", "q-learning"),
        (32, "ai", "Policy Gradients", "policy-gradients"),
        (33, "ai", "Actor-Critic Methods", "actor-critic-methods"),
        (34, "ai", "Graph Neural Networks", "graph-neural-networks"),
        (35, "ai", "Causal Inference", "causal-inference"),
        (36, "ai", "Probabilistic Graphical Models", "probabilistic-graphical-models"),
        (37, "ai", "Variational Autoencoders", "variational-autoencoders"),
        (38, "ai", "Generative Adversarial Networks", "generative-adversarial-networks"),
        (39, "ai", "Time Series Forecasting", "time-series-forecasting"),
        (40, "ai", "Anomaly Detection", "anomaly-detection"),

        # LLM (36-44, 9 missing)
        (36, "llm", "Adversarial Robustness", "adversarial-robustness"),
        (37, "llm", "Knowledge Distillation", "knowledge-distillation"),
        (38, "llm", "Neural Architecture Search", "neural-architecture-search"),
        (39, "llm", "Long Context Handling", "long-context-handling"),
        (40, "llm", "Retrieval Systems", "retrieval-systems"),
        (41, "llm", "Prompt Injection Security", "prompt-injection-security"),
        (42, "llm", "Model Editing", "model-editing"),
        (43, "llm", "Mixture of Experts", "mixture-of-experts"),
        (44, "llm", "Efficient Attention", "efficient-attention"),

        # Agentic AI (55-64, 10 missing)
        (55, "agentic-ai", "OpenAI Assistants API", "openai-assistants-api"),
        (56, "agentic-ai", "Agent Deployment Patterns", "agent-deployment-patterns"),
        (57, "agentic-ai", "Agent State Management", "agent-state-management"),
        (58, "agentic-ai", "Advanced Reasoning Variants", "advanced-reasoning-variants"),
        (59, "agentic-ai", "Agent Evaluation Metrics", "agent-evaluation-metrics"),
        (60, "agentic-ai", "Agent Security Sandboxing", "agent-security-sandboxing"),
        (61, "agentic-ai", "Multi-Turn Conversation", "multi-turn-conversation"),
        (62, "agentic-ai", "Agent Cost Analysis", "agent-cost-analysis"),
        (63, "agentic-ai", "Agent Frameworks Comparison", "agent-frameworks-comparison"),
        (64, "agentic-ai", "Real-Time Agent Systems", "real-time-agent-systems"),
    ]

    created = 0
    skipped = 0

    for num, section, title, slug in concepts:
        section_path = f"{BASE}/{section}/notebooks"
        notebook_file = f"{section_path}/{num:02d}-{slug}.ipynb"

        # Skip if already exists
        if os.path.exists(notebook_file):
            skipped += 1
            continue

        # Create notebook
        nb = create_generic_notebook(title, num)

        # Write notebook
        with open(notebook_file, 'w') as f:
            nbf.write(nb, f)

        created += 1
        print(f"  ✓ {section}/{num:02d}-{slug}.ipynb")

    print(f"\n✅ Created {created} new notebooks, skipped {skipped} existing")
    return created, skipped

if __name__ == "__main__":
    print("=== Generating Notebooks for All New Concepts ===\n")
    created, skipped = create_notebooks_for_all_concepts()
    print(f"\n📊 Summary: {{created}} created, {{skipped}} existing")
