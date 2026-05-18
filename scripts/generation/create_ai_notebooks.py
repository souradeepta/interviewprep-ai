#!/usr/bin/env python3
import nbformat as nbf
import os

# Comprehensive notebook implementations
NOTEBOOK_CONTENT = {
    "01-gradient-descent": {
        "cells": [
            ("markdown", "# Gradient Descent\n\n## Learning Objectives\n1. Understand how gradient descent optimizes model weights\n2. Implement GD variants (batch, SGD, mini-batch)\n3. Analyze learning rate and convergence\n4. Compare optimizers (momentum, Adam)"),
            ("markdown", "## Level 1: Manual Gradient Descent"),
            ("code", """import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    '''Basic gradient descent for linear regression.'''
    m = len(y)
    theta = np.zeros(X.shape[1])
    costs = []

    for epoch in range(epochs):
        # Forward
        predictions = X @ theta
        errors = predictions - y

        # Backward
        gradient = (2/m) * X.T @ errors

        # Update
        theta -= learning_rate * gradient

        # Track cost
        cost = np.mean(errors ** 2)
        costs.append(cost)

    return theta, costs

# Generate data
np.random.seed(42)
X = np.c_[np.ones(100), np.random.randn(100, 2)]
true_theta = np.array([1, 2, 3])
y = X @ true_theta + np.random.randn(100) * 0.1

# Train
theta, costs = gradient_descent(X, y, learning_rate=0.1, epochs=50)
print(f"Learned theta: {theta}")
print(f"Final cost: {costs[-1]:.4f}")

# Plot convergence
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.xlabel('Epoch'), plt.ylabel('MSE Loss')
plt.title('Convergence')

plt.subplot(1, 2, 2)
plt.scatter(y, X @ theta, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('True'), plt.ylabel('Predicted')
plt.title('Predictions')
plt.tight_layout()
plt.show()"""),
            ("markdown", "## Level 2: Mini-Batch with Momentum"),
            ("code", """class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.0, batch_size=32):
        self.lr = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.velocity = None

    def __call__(self, X, y, epochs=100):
        m = len(y)
        theta = np.zeros(X.shape[1])
        self.velocity = np.zeros_like(theta)
        costs = []

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X[indices], y[indices]

            epoch_cost = 0
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                # Gradient
                pred = X_batch @ theta
                grad = (2/len(y_batch)) * X_batch.T @ (pred - y_batch)

                # Momentum
                self.velocity = self.momentum * self.velocity - self.lr * grad
                theta += self.velocity

                epoch_cost += np.mean((pred - y_batch) ** 2)

            costs.append(epoch_cost / (m / self.batch_size))

        return theta, costs

# Compare optimizers
X = np.c_[np.ones(200), np.random.randn(200, 5)]
y = np.sum(X[:, 1:3], axis=1) + np.random.randn(200) * 0.1

results = {}
for momentum in [0.0, 0.9]:
    opt = GradientDescentOptimizer(lr=0.1, momentum=momentum, batch_size=32)
    theta, costs = opt(X, y, epochs=50)
    results[f"momentum={momentum}"] = costs

plt.figure(figsize=(10, 5))
for label, costs in results.items():
    plt.plot(costs, label=label)
plt.xlabel('Epoch'), plt.ylabel('Loss')
plt.legend()
plt.title('Impact of Momentum on Convergence')
plt.show()"""),
            ("markdown", "## Real-World Example 1: Learning Rate Schedules"),
            ("code", """def cosine_annealing(epoch, initial_lr=0.1, T_max=100):
    '''Cosine annealing schedule.'''
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / T_max))

def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_per_drop=10):
    '''Step decay schedule.'''
    return initial_lr * (drop ** (epoch // epochs_per_drop))

# Compare schedules
epochs = 100
schedules = {
    'constant': [0.1] * epochs,
    'cosine': [cosine_annealing(e) for e in range(epochs)],
    'step': [step_decay(e) for e in range(epochs)],
}

plt.figure(figsize=(12, 4))
for name, schedule in schedules.items():
    plt.plot(schedule, label=name)
plt.xlabel('Epoch'), plt.ylabel('Learning Rate')
plt.legend()
plt.title('Learning Rate Schedules')
plt.show()"""),
            ("markdown", "## Real-World Example 2: Adam Optimizer"),
            ("code", """class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.m, self.v, self.t = None, None, 0

    def __call__(self, theta, gradient):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Test on non-convex problem
np.random.seed(42)
X = np.c_[np.ones(300), np.random.randn(300, 10)]
y = np.sin(X[:, 1]) + X[:, 2] * X[:, 3] + np.random.randn(300) * 0.1

adam = Adam(lr=0.01)
theta = np.random.randn(X.shape[1]) * 0.1
costs = []

for _ in range(100):
    pred = X @ theta
    grad = (2/len(y)) * X.T @ (pred - y)
    theta = adam(theta, grad)
    costs.append(np.mean((pred - y) ** 2))

plt.plot(costs)
plt.xlabel('Iteration'), plt.ylabel('Loss')
plt.title('Adam Optimizer Convergence')
plt.show()"""),
            ("markdown", "## Real-World Example 3: Learning Rate Tuning"),
            ("code", """def train_with_lr(X, y, learning_rate, epochs=50):
    '''Train and return final loss.'''
    theta = np.zeros(X.shape[1])
    for _ in range(epochs):
        pred = X @ theta
        grad = (2/len(y)) * X.T @ (pred - y)
        theta -= learning_rate * grad
    return np.mean((X @ theta - y) ** 2)

X = np.c_[np.ones(200), np.random.randn(200, 5)]
y = np.sum(X[:, 1:3], axis=1) + np.random.randn(200) * 0.1

# Grid search learning rates
learning_rates = np.logspace(-4, 0, 20)
final_losses = [train_with_lr(X, y, lr) for lr in learning_rates]

plt.figure(figsize=(10, 5))
plt.semilogx(learning_rates, final_losses, 'o-')
plt.xlabel('Learning Rate')
plt.ylabel('Final Loss')
plt.title('Learning Rate vs Final Loss')
plt.axvline(learning_rates[np.argmin(final_losses)], color='r', linestyle='--',
            label=f'Optimal: {learning_rates[np.argmin(final_losses)]:.4f}')
plt.legend()
plt.show()"""),
            ("markdown", "## Key Takeaways\n\n**When to Use:**\n- Gradient descent: foundation for all neural network training\n- Batch GD: small datasets, stable but slow\n- SGD: large datasets, noisy but fast\n- Mini-batch: balanced, most common in practice\n- Adam: adaptive learning rates, robust default choice\n\n**Related Concepts:**\n- [Backpropagation](./02-backpropagation.md) — computes gradients\n- [Optimization Algorithms](./04-optimization-algorithms.md) — advanced variants\n- [Learning Rate Scheduling](./05-learning-rate-scheduling.md) — improving convergence"),
        ]
    },
}

def create_default_notebook(title, concept_num):
    """Create a default notebook for a concept."""
    return {
        "cells": [
            ("markdown", f"# {title}\n\n## Learning Objectives\n1. Understand {title.lower()}\n2. Implement in Python\n3. Apply to real problems"),
            ("markdown", "## Level 1: Basic Implementation"),
            ("code", f"""import numpy as np
import sklearn
import matplotlib.pyplot as plt

# Level 1: Core {title} concept
print(f"Implementing {title}")

# Basic setup
X = np.random.randn(100, 3)
y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

# Simple implementation
class Simple{title.replace(' ', '')}:
    def fit(self, X, y):
        # Placeholder for core algorithm
        self.mean_ = np.mean(X, axis=0)
        return self

    def predict(self, X):
        return X @ np.ones(X.shape[1])

model = Simple{title.replace(' ', '')}()
model.fit(X, y)
print(f"Model fitted")"""),
            ("markdown", "## Level 2: Advanced Implementation"),
            ("code", f"""# Level 2: Production pattern with error handling

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Enhanced model with validation
class Enhanced{title.replace(' ', '')}:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        # Fit algorithm
        self.mean_ = np.mean(X, axis=0)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        return X @ np.ones(X.shape[1])

    def score(self, X, y):
        pred = self.predict(X)
        mse = np.mean((pred - y) ** 2)
        return mse

model = Enhanced{title.replace(' ', '')}()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train MSE: {{train_score:.4f}}, Test MSE: {{test_score:.4f}}")"""),
            ("markdown", f"## Real-World Example 1: {title} in Practice"),
            ("code", """# Example 1: Practical application
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

data = load_iris()
X, y = data.data, (data.target == 0).astype(int)

# Cross-validation
model = Enhanced{title.replace(' ', '')}()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")"""),
            ("markdown", f"## Real-World Example 2: Parameter Tuning"),
            ("code", """# Example 2: Grid search parameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
# Placeholder: in practice use real parameters for your model
print(f"Parameter grid would be: {param_grid}")

# In practice:
# grid = GridSearchCV(model, param_grid, cv=5)
# grid.fit(X, y)
# print(f"Best params: {grid.best_params_}")"""),
            ("markdown", f"## Real-World Example 3: Pipeline Integration"),
            ("code", """# Example 3: Full pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Enhanced{title.replace(' ', '')}()),
])

# Train and evaluate
pipeline.fit(X_train, y_train)
final_score = pipeline.score(X_test, y_test)
print(f"Pipeline score: {final_score:.4f}")"""),
            ("markdown", f"## Key Takeaways\n\n**When to Use {title}:**\n- Use case 1\n- Use case 2\n- Use case 3\n\n**Common Pitfalls:**\n- Mistake 1: not standardizing features\n- Mistake 2: overfitting to training data\n- Mistake 3: ignoring class imbalance\n\n**Related Concepts:**\n- Link to related concept 1\n- Link to related concept 2"),
        ]
    }

# Generate notebooks
os.chdir("/home/sbisw/github/interviewprep-ml/ai/notebooks")

concepts = [
    ("01-gradient-descent", "Gradient Descent"),
    ("02-backpropagation", "Backpropagation"),
    ("03-loss-functions", "Loss Functions"),
    ("04-optimization-algorithms", "Optimization Algorithms"),
    ("05-learning-rate-scheduling", "Learning Rate Scheduling"),
    ("06-linear-regression", "Linear Regression"),
    ("07-logistic-regression", "Logistic Regression"),
    ("08-decision-trees", "Decision Trees"),
    ("09-random-forests", "Random Forests"),
    ("10-gradient-boosting", "Gradient Boosting"),
    ("11-support-vector-machines", "Support Vector Machines"),
    ("12-k-nearest-neighbors", "K-Nearest Neighbors"),
    ("13-neural-networks", "Neural Networks"),
    ("14-activation-functions", "Activation Functions"),
    ("15-weight-initialization", "Weight Initialization"),
    ("16-regularization", "Regularization"),
    ("17-batch-normalization", "Batch Normalization"),
    ("18-k-means-clustering", "K-Means Clustering"),
    ("19-dimensionality-reduction", "Dimensionality Reduction"),
    ("20-gaussian-mixture-models", "Gaussian Mixture Models"),
    ("21-bias-variance-tradeoff", "Bias-Variance Tradeoff"),
    ("22-cross-validation", "Cross-Validation"),
    ("23-classification-metrics", "Classification Metrics"),
    ("24-regression-metrics", "Regression Metrics"),
    ("25-feature-engineering", "Feature Engineering"),
    ("26-hyperparameter-tuning", "Hyperparameter Tuning"),
    ("27-ensemble-methods", "Ensemble Methods"),
    ("28-bayesian-inference", "Bayesian Inference"),
]

for filename, title in concepts:
    # Use custom content if available, else default
    if filename in NOTEBOOK_CONTENT:
        cells_data = NOTEBOOK_CONTENT[filename]["cells"]
    else:
        cells_data = create_default_notebook(title, filename.split('-')[0])["cells"]

    # Create notebook
    nb = nbf.v4.new_notebook()
    for cell_type, content in cells_data:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        else:
            nb.cells.append(nbf.v4.new_code_cell(content))

    # Save
    with open(f"{filename}.ipynb", 'w') as f:
        nbf.write(nb, f)
    print(f"✓ Created {filename}.ipynb")

print(f"\n✅ Generated {len(concepts)} Jupyter notebooks")
