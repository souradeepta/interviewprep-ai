# Decision Trees

## Detailed Explanation

Recursively partitions feature space with axis-aligned splits...

## Core Intuition

A key technique in machine learning.

## How It Works

1. Step 1
2. Step 2
3. Step 3

```mermaid
graph TD
    A[Input] --> B[Process]
    B --> C[Output]
```

## Architecture / Trade-offs

Trade-off 1 vs trade-off 2

## Interview Q&A

**Q: When would you use Decision Trees?**
A: Context-dependent, varies by problem type.

**Q: What are the main trade-offs?**
A: Refer to Architecture / Trade-offs section above.

**Q: How do you choose hyperparameters?**
A: Cross-validation, grid/random/Bayesian search, domain knowledge.

**Q: What are common failure modes?**
A: Refer to Common Pitfalls section below.

## Best Practices

- Practice 1
- Practice 2
- Practice 3

## Common Pitfalls

- Pitfall 1
- Pitfall 2


## Code Examples

### Example 1: CART Algorithm with Gini

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.load_iris(return_X_y=True)
X, y = X[:, :2], y  # Use first 2 features for visualization

# Train decision tree
dt = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
dt.fit(X, y)

# Feature importance
print("Feature importances:")
for i, imp in enumerate(dt.feature_importances_):
    print(f"  Feature {i}: {imp:.4f}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=['SepalLength', 'SepalWidth'], class_names=['Setosa', 'Versicolor', 'Virginica'])
plt.show()
```

### Example 2: Pruning to Prevent Overfitting

```python
from sklearn.tree import DecisionTreeClassifier

# Train unpruned tree
dt_deep = DecisionTreeClassifier(random_state=42)
dt_deep.fit(X, y)

# Train pruned tree
dt_pruned = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
dt_pruned.fit(X, y)

train_score_deep = dt_deep.score(X, y)
train_score_pruned = dt_pruned.score(X, y)

print(f"Deep tree - Depth: {dt_deep.get_depth()}, Train accuracy: {train_score_deep:.4f}")
print(f"Pruned tree - Depth: {dt_pruned.get_depth()}, Train accuracy: {train_score_pruned:.4f}")
```

### Example 3: Classification Tree

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)
scores = cross_val_score(dt, X, y, cv=5)

print(f"5-fold CV scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Related Concepts

- [Gradient Descent](./01-gradient-descent.md)
- [Cross-Validation](./22-cross-validation.md)
- [Hyperparameter Tuning](./26-hyperparameter-tuning.md)
