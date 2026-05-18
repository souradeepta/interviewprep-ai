# Decision Trees

## Detailed Explanation

Recursively partitions feature space with axis-aligned splits...

## Core Intuition

A key technique in machine learning.

## How It Works

1. Start with the full training dataset at the root node
2. For each candidate feature and split threshold, compute impurity (Gini or entropy) of resulting child nodes
3. Select the split that maximizes information gain (parent impurity − weighted child impurity)
4. Recursively split each child node using the same procedure
5. Stop when a stopping criterion is met: max_depth reached, min_samples_leaf, or no impurity improvement
6. Assign each leaf node the majority class (classification) or mean value (regression) of its training samples
7. Optionally prune the tree post-hoc by removing splits that don't improve validation performance

```mermaid
graph TD
    A[Input] --> B[Process]
    B --> C[Output]
```

## Architecture / Trade-offs

Trade-off 1 vs trade-off 2

## Interview Q&A

**Q: Why do decision trees overfit, and what are the main ways to prevent it?**
A: Unpruned trees can grow to depth n-1 (one leaf per training point), perfectly memorizing labels. The fix is controlling capacity: set max_depth (3-6), min_samples_leaf (5-20), or min_impurity_decrease. These hyperparameters should be tuned with cross-validation. Alternatively, use tree ensembles (Random Forest, GBM) which inherently reduce overfitting through averaging.

**Q: When would you prefer a decision tree over a Random Forest?**
A: When interpretability is critical — a single shallow tree (depth 3-4) is fully explainable to non-technical stakeholders and can be printed as a flowchart. Decision trees are also faster to train and predict, and better for rule extraction (convert tree to if-else rules). For predictive performance, Random Forests almost always win; for transparency, single trees are preferred.

**Q: What is the difference between Gini impurity and entropy as splitting criteria?**
A: Both measure class impurity and usually yield very similar trees. Gini is slightly faster to compute (no log). Entropy (information gain) can sometimes create more balanced splits. In practice the difference is negligible — sklearn uses Gini by default. Choose entropy if you want information-theoretic interpretability; otherwise Gini is fine.

**Q: How are decision trees biased toward certain feature types?**
A: Trees are biased toward features with many possible split points (high-cardinality continuous features) and features with many unique values, because they offer more chances for a high-information-gain split purely by chance. This inflates their apparent importance. Use permutation importance instead of Gini importance to get unbiased feature rankings.

**Q: How would you debug a decision tree that performs well on training but poorly on validation?**
A: The tree is overfitting. Check tree depth (if unlimited, start by setting max_depth=5); check if training accuracy is near 100% while validation is much lower; plot validation accuracy vs max_depth (should peak then decline). Also check for data leakage — a feature that's a proxy for the target will look great in training but fail in production.

**Q: Can decision trees handle missing values natively?**
A: Standard CART (sklearn's implementation) cannot — you must impute before fitting. However, some implementations like XGBoost and LightGBM handle missing values natively by learning the best direction to send missing values at each split. If using sklearn trees, impute with median (numeric) or mode (categorical), or add a missing indicator feature.
## Best Practices

- Set max_depth (3-6) or min_samples_leaf to prevent overfitting
- Use feature importances to identify noisy features
- Prune trees post-training for interpretability
- Visualize with sklearn.tree.plot_tree or export_graphviz
- Use Gini for classification speed, entropy when you need information gain interpretation
- Always validate depth with cross-validation, not just training accuracy
- Use min_impurity_decrease to stop splits below a threshold

## Common Pitfalls

- Unpruned trees perfectly memorize training data (max depth=None means overfit)
- Biased toward high-cardinality features in splitting criteria
- Unstable — small data changes create very different trees
- Single trees are weak learners — use ensembles (RF, GBM) in production


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
