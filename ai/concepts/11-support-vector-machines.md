# Support Vector Machines

## Detailed Explanation

Finds maximum-margin hyperplane with kernel trick...

## Core Intuition

A key technique in machine learning.

## How It Works

1. Map inputs to a high-dimensional feature space φ(x) using the kernel trick (no explicit computation needed)
2. Find the hyperplane w·φ(x) + b = 0 that maximizes the margin 2/‖w‖ between the two classes
3. Formulate as a convex optimization: minimize ½‖w‖² subject to yᵢ(w·φ(xᵢ)+b) ≥ 1
4. Introduce slack variables ξᵢ for soft margin (allow some misclassifications): minimize ½‖w‖² + C·Σξᵢ
5. Solve the dual problem using Lagrange multipliers — only support vectors (points on or inside the margin) have non-zero multipliers
6. Use the kernel function K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ) to compute inner products implicitly (RBF: e^(−γ‖xᵢ−xⱼ‖²))
7. Predict: ŷ = sign(Σᵢ αᵢyᵢK(xᵢ,x) + b), summing only over support vectors

```mermaid
graph TD
    A[Input] --> B[Process]
    B --> C[Output]
```

## Architecture / Trade-offs

Trade-off 1 vs trade-off 2

## Interview Q&A

**Q: When would you use Support Vector Machines?**
A: Context-dependent, varies by problem type.

**Q: What are the main trade-offs?**
A: Refer to Architecture / Trade-offs section above.

**Q: How do you choose hyperparameters?**
A: Cross-validation, grid/random/Bayesian search, domain knowledge.

**Q: What are common failure modes?**
A: Refer to Common Pitfalls section below.

## Best Practices

- Always scale features — SVMs are distance-based and scale-sensitive
- Use RBF kernel as default; only use linear kernel for text/very high-dimensional data
- Tune C and gamma together on log-scale grid
- Use probability=True only when calibrated probabilities needed (it's slower)
- For n > 50k samples consider LinearSVC or SGDClassifier instead
- Use class_weight='balanced' for imbalanced data
- Cache kernel computations with cache_size=2000 for large datasets

## Common Pitfalls

- Forgetting to scale features completely breaks SVM performance
- Training O(n²) to O(n³) makes vanilla SVM impractical beyond ~50k samples
- Tuning C and gamma independently — they interact and need joint grid search
- Using probability=True adds Platt scaling overhead — avoid when not needed


## Code Examples

### Example 1: Linear SVM

```python
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)

print(f"Support vectors: {svm_linear.n_support_}")
print(f"Train: {svm_linear.score(X_train, y_train):.4f}")
print(f"Test: {svm_linear.score(X_test, y_test):.4f}")
```

### Example 2: RBF Kernel SVM

```python
from sklearn.svm import SVC

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)

print(f"Linear kernel score: {svm_linear.score(X_test, y_test):.4f}")
print(f"RBF kernel score: {svm_rbf.score(X_test, y_test):.4f}")
```

### Example 3: Tuning C Parameter

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best C: {grid.best_params_['C']}")
print(f"Best CV score: {grid.best_score_:.4f}")
print(f"Test score: {grid.score(X_test, y_test):.4f}")
```

## Related Concepts

- [Gradient Descent](./01-gradient-descent.md)
- [Cross-Validation](./22-cross-validation.md)
- [Hyperparameter Tuning](./26-hyperparameter-tuning.md)
