# Support Vector Machines (SVMs)

## TL;DR
SVMs find the maximum-margin hyperplane separating two classes. The kernel trick maps data to high-dimensional space where separation is possible, without explicitly materializing the mapping. Powerful for small-to-medium datasets, interpretable margins, but harder to scale than trees.

## Core Intuition
Draw the widest possible line between two clouds of points. Points on the margin are "support vectors" — they define the boundary. Move one support vector slightly and the line moves. Everything else is irrelevant.

## How It Works

**Hard margin:** $\min_{w,b} \frac{1}{2}\|w\|^2$ subject to $y_i(w^Tx_i + b) \geq 1$.

**Soft margin (C-SVM):** allows misclassification via slack variables $\xi_i$. 
$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$ subject to $y_i(w^Tx_i + b) \geq 1 - \xi_i$.

**Kernel trick:** use $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ to compute dot products in high-d space without materializing $\phi$.

Common kernels: linear, polynomial $(x^Tz+c)^d$, RBF $\exp(-\gamma\|x-z\|^2)$.

## Key Properties / Trade-offs
- **Interpretability:** margin and support vectors have geometric meaning
- **Scalability:** O(n²) to O(n³) training — slow for n > 100k
- **No probability output:** outputs score, not probability (use Platt scaling)
- **Hyperparameter C:** larger C penalizes misclassification more (lower training error, risk overfitting)

## Common Mistakes / Gotchas
- **Scale sensitivity:** normalize features! RBF is distance-based
- **C and γ tuning:** grid search required. C ∈ [0.01, 100], γ ∈ [0.001, 10]
- **Linear vs RBF:** try linear first (simpler, faster). Only use RBF if nonlinearity needed

## Code Example
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X = [[0, 0], [1, 1], [2, 2], [1, 0]]
y = [0, 1, 1, 0]
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X, y)
print(svm.support_vectors_)  # Points on the margin
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What is SVM?" | Maximum-margin classifier. Kernel trick enables nonlinear separation. |
| "Why kernel trick?" | Maps to high-d space implicitly. Avoid explicit $\phi$ computation. |
| "SVM vs logistic regression?" | SVM: geometric margin. LR: probabilistic. SVM scales worse for large n. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Regularization](regularization.md)

## Resources
- [Support Vector Machines Explained](https://towardsdatascience.com/support-vector-machine-simplified-ccb8c91dd32c)
