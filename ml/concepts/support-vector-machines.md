# Support Vector Machines

**TL;DR:** Support vector machines choose a decision boundary with a large margin, using slack penalties for imperfect separation and kernels for nonlinear boundaries.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), linear algebra, and [regularization](regularization.md). You can explain margin and support vectors, derive the soft-margin objective, interpret C and kernel parameters, identify scaling and calibration needs, and decide when an SVM is computationally unsuitable. This ML lesson covers the classifier's geometry; the AI roadmap introduces classification options more broadly.

## Learning Outcomes

- Explain the central assumptions and mechanism of support vector machines.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Imagine two classes of points with many possible separating lines. A support vector machine chooses a boundary that leaves the widest safety margin between the classes, rather than merely finding any line that classifies the training points correctly. Points nearest the boundary—the support vectors—determine its location. Farther points generally do not affect a linear solution unless they become close enough to constrain the margin.

The margin can provide a useful regularization principle, but it does not guarantee calibrated probabilities or immunity to poor features. Outliers may be tolerated through a soft margin, and the kernel choice determines which nonlinear patterns are representable. A complex kernel can fit noise just as another flexible model can.

## How It Works

For binary labels yᵢ∈{−1,+1}, a hard-margin linear SVM minimizes ½||w||² subject to yᵢ(wᵀxᵢ+b)≥1. The distance between the two margin boundaries is 2/||w||, so minimizing the norm maximizes the geometric margin. Perfect separation is often unrealistic, so the soft-margin primal adds slack ξᵢ≥0 and penalty CΣᵢξᵢ. Large C punishes violations strongly and can fit training data more closely; small C accepts more violations for a wider/more regularized margin.

The dual optimization uses examples through dot products xᵢᵀxⱼ. A kernel K(x,z)=φ(x)ᵀφ(z) computes an inner product in an implicit feature space without constructing φ. Common choices are linear K=xᵀz, polynomial K=(γxᵀz+r)ᵈ, and radial basis K=exp(−γ||x−z||²). For RBF, large γ makes each point's influence local and can overfit; small γ makes the boundary smoother. Standardizing features is essential for distance-based kernels because units otherwise dominate distances.

Training kernel SVMs often requires O(n²) kernel storage and between O(n²) and O(n³) optimization time depending on implementation and data, which can be prohibitive at large n. Linear SVM solvers scale much better for sparse high-dimensional features such as text. Prediction may depend on many support vectors, affecting latency and model storage. Multiclass classification commonly uses one-vs-rest or one-vs-one reductions. The raw decision score is a signed margin, not a calibrated class probability; Platt scaling or another calibration method needs held-out data.

The support-vector interpretation is local to the optimization formulation. With a kernel, a prediction is a weighted sum of similarities to support vectors. Feature importance is not directly available in a nonlinear kernel model. Linear coefficients can be inspected, but correlated features and scaling still affect interpretation. Kernel choice is an inductive bias, not an automatic cure for nonlinear data.

The parameter C is inversely related to regularization in common SVM formulations, but its useful numeric range depends on feature scale and the loss normalization used by an implementation. If the training objective sums hinge losses in one library and averages them in another, the same C may represent a different effective penalty. Standardize numeric features inside each training fold, then tune C and kernel parameters on a logarithmic grid or randomized search. Do not compare a tuned RBF model with an untuned logistic baseline and call the difference an algorithm effect.

SVM scores have a geometric interpretation: the signed decision value is proportional to distance from the hyperplane for a linear model when the weight norm is accounted for. A score of two is not “twice the probability” of a score of one. If an operating threshold is important, select it on validation data; if probabilities are required, fit a calibration model using data kept separate from the SVM's training. Calibration needs its own monitoring because prevalence changes can invalidate the mapping.

For multiclass problems, one-vs-rest trains a classifier per class, while one-vs-one trains pairwise classifiers. Their memory, class imbalance behavior, and inference costs differ. Inspect the library's strategy and decision-function shape before interpreting outputs. Some SVM implementations also expose class weights that address unequal misclassification cost, but class weighting does not remove the need for suitable metrics and thresholds.

## Trade-offs and Selection Boundary

| SVM variant | Useful when | Boundary |
|---|---|---|
| Linear SVM | Many sparse features, moderate sample count, margin baseline | Decision boundary is linear in current representation |
| RBF SVM | Small/medium scaled data with smooth nonlinear boundary | Kernel training can become expensive as sample count grows |
| Polynomial kernel | Interactions up to a chosen degree are meaningful | Degree/offset choices can cause unstable scale and overfit |
| Soft margin C | Data has noise/outliers or overlap | Must tune jointly with kernel scale |
| Calibrated SVM | Probability decisions are required | Calibration adds a held-out procedure and may drift with prevalence |

Prefer a linear SVM for sparse text or a high-dimensional small-data baseline. Try RBF when the sample count is manageable, features are scaled, and validation indicates nonlinearity beyond a linear model. Prefer logistic regression when calibrated probabilities, online updates, or a simpler probabilistic interpretation matter. For very large datasets, tree boosting or scalable linear methods may be more practical.

## Implementation

```python
def rbf(x, z, gamma):
    if gamma < 0: raise ValueError("gamma must be nonnegative")
    return pow(2.718281828, -gamma*sum((a-b)**2 for a,b in zip(x,z)))

points = [[0.,0.], [1.,1.], [2.,2.]]
gram = [[rbf(x,z,.5) for z in points] for x in points]
assert all(abs(gram[i][i]-1.0) < 1e-12 for i in range(3))
assert gram[0][1] > gram[0][2]
assert all(abs(gram[i][j]-gram[j][i]) < 1e-12 for i in range(3) for j in range(3))
```

## Failure Modes and Diagnostics

1. **Features not scaled:** one large-unit feature dominates RBF distances. Fit a scaler within each training fold and inspect per-feature ranges.
2. **C/γ jointly mis-set:** high C and high γ create a jagged boundary; low values underfit. Search log-spaced combinations and inspect train-validation gaps.
3. **Kernel computation overwhelms memory:** training stalls or exhausts RAM as n grows. Estimate kernel matrix size early and switch to linear/approximate methods when needed.
4. **Margin score used as probability:** threshold decisions are unjustified. Calibrate on held-out data and monitor prevalence shift.
5. **Validation leakage through scaling or feature selection:** scores are optimistic. Fit every transform inside CV folds.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What are support vectors? | Training points that constrain the margin and determine the learned boundary. |
| What does the kernel trick do? | Computes feature-space inner products implicitly, enabling nonlinear boundaries without constructing mapped features. |
| Design follow-up: nonlinear task grows from 10k to 10m rows. | Kernel SVM may not scale; benchmark a linear model, approximate kernel, or tree/boosting alternative and preserve a realistic validation split. |

## Related Concepts

- [Regularization](regularization.md), [evaluation metrics](evaluation-metrics.md), and [model selection](model-selection.md)
- [Matching notebook](../notebooks/36-support-vector-machines.ipynb)
- [Interview practice: kernel trick](../interview-prep/ml-theory-questions.md#q-what-is-the-kernel-trick-and-when-do-svms-use-it)

## References

- Cortes and Vapnik, [Support-Vector Networks](https://doi.org/10.1007/BF00994018).
- scikit-learn, [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html).
