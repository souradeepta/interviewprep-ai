# Regularization

**TL;DR:** Regularization constrains effective model complexity or training behavior to reduce sensitivity to sample-specific noise, usually trading some training fit for better generalization.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), [optimization](optimization.md), and training/validation splits. You can distinguish L1 from L2 penalties, relate regularization strength to bias and variance, compare penalty, stochastic, and data-based approaches, and select interventions from learning curves. This ML lesson unifies practical constraints; the AI overview covers model training at a higher level.

## Learning Outcomes

- Explain the central assumptions and mechanism of regularization.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A model with enough flexibility can memorize accidental details in its training sample. Regularization limits how readily the model can use those details. It is like fitting a smooth curve through noisy measurements: a curve that passes through every point may capture measurement error, while a simpler curve can better approximate the underlying relation. The appropriate simplicity depends on the task; excessive constraint removes useful signal and causes underfitting.

Regularization is not synonymous with one penalty or with making every parameter small. Some methods shrink weights, some randomize activations, some stop training early, and some supply valid transformations of data. They influence different parts of learning and can complement one another, but stacking them without measurement can over-constrain the model.

## How It Works

For empirical loss L(θ), a penalized objective is J(θ)=L(θ)+λΩ(θ). L2 uses Ω=Σⱼθⱼ² and smoothly shrinks weights; L1 uses Ω=Σⱼ|θⱼ| and can produce exact zeros because of its geometry and subgradient around zero. Elastic Net combines both, encouraging sparsity while behaving more stably with correlated features. λ controls the trade-off and must be selected without using the final test set. In linear regression, Ridge has a closed form under common assumptions; Lasso generally requires iterative optimization.

Weight decay multiplies parameters by a factor during updates. For plain SGD this can correspond to L2 regularization under particular conventions, but for adaptive optimizers such as Adam the update interactions differ; AdamW decouples decay from the adaptive gradient. Exclude or include biases and normalization parameters deliberately rather than applying decay indiscriminately.

Dropout samples hidden-unit masks during training, reducing reliance on particular activations. Data augmentation transforms examples while preserving their labels, encouraging invariance to relevant changes. Early stopping selects the point where validation stops improving. Label smoothing softens one-hot targets and discourages extreme probabilities. Mixup trains on interpolations of examples and labels, imposing a smoother behavior between samples. Each method encodes an assumption about which complexity or invariance is undesirable.

Regularization changes the effective learning problem and can improve validation behavior even when training loss worsens. It cannot recover information absent from the features, fix leakage, repair invalid labels, or correct a validation set that does not reflect deployment. Choose methods from error patterns and domain knowledge, then measure them on an honest split.

## Trade-offs and Selection Boundary

| Method | Useful effect | Boundary |
|---|---|---|
| L2 / weight decay | Shrinks parameters smoothly and stabilizes linear/deep models | Does not produce sparse weights; optimizer semantics matter |
| L1 | Encourages sparse linear solutions and feature selection | Unstable feature choice among correlated predictors |
| Dropout | Reduces activation co-adaptation | Can slow training or hurt already strongly regularized models |
| Data augmentation | Adds label-preserving variation and invariance | Invalid transformations corrupt labels or erase meaningful cues |
| Early stopping | Limits late-stage fit and compute | Validation noise and repeated selection can bias the result |
| Label smoothing / Mixup | Softens targets or interpolates examples | Can reduce calibration or harm tasks with sharp boundaries |

Use L2/weight decay as a robust starting point for many neural and linear models; use L1 when sparsity or feature selection is desired and its instability is understood. Favor augmentation when domain-valid transformations exist. If both training and validation scores are poor, investigate underfitting/optimization before adding more regularization. If only validation is poor, first confirm there is no distribution shift or leakage.

The geometry of a penalty helps explain its behavior. An L2 constraint forms a smooth ball, so a loss contour often touches it where many coefficients are small but nonzero. An L1 constraint has corners aligned with coordinate axes, so the optimum can land exactly on a zero coefficient. With strongly correlated features, there may be many nearly equivalent sparse solutions; small data changes can select a different feature even if prediction quality stays similar. Elastic Net's L2 component can stabilize these correlated groups while retaining some sparsity.

Regularization strength should be examined across a range rather than chosen by convention. Very weak penalties may leave a high-variance model, while very strong ones flatten meaningful patterns. Plot training and validation behavior against λ and check both predictive quality and operational properties such as coefficient stability. For neural models, learning rate and penalty strength interact: large updates can overwhelm small shrinkage, and decoupled weight decay is applied on a different path than a loss penalty. Record the exact optimizer and decay semantics so experiments can be repeated.

Regularization assumptions should make sense for the input domain. Horizontal flips are reasonable for some natural images but invalid for text, directional arrows, or medical laterality. Mixup may create physically impossible examples in domains with discrete or constrained features. Domain review is needed to determine whether a transformation preserves the label and whether its invariance is desirable at serving time.

## Implementation

```python
def penalized_loss(residuals, weights, strength=.1):
    data = sum(r*r for r in residuals)/len(residuals)
    l1 = strength*sum(abs(w) for w in weights)
    l2 = strength*sum(w*w for w in weights)
    return data, data+l1, data+l2

plain, with_l1, with_l2 = penalized_loss([1., -1.], [0., 2.])
assert plain == 1.0 and with_l1 > plain and with_l2 > plain
assert with_l1 != with_l2
```

## Failure Modes and Diagnostics

1. **Penalty too strong:** training and validation both degrade. Plot both curves over λ and reduce regularization or increase useful capacity.
2. **Penalty applied to every parameter:** biases or normalization scales are unnecessarily shrunk. Inspect parameter groups and compare explicit decay policies.
3. **Augmentation changes labels:** model learns from invalid transformed targets. Review transformations with domain experts and inspect transformed examples plus per-class errors.
4. **Dropout active in evaluation:** scores vary across repeated inference. Check training/evaluation mode and assert deterministic predictions where expected.
5. **Validation becomes a repeated tuning target:** many trials overfit one held-out set. Use nested CV or a final untouched test set and log the selection process.

## Interview Quick Reference

| Question | Answer |
|---|---|
| L1 versus L2? | L1 can drive coefficients exactly to zero; L2 shrinks smoothly and usually retains all features. |
| What does λ control? | The strength of the complexity penalty relative to data fit; select it with validation, not the test set. |
| Debug follow-up: high training error after adding dropout and decay. | The model may be over-regularized; reduce one constraint at a time and compare training as well as validation curves. |

## Related Concepts

- [Overfitting and underfitting](overfitting-underfitting.md), [dropout](dropout.md), and [early stopping](early-stopping.md)
- [Matching notebook](../notebooks/33-regularization.ipynb)
- [Interview practice: L1 regularization](../interview-prep/ml-theory-questions.md#q-how-does-l1-regularization-produce-sparse-weights)

## References

- Tibshirani, [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x).
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
