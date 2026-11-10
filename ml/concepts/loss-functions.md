# Loss Functions

**TL;DR:** A loss function turns predictions and targets into an optimization signal, so its mathematical behavior and assumptions must match the task.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), probabilities, and [optimization](optimization.md). You can distinguish losses from metrics, derive common regression and classification objectives, select a loss for label structure, and diagnose unstable or mismatched objectives. This ML lesson focuses on objective design; the AI overview introduces task families and should not substitute for this detail.

## Learning Outcomes

- Explain the central assumptions and mechanism of loss functions.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A loss is the training-time price assigned to a prediction. Squared error charges large residuals quadratically, so one outlier can dominate; absolute error charges linearly and is more robust. Cross-entropy asks the model to assign high probability to the observed class and heavily penalizes confident mistakes. The best training loss is a useful differentiable surrogate, but the metric used to judge deployment may be different.

Choosing a familiar loss without checking label semantics can create a system that optimizes the wrong question. Mutually exclusive labels, independent labels, and ordered outcomes require different outputs and objectives.

## How It Works

For regression, MSE=(1/n)Σ(y−ŷ)² corresponds to Gaussian noise with fixed variance and has gradient proportional to residual; MAE=(1/n)Σ|y−ŷ| corresponds to Laplace noise and is less sensitive to outliers but non-smooth at zero. Huber loss is quadratic near zero and linear beyond δ. For binary classification with logit z and y∈{0,1}, binary cross-entropy is −y log σ(z)−(1−y)log(1−σ(z)); implementations should consume logits stably. Multiclass cross-entropy uses log-softmax for a single class among K. Ranking losses compare ordered pairs; contrastive losses pull positive pairs together and separate negatives.

Loss reduction (sum versus mean) changes gradient scale with batch size. Class weights alter contributions; masking must exclude padding/unknown targets from both numerator and denominator. Multi-task objectives add weighted losses, and those weights determine which task dominates. Loss values from different formulations are not directly comparable.

The statistical interpretation helps make the choice concrete. Under squared error, minimizing expected loss estimates the conditional mean E[Y|X]; under absolute error it estimates the conditional median. These estimands differ for skewed targets, so changing MSE to MAE changes what prediction is considered optimal. Quantile loss estimates a chosen conditional quantile and is useful when underestimation and overestimation have asymmetric costs. For a demand forecast where stockouts cost more than surplus, a high quantile can match the decision better than a mean, though downstream inventory cost should still be evaluated directly.

Classification cross-entropy is a proper scoring rule: in expectation it rewards honest probability estimates. The zero-one accuracy metric only checks the final argmax and ignores confidence. A model may improve accuracy while becoming poorly calibrated, or improve log loss by refining probabilities without changing any predicted labels. When the system acts at a threshold, choose the threshold on validation data using the actual costs, and report calibration and decision utility alongside loss. Do not tune that threshold on the final test set.

The reduction and masking rules are part of the objective definition. If one batch has 20 valid tokens and another has 200, averaging each batch then averaging batch losses weights the batches equally, while dividing total loss by the number of valid tokens weights tokens equally. Distributed training can introduce the same issue when workers have unequal valid counts. Check whether the framework averages over batch elements, pixels, tokens, or nonignored labels. A zero-valid-target batch needs explicit handling to avoid division by zero and NaN gradients.

Class weighting and resampling alter the effective training distribution. They may improve minority recall, but weighted cross-entropy can make raw model scores poorly calibrated for the deployment prevalence. Evaluate on the natural validation distribution, recalibrate if probability quality matters, and choose thresholds separately. Focal loss emphasizes hard examples; if “hard” examples are mostly mislabeled or ambiguous, that emphasis can amplify noise. Compare it to a simpler weighted baseline and inspect errors by class and label confidence.

For contrastive objectives, the negative sampling scheme defines which distinctions the representation must learn. False negatives occur when two examples treated as negatives are actually semantically related. Temperature controls the sharpness of similarity weighting, and batch composition changes the number and difficulty of negatives. For ranking losses, sampled pairs and tie handling define the target order. These methods should be evaluated with retrieval or ranking metrics on a realistic candidate set, not only by a decreasing training loss. The selection boundary is determined by label semantics and downstream use, not by whichever loss is most familiar.

## Trade-offs and Selection Boundary

| Loss | Use when | Boundary |
|---|---|---|
| MSE | Continuous targets and squared-error cost | Outliers dominate; estimates conditional mean |
| MAE | Absolute-error cost or heavy-tailed noise | Less smooth optimization; estimates conditional median |
| Huber | Want quadratic near zero and robust tails | δ needs units-aware tuning |
| Cross-entropy | Probabilistic classification | Confident label noise can produce large penalties |
| Focal loss | Dense tasks with many easy negatives | Can hurt calibration and needs γ/α tuning |
| Pairwise ranking | Ordering matters more than calibrated scores | Pair construction and sampling define the task |

Choose the loss from the target distribution and decision objective. For class imbalance, compare weighting or focal loss to threshold adjustment; these solve different problems. Use a stable logits-based API and separately report deployment metrics.

## Implementation

```python
import math

def bce_with_logits(logit, target):
    # Numerically stable softplus(logit) - target*logit
    return max(logit, 0) - logit*target + math.log1p(math.exp(-abs(logit)))

def huber(error, delta=1.0):
    a = abs(error)
    return 0.5*error*error if a <= delta else delta*(a-0.5*delta)

assert abs(bce_with_logits(0, 1)-math.log(2)) < 1e-12
assert bce_with_logits(10, 0) > bce_with_logits(0, 0)
assert huber(.5) == .125 and huber(3) == 2.5
```

## Failure Modes and Diagnostics

1. **Probability passed to logits loss (or vice versa):** gradients and loss are wrong despite plausible values. Check model output contract and use a logits-aware loss exactly once.
2. **Loss reduction changes with batch size:** summed losses make gradients scale with examples. Confirm mean/sum semantics when changing batch or accumulation.
3. **Padding contributes to sequence loss:** model learns padding distribution. Apply target mask and divide by valid-token count, including edge cases with zero valid tokens.
4. **Surrogate improves while business metric worsens:** objective is misaligned. Track both, inspect slices, and consider threshold/cost-aware training or post-training decisions.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Loss versus metric? | Loss provides an optimization signal, often differentiable; metric evaluates task performance and may be nondifferentiable. |
| MSE versus MAE? | MSE emphasizes large residuals and estimates a conditional mean; MAE is more robust and targets a conditional median. |
| Debug follow-up: loss falls but accuracy does not. | Inspect labels, logits/probability contract, class balance, threshold, and whether the optimization objective matches accuracy. |

## Related Concepts

- [Evaluation metrics](evaluation-metrics.md), [class imbalance](class-imbalance.md), and [optimization](optimization.md)
- [Matching notebook](../notebooks/20-loss-functions.ipynb)
- [Interview practice: gradient descent](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Goodfellow, Bengio, and Courville, [Deep Learning, Chapter 6](https://www.deeplearningbook.org/contents/mlp.html).
- Lin et al., [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
