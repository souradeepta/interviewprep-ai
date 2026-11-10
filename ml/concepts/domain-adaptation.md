# Domain Adaptation

**TL;DR:** Domain adaptation adjusts a model trained on a source distribution so it works on a related target distribution whose inputs, labels, or both differ.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), [transfer learning](transfer-learning.md), and distribution shift. You can distinguish covariate, label, and concept shift; compare reweighting, feature alignment, and fine-tuning; and identify when target labels or assumptions are required. The ML track focuses on statistical assumptions and evaluation; the AI overview covers applications without duplicating these methods.

## Learning Outcomes

- Explain the central assumptions and mechanism of domain adaptation.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A model trained on daytime road images may fail at night even though the task—detect the road—has not changed. The camera, illumination, and feature distribution changed. Domain adaptation asks what can be reused from the source and which target evidence is available. Transfer learning is the broader idea of reusing learned representations; adaptation specifically addresses source-to-target distribution mismatch.

There is no universal correction for “shift.” If the relationship between features and labels changes, aligning input distributions alone can make things worse. The shift mechanism must be diagnosed and the evaluation population must match deployment.

## How It Works

Let source data follow Pₛ(X,Y) and target data Pₜ(X,Y). Under covariate shift, Pₛ(Y|X)=Pₜ(Y|X) but P(X) differs; importance weights w(x)=pₜ(x)/pₛ(x) can correct source risk if support overlaps and density ratios are estimable. Under label shift, Pₛ(X|Y)=Pₜ(X|Y), allowing prior correction. Under concept shift, P(Y|X) changes and unlabeled alignment alone is insufficient.

Methods include reweighting, feature-space alignment, self-training with pseudo-labels, and supervised fine-tuning on target labels. Unsupervised domain adaptation uses labeled source and unlabeled target examples; domain generalization uses source domains but no target examples during training. Target data must be separated by time/entity from final evaluation to avoid adaptation leakage.

The shift taxonomy helps select evidence, but real shifts can be mixtures. A new camera may change pixel statistics (covariate shift) and the relationship between a visual cue and diagnosis (concept shift). A policy change can alter which users receive labels, changing both observed prevalence and the labeling mechanism. Plot feature and prediction distributions over time, but do not infer that P(Y|X) stayed constant merely because unlabeled feature histograms look similar. Obtaining even a small, representative target-labeled audit set can distinguish assumptions that are otherwise untestable.

Importance weighting can have severe variance. If a few source observations receive extremely large weights, they dominate the objective and effective sample size collapses. Clipping weights reduces variance but introduces bias; it is a trade-off to report, not a free stabilizer. Inspect the distribution of raw and clipped weights, compute effective sample size, and compare with an unweighted source-only baseline. If target density lies outside source support, no reweighting can create information about those regions; collect labeled target examples or change the product scope.

Feature alignment also has a semantic risk. Distributional alignment objectives can map different target classes onto a source class if class proportions or conditional structure differ. Global alignment may obscure subgroup distinctions. Validate class-wise and subgroup-wise outcomes whenever labels are available, and use class-conditional alignment only when its assumptions are justified. Pseudo-labeling should include confidence thresholds, class-balance safeguards, and a comparison against teacher-only predictions; confident errors can reinforce themselves over repeated rounds.

Operationally, adaptation is a model update with its own data governance and rollback needs. Record which target inputs were used, whether labels were present, the adaptation method and hyperparameters, and the resulting artifact version. Monitor target performance as labels arrive and define a fallback if adaptation worsens a cohort. A one-time benchmark does not prove that future target batches follow the same shift process.

## Trade-offs and Selection Boundary

| Method | Assumption / benefit | Boundary |
|---|---|---|
| Importance weighting | Conditional label relation stable; target covariates available | High-variance weights and no correction outside source support |
| Feature alignment | Domains differ in representation but share task structure | Can align class-mismatched features and hurt accuracy |
| Pseudo-labeling | Target model predictions become extra training labels | Confident systematic errors reinforce themselves |
| Labeled target fine-tuning | Direct evidence about target task | Label cost and small-sample overfit |
| Domain generalization | Must work on unseen future domains | Requires representative diverse source environments |

Prefer target-labeled fine-tuning when labels are affordable and shift is material. If only unlabeled target data exist, verify the adaptation assumption and monitor target slices; if support or label semantics differ, collect labels or redesign rather than blindly align.

## Implementation

```python
def clipped_importance_weights(source_density, target_density, cap=5.0):
    if any(p <= 0 for p in source_density):
        raise ValueError("source density must be positive on target support")
    raw = [min(cap, t/s) for s,t in zip(source_density, target_density)]
    mean = sum(raw)/len(raw)
    return [w/mean for w in raw]

w = clipped_importance_weights([.5, .4, .1], [.2, .5, .3])
assert all(0 < x <= 5 for x in w)
assert abs(sum(w)/len(w)-1) < 1e-12
# Extreme ratios reveal weak overlap; clipping reduces variance but adds bias.
assert max(w) < 5
```

## Failure Modes and Diagnostics

1. **Wrong shift assumption:** covariate correction is applied despite changing P(Y|X). Evaluate labeled target slices and inspect confusion changes.
2. **Poor support overlap:** target examples have no similar source examples, producing huge weights. Plot weight distribution/effective sample size; collect target labels or new source data.
3. **Negative transfer:** alignment degrades target classes. Compare against source-only and simple target fine-tuning baselines by class/domain.
4. **Adaptation uses test information:** thresholds or representations are repeatedly tuned on evaluation target labels. Define allowed unlabeled target use and keep a separate final target test set.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Transfer learning versus domain adaptation? | Transfer is broad reuse; adaptation explicitly handles source/target distribution differences. |
| When does importance weighting work? | Under covariate shift with overlap and estimable density ratio, while P(Y|X) remains stable. |
| Design follow-up: night performance drops and no target labels exist. | Compare input drift, test unsupervised adaptation cautiously, monitor slices, and prioritize obtaining a labeled target audit. |

## Related Concepts

- [Transfer learning](transfer-learning.md), [evaluation metrics](evaluation-metrics.md), and [data leakage](data-leakage.md)
- [Matching notebook](../notebooks/09-domain-adaptation.ipynb)
- [Interview practice: evaluation and experiments](../interview-prep/evaluation-experiment-questions.md)

## References

- Ben-David et al., [A Theory of Learning from Different Domains](https://doi.org/10.1007/978-3-540-75525-4_5).
- Sugiyama et al., [Covariate Shift Adaptation by Importance Weighted Cross Validation](https://doi.org/10.1162/089976604773135954).
