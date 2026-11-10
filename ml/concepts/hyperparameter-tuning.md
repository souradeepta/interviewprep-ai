# Hyperparameter Tuning

**TL;DR:** Hyperparameter tuning searches choices set outside model fitting, using validation procedures that account for compute, randomness, and selection bias.

## Prerequisites

Prerequisites: [model selection](model-selection.md), [cross-validation](cross-validation-strategies.md), and evaluation metrics. You can define a search space, compare grid/random/Bayesian strategies, avoid tuning leakage, and allocate trials under a budget. This ML note addresses practical search design; the AI track's model overview supplies context on model families.

## Learning Outcomes

- Explain the central assumptions and mechanism of hyperparameter tuning.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Training learns parameters from examples; tuning chooses the recipe—tree depth, regularization, learning rate, or architecture. A search is an experiment: each trial consumes compute and validation exposure. If the metric is noisy, selecting the single highest score can select luck rather than a robust configuration. A disciplined search records seeds, split, budget, and all tried configurations.

## How It Works

Define an objective J(λ) (validation score or loss) over hyperparameters λ, a feasible search domain, and a fixed compute budget. Grid search evaluates a Cartesian product; it wastes trials when many dimensions are irrelevant. Random search samples distributions and often covers important dimensions more efficiently. Bayesian optimization fits a surrogate to prior trials and chooses promising or uncertain candidates, which helps when each evaluation is expensive but adds tuning machinery. Successive halving/Hyperband allocates small budgets broadly, then more resources to promising runs.

Use log-scaled sampling for multiplicative quantities such as learning rate and weight decay. Compare candidates on the same splits and training budgets; otherwise changes are confounded. A final test set should not guide search. For noisy tasks, repeat promising configurations across seeds and report variability. Search cost is approximately trials × model-fit cost (or less with early pruning).

Tuning is an adaptive experiment: every validation result influences the next trial, so repeated search can overfit the validation set. Keep a final test set outside the search loop, and use nested cross-validation when the dataset is small and an unbiased estimate of the whole selection procedure is needed. A search budget should include training failures and early stopping rules; otherwise the optimizer may spend most of its time on implausible configurations. Log the full configuration, seed, data version, metric, and resource use so the winning run can be reproduced. Random search often beats a dense grid when only a few dimensions matter. Bayesian optimization can use past observations efficiently for expensive low-dimensional spaces, while successive halving allocates more compute to promising trials. Select by the metric and operating threshold that match the deployment goal, then evaluate the selected procedure once on held-out data.

## Trade-offs and Selection Boundary

| Strategy | Strength | Boundary |
|---|---|---|
| Grid | Exhaustive over a small discrete set | Combinatorial explosion; poor continuous coverage |
| Random | Efficient baseline for mixed spaces | Does not exploit prior trial outcomes |
| Bayesian | Uses past trials to guide expensive search | Surrogate assumptions and parallelism can be limiting |
| Successive halving | Prunes weak trials early | Early metric may not predict final quality |
| Manual tuning | Uses domain understanding | Hard to reproduce and vulnerable to confirmation bias |

Use a small random search as a strong baseline, then add adaptive methods if evaluations are expensive. Tune only parameters that plausibly matter, and preserve an untouched test set for the final selected procedure.

Search-space design can matter more than the search algorithm. Constrain combinations that are invalid or redundant, use log distributions for quantities spanning orders of magnitude, and encode conditional parameters explicitly—for example, a kernel degree matters only when a polynomial kernel is selected. Include a reasonable default configuration so the search has a known baseline. A broad space with mostly implausible values wastes compute and can make a competent model appear poor.

Noisy objectives complicate selection. A candidate can score highly because of one favorable seed, a small validation set, or an unusually easy fold. Use common splits and seeds where possible to make paired comparisons, then repeat finalists across independent seeds. If the training procedure itself is stochastic, record both average performance and variability. Early-stopping and successive-halving methods may favor models that learn quickly at small budgets rather than models that achieve the best final quality; validate the pruning rule on representative trials.

Hyperparameter optimization can overfit the validation set even when each trial is individually reasonable. As trial count grows, the maximum observed score increasingly reflects noise. Nested cross-validation estimates the complete selection process, while a locked test set provides a final check. Neither removes uncertainty caused by a different future population. Keep a search log with every attempted configuration, data split, software version, compute budget, and stopping reason so the winning result can be reproduced rather than recalled from a notebook cell.

Compute allocation should match decision value. A coarse randomized search can identify influential dimensions; then refine only around promising regions. If the model is cheap, a simple grid over a small set may be more transparent than an elaborate optimizer. If trials are expensive, Bayesian optimization can reuse observations, but parallel workers introduce stale suggestions and resource contention. Compare methods by the best validation quality achieved at equal wall-clock or accelerator cost, not by the number of trials alone.

## Implementation

```python
import random
random.seed(23)

def sample_log_uniform(low, high):
    return 10 ** random.uniform(low, high)

trials = [{"lr": sample_log_uniform(-4, -1), "weight_decay": sample_log_uniform(-6, -2)}
          for _ in range(8)]
assert len({round(t["lr"], 8) for t in trials}) > 1
assert all(1e-4 <= t["lr"] <= 1e-1 and 1e-6 <= t["weight_decay"] <= 1e-2 for t in trials)
# Trial outcomes must use comparable splits/budgets before ranking.
results = [(0.82, "a"), (0.84, "b"), (0.83, "c")]
assert max(results)[0] == .84
```

## Failure Modes and Diagnostics

1. **Test-set tuning:** final metric becomes optimistically selected. Record test access and freeze it until the procedure is chosen.
2. **Incomparable trial budgets:** one model gets more epochs or different folds. Normalize compute/splits and log every configuration.
3. **Search bounds miss useful region:** all trials look similarly bad. Inspect distributions and use learning-rate range tests or domain constraints to revise bounds.
4. **Winner is seed luck:** top score has high variance. Rerun finalists on multiple seeds and prefer robust performance under the compute budget.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Random search versus grid? | Random search covers continuous spaces efficiently when only a few dimensions strongly affect results. |
| Why use log scale for learning rate? | Meaningful values span orders of magnitude; equal ratios matter more than equal differences. |
| Design follow-up: thousands of trials are too expensive. | Use random baseline plus early stopping/successive halving, prune irrelevant dimensions, and report compute-normalized results. |

## Related Concepts

- [Model selection](model-selection.md), [cross-validation](cross-validation-strategies.md), and [early stopping](early-stopping.md)
- [Matching notebook](../notebooks/16-hyperparameter-tuning.ipynb)
- [Interview practice: cross-validation](../interview-prep/ml-theory-questions.md#q-how-does-cross-validation-work-and-when-do-you-use-it)

## References

- Bergstra and Bengio, [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html).
- Li et al., [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://jmlr.org/papers/v18/16-558.html).
