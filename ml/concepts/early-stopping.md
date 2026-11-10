# Early Stopping

**TL;DR:** Early stopping selects a training checkpoint when validation performance stops improving, limiting wasted compute and often reducing overfit.

## Prerequisites

Prerequisites: [supervised learning](supervised-learning.md), train/validation/test splits, and [overfitting](overfitting-underfitting.md). You can distinguish the selection split from the final test set, define patience and minimum improvement, choose a monitored metric, and explain why restoring the best weights matters. This ML lesson treats stopping as a practical regularizer and model-selection rule; it complements rather than duplicates the general AI training overview.

## Learning Outcomes

- Explain the central assumptions and mechanism of early stopping.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Training loss measures how well the model fits examples it sees. Validation loss estimates behavior on held-out examples. A learner may keep memorizing training details after its ability to generalize has peaked. Early stopping says: keep a copy of the best validation checkpoint and stop when further training has not helped for a chosen period. It is like ending practice when repeated rehearsal improves the answer key score but harms performance on new questions.

Validation curves are noisy, so one bad epoch is not necessarily a reason to stop. Patience allows temporary fluctuations; minimum delta avoids treating tiny numerical changes as meaningful. Early stopping is not proof the model is optimal and does not eliminate the need for a held-out final test.

## How It Works

After each epoch t, measure validation metric mₜ. For a loss to minimize, improve if mₜ < m_best−δ; for a score to maximize, reverse the comparison. Save parameters when improvement occurs, reset a counter, otherwise increment it. Stop when the counter reaches patience P, then restore the saved parameters. Overhead is one validation pass per check and checkpoint storage; metric evaluation may dominate training for large validation sets.

Choose the validation cadence carefully. Checking every mini-batch makes the metric expensive and noisy; checking only every many epochs may waste substantial compute. In distributed training, synchronize validation and stopping decisions across workers. For non-stationary or time-ordered data, validation must mimic the deployment horizon; random splits may stop on an unrealistically easy sample.

Patience is measured in validation checks or epochs depending on the implementation, so document the unit. Minimum delta can be absolute or relative and should reflect meaningful metric resolution; a threshold larger than plausible incremental gains can stop too early. Some implementations stop after a fixed number of checks without improvement, while others also smooth the monitored metric. These choices affect the selected checkpoint and should be treated as hyperparameters. A validation curve with a long plateau and occasional improvements may need more patience than one with a clear overfit turn.

Checkpoint management matters as much as the stopping rule. Save the best state, not only the latest epoch, and ensure it includes optimizer-independent model buffers such as batch-normalization running means/variances. If training resumes, restore the best checkpoint or the latest checkpoint according to the intended continuation semantics; these are not interchangeable. Logging the best epoch, value, patience counter, and metric direction makes reproduction and incident debugging easier. In asynchronous/distributed settings, the selected model should correspond to a fully synchronized training step.

Early stopping introduces adaptive reuse of the validation set. Choosing an epoch based on validation values is a model-selection decision, and repeating the process across many architectures increases selection bias. A final test set can evaluate the entire selected procedure once. When data is too limited for a dedicated validation split, cross-validation can be used, but each fold needs its own stopping data or a nested structure to avoid stopping on the fold used for final scoring. For time series, the validation window should be later than training and should match the expected forecast horizon.

Early stopping may not be appropriate when the training schedule itself has phases or when validation metrics are too delayed/noisy to guide an online training process. Some self-supervised objectives continue improving without clear downstream validation movement; a fixed compute budget may be more reproducible. Use the mechanism when the validation signal is meaningful and the selected endpoint is part of a clearly defined training recipe.

## Trade-offs and Selection Boundary

| Choice | Advantage | Boundary |
|---|---|---|
| Early stopping | Simple, low-cost regularization and compute control | Depends on a representative validation set and noisy metric choices |
| Weight decay | Directly discourages large parameters | Requires tuning strength; may complement early stopping |
| Fixed training budget | Predictable runtime and reproducibility | Can stop too early or waste compute across tasks |
| Learning-rate schedule | Moves optimization toward convergence | A falling learning rate is not itself a stopping criterion |

Use early stopping when validation data reflects expected deployment and repeated evaluation is affordable. In hyperparameter search, keep the final test untouched; otherwise repeated decisions overfit the validation set. For data-starved tasks, cross-validation or a carefully justified fixed schedule may be preferable.

## Implementation

```python
def early_stop(losses, patience=2, min_delta=0.0):
    best, best_epoch, stale = float("inf"), None, 0
    for epoch, loss in enumerate(losses):
        if loss < best - min_delta:
            best, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                return best_epoch, best
    return best_epoch, best

epoch, loss = early_stop([.9, .7, .6, .61, .62, .59, .60, .61], patience=2)
assert epoch == 5 and loss == .59  # caller restores checkpoint from epoch 5
assert early_stop([1.0, .8, .7], patience=2)[0] == 2
```

## Failure Modes and Diagnostics

1. **Stop without restoring best checkpoint:** final model is worse than validation optimum. Store and reload the selected state; log best epoch explicitly.
2. **Validation leakage from repeated tuning:** many experiments indirectly fit the validation set. Reserve an untouched test set or use nested evaluation for unbiased estimates.
3. **Noisy metric triggers premature stop:** inspect the curve and use patience, smoothing, larger validation samples, or a stable metric.
4. **Wrong monitored objective:** loss improves while business utility falls (or vice versa). Monitor the prespecified deployment metric and retain loss as a diagnostic.

## Interview Quick Reference

| Question | Answer |
|---|---|
| How does early stopping regularize? | It limits optimization before the model increasingly fits training-specific detail and selects a validation checkpoint. |
| Why use patience? | Validation metrics fluctuate; patience tolerates short regressions before stopping. |
| Debug follow-up: reported score is worse than the best logged score. | Check that best weights were saved and restored, and that the logged metric matches evaluation mode. |

## Related Concepts

- [Overfitting and underfitting](overfitting-underfitting.md), [regularization](regularization.md), and [model selection](model-selection.md)
- [Matching notebook](../notebooks/11-early-stopping.ipynb)
- [Interview practice: evaluation and experiments](../interview-prep/evaluation-experiment-questions.md)

## References

- Prechelt, [Early Stopping—but When?](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf).
- Goodfellow, Bengio, and Courville, [Deep Learning, Chapter 7](https://www.deeplearningbook.org/contents/regularization.html).
