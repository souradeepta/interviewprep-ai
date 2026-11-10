# Dropout

**TL;DR:** Dropout randomly masks activations during training to discourage fragile co-adaptation, then disables masking at inference while preserving the expected activation scale.

## Prerequisites

Prerequisites: [neural networks](neural-networks.md), stochastic training, and [overfitting](overfitting-underfitting.md). You should be able to derive inverted dropout scaling, explain why training and inference modes differ, compare dropout with weight decay and data augmentation, and decide when masking is likely to help. This ML note covers generalization and implementation; the AI track's neural-network overview is broader and should not be repeated as a substitute.

## Learning Outcomes

- Explain the central assumptions and mechanism of dropout.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Suppose a team succeeds only when one particular person is present. Randomly making team members unavailable during practice forces everyone to learn useful combinations instead of relying on a single specialist. Dropout does something similar to hidden units: each training pass uses a sampled subnetwork, so downstream units cannot count on one exact activation pattern. At inference, all units are available.

The analogy has limits. Dropout does not create truly independent models, and it does not guarantee robustness to arbitrary input corruption. It is a stochastic regularizer whose value depends on model, data, and placement.

## How It Works

For activation hⱼ and drop probability p, sample mⱼ~Bernoulli(1−p). Inverted dropout computes h̃ⱼ=mⱼhⱼ/(1−p) during training. Since E[mⱼ/(1−p)]=1, the expected activation matches the unmasked activation, and inference simply uses h. The scale factor would otherwise need to be applied at inference. A new mask is sampled on each training forward pass; mask sharing across time or spatial dimensions changes the regularizer and is used in some architectures.

Dropout adds little asymptotic cost: O(number of activations) random draws and multiplies. In sequence models, independently dropping every timestep can destroy temporal coherence; variational/recurrent dropout may share a mask across time. In convolutional networks, channel or spatial dropout can better match feature-map structure. Modern large models often rely more on data scale, weight decay, augmentation, or stochastic depth, so defaulting to high dropout is not justified.

The ensemble intuition is useful but should be stated carefully. With dropout, each minibatch trains a different thinned network, and inference uses an approximation to averaging many such networks. Those subnetworks share most parameters and are not independent models trained from separate samples. The method changes the training objective and introduces noise; it does not literally materialize or average every possible subnetwork. Monte Carlo dropout at inference keeps masks active and averages repeated predictions for approximate uncertainty, but this has additional latency and requires a separate evaluation of whether that uncertainty is meaningful.

Mask structure changes which dependencies are discouraged. Element-wise masks independently remove activations; spatial dropout may remove whole feature channels; recurrent dropout may hold a mask fixed across a sequence to avoid injecting unrelated noise at every timestep. Transformer implementations sometimes apply dropout to attention probabilities, residual branches, or embeddings. Each location has a different effect, so “dropout rate 0.1” is incomplete without saying where the mask is applied. When reproducing a checkpoint or paper, match the locations and training/evaluation behavior, not just a scalar rate.

The right rate depends on sample size, architecture, augmentation, weight decay, and training duration. A high rate can act like reducing effective capacity but may require more updates to converge. If batch size changes, the number of masks sampled per epoch changes. For fine-tuning a pretrained model, the source representation may already be regularized by massive pretraining; extra dropout can erase useful information. Sweep a small range on validation data and monitor train and validation curves, calibration, and per-slice results. Do not use dropout to patch data leakage or a misspecified target.

## Trade-offs and Selection Boundary

| Method | Best fit | Boundary |
|---|---|---|
| Dropout | Dense layers with overfitting and sufficient data to average stochastic masks | Can slow convergence and harm models already strongly regularized |
| Weight decay | Smoothly constrain parameter magnitude | Does not directly randomize activations; decoupled AdamW is not identical to L2 loss penalty |
| Data augmentation | Known label-preserving input transformations | Requires domain-valid transformations |
| Stochastic depth | Very deep residual networks | Drops whole residual branches, not individual activations |

Use dropout when validation evidence indicates overfitting and a representative tuning split is available. Start with a modest rate, often 0.1–0.3 in relevant layers, and validate. Avoid applying it to every layer or to a pretrained model without checking the original architecture.

## Implementation

```python
import random

def inverted_dropout(values, p, rng):
    if not 0 <= p < 1:
        raise ValueError("p must be in [0, 1)")
    if p == 0:
        return list(values)
    return [v / (1-p) if rng.random() >= p else 0.0 for v in values]

rng = random.Random(7)
values = [2.0, 4.0, 6.0]
masked = inverted_dropout(values, p=0.5, rng=rng)
assert len(masked) == len(values)
assert all(v == 0 or v in (4.0, 8.0, 12.0) for v in masked)
assert inverted_dropout(values, 0.0, rng) == values
# Monte Carlo expectation approaches the original activation.
rng = random.Random(11)
draws = [inverted_dropout([4.0], 0.25, rng)[0] for _ in range(20_000)]
assert abs(sum(draws)/len(draws) - 4.0) < 0.2
```

## Failure Modes and Diagnostics

1. **Dropout remains active at inference:** repeated predictions vary and are systematically noisy. Verify `eval()`/inference mode and test repeated predictions for determinism.
2. **Too much regularization:** training and validation performance are both poor. Reduce dropout and compare learning curves; distinguish underfitting from variance.
3. **Wrong scaling convention:** training activation means drift downward or upward. Use inverted dropout consistently, or ensure inference scaling matches the chosen convention.
4. **Masks break sequence structure:** a recurrent model loses persistent signals when each timestep is masked independently. Consider time-shared masks and compare on a held-out temporal split.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Why does dropout regularize? | Each update trains a sampled subnetwork, discouraging co-adaptation and dependence on individual activations. |
| Why scale by 1/(1−p)? | It keeps expected training activations equal to inference activations, so inference needs no scaling. |
| Debug follow-up: training accuracy is low and validation accuracy is also low. | Dropout may be too strong; reduce it and inspect both curves before adding capacity. |

## Related Concepts

- [Regularization](regularization.md), [early stopping](early-stopping.md), and [overfitting](overfitting-underfitting.md)
- [Matching notebook](../notebooks/10-dropout.ipynb)
- [Interview practice: L1 regularization](../interview-prep/ml-theory-questions.md#q-how-does-l1-regularization-produce-sparse-weights)

## References

- Srivastava et al., [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html).
- Gal and Ghahramani, [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287).
