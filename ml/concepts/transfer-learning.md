# Transfer Learning

**TL;DR:** Transfer learning reuses representations or parameters learned on a source task to improve a related target task, often reducing labeled-data and training requirements.

## Prerequisites

Prerequisites: [neural networks](neural-networks.md), supervised learning, and model evaluation. You can distinguish frozen-feature extraction from fine-tuning, choose a layer adaptation strategy, identify negative transfer, and avoid target-data leakage during model selection. This ML lesson focuses on transfer mechanics and assumptions; the AI roadmap discusses pretrained-model applications at a higher level.

## Learning Outcomes

- Explain the central assumptions and mechanism of transfer learning.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A vision model trained on millions of natural images may already detect edges, textures, and shapes. A small target dataset for plant disease can reuse those features instead of learning every visual primitive from scratch. The source training has supplied a useful starting representation; target examples adapt it to the new label space. Transfer works when useful structure is shared. If source and target differ in sensor, language, population, or task semantics, transferred features may be irrelevant or harmful.

Transfer learning is broader than “load weights and fine-tune.” It includes fixed feature extraction, parameter-efficient adapters, partial updates, and reuse of embeddings or preprocessing. The target label budget, source-target similarity, compute, and deployment constraints determine which form to test.

## How It Works

Let a source model have representation hφ(x) and prediction head gψ. Feature extraction freezes φ and trains a new target head ψₜ using labeled target examples. Fine-tuning initializes target parameters from source values and updates some or all layers. Early layers often encode generic low-level patterns while later layers are more task-specific, but this is not a guarantee: domain gap and architecture matter. Gradual unfreezing starts with the head, then allows selected backbone layers to adapt.

Fine-tuning often uses a smaller learning rate for pretrained parameters than for a newly initialized head. Weight decay, layer-wise learning rates, and discriminative schedules can reduce catastrophic forgetting, where target updates erase useful source behavior. Parameter-efficient fine-tuning updates small adapters or low-rank parameter deltas while keeping the base fixed; this lowers trainable parameter memory but may still require loading the full model at inference.

Model selection must be target-aware. Split target data before fitting scalers, choosing layers, or selecting checkpoints. If target labels are very limited, cross-validation or nested evaluation may be needed. Compare against a target-only baseline where feasible and a frozen-feature baseline. Measure performance by target subgroup, not just aggregate score. If target data is unlabeled, the task becomes domain adaptation or unsupervised transfer and requires additional assumptions.

Transfer has resource trade-offs. Reusing a pretrained checkpoint reduces training from scratch but can add dependency, license, security, and artifact-size considerations. Fine-tuning may require optimizer states for all weights. Freezing layers lowers gradient memory and compute, while adapters reduce trainable state. At serving time, a separate head or adapter must be versioned with the exact source checkpoint and preprocessing specification.

The source model's pretraining objective also affects what its representation can support. A vision encoder trained with supervised categories, contrastive image-text pairs, or self-supervised reconstruction may expose different strengths. A language representation trained on broad text can carry both useful syntax and unwanted source-domain associations. Inspect provenance, label ontology, training population, known limitations, and terms of use before treating a public checkpoint as a neutral starting point. Transfer can import biases and failure modes along with useful features.

Fine-tuning should be controlled with comparisons that isolate where adaptation helps. Train a fresh target head on frozen representations first; then unfreeze later blocks, and only then consider full fine-tuning. Use separate learning rates or schedules for the head and pretrained backbone when their parameter histories differ. Keep an untouched target test set because choices such as which layers to unfreeze are model-selection decisions. If training data is extremely small, report fold-to-fold variability and avoid presenting one lucky split as reliable evidence.

Deployment compatibility also matters. Tokenizer vocabulary, image resizing, normalization, positional embedding length, and class mapping are part of the checkpoint contract. A model's weights alone are not a complete transferable artifact. Validate the exact serialized model after export and ensure any adapter is applied to the intended base revision. If a downstream team cannot reproduce the transformation, an apparently successful research result may not transfer operationally.

## Trade-offs and Selection Boundary

| Strategy | Benefit | Boundary |
|---|---|---|
| Frozen feature extractor | Low training cost and fewer target labels | Representation cannot adapt when target domain is substantially different |
| Full fine-tuning | Maximum flexibility for target distribution | More compute/data and risk of forgetting or overfit |
| Partial/unfreeze schedule | Balances cost and adaptation | Requires layer choices and schedule tuning |
| Adapters/low-rank updates | Small trainable artifact and efficient task-specific updates | Base model still needed; merge/runtime behavior varies |
| Train from scratch | No source mismatch or inherited behavior | Needs more labels and compute; may be infeasible |

Start with frozen features and a simple head for small target datasets, then fine-tune if validation indicates representation mismatch and enough data exists. If source and target are unrelated, train a suitable target model rather than forcing transfer. Check source model provenance and license before use in a product.

## Implementation

```python
def linear_head(features, weights, bias):
    return [sum(x*w for x,w in zip(features,weights))+bias]

# Frozen representation vectors stand in for a pretrained encoder.
embeddings = [[1.,0.], [0.,1.], [1.,1.]]
labels = [1., 0., 1.]
weights, bias = [0.,0.], 0.0
lr = .1
for x,y in zip(embeddings,labels):
    pred = linear_head(x,weights,bias)[0]
    err = pred-y
    weights = [w-lr*err*v for w,v in zip(weights,x)]
    bias -= lr*err
assert weights[0] > 0 and weights[1] < weights[0]
assert len(embeddings) == len(labels)
```

## Failure Modes and Diagnostics

1. **Negative transfer:** source initialization underperforms a target-only baseline. Compare frozen, partial, and scratch models; inspect errors by domain and stop transfer if evidence is consistently negative.
2. **Catastrophic forgetting:** target metrics rise while source capability disappears. Measure both target and required source tasks; use smaller backbone rates, adapters, or replay data where allowed.
3. **Overfitting a tiny target set:** full fine-tuning memorizes examples. Use a realistic split, freeze more layers, regularize, augment appropriately, and report uncertainty.
4. **Head/backbone preprocessing mismatch:** target input normalization differs from the source model contract. Reproduce source preprocessing exactly and validate tensor parity.
5. **Checkpoint ambiguity:** adapter weights are loaded onto the wrong base revision. Pin immutable model revision, tokenizer/preprocessor, adapter config, and artifact checksums.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Feature extraction versus fine-tuning? | Feature extraction freezes the pretrained encoder and learns a target head; fine-tuning updates some or all source parameters. |
| When can transfer hurt? | Source and target representations or labels conflict, causing negative transfer or forgetting. |
| Design follow-up: only 500 labeled target images. | Start with a frozen pretrained backbone and head, use leakage-safe validation, then selectively fine-tune if the domain gap warrants it. |

## Related Concepts

- [Domain adaptation](domain-adaptation.md), [meta-learning](meta-learning.md), and [knowledge distillation](knowledge-distillation.md)
- [Matching notebook](../notebooks/37-transfer-learning.ipynb)
- [Interview practice: model selection](../interview-prep/ml-theory-questions.md#q-how-does-cross-validation-work-and-when-do-you-use-it)

## References

- Yosinski et al., [How Transferable Are Features in Deep Neural Networks?](https://arxiv.org/abs/1411.1792).
- Pan and Yang, [A Survey on Transfer Learning](https://doi.org/10.1109/TKDE.2009.191).
