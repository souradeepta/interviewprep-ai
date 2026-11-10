# Knowledge Distillation

**TL;DR:** Knowledge distillation trains a student model to match a teacher's softened predictions or intermediate signals, often trading some accuracy for lower serving cost.

## Prerequisites

Prerequisites: classification logits, cross-entropy, and [model compression](model-compression.md). You can derive temperature-softened targets, combine teacher and label losses, distinguish distillation from quantization/pruning, and design evaluation that measures task quality and serving cost. This ML lesson covers compression through transferred behavior; the AI track's model summaries may mention it but do not replace the mechanics here.

## Learning Outcomes

- Explain the central assumptions and mechanism of knowledge distillation.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

A teacher may know that an image is mostly “wolf” but also resembles “dog”; a hard label hides this structure, while the teacher's probability distribution conveys similarity. A smaller student learns both the correct class and some of the teacher's relationships between classes. Distillation is a training signal, not a guarantee the student inherits all teacher capabilities.

The student's useful knowledge can include calibrated class relationships, token distributions, or hidden representations. But a teacher's mistakes and biases can transfer too, and the smaller architecture may be unable to represent the desired behavior.

## How It Works

For logits z and temperature τ>0, probabilities are pᵢ=softmax(zᵢ/τ). Larger τ produces softer distributions and exposes relative alternatives. A common objective is L=α·CE(y,p_student)+(1−α)·τ²·KL(p_teacher^τ || p_student^τ). The τ² factor compensates for gradient scaling in common formulations. Teacher and student must use matching class order and compatible preprocessing. Teacher outputs can be precomputed to save training time, but cached outputs must correspond to the exact examples and augmentation policy.

Distillation cost depends on teacher inference and student training; a pretrained teacher can be frozen and evaluated once per example. Intermediate-feature matching requires an adapter when dimensions differ. For language models, tokenizers and vocabularies must align or targets need a mapping. At deployment, the teacher can be discarded if no ensemble fallback is required.

Teacher targets may be generated online during student training or cached in advance. Online teacher inference increases compute but allows fresh augmentation and avoids a large target store. Cached logits reduce repeated computation and make training deterministic, but the cache must be keyed to the exact example, augmentation view, tokenizer, teacher checkpoint, and class order. If randomized image crops change each epoch, one cached output per original image may not correspond to the augmented input. A wrong cache can quietly train the student toward irrelevant outputs.

The soft-target coefficient and temperature determine how much the student follows the teacher versus labels. A teacher can be strong on average while systematically weak on a rare subgroup. Mixing ground-truth loss protects against some errors but does not guarantee subgroup quality, especially if labels are scarce or noisy. Measure student and teacher performance on the same subgroup slices, report coverage and calibration, and consider class-conditional distillation only when enough trusted data supports it. The compression objective should include the actual task metric, not just KL divergence to the teacher.

Distillation can also transfer behavior between architectures, but the output interface must be aligned. For classification, teacher and student need corresponding class logits. For sequence models, tokenization differences change the position-by-position targets; alignment may require string-level spans or a common vocabulary. Feature distillation between layers of different widths can use a projection adapter, though choosing corresponding semantic layers is nontrivial. The adapter is often discarded at inference, but its training cost and objective still matter.

Evaluate the complete deployment design. If the teacher remains as an ensemble member or provides a fallback, the system has not eliminated its serving cost. If the student is exported to lower precision, measure the combination of distillation and quantization rather than assuming their errors are independent. Compare the small student trained directly on labels, the distilled student, and a compressed teacher to identify which method provides the best quality-cost balance.

## Trade-offs and Selection Boundary

| Signal | Benefit | Boundary |
|---|---|---|
| Soft logits | Transfers class similarity and ranking | Student can mimic teacher errors; requires same output space |
| Hard pseudo-labels | Cheap target generation and broad unlabeled data | Loses uncertainty; confirmation bias from confident errors |
| Feature matching | Transfers intermediate representations | Architecture alignment and loss weighting are difficult |
| Self-distillation | Improves a model using its own earlier/larger version | Extra training and gains are task-dependent |

Use distillation when serving constraints justify a smaller student and a strong teacher exists. Compare against the student trained on labels alone, and measure latency/memory on target hardware. If a quantized teacher meets constraints with less engineering effort, distillation may be unnecessary.

## Implementation

```python
import math

def softmax(logits, temperature=1.0):
    scaled = [x/temperature for x in logits]
    peak = max(scaled)
    exps = [math.exp(x-peak) for x in scaled]
    return [x/sum(exps) for x in exps]

teacher = [4.0, 1.0, 0.0]
hard = softmax(teacher, 1.0)
soft = softmax(teacher, 3.0)
assert soft[1] > hard[1]  # softened targets expose alternatives
student = [2.8, 1.3, 0.2]
p, q = softmax(teacher, 3), softmax(student, 3)
kl = sum(a*math.log(a/b) for a, b in zip(p, q) if a > 0)
assert kl >= 0
```

## Failure Modes and Diagnostics

1. **Teacher and student label order differs:** apparently valid soft targets train the wrong class mapping. Assert class vocabulary equality before generating targets.
2. **Teacher is wrong or overconfident:** student reproduces systematic errors. Slice teacher/student errors by subgroup and retain ground-truth loss.
3. **Temperature or α dominates:** student ignores labels or receives nearly hard targets. Log component losses and tune on a held-out validation set.
4. **Claimed compression has no deployment gain:** student is smaller but kernel/runtime overhead raises latency. Benchmark end-to-end on target hardware, including preprocessing and batch size.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What does temperature do? | Dividing logits by τ softens the distribution and reveals relative teacher alternatives. |
| Why combine hard and soft targets? | Ground-truth labels anchor correctness while teacher targets transfer useful structure. |
| Design follow-up: student matches accuracy but is no faster. | Benchmark target hardware; architecture, batching, and supported kernels determine real latency. |

## Related Concepts

- [Model compression](model-compression.md), [quantization](quantization.md), and [pruning](pruning.md)
- [Matching notebook](../notebooks/17-knowledge-distillation.ipynb)
- [Interview practice: model compression and optimization](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Hinton, Vinyals, and Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
- Gou et al., [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525).
