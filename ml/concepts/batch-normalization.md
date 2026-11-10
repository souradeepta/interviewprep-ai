# Batch Normalization

**TL;DR:** Batch normalization standardizes activations using minibatch statistics in training and running estimates at inference, often making convolutional networks easier to optimize.

## Prerequisites

Prerequisites: [neural networks](neural-networks.md), mean/variance, and minibatch training. You can explain training versus inference behavior, calculate the transform, compare it with [layer normalization](layer-normalization.md), and choose a safe split and batch arrangement. The AI track introduces neural networks broadly; this ML note covers normalization's optimization, statistical, and deployment consequences.

## Learning Outcomes

- Explain the central assumptions and mechanism of batch normalization.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

If a layer's incoming values drift in scale while its weights are learning, later layers must continually adapt to a moving target. Batch norm recenters and rescales each feature using the current minibatch, then learns γ and β so the network can restore a useful representation. In image models, a channel has many spatial samples, which can provide useful statistics even when the number of images is moderate.

Batch norm is not simply a guarantee that all activations stay Gaussian, nor a substitute for good data splitting. Its batch statistics introduce noise and its stored estimates become part of the deployed model.

## How It Works

For feature j and minibatch B, μ_B=(1/|B|)Σᵢxᵢⱼ and σ²_B=(1/|B|)Σᵢ(xᵢⱼ−μ_B)². Normalize x̂ᵢⱼ=(xᵢⱼ−μ_B)/√(σ²_B+ε), then yᵢⱼ=γⱼx̂ᵢⱼ+βⱼ. During training, update running estimates of mean and variance; inference uses those saved estimates and does not depend on other inference examples. For convolution, statistics are typically computed per channel across batch and spatial positions. Cost is linear in activation count and memory bandwidth often dominates.

The precise estimator and running-average momentum convention vary by framework. Ensure the checkpoint includes running statistics, and set training/evaluation mode correctly. Small or non-representative batches make estimates noisy; distributed training may aggregate statistics across devices (synchronized BN) at communication cost.

The stochastic minibatch statistics change the training computation. A sample's normalized value depends on other samples in that minibatch, so shuffling and batch composition can alter the signal. This noise can have a regularizing effect, but it also means a training example's output is not purely a function of that example and the learned parameters. At inference, fixed running estimates restore a deterministic transformation. Test-time augmentation or online batching should not silently put the layer back into training mode, since then one request can affect another request's score.

Running-statistic updates are commonly exponential moving averages, but their momentum definition differs across libraries: a stated momentum may be the weight on the new batch or the retained history. A high momentum on batch statistics reacts quickly but is noisy; a low momentum is smoother but may lag a changing distribution. If training is extremely short or batches are unrepresentative, the running values can be poor even if training-mode metrics look strong. Re-estimating statistics on a training-only calibration set can be considered, but the set must follow the feature and population contract and must not reuse final evaluation data.

Distributed training adds another choice. Each worker can normalize from local examples, causing its activations to differ from those of other workers, or synchronize sufficient statistics across devices. Synchronized batch norm better approximates a larger batch but incurs communication and can become a bottleneck. Gradient accumulation does not increase the batch norm sample count because statistics are computed separately for each forward pass. If memory forces tiny microbatches, group norm or layer norm may be more suitable than pretending accumulation makes batch norm equivalent to a large batch.

Batch normalization can often be folded into a preceding linear or convolutional layer for inference. Given frozen mean μ, variance σ², learned γ and β, and preceding affine weights W and bias b, the transformation can be rewritten with scaled weights and shifted bias. Export runtimes may perform this fold automatically. Verify numerical equivalence and check that the model is in evaluation mode before exporting; folding current minibatch statistics would be incorrect.

## Trade-offs and Selection Boundary

| Method | Advantage | Boundary |
|---|---|---|
| Batch norm | Effective for many CNNs; permits useful batch-statistic noise | Needs representative batch statistics and train/eval state |
| Layer norm | Independent of batch composition; natural for sequences | Normalizes features within each example, changing inductive bias |
| Group norm | Stable with small image batches | More groups/compute choices; may not match pretrained checkpoints |
| No normalization | Simpler for shallow or specially designed architectures | Deep optimization can be more sensitive |

Prefer batch norm for conventional convolutional vision training with reasonably stable batches. Prefer group/layer norm for tiny batches, variable-length sequences, or batch-composition-sensitive online inference. Keep normalization statistics isolated from validation/test data.

## Implementation

```python
def batch_norm(values, gamma=1.0, beta=0.0, eps=1e-5):
    """Normalize one feature over a training mini-batch."""
    mean = sum(values) / len(values)
    var = sum((v-mean)**2 for v in values) / len(values)
    return [gamma*(v-mean)/(var+eps)**0.5 + beta for v in values]

train = [1.0, 2.0, 7.0, 10.0]
out = batch_norm(train)
assert abs(sum(out)/len(out)) < 1e-9
assert abs(sum(v*v for v in out)/len(out) - 1.0) < 1e-4
# Inference must use frozen training estimates, not the incoming request batch.
running_mean, running_var = 5.0, 14.5
def inference(x): return (x-running_mean)/(running_var+1e-5)**0.5
assert inference(6.0) == inference(6.0)
```

## Failure Modes and Diagnostics

1. **Training mode used for serving:** predictions depend on which requests share a batch. Confirm evaluation mode and test one record alone and in a mixed batch.
2. **Tiny or skewed batches:** running variance is noisy or differs from serving data. Inspect saved statistics and compare batch/group/layer norm.
3. **Train/test contamination:** estimates or preprocessing are fit using held-out examples. Fit the model only on training data and preserve a truly held-out evaluation split.
4. **Unfrozen pretrained statistics during fine-tuning:** small target batches corrupt useful source estimates. Compare freezing statistics with updating them on target training data only.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What changes between training and inference? | Training normalizes by batch statistics and updates running estimates; inference uses the saved estimates. |
| Why is batch norm less common in transformers? | Sequence batches vary and token-wise feature normalization is simpler and independent of batch composition. |
| Debug follow-up: serving scores shift after export. | Check normalization running buffers, train/eval mode, export folding, and input preprocessing parity. |

## Related Concepts

- [Layer normalization](layer-normalization.md), [weight initialization](weight-initialization.md), [overfitting](overfitting-underfitting.md)
- [Matching notebook](../notebooks/03-batch-normalization.ipynb)
- [Interview practice: gradient descent](../interview-prep/ml-theory-questions.md#q-explain-gradient-descent-and-its-variants)

## References

- Ioffe and Szegedy, [Batch Normalization](https://arxiv.org/abs/1502.03167).
- Wu and He, [Group Normalization](https://arxiv.org/abs/1803.08494).
