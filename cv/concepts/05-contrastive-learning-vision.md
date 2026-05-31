# Contrastive Learning in Vision

## Detailed Explanation

Contrastive learning is a self-supervised technique that trains visual representations
by pulling similar pairs of examples together in embedding space while pushing dissimilar
pairs apart. Rather than requiring labeled data, it creates its own supervision signal
from the structure of the data itself.

The two landmark frameworks are SimCLR and CLIP. SimCLR (Chen et al., 2020) takes a
single image, applies two random augmentations to produce a pair, and trains an encoder
to make those two views similar while treating all other images in the batch as negatives.
The key insight is that augmentation-invariant features are semantically meaningful.
CLIP (Radford et al., 2021) extends this to cross-modal pairs: an image and its caption
form a positive pair, while all other image-caption combinations in the batch are
negatives. This yields vision encoders that understand semantic content well enough for
zero-shot classification.

The NT-Xent (normalized temperature-scaled cross-entropy) loss formalizes this:

```
L = -log [ exp(sim(z_i, z_j) / tau) / sum_k exp(sim(z_i, z_k) / tau) ]
```

Temperature tau controls how hard the loss focuses on difficult negatives. Low tau
(tau=0.07) creates sharp distributions that emphasize hard negatives but can be
numerically unstable. High tau (tau=2.0) produces a soft loss that ignores fine-grained
structure.

A critical practical constraint: contrastive learning requires large batch sizes (B>=256
for SimCLR) to provide enough negative examples. With small batches, the model can
trivially solve the task using superficial cues. Production CLIP training uses batch
sizes of 32,768 or larger.

Contrastive pre-training produces transfer-friendly representations that generalize across
tasks, reducing labeling requirements by 10x or more in many downstream applications.

## Core Intuition

Contrastive learning trains encoders by playing a "spot the odd one out" game at scale:
the model must recognize that two different views of the same photo are more similar than
10,000 other photos in the batch. The surprise is that this pressure alone — no labels,
no supervision — forces the model to learn about objects, textures, and scenes, because
those are the only stable features across augmentations.

## How It Works

```
graph TD
    A[Input Image x] --> B[Augmentation 1]
    A --> C[Augmentation 2]
    B --> D[Encoder f]
    C --> E[Encoder f - shared weights]
    D --> F[Projection Head g]
    E --> G[Projection Head g - shared weights]
    F --> H[z_i embedding]
    G --> I[z_j embedding]
    H --> J[NT-Xent Loss]
    I --> J
    J --> K[Pull z_i and z_j together]
    J --> L[Push all other z_k apart]
```

1. **Augment**: Apply two independent random transformations to each image in the batch.
   Common augmentations: random crop, color jitter, Gaussian blur, horizontal flip.
2. **Encode**: Pass both augmented views through a shared encoder (CNN or ViT) to get
   feature vectors h_i and h_j.
3. **Project**: Apply a small MLP projection head to obtain embeddings z_i and z_j.
   The projection head is discarded at inference time; only the encoder is kept.
4. **Compute similarity**: Calculate cosine similarity between all pairs in the batch.
   For a batch of N images, this yields a 2N x 2N similarity matrix.
5. **Apply NT-Xent loss**: For each anchor z_i, the positive is its augmented pair z_j.
   All 2(N-1) other embeddings are negatives. Temperature tau sharpens or softens
   the contrastive signal.
6. **Backpropagate**: Gradients flow through both augmented paths equally; the shared
   weights update to align positive pairs and separate negatives.

## Architecture and Trade-offs

### Contrastive Frameworks Comparison

| Framework | Positive Pair | Negative Source | Batch Size Needed | Use Case |
|-----------|---------------|-----------------|-------------------|----------|
| SimCLR | Two augmented views of same image | Other images in batch | 256-8192 | Vision pre-training |
| CLIP | Image + matched caption | Other pairs in batch | 32768 | Cross-modal understanding |
| MoCo | Two views, one via momentum encoder | Memory queue (65536) | 256 | Memory-efficient contrast |
| BYOL | Two views, online vs target network | None (no negatives) | 256 | Collapse-free contrast |

### Temperature Effects

| tau Value | Gradient Behavior | Risk | Best Use |
|-----------|-------------------|------|----------|
| 0.01 | Very sharp, hard negatives dominate | Gradient explosion, instability | Rarely used |
| 0.07 | Sharp, focuses on hardest negatives | May collapse early | SimCLR default |
| 0.1 | Balanced sharpness | Low | Most production use |
| 0.5 | Soft, all negatives contribute | Slow convergence | Small batches |
| 2.0 | Very flat distribution | Ignores hard negatives | Not recommended |

### Alignment vs Uniformity Trade-off

Contrastive representations can be measured on two axes:
- **Alignment**: how close positive pairs are in embedding space (want high)
- **Uniformity**: how uniformly embeddings are spread on the unit hypersphere (want high)

| Method | Alignment | Uniformity | Notes |
|--------|-----------|------------|-------|
| No training (random) | Low | High | Uniform but meaningless |
| Supervised CE | High | Medium | Task-specific clusters |
| SimCLR (tau=0.07) | High | High | Best general representations |
| SimCLR (tau=2.0) | Medium | Low | Embeddings cluster, poor uniformity |
| Collapse | High | Very Low | All embeddings identical — failure mode |

## Interview Q&A

**Q: Why does SimCLR need such large batch sizes (256-8192)?**

A: The batch provides negatives. With B=256 images and 2 augmentations each, you have
2*255=510 negatives per anchor — enough diversity that the model cannot solve the task
with spurious shortcuts. With B=32, the model may memorize batch-level patterns instead
of learning semantic structure. The memory bank in MoCo solves this by decoupling batch
size from negative count.

**Q: Why is the projection head discarded after training?**

A: The projection head learns features specifically optimized for the contrastive
objective — it compresses information needed to distinguish pairs. These features are not
optimal for downstream tasks like classification, which benefit from richer representations
in the encoder. Empirically, features from before the projection head transfer better.

**Q: What happens when tau is too small (e.g., 0.01)?**

A: The loss distribution becomes extremely peaked. A single hard negative dominates the
denominator, producing very large gradients from that one example. This causes gradient
explosion or instability early in training when representations are random and many
pairs are hard. Start with tau=0.1, then anneal down.

**Q: How would you adapt SimCLR for a medical imaging dataset where augmentations must
preserve diagnostic features?**

A: Be conservative — avoid color jitter (HU values matter in CT), aggressive crops
(lesions are small), and flips (anatomical orientation is meaningful). Use mild noise,
slight rotations (less than 15 degrees), and elastic deformations that preserve structure.
Also consider supervised contrastive learning where positive pairs come from same-patient
scans rather than augmentations.

**Q: How do you detect embedding collapse?**

A: Monitor uniformity = (1/N^2) sum_ij exp(-2 * ||z_i - z_j||^2). If uniformity stays
high (near 0.0) while training, collapse is occurring. Also plot the std of each
embedding dimension — collapse shows near-zero std across all dimensions. Early warning:
loss drops to near 0 too quickly (before 10 epochs on standard datasets).

**Q: CLIP uses 400M image-text pairs. Can you train a useful CLIP with less data?**

A: Yes, but quality degrades gracefully. With 1M pairs, zero-shot accuracy on ImageNet
drops from 76% to roughly 40-50%, still useful for many applications. Key mitigations:
curate data quality over quantity, use stronger augmentations to multiply effective pairs,
and initialize visual encoder from a pre-trained vision model rather than random weights.

## Best Practices

- Use tau=0.07-0.1 for SimCLR; anneal from 0.1 to 0.07 over the first 30% of training
  to avoid early instability from random representations.
- Normalize embeddings to unit sphere before computing cosine similarity — this prevents
  the norm magnitude from dominating the similarity score and makes tau calibration
  consistent across training.
- Include at least 4 augmentation types in SimCLR (crop, color, blur, flip) — ablation
  studies show each contributes independently. Removing any one drops accuracy by 2-5%.
- For CLIP-style training, stratify the batch so each class appears at most once — this
  ensures informative negatives and prevents easy shortcuts from class imbalance.
- Monitor the ratio of positive similarity to mean negative similarity throughout
  training. If this ratio is not increasing, the model is not learning.
- When fine-tuning a contrastive model for classification, use a learning rate 10x lower
  than pre-training — representations are already good and large updates destroy them.
- Use mixed precision (fp16) for contrastive training — the main bottleneck is the
  similarity matrix computation which is a matrix multiply that benefits greatly from
  Tensor Core acceleration.

## Common Pitfalls

- **Trivial solution via augmentation artifacts**: If augmentation parameters leak
  information (e.g., two crops always from the same half of the image), the model learns
  to match artifacts rather than semantics. Fix: verify that a random baseline (matching
  random pairs) cannot exploit the pattern.

- **Too-small batch with NT-Xent**: With B<64, you have fewer than 126 negatives.
  The loss becomes easy and gradients are small. Model appears to train (loss decreases)
  but representations are poor. Fix: use MoCo-style memory bank or increase batch size.
  Symptom: linear probe accuracy stays below 50% on CIFAR-10 after 100 epochs.

- **Not L2-normalizing before similarity computation**: Unnormalized dot products give
  high similarity to high-norm embeddings regardless of direction. The contrastive loss
  then learns to minimize embedding norms rather than align directions. Fix: always apply
  F.normalize(z, dim=-1) before computing the similarity matrix.

- **Keeping the projection head for downstream tasks**: The projection head discards
  useful information for the sake of the contrastive objective. Using it for fine-tuning
  costs 3-5% accuracy on standard benchmarks. Fix: always detach at the encoder output.

- **Symmetric loss computation omitted**: NT-Xent should be computed symmetrically
  (treating z_i as anchor with z_j as positive AND z_j as anchor with z_i as positive,
  then averaging). Asymmetric loss loses half the gradient signal and requires twice the
  epochs to converge.

## Related Concepts

- [06-diffusion-models.md](./06-diffusion-models.md) — generative models that also learn
  from image distributions, complementary to discriminative contrastive methods
- [08-multimodal-vision-llm.md](./08-multimodal-vision-llm.md) — CLIP's contrastive
  pre-training is the foundation for multimodal vision-language models
- [07-video-understanding.md](./07-video-understanding.md) — contrastive learning
  extended to temporal pairs for video representation learning
