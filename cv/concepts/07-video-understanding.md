# Video Understanding

## Detailed Explanation

Video understanding extends image recognition to the temporal dimension, capturing not
just what is in a scene but how it changes over time. This is fundamental for action
recognition, anomaly detection, video retrieval, and autonomous driving — all requiring
models to reason about motion and dynamics, not just static appearance.

The core challenge is computational: a 10-second video at 30fps has 300 frames, each
potentially a 224x224 image. Processing all frames through a 2D CNN would be 300x the
compute of a single image. Practical approaches make trade-offs between temporal resolution
and computational budget.

Three main architectural families address this:

**Optical flow** (Lucas-Kanade, Farneback) estimates per-pixel motion vectors between
consecutive frames. Two-stream networks (Simonyan & Zisserman, 2014) feed RGB frames
through a spatial stream and optical flow through a temporal stream, then fuse predictions.
Optical flow provides explicit motion representations but is expensive to compute.

**3D CNNs** (C3D, I3D) extend 2D convolution kernels from (k, k) to (t, k, k), learning
spatial and temporal patterns jointly. I3D inflates ImageNet-pretrained 2D weights into
3D, enabling powerful transfer learning. Memory and compute scale as O(T*H*W).

**Temporal attention** (TimeSformer, Video Swin) applies self-attention across frame
patches. "Divided Space-Time Attention" attends within frames and across frames separately,
reducing quadratic cost. Transformers outperform CNNs on Kinetics-400 at scale but need
more data.

Action recognition benchmarks: Kinetics-400 (400 classes, 240k clips), UCF-101 (101
classes), Something-Something v2 (motion-centric, temporal reasoning required).

## Core Intuition

A video is a stack of photos that forms a flipbook — understanding it means recognizing
not just what each page contains, but the story told by the sequence. Optical flow is
like tracing where each pixel moves between pages; 3D CNNs look at a small cube of pages
at once; temporal attention asks each frame "which other frames should I pay attention to?"

## How It Works

```
graph TD
    A[Video clip T frames] --> B[Frame sampling strategy]
    B --> C[RGB spatial stream]
    B --> D[Optical flow temporal stream]
    C --> E[2D CNN or 3D CNN encoder]
    D --> F[2D CNN encoder]
    E --> G[Spatial features]
    F --> H[Temporal features]
    G --> I[Late fusion or score fusion]
    H --> I
    I --> J[Action classification logits]
    A --> K[3D CNN alternative path]
    K --> L[3D Conv spatial and temporal jointly]
    L --> J
```

1. **Frame sampling**: Choose T frames from the full video. Uniform sampling (every
   interval of floor(total/T) frames) is common. Dense sampling (T=64 frames, short
   clip) captures motion; sparse sampling (T=8 frames, full video) captures scene context.

2. **Input representation**: RGB frames (H x W x 3) for spatial; optical flow
   (H x W x 2, dx and dy displacement) for temporal. Stack 10 consecutive flow frames
   to form a 20-channel temporal input.

3. **Spatial processing (2D CNN path)**: Apply CNN independently to each frame, pool
   features across time (mean pool, max pool, or learned temporal aggregation).

4. **Spatiotemporal processing (3D CNN path)**: Input tensor is (B, C, T, H, W).
   Conv3d with kernel (t, h, w) slides across time and space simultaneously, learning
   motion patterns directly from raw pixels.

5. **Temporal attention**: Divide video into N patches per frame, resulting in T*N tokens.
   Apply space-time self-attention (full) or factored attention (space then time separately)
   using transformer encoder blocks.

6. **Classification head**: Pool over spatial and temporal dimensions, pass through
   linear classifier. For action recognition, global average pooling (T, H, W) -> FC.

## Architecture and Trade-offs

### Temporal Modeling Approaches

| Method | Representation | Temporal Receptive Field | Memory | Strengths |
|--------|---------------|--------------------------|--------|-----------|
| 2D CNN + mean pool | Per-frame features | Whole video (pool) | Low | Fast, works well for appearance |
| Optical flow | Explicit motion | Adjacent frames | Medium | Strong temporal signal |
| Two-stream | RGB + flow | Adjacent frames | High | Complementary streams |
| 3D CNN (C3D) | Volumetric | T=8-16 frames | High | Joint spatiotemporal |
| I3D | Inflated 3D | T=64 frames | Very High | Best with ImageNet init |
| TimeSformer | Patch attention | Whole video | High | Flexible, data-hungry |

### Frame Sampling Strategy

| Strategy | Description | Best For | Risk |
|----------|-------------|----------|------|
| Uniform sampling | Evenly spaced frames | Scene-level understanding | Misses fast motion |
| Dense sampling | Short clip, many frames | Short actions, fine motion | Loses temporal context |
| Keyframe extraction | Scene change detection | Efficiency, long videos | Misses transitions |
| Multi-scale | Both dense and sparse | All action durations | Compute cost |

### Action Recognition Benchmark Performance

| Model | Kinetics-400 Top-1 | Parameters | Speed (clips/sec) |
|-------|-------------------|------------|-------------------|
| 2D CNN (ResNet-50) | ~72% | 25M | Fast (300+) |
| Two-stream (VGG) | ~88% | 138M | Slow (needs flow) |
| I3D (64-frame) | ~72-80% | 25M | Medium (50-100) |
| SlowFast (8+32 frames) | ~79% | 35M | Medium |
| TimeSformer-L | ~80% | 121M | Slow (10-20) |

## Interview Q&A

**Q: Two-stream networks are expensive because they require optical flow computation.
When would you use them anyway?**

A: When temporal dynamics are the primary cue — sports analytics (ball trajectory, player
movement), sign language recognition, fine-grained action recognition where appearance
alone is ambiguous (e.g., "throwing" vs "catching"). Optical flow also generalizes better
across cameras and lighting than RGB-based motion estimation. For production, precompute
and cache flow at dataset creation time; flow adds ~100ms per frame but only once.

**Q: What is the difference between something-something performance and kinetics
performance, and why do they come apart?**

A: Kinetics tests object-centric recognition ("playing guitar") where visual appearance
matters most — even a single keyframe often suffices. Something-Something tests temporal
reasoning ("moving object from left to right") where motion direction determines the
correct label, not the objects shown. Models that excel on Kinetics often rely on object
appearance shortcuts. Something-Something specifically measures whether the model
understands temporal dynamics.

**Q: A 3D CNN trained on 8-frame clips fails on 60-second videos at inference. How do
you adapt it?**

A: Sample multiple 8-frame clips uniformly from the video, run inference on each, and
aggregate predictions (max pool or mean pool logits). Empirically, 10-25 clips at 3 crops
per clip gives stable predictions. For actions spanning different timescales, use a
hierarchical approach: one pass for short clips, another for the full video context
(slow-fast architecture). Avoid feeding the full 60s at once — temporal receptive field
of the 3D CNN is fixed.

**Q: What are the first signs that your video model is using appearance shortcuts
instead of learning temporal dynamics?**

A: Single-frame accuracy is close to full-video accuracy (within 5%). Permuting frame
order does not degrade performance significantly. Class activation maps show the model
attending to objects rather than motion regions. Fix: test with shuffled frames
explicitly and report the accuracy drop. If under 5%, temporal modeling is not working.
Use Something-Something-style evaluation or introduce frame-order prediction as auxiliary.

**Q: How would you handle variable-length videos in a 3D CNN framework?**

A: Three approaches: (1) Pad/trim to fixed T (simplest, information loss). (2) Sample K
clips of fixed T and aggregate at inference (best quality). (3) Use temporal average
pooling after the last 3D conv layer — this is resolution-invariant and allows variable-T
inputs with the same model weights. Option 2 is standard in research (TCN sampling);
option 3 is used in production for simplicity.

**Q: Optical flow seems redundant given that 3D CNNs learn temporal features from raw
pixels. When does explicit flow help?**

A: Flow explicitly disentangles camera motion from object motion, which 3D CNNs cannot
do from pixels alone (a panning camera creates strong motion signal even with no action).
Flow helps especially for egocentric videos (camera is constantly moving), drone footage,
and sports with fast camera pans. For fixed-camera surveillance or controlled studio
recording, 3D CNNs capture sufficient motion without explicit flow.

## Best Practices

- Always report FLOPs per clip and clips per second alongside accuracy — models that
  process 8 frames need 3-10 clips per video for stable predictions, multiplying actual
  compute by 3-10x.
- Use temporal data augmentation: random clip extraction (not just center crop), temporal
  jitter (vary start frame by 2-5 frames), and occasional temporal reversal to prevent
  models from exploiting clip-boundary artifacts.
- Initialize 3D CNN weights from ImageNet-pretrained 2D models (I3D-style inflation):
  duplicate 2D weights T times along the temporal axis and divide by T to preserve
  activation scale. This provides a 5-10% absolute accuracy boost vs random init.
- For action recognition, uniform frame sampling works for most actions. For fine-grained
  temporal reasoning (Something-Something), use T=8 dense frames (short clip, 32-64 frames
  total at 30fps = 1-2 seconds) rather than sparse sampling across the full video.
- Apply spatial augmentations (crop, flip, color jitter) consistently across all frames in
  a clip — applying different crops per frame creates artificial motion that pollutes the
  temporal signal.
- Separate spatial and temporal learning rates when fine-tuning: 1e-4 for temporal layers,
  1e-5 for spatial layers that were well-initialized from image pre-training.

## Common Pitfalls

- **Not using temporal augmentation**: Models overfit to the first or center clip
  extracted during training. Always randomly sample the start frame during training.
  Symptom: large gap between train and val accuracy that does not close with more data.
  Fix: add random temporal crop and temporal jitter to the data pipeline.

- **Evaluating with single-clip single-crop**: Single-clip accuracy underestimates true
  model quality by 3-8% on Kinetics. Always evaluate with 10-25 clips x 3 spatial crops
  and report mean accuracy. Comparing single-clip models to multi-clip baselines will
  produce misleading conclusions.

- **Input shape confusion for Conv3d**: PyTorch Conv3d expects (B, C, T, H, W). Permuting
  T and C dimensions is a silent bug — the model runs without error but learns nothing.
  Always print input shapes at the start of training and verify with small dummy batches.

- **Optical flow scale normalization**: Raw optical flow magnitudes vary with video
  resolution and camera speed. Not normalizing to [-1, 1] before input causes the temporal
  stream to see wildly different input scales. Fix: clip flow to [-20, 20] pixels,
  normalize to [-1, 1]. Apply per-video normalization for best results.

- **Using 3D CNNs for very long videos (>60 frames) without hierarchical pooling**:
  Memory scales linearly with T. At T=64, a single batch of 8 clips at 224x224 requires
  ~16 GB. Fix: use SlowFast (8 frames slow + 32 frames fast) or segment the video and
  aggregate features post-processing.

## Related Concepts

- [05-contrastive-learning-vision.md](./05-contrastive-learning-vision.md) — contrastive
  learning applied to video pairs enables self-supervised video representation learning
- [06-diffusion-models.md](./06-diffusion-models.md) — video diffusion extends DDPM to
  temporal sequences using 3D convolutions and temporal attention in the denoising U-Net
- [08-multimodal-vision-llm.md](./08-multimodal-vision-llm.md) — video understanding is
  increasingly combined with language for video captioning and question answering
