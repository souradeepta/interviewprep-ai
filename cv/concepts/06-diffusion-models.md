# Diffusion Models

## Detailed Explanation

Diffusion models are a class of generative models that learn to reverse a gradual
noising process. The forward process systematically destroys structure in data by
adding Gaussian noise over T timesteps until the data becomes indistinguishable from
pure noise. The reverse process learns to reconstruct structure by denoising, one step
at a time.

The key insight from DDPM (Ho et al., 2020) is that the forward process has a closed-form
solution: you can jump directly from x_0 to x_t at any timestep without simulating all
intermediate steps. This makes training efficient — randomly sample a timestep t, add the
corresponding amount of noise, and train a neural network to predict that noise.

The noise-prediction network (typically a U-Net with time conditioning) takes a noisy
image x_t and timestep t as input and outputs the noise that was added. At inference,
the model starts from pure Gaussian noise and iteratively removes predicted noise,
yielding high-quality samples after T=1000 steps.

DDIM (Song et al., 2020) showed that the same trained model can generate samples in
50 steps instead of 1000 by using a deterministic non-Markovian sampling trajectory.
This accelerated inference without retraining made diffusion models practical.

Diffusion models now underpin the most capable image generation systems (Stable Diffusion,
DALL-E 3, Imagen) and are being extended to video, 3D, and protein structure generation.
Key advantages over GANs: stable training with no mode collapse, no adversarial objective,
and direct likelihood evaluation. Key disadvantage: slow multi-step inference compared to
single-pass generative models.

## Core Intuition

Imagine repeatedly photocopying a photo in a noisy machine until only static remains.
A diffusion model learns to run this process backward: starting from static noise, it
makes thousands of tiny corrections until a coherent image emerges. The secret is that
the denoising model never sees the original photo during sampling — it only learns to
make locally plausible corrections based on the current noisy image and how noisy it is.

## How It Works

```
graph TD
    A[x_0 clean image] --> B[Add beta_1 noise]
    B --> C[x_1]
    C --> D[Add beta_2 noise]
    D --> E[x_2 ...]
    E --> F[x_T pure Gaussian noise]
    F --> G[Reverse: predict noise eps_theta at step T]
    G --> H[x_T-1 slightly denoised]
    H --> I[Reverse: predict noise at step T-1]
    I --> J[x_T-2]
    J --> K[... 1000 steps ...]
    K --> L[x_0 hat reconstructed image]
```

1. **Forward process (fixed)**: Apply linear noise schedule beta_t from 0.0001 to 0.02.
   At each step t, x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon where
   epsilon ~ N(0, I). Intuitively: slightly scale down signal and add a small noise bump.

2. **Closed-form shortcut**: Define alpha_bar_t = product from s=1 to t of (1 - beta_s).
   Then x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.
   This skips all intermediate steps for training.

3. **Training**: Sample (x_0, t, epsilon). Compute x_t from x_0 using the closed form.
   Train network epsilon_theta to predict epsilon from (x_t, t). Loss = ||epsilon -
   epsilon_theta(x_t, t)||^2.

4. **U-Net architecture**: Encoder-decoder with skip connections. Time t is embedded as
   a sinusoidal positional encoding, then added to each residual block as a shift+scale.
   Attention at the bottleneck captures global structure.

5. **DDPM sampling**: Start with x_T ~ N(0, I). For t from T down to 1: predict epsilon
   using network, compute x_{t-1} using the DDPM reverse formula, add stochastic noise
   (except at t=1). Takes 1000 forward passes.

6. **DDIM sampling**: Deterministic — use the predicted epsilon to estimate x_0 directly,
   then re-noise to x_{t-1} according to a schedule. Can skip 95% of timesteps (50 steps
   instead of 1000) with nearly identical quality.

## Architecture and Trade-offs

### Noise Schedule Comparison

| Schedule | beta_1 | beta_T | Characteristics | Best For |
|----------|--------|--------|-----------------|----------|
| Linear | 0.0001 | 0.02 | Simple, original DDPM | 256x256 images |
| Cosine | adaptive | adaptive | Gentler at extremes, better quality | High-res images |
| Sqrt | adaptive | adaptive | Aggressive noise, faster training | Research |
| Sigmoid | 0.0001 | 0.02 | Smooth S-curve, stable | General use |

### DDPM vs DDIM vs Score-Based

| Method | Steps | Deterministic | Quality | Speed | Notes |
|--------|-------|---------------|---------|-------|-------|
| DDPM | 1000 | No (stochastic) | High | Slow (~1 min/image) | Original formulation |
| DDIM | 50 | Yes | High | 20x faster | Same trained model |
| DDIM | 10 | Yes | Medium | 100x faster | Quality degrades visibly |
| DPM-Solver | 20 | Yes | High | 50x faster | Better ODE solver |
| Flow Matching | 30 | Yes | Very High | Fast | Rectified flows, newer |

### Guidance Techniques

| Method | Description | Trade-off |
|--------|-------------|-----------|
| Classifier guidance | Scale gradient from external classifier | Needs classifier trained on noisy data |
| Classifier-free guidance (CFG) | Train conditional + unconditional jointly | No external model, tunable strength |
| CFG scale=1 | No guidance | Lower quality but more diversity |
| CFG scale=7.5 | Standard guidance | Good balance |
| CFG scale=15+ | Strong guidance | High fidelity but lower diversity |

## Interview Q&A

**Q: Why does DDPM use noise prediction instead of directly predicting x_0?**

A: Empirically, predicting noise (epsilon parameterization) is more stable. Predicting x_0
directly requires the model to guess the full clean image from highly noisy input, which
is ill-conditioned at high noise levels. Predicting noise is a more uniform task across
timesteps — at any noise level, the model just needs to identify the noise pattern added.
The v-prediction parameterization is a newer alternative that blends both.

**Q: How would you condition a diffusion model on text prompts?**

A: Encode the text with a frozen language model (typically CLIP or T5). Cross-attend
image features to text embeddings at each U-Net resolution level. During training, randomly
drop the conditioning with probability p=0.1-0.2 (classifier-free guidance). At inference,
run two forward passes — conditioned and unconditioned — and interpolate: output =
unconditioned + scale * (conditioned - unconditioned), scale=7.5 is a common default.

**Q: What causes diffusion models to hallucinate details, and how do you mitigate it?**

A: Hallucinations occur when the model lacks sufficient conditioning signal for a specific
region (ambiguous prompt or large uniform area). Mitigation: increase CFG scale to
strengthen prompt adherence, use negative prompts to suppress unwanted content, employ
inpainting-style conditioning to anchor specific regions, and use higher resolution
(more spatial tokens = more detail capacity).

**Q: DDIM sampling skips 95% of timesteps but maintains quality. Why?**

A: DDPM adds stochastic noise at each step, making exact reversal impossible without many
small steps. DDIM reformulates the reverse process as a deterministic ODE. The key insight
is that the trained noise predictor epsilon_theta provides enough information to estimate
x_0 at any timestep, so you can jump larger steps along a deterministic trajectory.
Quality degrades with too few steps (under 20) because the ODE approximation breaks down.

**Q: How would you debug a diffusion model that produces blurry images?**

A: Blurriness indicates insufficient denoising at low noise levels (small t). Diagnose by
plotting loss separately for small-t and large-t samples — if loss is low at large t but
high at small t, the model has not learned fine detail. Fix: increase training time,
use noise schedule that allocates more training time to low-noise levels (cosine schedule),
or reduce the minimum beta value.

**Q: What is the difference between image generation quality when using DDIM with 50 vs
10 steps?**

A: At 50 steps, quality is nearly identical to DDPM-1000 on standard benchmarks (FID
difference less than 1). At 10 steps, you see artifacts — over-sharpening, color banding,
loss of fine detail — because the ODE solver makes larger approximation errors per step.
For latent diffusion (Stable Diffusion), 20-25 steps is the practical minimum for clean
outputs. DPM-Solver++ achieves 10-step quality comparable to DDIM-50.

## Best Practices

- Use cosine noise schedule instead of linear for images larger than 64x64 — linear
  schedule causes x_T to be insufficiently noisy at high resolution, leading to visible
  artifacts. Check that alpha_bar_T < 0.001 to confirm full noising.
- Train with T=1000 timesteps but sample with DDIM-50 or DPM-Solver-20 at inference.
  Never retrain for faster sampling — DDIM and DPM-Solver use the same model weights.
- Monitor FID score (not just loss) during training — loss plateau does not mean FID
  plateau. Compute FID every 50k steps with 10k generated samples vs 10k real samples.
- For text-to-image models, use classifier-free guidance with scale 5-10. Below 5:
  prompt adherence is weak. Above 12: images look sharp but oversaturated (burned colors).
- Condition on time using sinusoidal embeddings of dimension 256-512, projected to match
  each U-Net block's channel width. Additive injection (not concatenation) is standard.
- Use gradient checkpointing for U-Net training — attention layers at full resolution are
  memory bottlenecks. Gradient checkpointing trades 30-40% compute for 2-4x memory savings.
- At inference, use float16 for speed with float32 for the final 20 steps to avoid
  accumulation errors at low noise levels.

## Common Pitfalls

- **Forgetting the sqrt(alpha_bar) normalization**: When computing x_t from x_0, using
  x_t = x_0 + sqrt(1 - alpha_bar_t) * epsilon instead of the correct
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon causes the signal-to-
  noise ratio to be wrong. Symptom: model converges to producing blurry mean-like images.
  Fix: verify your alpha_bar_t values and double-check the forward process formula.

- **Not conditioning U-Net on timestep t**: A model that ignores t cannot learn the noise
  level and produces incoherent outputs across different denoising stages. Symptom: images
  have chaotic noise patterns regardless of t. Fix: inject time embedding into every
  residual block, not just the bottleneck.

- **Using DDPM sampling at test time**: With 1000 steps and 1 second per forward pass,
  generation takes 16 minutes. Always use DDIM or DPM-Solver at inference unless studying
  stochastic sampling effects. Symptom: inference is impractically slow.

- **Applying to high-resolution images directly**: U-Net memory scales as O(H*W) per
  attention layer. At 512x512, full-resolution attention requires tens of GB. Fix: use
  latent diffusion (encode to 64x64 latent space first), or use hierarchical approaches
  with attention only at lower resolutions.

- **Evaluating only with loss, not FID/IS**: Training loss at T=1000 includes many trivial
  timesteps (very high noise, all structure lost). Loss can plateau while sample quality
  continues improving. Always evaluate generated images visually and with FID.

## Related Concepts

- [05-contrastive-learning-vision.md](./05-contrastive-learning-vision.md) — contrastive
  encoders are often used as image quality metrics for diffusion model evaluation
- [08-multimodal-vision-llm.md](./08-multimodal-vision-llm.md) — text conditioning in
  diffusion models uses the same cross-modal alignment concepts as vision-language models
- [07-video-understanding.md](./07-video-understanding.md) — video diffusion extends the
  spatial U-Net to temporal dimensions with 3D convolutions and temporal attention
