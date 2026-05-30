# Neural Network Architectures Quick Reference

## Master Comparison Table

| Architecture | Input Type | Core Operation | Strengths | Weaknesses | Key Use Cases |
|-------------|------------|----------------|-----------|------------|---------------|
| CNN | Images, local-structure data | Convolution + pooling (spatial weight sharing) | Translation invariance, parameter efficiency, hierarchical features | Struggles with long-range dependencies, fixed receptive field | Image classification, object detection, semantic segmentation |
| RNN / LSTM | Sequences (text, time series) | Gated recurrent state (hidden vector) | Handles variable-length sequences, temporal dynamics | Sequential computation (cannot parallelize), vanishing gradient (plain RNN) | Language modeling, speech, time-series forecasting |
| Transformer | Sequences, patches, graphs | Self-attention (global pairwise interactions) | Fully parallel, captures long-range deps, scales to billions of params | O(n²) memory/compute with sequence length, no built-in positional bias | NLP (BERT, GPT), vision (ViT), audio, protein folding |
| ViT (Vision Transformer) | Images (as patch sequences) | Patch embedding + Transformer encoder | State-of-art accuracy at scale, global context from first layer | Needs large data to beat CNNs; no inductive bias for locality | Image classification, dense prediction (with decoder head) |
| Diffusion Model | Noise → image/audio/molecule | Iterative denoising (U-Net backbone) | High-quality diverse samples, stable training | Very slow inference (many denoising steps), inference memory heavy | Image generation, audio synthesis, molecule design |
| GAN | Noise → output (generator) vs. real/fake (discriminator) | Adversarial minimax game | Sharp samples, no likelihood needed | Training instability (mode collapse, gradient vanishing), hard to evaluate | Image synthesis, super-resolution, data augmentation |

---

## Architecture Diagrams (ASCII)

### CNN
```
Input Image
    |
[Conv3x3 + ReLU]  <- local receptive field, shared weights
    |
[MaxPool 2x2]     <- downsampling, translation invariance
    |
[Conv3x3 + ReLU]
    |
[MaxPool 2x2]
    |
[Flatten]
    |
[Fully Connected]
    |
 Predictions
```

### RNN / LSTM
```
x_1   x_2   x_3   ...   x_T
 |     |     |           |
[h_0]->[h_1]->[h_2]->...->[h_T]
               |                |
           (optional         Output /
            output at t)   Final state

LSTM cell gates:
  forget gate: f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)
  input gate:  i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)
  cell update: C_t = f_t * C_{t-1} + i_t * tanh(W_c [h_{t-1}, x_t])
  output gate: o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)
  hidden:      h_t = o_t * tanh(C_t)
```

### Transformer (Encoder Block)
```
Input Tokens
    |
[Token Embedding + Positional Encoding]
    |
 +-----------+
 | Multi-Head|   Q, K, V projected from same input
 | Self-Attn |   Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
 +-----------+
    |
[Add & Norm]   <- residual connection
    |
 +-----------+
 | Feed-Fwd  |   2x Linear + GELU/ReLU
 |  Network  |   dim: d_model -> 4*d_model -> d_model
 +-----------+
    |
[Add & Norm]
    |
 (repeat N layers)
    |
Contextual Representations
```

### ViT (Vision Transformer)
```
Image (H x W x C)
    |
[Split into patches: (H/P x W/P) patches of size P x P]
    |
[Linear Projection of each patch -> embedding dim]
    |
[Prepend [CLS] token]
    |
[Add Learnable Position Embeddings]
    |
[Transformer Encoder x L layers]
    |
[CLS token output]
    |
[MLP Classification Head]
    |
Class Probabilities
```

### Diffusion Model (DDPM)
```
Forward process (add noise):
x_0 (real image) -> x_1 -> x_2 -> ... -> x_T (pure Gaussian noise)
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I)

Reverse process (denoise — learned):
x_T -> x_{T-1} -> ... -> x_0
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2*I)

Network: U-Net with time embedding t injected at each scale
Training: predict noise epsilon_theta(x_t, t) ~ actual noise epsilon
Loss: ||epsilon - epsilon_theta(x_t, t)||^2
```

### GAN
```
Noise z ~ N(0,I)
    |
[Generator G]     <- wants to fool D
    |
Fake samples G(z)
    |
              Real samples x ~ p_data
              |
         +----------+
         |Discriminator D|  -> P(real) in [0,1]
         +----------+
              |
         Adversarial Loss:
         max_D: E[log D(x)] + E[log(1 - D(G(z)))]
         min_G: E[log(1 - D(G(z)))]  (or: max E[log D(G(z))])
```

---

## Per-Architecture Details

### CNN

**Key papers:** LeNet (1989), AlexNet (2012), VGG (2014), ResNet (2015), EfficientNet (2019)

**Key hyperparameters:**
- Kernel size: 3x3 (most common), 1x1 (channel mixing), 5x5 (larger context)
- Filters per layer: 32, 64, 128, 256, 512 (doubles with each pool)
- Stride: 1 (preserve resolution), 2 (halve resolution instead of pool)
- Depth: ResNet-50 (50 layers), EfficientNet-B0 to B7

**Common pitfalls:**
- Training without BatchNorm on deep networks -> unstable
- Not using residual connections past 20 layers -> vanishing gradient
- Forgetting data augmentation -> overfit on small image datasets

---

### RNN / LSTM

**Key papers:** Elman (1990), LSTM Hochreiter & Schmidhuber (1997), GRU Cho et al. (2014)

**Key hyperparameters:**
- Hidden size: 128–1024
- Layers: 1–4 (bidirectional doubles effective hidden)
- Dropout: applied between layers, not inside recurrent connections

**Common pitfalls:**
- Forgetting to reset hidden state between unrelated sequences
- Plain RNN exploding/vanishing gradient beyond 20 timesteps -> use LSTM/GRU
- Bidirectional only valid for tasks with full sequence available (not autoregressive)

---

### Transformer

**Key papers:** Attention Is All You Need (2017), BERT (2019), GPT-2/3/4, T5, LLaMA

**Key hyperparameters:**
- d_model: 512–8192
- Heads: 8–128 (d_head = d_model / heads, typically 64)
- Layers: 6–96
- FFN expansion: 4x d_model typical
- Dropout: 0.0–0.1 (lower for large models)

**Common pitfalls:**
- Not using learning rate warmup -> unstable early training
- Positional encoding mismatch at inference (longer context than trained on)
- KV cache not enabled at inference -> 10x+ latency hit

---

### ViT

**Key papers:** ViT (Dosovitskiy 2020), DeiT (2020), Swin Transformer (2021), DINO (2021)

**Patch sizes:** 16x16 (standard), 32x32 (faster, lower accuracy), 8x8 (higher accuracy, 4x more patches)

**Common pitfalls:**
- ViT underperforms ResNet on small datasets without pretraining (lacks inductive bias)
- Not using ImageNet-21k or JFT pretraining -> needs DeiT distillation trick instead

---

### Diffusion Models

**Key papers:** DDPM Ho et al. (2020), DDIM (2020), Stable Diffusion (2022), DiT (2023)

**Inference tricks:**
- DDIM sampling: 50 steps instead of 1000 (deterministic, same quality)
- Classifier-free guidance: scale omega > 1 boosts prompt adherence, increases diversity loss
- LoRA fine-tuning for style/concept adaptation

**Common pitfalls:**
- More diffusion steps != always better; DDIM at 50 steps usually sufficient
- Very high guidance scale -> artifacts (faces distort, colors saturate)

---

### GAN

**Key papers:** GAN Goodfellow (2014), DCGAN (2015), StyleGAN2 (2020), BigGAN (2018)

**Training tricks:**
- Label smoothing on discriminator (real label = 0.9, not 1.0)
- Two-timescale update rule: different LRs for G and D (e.g., 1e-4 for G, 4e-4 for D)
- Spectral normalization on discriminator weights
- Progressive growing for high-resolution outputs

**Common pitfalls:**
- Mode collapse: G produces single or few samples -> use minibatch discrimination or unrolled GANs
- Discriminator too strong early -> G gets zero gradient; use equal capacity
- No standard likelihood evaluation -> use FID (Frechet Inception Distance) as proxy
