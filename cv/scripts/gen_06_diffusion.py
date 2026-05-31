"""Generator for CV notebook 06: Diffusion Models."""
import nbformat
import os


def make_cell(source, cell_type="code"):
    if cell_type == "markdown":
        return nbformat.v4.new_markdown_cell(source)
    return nbformat.v4.new_code_cell(source)


def build_notebook():
    cells = []

    # Cell 1: Title + Objectives
    cells.append(make_cell("""\
# Diffusion Models

## Learning Objectives
1. Implement the DDPM forward process (closed-form noising) from scratch
2. Train a U-Net denoiser on synthetic 28x28 patterns using PyTorch
3. Implement DDPM reverse (sampling) and DDIM accelerated sampling
4. Compare DDPM vs DDIM at 1000, 50, and 10 steps in quality and speed
""", "markdown"))

    # Cell 2: Imports + seeds + device
    cells.append(make_cell("""\
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
"""))

    # Cell 3: Level 1 header
    cells.append(make_cell("""\
## Level 1: DDPM Forward Process (NumPy)

The forward process adds Gaussian noise over T steps.
Key formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

where alpha_bar_t = product of (1 - beta_s) for s=1..t
""", "markdown"))

    # Cell 4: Level 1 code
    cells.append(make_cell("""\
def make_noise_schedule(T=1000, beta_start=0.0001, beta_end=0.02):
    \"\"\"Create linear noise schedule and precompute alpha_bar values.

    Args:
        T: total diffusion timesteps
        beta_start: initial noise level
        beta_end: final noise level

    Returns:
        betas: (T,) noise values
        alpha_bars: (T,) cumulative product of (1 - beta_t)
    \"\"\"
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)  # ᾱ_t = prod_{s=1}^{t} (1 - beta_s)
    return betas, alpha_bars


def forward_diffusion(x0, t, alpha_bars):
    \"\"\"Closed-form forward diffusion: x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*eps.

    Args:
        x0: np.ndarray, clean signal of any shape
        t: int, timestep (0-indexed, so t=0 means first noisy step)
        alpha_bars: (T,) precomputed cumulative products

    Returns:
        x_t: noisy version of x0 at timestep t
        epsilon: the noise added (for training)
    \"\"\"
    alpha_bar_t = alpha_bars[t]
    epsilon = np.random.randn(*x0.shape)  # standard Gaussian noise
    x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * epsilon
    return x_t, epsilon


T = 1000
betas, alpha_bars = make_noise_schedule(T=T)

print(f"Noise schedule summary:")
print(f"  beta[0]   = {betas[0]:.6f} (almost no noise)")
print(f"  beta[499] = {betas[499]:.6f} (moderate noise)")
print(f"  beta[999] = {betas[999]:.6f} (heavy noise)")
print(f"  alpha_bar[0]   = {alpha_bars[0]:.4f} (signal mostly intact)")
print(f"  alpha_bar[499] = {alpha_bars[499]:.4f} (50% signal, 50% noise)")
print(f"  alpha_bar[999] = {alpha_bars[999]:.6f} (almost pure noise)")

# Demonstrate on a simple 1D signal: a sine wave
t_axis = np.linspace(0, 2 * np.pi, 64)
x0_signal = np.sin(t_axis)  # clean signal

fig, axes = plt.subplots(1, 6, figsize=(15, 3))
timesteps_to_show = [0, 100, 250, 500, 750, 999]
for ax, t_show in zip(axes, timesteps_to_show):
    x_t, _ = forward_diffusion(x0_signal, t_show, alpha_bars)
    ax.plot(x_t, color='steelblue', lw=0.8)
    ax.set_title(f"t={t_show}")
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
axes[0].set_ylabel("Signal amplitude")
fig.suptitle("DDPM Forward Process: Sine Wave Degrading to Noise")
plt.tight_layout()
plt.savefig("/tmp/ddpm_forward.png", dpi=80, bbox_inches="tight")
plt.show()

# Show SNR (signal-to-noise ratio) across timesteps
snr = alpha_bars / (1.0 - alpha_bars + 1e-8)
print(f"\\nSNR at key timesteps: t=0: {snr[0]:.1f}, t=500: {snr[499]:.3f}, t=999: {snr[999]:.5f}")
"""))

    # Cell 5: Level 2 header
    cells.append(make_cell("""\
## Level 2: U-Net Denoiser on Synthetic 28x28 Patterns

Train a U-Net to predict noise from noisy images.
Synthetic dataset: 2 classes of 28x28 binary patterns
  - Class 0: horizontal gradient
  - Class 1: vertical gradient + noise
U-Net: encoder-decoder with skip connections and time embedding injection.
""", "markdown"))

    # Cell 6: Level 2 code
    cells.append(make_cell("""\
# ---- Synthetic image dataset ----
def make_synthetic_images(n=512, img_size=28):
    \"\"\"Generate two classes of simple 28x28 patterns.

    Class 0: horizontal gradient (value increases left to right)
    Class 1: vertical gradient (value increases top to bottom)

    Returns:
        images: (n, 1, img_size, img_size) float32 tensor in [-1, 1]
        labels: (n,) int tensor
    \"\"\"
    images = []
    labels_list = []
    for i in range(n):
        cls = i % 2
        img = np.zeros((img_size, img_size), dtype=np.float32)
        if cls == 0:
            # Horizontal gradient + mild noise
            for col in range(img_size):
                img[:, col] = (col / (img_size - 1)) * 2.0 - 1.0
        else:
            # Vertical gradient + mild noise
            for row in range(img_size):
                img[row, :] = (row / (img_size - 1)) * 2.0 - 1.0
        img += np.random.randn(img_size, img_size).astype(np.float32) * 0.05
        images.append(img)
        labels_list.append(cls)
    X = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
    y = torch.tensor(labels_list, dtype=torch.long)
    return X, y


# ---- Time embedding ----
def sinusoidal_time_embedding(t, dim=128):
    \"\"\"Sinusoidal positional encoding for timestep t.

    Args:
        t: (B,) int tensor of timesteps
        dim: embedding dimension

    Returns:
        emb: (B, dim) float tensor
    \"\"\"
    half_dim = dim // 2
    factor = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    freqs = torch.exp(-factor * torch.arange(half_dim, dtype=torch.float32)).to(t.device)
    angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)
    return emb


# ---- Simple U-Net ----
class ResBlock(nn.Module):
    \"\"\"Residual block with time embedding injection.

    Injects time embedding as a learned shift+scale after the first conv.
    \"\"\"
    def __init__(self, in_ch, out_ch, time_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)  # scale + shift
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        # Time conditioning: scale and shift
        ts = self.time_proj(t_emb)[:, :, None, None]  # (B, 2*out_ch, 1, 1)
        scale, shift = ts.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    \"\"\"Minimal U-Net for noise prediction in DDPM.

    Architecture:
        Encoder: 3 levels with stride-2 downsampling
        Bottleneck: ResBlock
        Decoder: 3 levels with bilinear upsampling + skip connections
        Output: predicts noise epsilon at each pixel
    \"\"\"
    def __init__(self, img_channels=1, base_ch=32, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        ch = base_ch
        # Encoder
        self.enc1 = ResBlock(img_channels, ch, time_dim)
        self.down1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.enc2 = ResBlock(ch, ch * 2, time_dim)
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1)
        self.enc3 = ResBlock(ch * 2, ch * 4, time_dim)
        self.down3 = nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1)
        # Bottleneck
        self.bot = ResBlock(ch * 4, ch * 4, time_dim)
        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ResBlock(ch * 4 + ch * 4, ch * 4, time_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ResBlock(ch * 4 + ch * 2, ch * 2, time_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ResBlock(ch * 2 + ch, ch, time_dim)
        # Output head
        self.out_conv = nn.Conv2d(ch, img_channels, 1)

    def forward(self, x, t):
        \"\"\"Forward pass: predict noise from noisy image.

        Args:
            x: (B, C, H, W) noisy image
            t: (B,) int timestep

        Returns:
            predicted_noise: (B, C, H, W)
        \"\"\"
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        # Encoder
        e1 = self.enc1(x, t_emb)                    # (B, ch, H, W)
        e2 = self.enc2(self.down1(e1), t_emb)       # (B, 2ch, H/2, W/2)
        e3 = self.enc3(self.down2(e2), t_emb)       # (B, 4ch, H/4, W/4)
        # Bottleneck
        b = self.bot(self.down3(e3), t_emb)         # (B, 4ch, H/8, W/8)
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1), t_emb)   # (B, 4ch, H/4, W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)  # (B, 2ch, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)  # (B, ch, H, W)
        return self.out_conv(d1)


# ---- Training ----
# Setup noise schedule as tensors
betas_torch = torch.tensor(betas, dtype=torch.float32).to(device)
alpha_bars_torch = torch.tensor(alpha_bars, dtype=torch.float32).to(device)

X_train, y_train = make_synthetic_images(n=512, img_size=28)
X_train = X_train.to(device)

unet = SimpleUNet(img_channels=1, base_ch=32, time_dim=128).to(device)
optimizer = optim.Adam(unet.parameters(), lr=1e-3)
n_params = sum(p.numel() for p in unet.parameters())
print(f"U-Net parameters: {n_params:,}")

# Training loop
batch_size = 32
n_steps = 300
loss_history = []

unet.train()
for step in range(n_steps):
    idx = torch.randperm(X_train.shape[0])[:batch_size]
    x0 = X_train[idx]

    # Sample random timestep for each image in batch
    t = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)
    alpha_bar_t = alpha_bars_torch[t].view(-1, 1, 1, 1)

    # Sample noise and compute x_t
    epsilon = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * epsilon

    # Predict noise
    eps_pred = unet(x_t, t)
    loss = F.mse_loss(eps_pred, epsilon)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
    optimizer.step()
    loss_history.append(loss.item())

    if (step + 1) % 50 == 0:
        print(f"Step {step+1:3d} | Loss: {loss.item():.6f}")

# Visualize denoising at different noise levels
unet.eval()
x0_test = X_train[:1]  # single test image
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, t_show in enumerate([100, 300, 500, 700, 900]):
    t_tensor = torch.tensor([t_show], device=device, dtype=torch.long)
    alpha_bar_t = alpha_bars_torch[t_show].item()
    eps_actual = torch.randn_like(x0_test)
    x_t = (torch.sqrt(torch.tensor(alpha_bar_t)) * x0_test +
           torch.sqrt(torch.tensor(1 - alpha_bar_t)) * eps_actual)
    with torch.no_grad():
        eps_pred = unet(x_t, t_tensor)
    # Estimated x0 from predicted noise
    x0_est = ((x_t - torch.sqrt(torch.tensor(1 - alpha_bar_t)) * eps_pred) /
               torch.sqrt(torch.tensor(alpha_bar_t)))
    axes[i].imshow(x0_est[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[i].set_title(f"t={t_show}\\ndenoised est")
    axes[i].axis('off')
fig.suptitle("U-Net Denoising Estimates at Different Noise Levels")
plt.tight_layout()
plt.savefig("/tmp/unet_denoising.png", dpi=80, bbox_inches="tight")
plt.show()
print(f"Training loss: {loss_history[0]:.6f} -> {loss_history[-1]:.6f}")
"""))

    # Cell 7: RW1 header
    cells.append(make_cell("""\
## Real-World Example 1: DDPM Reverse Process Sampling

Start from x_T ~ N(0, I) and apply T denoising steps to generate a new image.
DDPM reverse formula:
  x_{t-1} = (1/sqrt(1-beta_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta(x_t, t))
             + sqrt(beta_t) * z   where z ~ N(0,I) for t > 1
""", "markdown"))

    # Cell 8: RW1 code
    cells.append(make_cell("""\
@torch.no_grad()
def ddpm_sample(model, T, betas_t, alpha_bars_t, img_shape, device, n_vis_steps=10):
    \"\"\"Generate a sample via full DDPM reverse process.

    Args:
        model: trained U-Net
        T: total timesteps
        betas_t: (T,) noise schedule tensor
        alpha_bars_t: (T,) cumulative products tensor
        img_shape: tuple (B, C, H, W)
        device: torch device
        n_vis_steps: how many intermediate images to save

    Returns:
        x_final: (B, C, H, W) generated images
        intermediates: list of (t, image) tuples for visualization
    \"\"\"
    model.eval()
    B, C, H, W = img_shape
    x = torch.randn(B, C, H, W, device=device)  # start from pure noise

    alphas_t = 1.0 - betas_t
    intermediates = []
    vis_interval = T // n_vis_steps

    for t_idx in reversed(range(T)):
        t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
        alpha_bar_t = alpha_bars_t[t_idx].item()
        alpha_t = alphas_t[t_idx].item()
        beta_t = betas_t[t_idx].item()

        # Predict noise
        eps_pred = model(x, t_tensor)

        # DDPM reverse step
        # Estimate x0 from predicted noise
        x0_est = (x - np.sqrt(1 - alpha_bar_t) * eps_pred) / (np.sqrt(alpha_bar_t) + 1e-8)
        x0_est = x0_est.clamp(-1, 1)  # clip estimated x0 to valid range

        # Compute mean of posterior q(x_{t-1}|x_t, x0_est)
        if t_idx > 0:
            alpha_bar_prev = alpha_bars_t[t_idx - 1].item()
        else:
            alpha_bar_prev = 1.0  # at t=0, no prior noise

        mean = (
            np.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t) * x0_est
            + np.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * x
        )

        if t_idx > 0:
            # Add noise (stochastic)
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)
            noise = torch.randn_like(x)
            x = mean + np.sqrt(max(posterior_var, 1e-10)) * noise
        else:
            x = mean  # no noise at final step

        if t_idx % vis_interval == 0 or t_idx == 0:
            intermediates.append((t_idx, x.cpu().clone()))

    return x, intermediates


# Generate samples with DDPM
t0 = time.time()
x_gen, intermediates = ddpm_sample(
    unet, T, betas_torch, alpha_bars_torch,
    img_shape=(4, 1, 28, 28), device=device, n_vis_steps=8
)
ddpm_time = time.time() - t0
print(f"DDPM sampling ({T} steps): {ddpm_time:.2f}s for 4 images")

# Visualize the generation trajectory for first image
fig, axes = plt.subplots(1, len(intermediates), figsize=(15, 3))
for ax, (t_vis, img_vis) in zip(axes, intermediates):
    ax.imshow(img_vis[0, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
    ax.set_title(f"t={t_vis}")
    ax.axis('off')
fig.suptitle("DDPM Reverse Process: x_T -> x_0 (one sample)")
plt.tight_layout()
plt.savefig("/tmp/ddpm_sampling.png", dpi=80, bbox_inches="tight")
plt.show()

# Show final 4 generated images
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(x_gen[i, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
    ax.set_title(f"Generated {i}")
    ax.axis('off')
plt.suptitle("DDPM Generated Images (4 samples)")
plt.tight_layout()
plt.savefig("/tmp/ddpm_generated.png", dpi=80, bbox_inches="tight")
plt.show()
"""))

    # Cell 9: RW2 header
    cells.append(make_cell("""\
## Real-World Example 2: DDIM Accelerated Sampling

DDIM (Denoising Diffusion Implicit Models) uses the same trained U-Net but
samples deterministically in far fewer steps by following an ODE trajectory.

DDIM step: x_{t_prev} = sqrt(alpha_bar_prev) * x0_pred
            + sqrt(1 - alpha_bar_prev) * eps_pred

where x0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

Compare: DDPM-1000 vs DDIM-50 vs DDIM-10
""", "markdown"))

    # Cell 10: RW2 code
    cells.append(make_cell("""\
@torch.no_grad()
def ddim_sample(model, T, alpha_bars_t, img_shape, device, n_ddim_steps=50):
    \"\"\"Accelerated DDIM sampling using a subset of timesteps.

    Args:
        model: trained U-Net
        T: total training timesteps
        alpha_bars_t: (T,) cumulative products tensor
        img_shape: tuple (B, C, H, W)
        device: torch device
        n_ddim_steps: number of DDIM steps (<<T)

    Returns:
        x_final: (B, C, H, W) generated images
        elapsed: float, wall-clock seconds
    \"\"\"
    model.eval()
    B, C, H, W = img_shape

    # Select evenly spaced timesteps for DDIM
    step_size = T // n_ddim_steps
    ddim_timesteps = list(range(0, T, step_size))[::-1]  # descending

    x = torch.randn(B, C, H, W, device=device)

    t0 = time.time()
    for i, t_idx in enumerate(ddim_timesteps):
        t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
        alpha_bar_t = alpha_bars_t[t_idx].item()

        # Predict noise
        eps_pred = model(x, t_tensor)

        # Estimate clean x0
        sqrt_ab = np.sqrt(max(alpha_bar_t, 1e-8))
        x0_pred = (x - np.sqrt(1.0 - alpha_bar_t) * eps_pred) / sqrt_ab
        x0_pred = x0_pred.clamp(-1, 1)

        # Previous timestep alpha_bar
        if i + 1 < len(ddim_timesteps):
            t_prev = ddim_timesteps[i + 1]
            alpha_bar_prev = alpha_bars_t[t_prev].item()
        else:
            alpha_bar_prev = 1.0  # t=0: no noise

        # DDIM update (deterministic)
        x = (np.sqrt(alpha_bar_prev) * x0_pred +
             np.sqrt(1.0 - alpha_bar_prev) * eps_pred)

    elapsed = time.time() - t0
    return x, elapsed


# Run DDPM and DDIM with different step counts
torch.manual_seed(42)
x_ddpm, t_ddpm = ddpm_sample(
    unet, T, betas_torch, alpha_bars_torch,
    img_shape=(2, 1, 28, 28), device=device, n_vis_steps=1
)

torch.manual_seed(42)
x_ddim_50, t_ddim_50 = ddim_sample(unet, T, alpha_bars_torch, (2, 1, 28, 28), device, 50)

torch.manual_seed(42)
x_ddim_10, t_ddim_10 = ddim_sample(unet, T, alpha_bars_torch, (2, 1, 28, 28), device, 10)

print(f"Sampling times (2 images):")
print(f"  DDPM-{T:4d} steps: {t_ddpm:.3f}s")
print(f"  DDIM-50 steps:  {t_ddim_50:.3f}s ({t_ddpm/(t_ddim_50+1e-6):.1f}x faster)")
print(f"  DDIM-10 steps:  {t_ddim_10:.3f}s ({t_ddpm/(t_ddim_10+1e-6):.1f}x faster)")

# Compare generated images visually
methods = [
    (f"DDPM-{T}", x_ddpm.cpu()),
    ("DDIM-50", x_ddim_50.cpu()),
    ("DDIM-10", x_ddim_10.cpu()),
]
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for col, (name, imgs) in enumerate(methods):
    for row in range(2):
        axes[row, col].imshow(imgs[row, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[row, col].set_title(f"{name} - sample {row}")
        axes[row, col].axis('off')
fig.suptitle("DDPM vs DDIM: Same Model, Different Sampling Schedules")
plt.tight_layout()
plt.savefig("/tmp/ddpm_vs_ddim.png", dpi=80, bbox_inches="tight")
plt.show()

# MSE between DDPM and DDIM samples (lower = more similar)
mse_50 = F.mse_loss(x_ddpm.cpu(), x_ddim_50.cpu()).item()
mse_10 = F.mse_loss(x_ddpm.cpu(), x_ddim_10.cpu()).item()
print(f"MSE vs DDPM reference: DDIM-50={mse_50:.4f}, DDIM-10={mse_10:.4f}")
"""))

    # Cell 11: RW3 header
    cells.append(make_cell("""\
## Real-World Example 3: Conditional Generation (Class-Conditional DDPM)

Train a class-conditional denoiser: the noise prediction network also receives
a class label embedding. At sampling time, specify which class to generate.

Implementation: add learned class embedding to time embedding before injection.
""", "markdown"))

    # Cell 12: RW3 code + comparison
    cells.append(make_cell("""\
class ConditionalUNet(SimpleUNet):
    \"\"\"Class-conditional U-Net: adds a class embedding to the time embedding.

    The class embedding is added to the time embedding before injection into
    each ResBlock, conditioning generation on a specific class.
    \"\"\"
    def __init__(self, n_classes=2, img_channels=1, base_ch=32, time_dim=128):
        super().__init__(img_channels=img_channels, base_ch=base_ch, time_dim=time_dim)
        # Class embedding: maps class index to time_dim vector
        self.class_emb = nn.Embedding(n_classes, time_dim)

    def forward(self, x, t, c):
        \"\"\"Forward pass with class conditioning.

        Args:
            x: (B, C, H, W) noisy image
            t: (B,) int timestep
            c: (B,) int class label

        Returns:
            predicted_noise: (B, C, H, W)
        \"\"\"
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        c_emb = self.class_emb(c)
        # Combine time and class embeddings
        combined = t_emb + c_emb  # additive conditioning

        # Encoder (same as parent but using combined embedding)
        e1 = self.enc1(x, combined)
        e2 = self.enc2(self.down1(e1), combined)
        e3 = self.enc3(self.down2(e2), combined)
        b = self.bot(self.down3(e3), combined)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1), combined)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), combined)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), combined)
        return self.out_conv(d1)


# Train conditional model
cond_unet = ConditionalUNet(n_classes=2, img_channels=1, base_ch=32, time_dim=128).to(device)
opt_cond = optim.Adam(cond_unet.parameters(), lr=1e-3)
n_params_cond = sum(p.numel() for p in cond_unet.parameters())
print(f"Conditional U-Net parameters: {n_params_cond:,}")

cond_unet.train()
cond_losses = []
for step in range(300):
    idx = torch.randperm(X_train.shape[0])[:batch_size]
    x0 = X_train[idx]
    y_batch = y_train[idx].to(device)
    t = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)
    alpha_bar_t = alpha_bars_torch[t].view(-1, 1, 1, 1)
    epsilon = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * epsilon
    eps_pred = cond_unet(x_t, t, y_batch)
    loss = F.mse_loss(eps_pred, epsilon)
    opt_cond.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(cond_unet.parameters(), max_norm=1.0)
    opt_cond.step()
    cond_losses.append(loss.item())

    if (step + 1) % 100 == 0:
        print(f"Step {step+1} | Cond Loss: {loss.item():.6f}")


@torch.no_grad()
def ddim_sample_conditional(model, T, alpha_bars_t, img_shape, device, class_label, n_steps=50):
    \"\"\"DDIM sampling with class conditioning.

    Args:
        model: ConditionalUNet
        class_label: int, which class to generate

    Returns:
        x_final: (B, C, H, W)
    \"\"\"
    B, C, H, W = img_shape
    x = torch.randn(B, C, H, W, device=device)
    c = torch.full((B,), class_label, dtype=torch.long, device=device)
    step_size = T // n_steps
    ddim_ts = list(range(0, T, step_size))[::-1]

    for i, t_idx in enumerate(ddim_ts):
        t_tensor = torch.full((B,), t_idx, dtype=torch.long, device=device)
        alpha_bar_t = alpha_bars_t[t_idx].item()
        eps_pred = model(x, t_tensor, c)
        x0_pred = (x - np.sqrt(1 - alpha_bar_t) * eps_pred) / (np.sqrt(alpha_bar_t) + 1e-8)
        x0_pred = x0_pred.clamp(-1, 1)
        alpha_bar_prev = alpha_bars_t[ddim_ts[i + 1]].item() if i + 1 < len(ddim_ts) else 1.0
        x = np.sqrt(alpha_bar_prev) * x0_pred + np.sqrt(1 - alpha_bar_prev) * eps_pred
    return x


# Generate class-conditional samples
cond_unet.eval()
x_class0 = ddim_sample_conditional(cond_unet, T, alpha_bars_torch, (4, 1, 28, 28), device, 0)
x_class1 = ddim_sample_conditional(cond_unet, T, alpha_bars_torch, (4, 1, 28, 28), device, 1)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for col in range(4):
    axes[0, col].imshow(x_class0[col, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[0, col].set_title(f"Class 0 (horiz grad)")
    axes[0, col].axis('off')
    axes[1, col].imshow(x_class1[col, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[1, col].set_title(f"Class 1 (vert grad)")
    axes[1, col].axis('off')
fig.suptitle("Conditional Generation: Class 0 (horizontal) vs Class 1 (vertical)")
plt.tight_layout()
plt.savefig("/tmp/conditional_generation.png", dpi=80, bbox_inches="tight")
plt.show()

# ====== Comparison Summary ======
print("\\n=== Diffusion Models: Method Comparison ===")
print(f"{'Method':20s} | {'Steps':6s} | {'Time(s)':8s} | {'Speedup':8s} | {'Deterministic'}")
print("-" * 68)
print(f"{'DDPM':20s} | {T:6d} | {t_ddpm:8.3f} | {'1x':8s} | No")
print(f"{'DDIM-50':20s} | {50:6d} | {t_ddim_50:8.3f} | {t_ddpm/(t_ddim_50+1e-6):7.1f}x | Yes")
print(f"{'DDIM-10':20s} | {10:6d} | {t_ddim_10:8.3f} | {t_ddpm/(t_ddim_10+1e-6):7.1f}x | Yes")
print(f"{'DDIM-50 + cond':20s} | {50:6d} | {'N/A':8s} | {'~20x':8s} | Yes")

# Final training loss plot
fig, ax = plt.subplots(figsize=(10, 4))
smooth = np.convolve(loss_history, np.ones(20) / 20, mode='valid')
ax.plot(smooth, label='Unconditional U-Net', color='steelblue')
smooth_c = np.convolve(cond_losses, np.ones(20) / 20, mode='valid')
ax.plot(smooth_c, label='Conditional U-Net', color='coral', linestyle='--')
ax.set_xlabel("Training Step")
ax.set_ylabel("MSE Loss (smoothed, window=20)")
ax.set_title("Unconditional vs Conditional U-Net Training")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/diffusion_training.png", dpi=80, bbox_inches="tight")
plt.show()
"""))

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    return nb


if __name__ == "__main__":
    os.makedirs("/home/sbisw/github/interviewprep-ml/cv/notebooks", exist_ok=True)
    nb = build_notebook()
    path = "/home/sbisw/github/interviewprep-ml/cv/notebooks/06-diffusion-models.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"Written: {path}")
    print(f"  Cells: {len(nb.cells)}")
    code_lines = sum(
        len("".join(c["source"]).split("\n"))
        for c in nb.cells if c["cell_type"] == "code"
    )
    print(f"  Code lines: {code_lines}")
