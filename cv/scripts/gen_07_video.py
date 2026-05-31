"""Generator for CV notebook 07: Video Understanding."""
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
# Video Understanding

## Learning Objectives
1. Implement optical flow estimation (Lucas-Kanade gradient-based) from scratch
2. Build and train a 3D CNN on synthetic video clips for action recognition
3. Implement two-stream fusion (spatial RGB + temporal difference features)
4. Compare 2D CNN + pooling vs 3D CNN vs temporal attention on a synthetic task
""", "markdown"))

    # Cell 2: Imports + seeds + device
    cells.append(make_cell("""\
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
"""))

    # Cell 3: Level 1 header
    cells.append(make_cell("""\
## Level 1: Optical Flow Estimation (Lucas-Kanade Approximation)

Optical flow estimates per-pixel motion vectors between two consecutive frames.
Lucas-Kanade assumes constant flow within a local window and solves:
    I_x * u + I_y * v + I_t = 0  (brightness constancy constraint)

for each pixel, where:
- I_x, I_y are spatial image gradients
- I_t is the temporal (frame-to-frame) gradient
- (u, v) is the flow vector
""", "markdown"))

    # Cell 4: Level 1 code
    cells.append(make_cell("""\
def compute_gradients(frame):
    \"\"\"Compute spatial gradients of an image using Sobel filters.

    Args:
        frame: (H, W) float numpy array

    Returns:
        Ix: (H, W) x-gradient
        Iy: (H, W) y-gradient
    \"\"\"
    # Sobel kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    ky = kx.T
    # Manual 2D convolution via numpy (no torchvision or scipy)
    H, W = frame.shape
    Ix = np.zeros_like(frame)
    Iy = np.zeros_like(frame)
    padded = np.pad(frame, 1, mode='reflect')
    for i in range(H):
        for j in range(W):
            patch = padded[i:i+3, j:j+3]
            Ix[i, j] = (patch * kx).sum()
            Iy[i, j] = (patch * ky).sum()
    return Ix, Iy


def lucas_kanade_flow(frame1, frame2, window_size=5):
    \"\"\"Estimate optical flow using Lucas-Kanade method.

    For each pixel, solve A^T A [u,v]^T = -A^T b using a local window.
    A is the matrix of [Ix, Iy] values in the window,
    b is the vector of -It values.

    Args:
        frame1: (H, W) float numpy array, first frame
        frame2: (H, W) float numpy array, second frame
        window_size: local window size for LK computation

    Returns:
        u: (H, W) horizontal flow component
        v: (H, W) vertical flow component
    \"\"\"
    H, W = frame1.shape
    w = window_size // 2

    # Temporal gradient: It = frame2 - frame1
    It = frame2 - frame1

    # Spatial gradients (average of both frames for better accuracy)
    Ix, Iy = compute_gradients((frame1 + frame2) / 2.0)

    u = np.zeros((H, W), dtype=np.float32)
    v = np.zeros((H, W), dtype=np.float32)

    Ix_pad = np.pad(Ix, w, mode='reflect')
    Iy_pad = np.pad(Iy, w, mode='reflect')
    It_pad = np.pad(It, w, mode='reflect')

    for i in range(H):
        for j in range(W):
            Ix_win = Ix_pad[i:i+window_size, j:j+window_size].flatten()
            Iy_win = Iy_pad[i:i+window_size, j:j+window_size].flatten()
            It_win = It_pad[i:i+window_size, j:j+window_size].flatten()

            A = np.column_stack([Ix_win, Iy_win])  # (window^2, 2)
            b = -It_win                              # (window^2,)
            AtA = A.T @ A  # (2, 2)
            Atb = A.T @ b  # (2,)

            # Solve AtA [u, v]^T = Atb
            det = AtA[0, 0] * AtA[1, 1] - AtA[0, 1] * AtA[1, 0]
            if abs(det) > 1e-6:  # check for invertibility
                flow = np.linalg.solve(AtA, Atb)
                u[i, j] = flow[0]
                v[i, j] = flow[1]

    return u, v


# Test with synthetic frames: rectangle moving right
def make_frame_with_rectangle(H=32, W=32, rect_x=8, rect_y=8, rect_size=8):
    \"\"\"Create a frame with a bright rectangle at specified position.\"\"\"
    frame = np.zeros((H, W), dtype=np.float32) + 0.1
    y2 = min(rect_y + rect_size, H)
    x2 = min(rect_x + rect_size, W)
    frame[rect_y:y2, rect_x:x2] = 1.0
    return frame


# Generate two frames: rectangle moves 3 pixels right
H, W = 32, 32
frame1 = make_frame_with_rectangle(H, W, rect_x=5, rect_y=10, rect_size=8)
frame2 = make_frame_with_rectangle(H, W, rect_x=8, rect_y=10, rect_size=8)

# Compute optical flow (use small image for speed)
u, v = lucas_kanade_flow(frame1, frame2, window_size=5)

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(frame1, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("Frame 1")
axes[1].imshow(frame2, cmap='gray', vmin=0, vmax=1)
axes[1].set_title("Frame 2")
axes[2].imshow(u, cmap='RdBu', vmin=-2, vmax=2)
axes[2].set_title("Flow u (horizontal)")
axes[3].imshow(v, cmap='RdBu', vmin=-2, vmax=2)
axes[3].set_title("Flow v (vertical)")
for ax in axes:
    ax.axis('off')
plt.suptitle("Lucas-Kanade Optical Flow: Rectangle Moving Right")
plt.tight_layout()
plt.savefig("/tmp/optical_flow.png", dpi=80, bbox_inches="tight")
plt.show()

# Verify: mean horizontal flow in rectangle region should be positive
rect_u = u[10:18, 5:15].mean()
print(f"Mean horizontal flow in rectangle region: {rect_u:.4f} (expected: ~3.0 pixels)")
print(f"Mean vertical flow in rectangle region: {v[10:18, 5:15].mean():.4f} (expected: ~0.0)")
"""))

    # Cell 5: Level 2 header
    cells.append(make_cell("""\
## Level 2: 3D CNN for Action Recognition on Synthetic Videos

Input: [B, C, T, H, W] video tensor where T=8 frames.
Task: 2-class action recognition (object moving left vs moving right).
Architecture: 3D convolution blocks with temporal and spatial downsampling.
""", "markdown"))

    # Cell 6: Level 2 code
    cells.append(make_cell("""\
def make_action_video(label, T=8, H=24, W=24, n_samples=1):
    \"\"\"Generate synthetic action videos.

    Class 0: bright rectangle moves left to right across frames
    Class 1: bright rectangle moves right to left across frames

    Returns:
        video: (n_samples, 1, T, H, W) float tensor in [0, 1]
    \"\"\"
    videos = []
    for _ in range(n_samples):
        video = np.zeros((1, T, H, W), dtype=np.float32)
        x_start = 2 if label == 0 else (W - 10)  # start position
        direction = 1 if label == 0 else -1       # +1=right, -1=left
        for t in range(T):
            x_pos = x_start + direction * t * 2
            x_pos = max(0, min(W - 6, x_pos))
            video[0, t, 8:16, x_pos:x_pos+6] = 1.0
        # Add mild noise for realism
        video += np.random.randn(*video.shape).astype(np.float32) * 0.05
        video = np.clip(video, 0, 1)
        videos.append(video)
    return torch.tensor(np.array(videos), dtype=torch.float32)


# Generate dataset
n_per_class = 128
T_frames, H_vid, W_vid = 8, 24, 24
X_vid = torch.cat([
    make_action_video(0, T=T_frames, H=H_vid, W=W_vid, n_samples=n_per_class),
    make_action_video(1, T=T_frames, H=H_vid, W=W_vid, n_samples=n_per_class),
])
y_vid = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long()
print(f"Video dataset shape: {X_vid.shape} | Labels: {y_vid.shape}")
print(f"  Class 0 (left->right): {(y_vid==0).sum()} samples")
print(f"  Class 1 (right->left): {(y_vid==1).sum()} samples")


# 3D CNN model
class ThreeDCNN(nn.Module):
    \"\"\"3D CNN for video action recognition.

    Input: (B, C, T, H, W) video tensor
    Output: (B, n_classes) logits

    Architecture:
        Conv3d block 1: learn short-term spatiotemporal features
        Conv3d block 2: learn longer-range motion patterns
        Global average pool over (T, H, W)
        Fully connected classifier
    \"\"\"
    def __init__(self, in_channels=1, n_classes=2, base_ch=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_ch, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(),
            # Temporal stride=1, spatial stride=2
            nn.Conv3d(base_ch, base_ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(),
            nn.Conv3d(base_ch * 2, base_ch * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # (B, C, 1, 1, 1)
        self.classifier = nn.Linear(base_ch * 2, n_classes)

    def forward(self, x):
        \"\"\"Forward pass.

        Args:
            x: (B, C, T, H, W) video tensor

        Returns:
            logits: (B, n_classes)
        \"\"\"
        h = self.conv1(x)   # (B, base_ch, T, H/2, W/2)
        h = self.conv2(h)   # (B, 2*base_ch, T, H/4, W/4)
        h = self.global_pool(h).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 2*base_ch)
        return self.classifier(h)


model_3d = ThreeDCNN(in_channels=1, n_classes=2, base_ch=16).to(device)
n_params_3d = sum(p.numel() for p in model_3d.parameters())
print(f"3D CNN parameters: {n_params_3d:,}")

# Verify input/output shapes
dummy_vid = torch.randn(4, 1, T_frames, H_vid, W_vid).to(device)
dummy_out = model_3d(dummy_vid)
print(f"Input shape: {dummy_vid.shape} -> Output shape: {dummy_out.shape}")

# Train
opt_3d = optim.Adam(model_3d.parameters(), lr=1e-3)
X_vid_dev = X_vid.to(device)
y_vid_dev = y_vid.to(device)

batch_size_v = 32
n_steps_v = 200
losses_3d = []
model_3d.train()

for step in range(n_steps_v):
    idx = torch.randperm(X_vid_dev.shape[0])[:batch_size_v]
    logits = model_3d(X_vid_dev[idx])
    loss = F.cross_entropy(logits, y_vid_dev[idx])
    opt_3d.zero_grad()
    loss.backward()
    opt_3d.step()
    losses_3d.append(loss.item())

model_3d.eval()
with torch.no_grad():
    logits_all = model_3d(X_vid_dev)
    acc_3d = (logits_all.argmax(1) == y_vid_dev).float().mean().item()
print(f"3D CNN train accuracy: {acc_3d:.4f}")
print(f"Loss: {losses_3d[0]:.4f} -> {losses_3d[-1]:.4f}")
"""))

    # Cell 7: RW1 header
    cells.append(make_cell("""\
## Real-World Example 1: Two-Stream Fusion

Two-stream networks process spatial (RGB) and temporal (optical flow) streams separately,
then fuse predictions. Here we use frame differences as a proxy for optical flow
(faster to compute, similar signal).

Spatial stream: individual frame features
Temporal stream: frame-difference features (dI = I_t+1 - I_t)
Fusion: average softmax predictions from both streams
""", "markdown"))

    # Cell 8: RW1 code
    cells.append(make_cell("""\
class SpatialCNN(nn.Module):
    \"\"\"Spatial stream: 2D CNN applied per-frame, then mean pool over time.

    Processes each frame independently and aggregates temporal information
    by averaging features, not pixels.
    \"\"\"
    def __init__(self, in_channels=1, n_classes=2, base_ch=16):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(base_ch * 2 * 4 * 4, n_classes)

    def forward(self, x):
        \"\"\"Args: x: (B, C, T, H, W). Returns: (B, n_classes).\"\"\"
        B, C, T, H, W = x.shape
        # Reshape to process all frames simultaneously: (B*T, C, H, W)
        x_frames = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        features = self.frame_encoder(x_frames)         # (B*T, ch, 4, 4)
        features = features.view(B, T, -1)              # (B, T, ch*4*4)
        # Mean pool over time
        pooled = features.mean(dim=1)                   # (B, ch*4*4)
        return self.classifier(pooled)


class TemporalCNN(nn.Module):
    \"\"\"Temporal stream: 2D CNN on frame differences (motion proxy).

    Frame difference = I_{t+1} - I_t is a simple optical flow proxy.
    Stack T-1 difference frames as channels.
    \"\"\"
    def __init__(self, T=8, n_classes=2, base_ch=16):
        super().__init__()
        self.T = T
        in_ch = T - 1  # number of difference frames
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(base_ch * 2 * 4 * 4, n_classes)

    def forward(self, x):
        \"\"\"Args: x: (B, C, T, H, W). Returns: (B, n_classes).\"\"\"
        # Compute frame differences: shape (B, T-1, H, W)
        diffs = x[:, 0, 1:, :, :] - x[:, 0, :-1, :, :]  # (B, T-1, H, W)
        features = self.encoder(diffs)                     # (B, ch, 4, 4)
        features = features.view(features.shape[0], -1)
        return self.classifier(features)


# Train both streams
spatial_model = SpatialCNN(in_channels=1, n_classes=2, base_ch=16).to(device)
temporal_model = TemporalCNN(T=T_frames, n_classes=2, base_ch=16).to(device)

results_streams = {}
for name, model_s in [("Spatial", spatial_model), ("Temporal", temporal_model)]:
    opt_s = optim.Adam(model_s.parameters(), lr=1e-3)
    model_s.train()
    stream_losses = []
    for step in range(200):
        idx = torch.randperm(X_vid_dev.shape[0])[:batch_size_v]
        logits_s = model_s(X_vid_dev[idx])
        loss_s = F.cross_entropy(logits_s, y_vid_dev[idx])
        opt_s.zero_grad()
        loss_s.backward()
        opt_s.step()
        stream_losses.append(loss_s.item())

    model_s.eval()
    with torch.no_grad():
        logits_eval = model_s(X_vid_dev)
        acc_s = (logits_eval.argmax(1) == y_vid_dev).float().mean().item()
    results_streams[name] = {'losses': stream_losses, 'accuracy': acc_s}
    print(f"{name:10s} stream | acc={acc_s:.4f}")

# Two-stream fusion: average softmax scores
spatial_model.eval()
temporal_model.eval()
with torch.no_grad():
    logits_sp = spatial_model(X_vid_dev)
    logits_tp = temporal_model(X_vid_dev)
    # Late fusion: average class probabilities
    probs_fusion = (F.softmax(logits_sp, dim=-1) + F.softmax(logits_tp, dim=-1)) / 2.0
    preds_fusion = probs_fusion.argmax(dim=1)
    acc_fusion = (preds_fusion == y_vid_dev).float().mean().item()

print(f"{'Two-Stream':10s} fusion  | acc={acc_fusion:.4f}")
results_streams['Two-Stream'] = {'accuracy': acc_fusion}

# Plot results
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(list(results_streams.keys()),
              [r['accuracy'] for r in results_streams.values()],
              color=['steelblue', 'coral', 'green'])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title("Spatial vs Temporal vs Two-Stream Fusion")
for bar, (name, r) in zip(bars, results_streams.items()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{r['accuracy']:.3f}", ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("/tmp/two_stream_comparison.png", dpi=80, bbox_inches="tight")
plt.show()
"""))

    # Cell 9: RW2 header
    cells.append(make_cell("""\
## Real-World Example 2: Frame Sampling Strategies

How you sample frames from a video clip affects what information the model sees.
- Uniform sampling: evenly spaced frames across the clip
- Dense sampling: consecutive frames (captures fast motion)
- Sparse sampling: widely spaced frames (captures scene changes)

Compare: how sampling rate affects 3D CNN accuracy and loss on our synthetic videos.
""", "markdown"))

    # Cell 10: RW2 code
    cells.append(make_cell("""\
def sample_frames(video, strategy='uniform', n_frames=8):
    \"\"\"Sample T frames from a video using different strategies.

    Args:
        video: (C, T_full, H, W) tensor
        strategy: 'uniform', 'dense_start', 'dense_middle', 'sparse'
        n_frames: number of frames to select

    Returns:
        sampled: (C, n_frames, H, W) tensor
    \"\"\"
    T_full = video.shape[1]

    if strategy == 'uniform':
        # Evenly spaced indices
        indices = np.linspace(0, T_full - 1, n_frames, dtype=int)
    elif strategy == 'dense_start':
        # First n_frames consecutive frames
        indices = np.arange(min(n_frames, T_full))
    elif strategy == 'dense_middle':
        # n_frames frames from the center of the clip
        mid = T_full // 2
        start = max(0, mid - n_frames // 2)
        indices = np.arange(start, min(start + n_frames, T_full))
        # Pad if necessary
        while len(indices) < n_frames:
            indices = np.append(indices, indices[-1])
        indices = indices[:n_frames]
    elif strategy == 'sparse':
        # Every other frame
        step = max(1, T_full // n_frames)
        indices = np.arange(0, T_full, step)[:n_frames]
        while len(indices) < n_frames:
            indices = np.append(indices, indices[-1])
        indices = indices[:n_frames]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return video[:, indices, :, :]


# Generate longer videos (T=16 frames) for sampling comparison
T_long = 16
X_long = torch.cat([
    make_action_video(0, T=T_long, H=H_vid, W=W_vid, n_samples=128),
    make_action_video(1, T=T_long, H=H_vid, W=W_vid, n_samples=128),
])
y_long = torch.cat([torch.zeros(128), torch.ones(128)]).long()

sampling_results = {}
strategies = ['uniform', 'dense_start', 'dense_middle', 'sparse']

for strategy in strategies:
    # Sample T=8 frames from T=16 videos using this strategy
    X_sampled = torch.stack([
        sample_frames(vid, strategy=strategy, n_frames=8)
        for vid in X_long
    ])

    X_s_dev = X_sampled.to(device)
    y_s_dev = y_long.to(device)

    # Train fresh 3D CNN on sampled data
    torch.manual_seed(42)
    model_s = ThreeDCNN(in_channels=1, n_classes=2, base_ch=16).to(device)
    opt_s = optim.Adam(model_s.parameters(), lr=1e-3)
    final_losses_s = []

    model_s.train()
    for step in range(200):
        idx = torch.randperm(X_s_dev.shape[0])[:batch_size_v]
        logits_s = model_s(X_s_dev[idx])
        loss_s = F.cross_entropy(logits_s, y_s_dev[idx])
        opt_s.zero_grad()
        loss_s.backward()
        opt_s.step()
        final_losses_s.append(loss_s.item())

    model_s.eval()
    with torch.no_grad():
        logits_eval = model_s(X_s_dev)
        acc_s = (logits_eval.argmax(1) == y_s_dev).float().mean().item()
    sampling_results[strategy] = {
        'accuracy': acc_s,
        'final_loss': final_losses_s[-1]
    }
    print(f"Strategy {strategy:15s} | acc={acc_s:.4f} | loss={final_losses_s[-1]:.4f}")

# Visualize sampling strategy examples
fig, axes = plt.subplots(4, 8, figsize=(14, 8))
sample_vid = X_long[0]  # first video (class 0)
for row, strategy in enumerate(strategies):
    sampled = sample_frames(sample_vid, strategy=strategy, n_frames=8)
    for col in range(8):
        axes[row, col].imshow(sampled[0, col].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(strategy, rotation=45, ha='right', fontsize=9)
fig.suptitle("Frame Sampling Strategies (8 from 16 frames)")
plt.tight_layout()
plt.savefig("/tmp/frame_sampling.png", dpi=80, bbox_inches="tight")
plt.show()

print("\\nSampling Strategy Comparison:")
print(f"{'Strategy':15s} | {'Accuracy':>8s} | {'Final Loss':>10s}")
print("-" * 38)
for strat, res in sampling_results.items():
    print(f"{strat:15s} | {res['accuracy']:>8.4f} | {res['final_loss']:>10.4f}")
"""))

    # Cell 11: RW3 header
    cells.append(make_cell("""\
## Real-World Example 3: Temporal Attention vs Temporal Pooling

Compare two temporal aggregation strategies after per-frame encoding:
1. Mean pooling: average frame features (no temporal reasoning)
2. Temporal self-attention: each frame can attend to all other frames
   (learns which frames to emphasize and relationships between them)
""", "markdown"))

    # Cell 12: RW3 code + comparison
    cells.append(make_cell("""\
class TemporalAttentionModel(nn.Module):
    \"\"\"2D frame encoder with temporal self-attention for aggregation.

    Architecture:
        1. Per-frame 2D CNN encoder produces feature vectors
        2. Learnable temporal positional encodings added
        3. Multi-head self-attention across T frames
        4. Linear classifier on attended [CLS] token or mean
    \"\"\"
    def __init__(self, in_channels=1, n_classes=2, feat_dim=64, T=8, n_heads=4):
        super().__init__()
        self.T = T
        self.feat_dim = feat_dim
        # Per-frame encoder
        self.frame_enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(32, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # Temporal positional encoding (learned)
        self.temporal_pos = nn.Embedding(T, feat_dim)
        # Self-attention over time
        self.attn = nn.MultiheadAttention(feat_dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        \"\"\"Args: x: (B, C, T, H, W). Returns: (B, n_classes).\"\"\"
        B, C, T, H, W = x.shape
        # Encode each frame: (B*T, C, H, W) -> (B*T, feat_dim)
        x_frames = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        feats = self.frame_enc(x_frames).squeeze(-1).squeeze(-1)  # (B*T, feat_dim)
        feats = feats.view(B, T, self.feat_dim)                   # (B, T, feat_dim)

        # Add temporal positional encoding
        t_idx = torch.arange(T, device=x.device)
        feats = feats + self.temporal_pos(t_idx).unsqueeze(0)    # (B, T, feat_dim)

        # Temporal self-attention: each frame attends to all others
        attended, attn_weights = self.attn(feats, feats, feats)  # (B, T, feat_dim)
        attended = self.norm(attended + feats)                    # residual connection

        # Aggregate: mean over time dimension
        pooled = attended.mean(dim=1)                             # (B, feat_dim)
        return self.classifier(pooled)


class MeanPoolModel(nn.Module):
    \"\"\"2D frame encoder with naive mean pooling (no temporal reasoning).\"\"\"
    def __init__(self, in_channels=1, n_classes=2, feat_dim=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.frame_enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(32, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_frames = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        feats = self.frame_enc(x_frames).squeeze(-1).squeeze(-1)  # (B*T, feat_dim)
        feats = feats.view(B, T, self.feat_dim).mean(dim=1)       # (B, feat_dim)
        return self.classifier(feats)


X_dev = X_vid_dev
y_dev = y_vid_dev

comparison_results = {}
models_to_compare = {
    '2D CNN + MeanPool': MeanPoolModel(in_channels=1, n_classes=2, feat_dim=64).to(device),
    '3D CNN': model_3d,  # already trained above
    '2D CNN + TempAttn': TemporalAttentionModel(in_channels=1, n_classes=2, feat_dim=64,
                                                 T=T_frames, n_heads=4).to(device),
}

for name, model_c in models_to_compare.items():
    n_params_c = sum(p.numel() for p in model_c.parameters())
    if name == '3D CNN':
        # Already trained, just evaluate
        model_c.eval()
        with torch.no_grad():
            preds_c = model_c(X_dev).argmax(1)
        acc_c = (preds_c == y_dev).float().mean().item()
        comparison_results[name] = {'accuracy': acc_c, 'params': n_params_c}
        print(f"{name:25s} | acc={acc_c:.4f} | params={n_params_c:,}")
        continue

    opt_c = optim.Adam(model_c.parameters(), lr=1e-3)
    model_c.train()
    for step in range(200):
        idx = torch.randperm(X_dev.shape[0])[:batch_size_v]
        logits_c = model_c(X_dev[idx])
        loss_c = F.cross_entropy(logits_c, y_dev[idx])
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()

    model_c.eval()
    with torch.no_grad():
        preds_c = model_c(X_dev).argmax(1)
    acc_c = (preds_c == y_dev).float().mean().item()
    comparison_results[name] = {'accuracy': acc_c, 'params': n_params_c}
    print(f"{name:25s} | acc={acc_c:.4f} | params={n_params_c:,}")

# Final comparison chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
names = list(comparison_results.keys())
accs = [comparison_results[n]['accuracy'] for n in names]
params = [comparison_results[n]['params'] for n in names]

axes[0].bar(names, accs, color=['steelblue', 'coral', 'green'])
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Action Recognition Accuracy by Architecture")
for i, (n, a) in enumerate(zip(names, accs)):
    axes[0].text(i, a + 0.02, f"{a:.3f}", ha='center', fontsize=10)
axes[0].tick_params(axis='x', labelrotation=15)

axes[1].bar(names, [p/1000 for p in params], color=['steelblue', 'coral', 'green'])
axes[1].set_ylabel("Parameters (K)")
axes[1].set_title("Model Parameters by Architecture")
for i, (n, p) in enumerate(zip(names, params)):
    axes[1].text(i, p/1000 + 0.5, f"{p/1000:.1f}K", ha='center', fontsize=10)
axes[1].tick_params(axis='x', labelrotation=15)

plt.suptitle("Video Understanding: Architecture Comparison")
plt.tight_layout()
plt.savefig("/tmp/video_comparison.png", dpi=80, bbox_inches="tight")
plt.show()

print("\\n=== Final Summary ===")
print(f"{'Architecture':25s} | {'Accuracy':>8s} | {'Params':>10s}")
print("-" * 48)
for name, res in comparison_results.items():
    print(f"{name:25s} | {res['accuracy']:>8.4f} | {res['params']:>10,}")
"""))

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    return nb


if __name__ == "__main__":
    os.makedirs("/home/sbisw/github/interviewprep-ml/cv/notebooks", exist_ok=True)
    nb = build_notebook()
    path = "/home/sbisw/github/interviewprep-ml/cv/notebooks/07-video-understanding.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"Written: {path}")
    print(f"  Cells: {len(nb.cells)}")
    code_lines = sum(
        len("".join(c["source"]).split("\n"))
        for c in nb.cells if c["cell_type"] == "code"
    )
    print(f"  Code lines: {code_lines}")
