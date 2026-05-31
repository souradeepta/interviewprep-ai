"""Generator for CV notebook 05: Contrastive Learning in Vision."""
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
# Contrastive Learning in Vision

## Learning Objectives
1. Implement NT-Xent (normalized temperature-scaled cross-entropy) loss from scratch
2. Train a SimCLR-style encoder on synthetic 2D data and visualize embedding space
3. Build a CLIP-style zero-shot classifier using contrastive embeddings
4. Analyze the effect of temperature tau and hard negative mining on representation quality
""", "markdown"))

    # Cell 2: Imports + seeds + device
    cells.append(make_cell("""\
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
## Level 1: NT-Xent Loss from Scratch (NumPy)

NT-Xent is the contrastive loss used in SimCLR.
For a batch of N images with 2 augmentations each (2N embeddings total):
- Positive pair for z_i is z_j (the other augmentation of the same image)
- All other 2(N-1) embeddings are negatives

Loss: L_i = -log [ exp(sim(z_i, z_j) / tau) / sum_{k != i} exp(sim(z_i, z_k) / tau) ]
""", "markdown"))

    # Cell 4: Level 1 code
    cells.append(make_cell("""\
def cosine_similarity_matrix(Z):
    \"\"\"Compute pairwise cosine similarity matrix.

    Args:
        Z: np.ndarray of shape (2N, D) - L2-normalized embeddings

    Returns:
        sim_matrix: np.ndarray of shape (2N, 2N)
    \"\"\"
    # Z is assumed to be L2-normalized, so dot product == cosine similarity
    return Z @ Z.T  # (2N, 2N)


def nt_xent_loss_numpy(Z, tau=0.07):
    \"\"\"NT-Xent loss for SimCLR.

    Args:
        Z: np.ndarray of shape (2N, D), L2-normalized embeddings.
           First N rows = augmentation 1, last N rows = augmentation 2.
        tau: float, temperature scaling factor

    Returns:
        loss: float scalar
    \"\"\"
    N2 = Z.shape[0]  # = 2N
    N = N2 // 2

    # L2-normalize (ensure unit sphere)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

    # Full similarity matrix (2N x 2N)
    sim = cosine_similarity_matrix(Z) / tau  # scale by temperature

    # Build mask: True where (i, j) is a positive pair
    # Positive pairs: (i, i+N) and (i+N, i) for i in [0, N)
    labels = np.zeros(N2, dtype=int)
    labels[:N] = np.arange(N, N2)   # row i -> positive at row i+N
    labels[N:] = np.arange(0, N)    # row i+N -> positive at row i

    # Mask diagonal (self-similarity) - set to -inf to exclude from softmax
    mask_diag = np.eye(N2, dtype=bool)
    sim[mask_diag] = -1e9

    # For each row i, positive index is labels[i]
    # Numerator: sim[i, labels[i]]
    # Denominator: sum over all j != i of exp(sim[i,j])
    losses = []
    for i in range(N2):
        numerator = sim[i, labels[i]]
        # log-sum-exp for numerical stability
        log_denominator = np.log(np.sum(np.exp(sim[i])))
        losses.append(-(numerator - log_denominator))

    return float(np.mean(losses))


# Demo: 8 image pairs, 16-dimensional embeddings
N = 8
D = 16

# Initially random embeddings - loss should be high
Z_random = np.random.randn(2 * N, D)
loss_random = nt_xent_loss_numpy(Z_random, tau=0.07)
print(f"[Random embeddings]   NT-Xent loss: {loss_random:.4f}")

# Perfectly aligned pairs - loss should approach 0
Z_aligned = np.random.randn(N, D)
Z_aligned = np.vstack([Z_aligned, Z_aligned])  # exact copies = perfect alignment
# Add tiny noise to avoid numerical issues with identical vectors
Z_aligned[N:] += np.random.randn(N, D) * 1e-4
loss_aligned = nt_xent_loss_numpy(Z_aligned, tau=0.07)
print(f"[Aligned embeddings]  NT-Xent loss: {loss_aligned:.4f}")

# Show that loss decreases as embeddings become more similar
print("\\nLoss vs alignment degree:")
for noise_scale in [2.0, 1.0, 0.5, 0.1, 0.01]:
    Z_base = np.random.randn(N, D)
    Z_aug = Z_base + np.random.randn(N, D) * noise_scale
    Z_combined = np.vstack([Z_base, Z_aug])
    loss = nt_xent_loss_numpy(Z_combined, tau=0.07)
    print(f"  Noise scale {noise_scale:.2f} -> loss {loss:.4f}")
"""))

    # Cell 5: Level 2 header
    cells.append(make_cell("""\
## Level 2: SimCLR on Synthetic 2D Point Cloud (PyTorch)

Train a SimCLR-style encoder on 2D points grouped into 4 clusters.
Augmentations: Gaussian noise + random rotation (2D).
Show embedding space before vs after training.
""", "markdown"))

    # Cell 6: Level 2 code
    cells.append(make_cell("""\
# ---- Data generation ----
def generate_2d_clusters(n_per_class=50, n_classes=4, noise=0.3):
    \"\"\"Generate 2D Gaussian clusters for contrastive learning demo.

    Args:
        n_per_class: points per class
        n_classes: number of clusters
        noise: within-cluster spread

    Returns:
        X: (N, 2) float32 tensor
        y: (N,) long tensor of class labels
    \"\"\"
    centers = [
        [2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]
    ]
    Xs, ys = [], []
    for cls_idx in range(n_classes):
        cx, cy = centers[cls_idx % len(centers)]
        pts = np.random.randn(n_per_class, 2) * noise + [cx, cy]
        Xs.append(pts)
        ys.extend([cls_idx] * n_per_class)
    X = torch.tensor(np.vstack(Xs), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


# ---- Augmentations ----
def augment_2d(x, noise_std=0.2):
    \"\"\"Augment 2D points: Gaussian noise + random 2D rotation.

    Args:
        x: (N, 2) tensor

    Returns:
        augmented: (N, 2) tensor
    \"\"\"
    noisy = x + torch.randn_like(x) * noise_std
    # Random rotation angle in [-30, 30] degrees
    angle = (torch.rand(1).item() - 0.5) * 60 * (np.pi / 180)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
    rotated = noisy @ R.T
    return rotated


# ---- Encoder ----
class MLPEncoder(nn.Module):
    \"\"\"Small MLP encoder for 2D contrastive learning.

    Architecture: 2 -> 32 -> 64 -> embedding_dim
    Followed by L2 normalization onto the unit hypersphere.
    \"\"\"
    def __init__(self, input_dim=2, hidden_dim=32, embedding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # Projection head (discarded after training)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x, project=True):
        h = self.encoder(x)          # encoder output (use for downstream tasks)
        if project:
            z = self.projector(h)    # projection (used during contrastive training)
            z = F.normalize(z, dim=-1)  # unit sphere normalization
            return z
        return F.normalize(h, dim=-1)


# ---- NT-Xent loss (PyTorch) ----
def nt_xent_loss_torch(z1, z2, tau=0.07):
    \"\"\"Compute NT-Xent contrastive loss.

    Args:
        z1: (N, D) L2-normalized embeddings of augmentation 1
        z2: (N, D) L2-normalized embeddings of augmentation 2
        tau: temperature

    Returns:
        loss: scalar
    \"\"\"
    N = z1.shape[0]
    # Concatenate: first N = z1, last N = z2
    Z = torch.cat([z1, z2], dim=0)  # (2N, D)
    # Similarity matrix
    sim = (Z @ Z.T) / tau  # (2N, 2N)
    # Mask diagonal (self-similarity)
    mask = torch.eye(2 * N, device=Z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    # Labels: for row i (in z1), positive is i+N (in z2), and vice versa
    labels = torch.cat([
        torch.arange(N, 2 * N, device=Z.device),
        torch.arange(0, N, device=Z.device)
    ])
    loss = F.cross_entropy(sim, labels)
    return loss


# ---- Training ----
X, y = generate_2d_clusters(n_per_class=64, n_classes=4)
X = X.to(device)
y = y.to(device)

model = MLPEncoder(input_dim=2, hidden_dim=32, embedding_dim=16).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Capture embeddings before training
model.eval()
with torch.no_grad():
    emb_before = model(X.to(device), project=False).cpu().numpy()

# Train for 100 steps
batch_size = 64
losses = []
model.train()
for step in range(100):
    # Sample a batch
    idx = torch.randperm(X.shape[0])[:batch_size]
    x_batch = X[idx]
    # Two augmented views
    x1 = augment_2d(x_batch)
    x2 = augment_2d(x_batch)
    z1 = model(x1)
    z2 = model(x2)
    loss = nt_xent_loss_torch(z1, z2, tau=0.07)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (step + 1) % 20 == 0:
        print(f"Step {step+1:3d} | Loss: {loss.item():.4f}")

# Capture embeddings after training
model.eval()
with torch.no_grad():
    emb_after = model(X.to(device), project=False).cpu().numpy()

# ---- Visualization ----
y_np = y.cpu().numpy()
pca = PCA(n_components=2)
emb_before_2d = pca.fit_transform(emb_before)
emb_after_2d = pca.fit_transform(emb_after)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['blue', 'red', 'green', 'orange']
class_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
for cls in range(4):
    mask_cls = y_np == cls
    axes[0].scatter(emb_before_2d[mask_cls, 0], emb_before_2d[mask_cls, 1],
                    c=colors[cls], label=class_labels[cls], alpha=0.6, s=20)
    axes[1].scatter(emb_after_2d[mask_cls, 0], emb_after_2d[mask_cls, 1],
                    c=colors[cls], label=class_labels[cls], alpha=0.6, s=20)

axes[0].set_title("Embedding Space: Before SimCLR Training")
axes[1].set_title("Embedding Space: After 100 Steps SimCLR")
for ax in axes:
    ax.legend(fontsize=8)
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
plt.tight_layout()
plt.savefig("/tmp/simclr_embeddings.png", dpi=80, bbox_inches='tight')
plt.show()
print("Saved embedding plot to /tmp/simclr_embeddings.png")
print(f"Final loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")
"""))

    # Cell 7: RW1 header
    cells.append(make_cell("""\
## Real-World Example 1: CLIP-Style Zero-Shot Classification

CLIP learns a shared embedding space for images and text.
At inference, class names are encoded as text features; test images are
classified by finding the nearest class centroid — no fine-tuning needed.

Here we simulate this with:
- Image encoder: the trained SimCLR encoder above
- Text encoder: random projection (proxy for a real text encoder)
- Classification: cosine similarity to class prototype embeddings
""", "markdown"))

    # Cell 8: RW1 code
    cells.append(make_cell("""\
# Simulate CLIP-style zero-shot classification

def compute_class_prototypes(embeddings, labels, n_classes):
    \"\"\"Compute mean embedding for each class (text prototype proxy).

    Args:
        embeddings: (N, D) numpy array
        labels: (N,) numpy int array
        n_classes: int

    Returns:
        prototypes: (n_classes, D) numpy array, L2-normalized
    \"\"\"
    D = embeddings.shape[1]
    prototypes = np.zeros((n_classes, D))
    for cls in range(n_classes):
        mask_cls = labels == cls
        if mask_cls.sum() > 0:
            prototypes[cls] = embeddings[mask_cls].mean(axis=0)
    # L2-normalize prototypes
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
    return prototypes / norms


def zero_shot_classify(query_embeddings, prototypes):
    \"\"\"Classify queries by nearest prototype (cosine similarity).

    Args:
        query_embeddings: (M, D) numpy array, L2-normalized
        prototypes: (C, D) numpy array, L2-normalized

    Returns:
        predictions: (M,) int array of predicted class indices
        confidences: (M,) float array of max similarity scores
    \"\"\"
    # Normalize queries
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8
    queries_norm = query_embeddings / norms
    # Similarity matrix (M, C)
    sim = queries_norm @ prototypes.T
    predictions = sim.argmax(axis=1)
    confidences = sim.max(axis=1)
    return predictions, confidences


# Generate train (prototype) and test splits from our 2D cluster dataset
X_np = X.cpu().numpy()
y_np_all = y.cpu().numpy()
n_classes = 4

# Split: 75% train (prototypes), 25% test
n_total = X_np.shape[0]
n_train = int(0.75 * n_total)
idx_perm = np.random.permutation(n_total)
train_idx = idx_perm[:n_train]
test_idx = idx_perm[n_train:]

X_train_pt = torch.tensor(X_np[train_idx], dtype=torch.float32).to(device)
X_test_pt = torch.tensor(X_np[test_idx], dtype=torch.float32).to(device)
y_train = y_np_all[train_idx]
y_test = y_np_all[test_idx]

# Get embeddings using trained model (encoder only, no projection head)
model.eval()
with torch.no_grad():
    train_emb = model(X_train_pt, project=False).cpu().numpy()
    test_emb = model(X_test_pt, project=False).cpu().numpy()

# Build class prototypes from training embeddings
prototypes = compute_class_prototypes(train_emb, y_train, n_classes)
print(f"Prototype shape: {prototypes.shape}")

# Classify test set
preds, confs = zero_shot_classify(test_emb, prototypes)
accuracy = (preds == y_test).mean()
print(f"Zero-shot classification accuracy: {accuracy:.4f} ({int(accuracy*len(y_test))}/{len(y_test)} correct)")

# Breakdown by class
print("\nPer-class accuracy:")
for cls in range(n_classes):
    mask_cls = y_test == cls
    if mask_cls.sum() > 0:
        acc_cls = (preds[mask_cls] == y_test[mask_cls]).mean()
        print(f"  Class {cls}: {acc_cls:.4f} ({mask_cls.sum()} samples, "
              f"avg confidence {confs[mask_cls].mean():.3f})")

# Visualize: test points colored by predicted vs true class
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
pca2 = PCA(n_components=2)
test_emb_2d = pca2.fit_transform(test_emb)

for cls in range(n_classes):
    mask_true = y_test == cls
    mask_pred = preds == cls
    axes[0].scatter(test_emb_2d[mask_true, 0], test_emb_2d[mask_true, 1],
                    c=colors[cls], label=f"True {cls}", alpha=0.7, s=25)
    axes[1].scatter(test_emb_2d[mask_pred, 0], test_emb_2d[mask_pred, 1],
                    c=colors[cls], label=f"Pred {cls}", alpha=0.7, s=25, marker='x')

axes[0].set_title("True Classes")
axes[1].set_title("Predicted Classes (zero-shot)")
for ax in axes:
    ax.legend(fontsize=8)
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
plt.tight_layout()
plt.savefig("/tmp/zeroshot_classify.png", dpi=80, bbox_inches='tight')
plt.show()
"""))

    # Cell 9: RW2 header
    cells.append(make_cell("""\
## Real-World Example 2: Hard Negative Mining

Random negatives are easy once the model has learned basic structure.
Hard negatives (embeddings that are similar but from different classes)
provide stronger gradients and accelerate learning of fine-grained distinctions.

Measure: how quickly accuracy improves with random vs hard negatives.
""", "markdown"))

    # Cell 10: RW2 code
    cells.append(make_cell("""\
def find_hard_negatives(Z, labels, topk=5):
    \"\"\"Find hard negatives: most similar embeddings from different classes.

    Args:
        Z: (N, D) L2-normalized embeddings
        labels: (N,) int array
        topk: number of hard negatives to find per anchor

    Returns:
        hard_neg_indices: (N, topk) array of hard negative indices
    \"\"\"
    # Pairwise similarity
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    sim = Z_norm @ Z_norm.T  # (N, N)

    # For each anchor, find most similar embeddings from different classes
    N = Z.shape[0]
    hard_neg_indices = np.zeros((N, topk), dtype=int)
    for i in range(N):
        same_class_mask = (labels == labels[i])
        sim_row = sim[i].copy()
        sim_row[same_class_mask] = -1e9  # mask same-class
        sim_row[i] = -1e9               # mask self
        hard_neg_indices[i] = np.argsort(sim_row)[-topk:][::-1]
    return hard_neg_indices


def contrastive_loss_with_mining(z1, z2, labels, mode='random', tau=0.07):
    \"\"\"NT-Xent with optional hard negative mining.

    Args:
        z1: (N, D) embeddings view 1
        z2: (N, D) embeddings view 2
        labels: (N,) class labels for hard mining
        mode: 'random' or 'hard'
        tau: temperature

    Returns:
        loss: scalar tensor
    \"\"\"
    N = z1.shape[0]

    if mode == 'random':
        return nt_xent_loss_torch(z1, z2, tau=tau)

    # Hard negative mode: replace random negatives with hard negatives
    # Use numpy for mining (on CPU)
    with torch.no_grad():
        Z_np = F.normalize(torch.cat([z1, z2], dim=0), dim=-1).cpu().numpy()
        labels_np = np.concatenate([labels.cpu().numpy(), labels.cpu().numpy()])
        hard_idx = find_hard_negatives(Z_np, labels_np, topk=N // 2)

    Z = torch.cat([z1, z2], dim=0)  # (2N, D)
    Z = F.normalize(Z, dim=-1)
    sim = (Z @ Z.T) / tau  # (2N, 2N)

    # Mask self-similarity
    mask = torch.eye(2 * N, device=Z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Labels: standard positives
    pos_labels = torch.cat([
        torch.arange(N, 2 * N, device=Z.device),
        torch.arange(0, N, device=Z.device)
    ])

    # Compute loss using only hard negatives + positive
    losses = []
    for i in range(2 * N):
        pos_i = pos_labels[i].item()
        hard_neg_i = hard_idx[i][:N // 4]
        relevant_cols = [pos_i] + list(hard_neg_i)
        relevant_sims = sim[i, relevant_cols]
        # Cross-entropy: target = 0 (positive is first)
        target = torch.zeros(1, dtype=torch.long, device=Z.device)
        loss_i = F.cross_entropy(relevant_sims.unsqueeze(0), target)
        losses.append(loss_i)
    return torch.stack(losses).mean()


# Train two models: one with random negatives, one with hard negatives
X_data, y_data = generate_2d_clusters(n_per_class=64, n_classes=4)
X_data = X_data.to(device)

results = {}
for mining_mode in ['random', 'hard']:
    torch.manual_seed(42)
    model_m = MLPEncoder(input_dim=2, hidden_dim=32, embedding_dim=16).to(device)
    opt_m = optim.Adam(model_m.parameters(), lr=1e-3)
    step_losses = []

    model_m.train()
    for step in range(100):
        idx = torch.randperm(X_data.shape[0])[:64]
        x_batch = X_data[idx]
        y_batch = y_data[idx]
        x1 = augment_2d(x_batch)
        x2 = augment_2d(x_batch)
        z1 = model_m(x1)
        z2 = model_m(x2)
        loss = contrastive_loss_with_mining(z1, z2, y_batch, mode=mining_mode)
        opt_m.zero_grad()
        loss.backward()
        opt_m.step()
        step_losses.append(loss.item())

    # Eval: linear probe accuracy
    model_m.eval()
    with torch.no_grad():
        emb_all = model_m(X_data, project=False).cpu().numpy()
    protos = compute_class_prototypes(emb_all, y_data.numpy(), 4)
    preds_m, _ = zero_shot_classify(emb_all, protos)
    acc_m = (preds_m == y_data.numpy()).mean()
    results[mining_mode] = {'losses': step_losses, 'accuracy': acc_m}
    print(f"[{mining_mode:6s} negatives] Final loss: {step_losses[-1]:.4f} | "
          f"Accuracy: {acc_m:.4f}")

# Plot loss curves
fig, ax = plt.subplots(figsize=(10, 4))
for mode, res in results.items():
    smooth = np.convolve(res['losses'], np.ones(10) / 10, mode='valid')
    ax.plot(smooth, label=f"{mode} neg (acc={res['accuracy']:.2f})")
ax.set_xlabel("Step")
ax.set_ylabel("NT-Xent Loss (smoothed)")
ax.set_title("Random vs Hard Negative Mining")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/hard_neg_mining.png", dpi=80, bbox_inches='tight')
plt.show()
"""))

    # Cell 11: RW3 header
    cells.append(make_cell("""\
## Real-World Example 3: Temperature Sensitivity and Embedding Quality

Temperature tau is the most sensitive hyperparameter in contrastive learning.
- Low tau (0.01): sharp distribution, focuses on hardest negatives, may destabilize
- tau=0.07: SimCLR default, good balance
- High tau (2.0): flat distribution, ignores hard negatives, slow convergence

We measure two metrics proposed by Wang & Isola (2020):
- Alignment: how close positive pairs are (lower = better, range [-1, 1])
- Uniformity: how uniform the embedding distribution is on the unit sphere (lower = better)
""", "markdown"))

    # Cell 12: RW3 code + comparison
    cells.append(make_cell("""\
def alignment_metric(z1, z2):
    \"\"\"Measure alignment: mean squared distance between positive pairs.

    Lower is better. Perfectly aligned pairs give 0.

    Args:
        z1: (N, D) L2-normalized embeddings
        z2: (N, D) L2-normalized embeddings (positive pairs for z1)

    Returns:
        alignment: float (lower = better)
    \"\"\"
    return ((z1 - z2).pow(2).sum(dim=-1).mean()).item()


def uniformity_metric(z):
    \"\"\"Measure uniformity: average pairwise Gaussian potential on unit sphere.

    Lower (more negative) is better. Perfectly uniform = -inf.
    Wang & Isola (2020) formula: log(mean_ij exp(-2 ||z_i - z_j||^2))

    Args:
        z: (N, D) L2-normalized embeddings

    Returns:
        uniformity: float (lower = better)
    \"\"\"
    sq_dists = torch.cdist(z, z).pow(2)  # (N, N) pairwise squared distances
    # Exclude diagonal (self-distance = 0)
    N = z.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    sq_dists_flat = sq_dists[mask]  # (N*(N-1),)
    return torch.log(torch.exp(-2 * sq_dists_flat).mean()).item()


# Sweep temperature values
tau_values = [0.01, 0.07, 0.1, 0.5, 2.0]
tau_results = {}

X_data2, y_data2 = generate_2d_clusters(n_per_class=64, n_classes=4)
X_data2 = X_data2.to(device)

for tau in tau_values:
    torch.manual_seed(42)
    model_t = MLPEncoder(input_dim=2, hidden_dim=32, embedding_dim=16).to(device)
    opt_t = optim.Adam(model_t.parameters(), lr=1e-3)
    step_losses_t = []

    model_t.train()
    for step in range(100):
        idx = torch.randperm(X_data2.shape[0])[:64]
        x_batch = X_data2[idx]
        x1 = augment_2d(x_batch).to(device)
        x2 = augment_2d(x_batch).to(device)
        try:
            z1 = model_t(x1)
            z2 = model_t(x2)
            loss = nt_xent_loss_torch(z1, z2, tau=tau)
            if torch.isnan(loss) or torch.isinf(loss):
                # Instability at very low tau — skip step
                continue
            opt_t.zero_grad()
            loss.backward()
            # Gradient clipping for small tau to prevent explosion
            torch.nn.utils.clip_grad_norm_(model_t.parameters(), max_norm=1.0)
            opt_t.step()
            step_losses_t.append(loss.item())
        except RuntimeError as e:
            print(f"  tau={tau}: RuntimeError at step {step}: {e}")
            break

    # Compute alignment and uniformity
    model_t.eval()
    with torch.no_grad():
        idx_eval = torch.randperm(X_data2.shape[0])[:128]
        x_eval = X_data2[idx_eval]
        x1_eval = augment_2d(x_eval).to(device)
        x2_eval = augment_2d(x_eval).to(device)
        z1_eval = model_t(x1_eval)
        z2_eval = model_t(x2_eval)
        align = alignment_metric(z1_eval, z2_eval)
        unif = uniformity_metric(z1_eval)

    final_loss = step_losses_t[-1] if step_losses_t else float('nan')
    tau_results[tau] = {
        'final_loss': final_loss,
        'alignment': align,
        'uniformity': unif,
        'n_steps': len(step_losses_t)
    }
    print(f"tau={tau:.3f} | loss={final_loss:.4f} | align={align:.4f} | "
          f"unif={unif:.4f} | steps={len(step_losses_t)}")

# Summary table
print("\n=== Temperature Sensitivity Summary ===")
print(f"{'tau':>6} | {'Final Loss':>10} | {'Alignment':>10} | {'Uniformity':>10} | {'Stable?':>8}")
print("-" * 58)
for tau, res in tau_results.items():
    stable = "YES" if res['n_steps'] >= 80 else "NO"
    print(f"{tau:>6.3f} | {res['final_loss']:>10.4f} | {res['alignment']:>10.4f} | "
          f"{res['uniformity']:>10.4f} | {stable:>8}")

# Plot: alignment vs uniformity scatter (Wang & Isola 2020 style)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
taus_list = list(tau_results.keys())
aligns = [tau_results[t]['alignment'] for t in taus_list]
unifs = [tau_results[t]['uniformity'] for t in taus_list]
losses_list = [tau_results[t]['final_loss'] for t in taus_list]

axes[0].scatter(aligns, unifs, c='steelblue', s=80)
for i, t in enumerate(taus_list):
    axes[0].annotate(f"tau={t}", (aligns[i], unifs[i]), textcoords="offset points",
                     xytext=(5, 3), fontsize=9)
axes[0].set_xlabel("Alignment (lower=better)")
axes[0].set_ylabel("Uniformity (lower=better)")
axes[0].set_title("Alignment vs Uniformity by tau")

axes[1].bar([str(t) for t in taus_list], aligns, color='steelblue')
axes[1].set_xlabel("Temperature tau")
axes[1].set_ylabel("Alignment")
axes[1].set_title("Alignment by Temperature")

axes[2].bar([str(t) for t in taus_list], unifs, color='coral')
axes[2].set_xlabel("Temperature tau")
axes[2].set_ylabel("Uniformity")
axes[2].set_title("Uniformity by Temperature")

plt.tight_layout()
plt.savefig("/tmp/temperature_sweep.png", dpi=80, bbox_inches='tight')
plt.show()

# Final comparison: SimCLR vs supervised cross-entropy on same data
print("\n=== Comparison: Contrastive SimCLR vs Supervised CE ===")
from sklearn.linear_model import LogisticRegression

# Contrastive embeddings (tau=0.07 model from tau sweep)
# Re-train fresh supervised model for fair comparison
torch.manual_seed(42)
model_sup = MLPEncoder(input_dim=2, hidden_dim=32, embedding_dim=16).to(device)
# Add classification head
clf_head = nn.Linear(16, 4).to(device)
opt_sup = optim.Adam(list(model_sup.parameters()) + list(clf_head.parameters()), lr=1e-3)
X_sup = X_data2
y_sup = y_data2.to(device)

for step in range(100):
    logits = clf_head(model_sup(X_sup, project=False))
    loss_sup = F.cross_entropy(logits, y_sup)
    opt_sup.zero_grad()
    loss_sup.backward()
    opt_sup.step()

model_sup.eval()
with torch.no_grad():
    sup_emb = model_sup(X_sup, project=False).cpu().numpy()
    sup_logits = clf_head(model_sup(X_sup, project=False))
    sup_acc = (sup_logits.argmax(dim=1).cpu().numpy() == y_data2.numpy()).mean()

# Contrastive: linear probe (logistic regression on frozen embeddings)
model.eval()
with torch.no_grad():
    contrast_emb = model(X_sup, project=False).cpu().numpy()

lr_probe = LogisticRegression(max_iter=500)
lr_probe.fit(contrast_emb, y_data2.numpy())
contrast_probe_acc = lr_probe.score(contrast_emb, y_data2.numpy())

# Alignment and uniformity for both
with torch.no_grad():
    xv1 = augment_2d(X_sup).to(device)
    xv2 = augment_2d(X_sup).to(device)
    z_cont_1 = model(xv1)
    z_cont_2 = model(xv2)
    z_sup_1 = F.normalize(model_sup(xv1, project=False), dim=-1)
    z_sup_2 = F.normalize(model_sup(xv2, project=False), dim=-1)

print(f"\nMethod         | Accuracy | Alignment | Uniformity")
print("-" * 52)
print(f"SimCLR (tau=0.07) | {contrast_probe_acc:.4f}   | "
      f"{alignment_metric(z_cont_1, z_cont_2):.4f}    | "
      f"{uniformity_metric(z_cont_1):.4f}")
print(f"Supervised CE    | {sup_acc:.4f}   | "
      f"{alignment_metric(z_sup_1, z_sup_2):.4f}    | "
      f"{uniformity_metric(z_sup_1):.4f}")
"""))

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    return nb


if __name__ == "__main__":
    os.makedirs("/home/sbisw/github/interviewprep-ml/cv/notebooks", exist_ok=True)
    nb = build_notebook()
    path = "/home/sbisw/github/interviewprep-ml/cv/notebooks/05-contrastive-learning-vision.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"Written: {path}")
    print(f"  Cells: {len(nb.cells)}")
    code_lines = sum(
        len("".join(c["source"]).split("\n"))
        for c in nb.cells if c["cell_type"] == "code"
    )
    print(f"  Code lines: {code_lines}")
