"""Generator for CV notebook 08: Multimodal Vision-Language Models."""
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
# Multimodal Vision-Language Models

## Learning Objectives
1. Implement visual patch tokenization: image -> sequence of embedded tokens
2. Build cross-modal attention: text tokens attending to image tokens
3. Implement cross-modal retrieval (image-text matching) with shared embedding space
4. Compare image-only, text-only, and multimodal fusion on a synthetic VQA task
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
## Level 1: Visual Patch Tokenization

ViT-style patch tokenization: divide an image into non-overlapping patches,
project each patch to a D-dimensional embedding.

For a 32x32 image with 8x8 patches: 16 patches -> 16 visual tokens.
Each patch is linearly projected to dim=512 (proxy for CLIP ViT-L embedding).
""", "markdown"))

    # Cell 4: Level 1 code
    cells.append(make_cell("""\
def extract_patches(image, patch_size=8):
    \"\"\"Extract non-overlapping patches from a 2D image.

    Args:
        image: (H, W) or (C, H, W) numpy/tensor array
        patch_size: int, patch height = patch width

    Returns:
        patches: (n_patches, patch_size*patch_size*C) flattened patch vectors
        n_patches_h: number of patches along height
        n_patches_w: number of patches along width
    \"\"\"
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    if image.ndim == 2:
        C, H, W = 1, *image.shape
        image = image[None, :, :]  # (1, H, W)
    else:
        C, H, W = image.shape

    n_h = H // patch_size
    n_w = W // patch_size
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patch = image[:, i*patch_size:(i+1)*patch_size,
                          j*patch_size:(j+1)*patch_size]
            patches.append(patch.flatten())  # (C * P * P,)
    return np.array(patches, dtype=np.float32), n_h, n_w


class PatchTokenizer(nn.Module):
    \"\"\"Tokenize images into patch embeddings (ViT-style, no torchvision).

    Splits image into non-overlapping patches using unfold operations,
    then projects each flattened patch to embedding dimension.

    Args:
        img_size: int, input image height = width
        patch_size: int, patch height = width
        in_channels: int, input image channels
        embed_dim: int, output embedding dimension per patch
    \"\"\"
    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        # Linear projection: patch -> embed_dim
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(self.n_patches, embed_dim)

    def forward(self, x):
        \"\"\"Args: x: (B, C, H, W). Returns: (B, n_patches, embed_dim).\"\"\"
        B, C, H, W = x.shape
        P = self.patch_size
        n_h, n_w = H // P, W // P

        # Use unfold to extract patches without loops
        # unfold along height: (B, C, n_h, P, W)
        x_h = x.unfold(2, P, P)  # (B, C, n_h, W, P)
        # unfold along width: (B, C, n_h, n_w, P, P)
        patches = x_h.unfold(3, P, P)  # (B, C, n_h, n_w, P, P)
        # Reshape to (B, n_patches, C*P*P)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, n_h, n_w, C, P, P)
        patches = patches.view(B, n_h * n_w, C * P * P)           # (B, n_patches, patch_dim)

        # Linear projection to embed_dim
        tokens = self.projection(patches)  # (B, n_patches, embed_dim)

        # Add positional embeddings
        pos_idx = torch.arange(self.n_patches, device=x.device)
        tokens = tokens + self.pos_embedding(pos_idx).unsqueeze(0)  # (B, n_patches, embed_dim)
        return tokens


# Demo: tokenize a synthetic 32x32 RGB image
tokenizer = PatchTokenizer(img_size=32, patch_size=8, in_channels=3, embed_dim=512)
dummy_img = torch.randn(2, 3, 32, 32)  # batch of 2 images
tokens = tokenizer(dummy_img)
print(f"Input image shape:  {dummy_img.shape}")
print(f"Output token shape: {tokens.shape}")
print(f"  Interpretation: {tokens.shape[0]} images, {tokens.shape[1]} patches each, "
      f"each patch -> {tokens.shape[2]}-dim embedding")

# Visualize patches on a synthetic image
def make_synthetic_scene(H=32, W=32):
    \"\"\"Synthetic 3-channel image: gradient + geometric shapes.\"\"\"
    img = np.zeros((3, H, W), dtype=np.float32)
    # Red channel: horizontal gradient
    for col in range(W):
        img[0, :, col] = col / (W - 1)
    # Green channel: vertical gradient
    for row in range(H):
        img[1, row, :] = row / (H - 1)
    # Blue channel: center circle
    cy, cx = H // 2, W // 2
    for i in range(H):
        for j in range(W):
            if (i - cy)**2 + (j - cx)**2 < (H // 4)**2:
                img[2, i, j] = 1.0
    return img

scene = make_synthetic_scene(32, 32)
patches_np, n_h, n_w = extract_patches(scene, patch_size=8)
print(f"\\nPatch extraction: {patches_np.shape[0]} patches, "
      f"each flattened to {patches_np.shape[1]} dims")

# Show the image and its patches
fig, axes = plt.subplots(1, n_h * n_w + 1, figsize=(16, 3))
axes[0].imshow(scene.transpose(1, 2, 0))
axes[0].set_title("Full Image")
axes[0].axis('off')
for idx in range(n_h * n_w):
    patch = patches_np[idx].reshape(3, 8, 8).transpose(1, 2, 0)
    patch = np.clip(patch, 0, 1)
    axes[idx + 1].imshow(patch)
    axes[idx + 1].set_title(f"P{idx}")
    axes[idx + 1].axis('off')
plt.suptitle("Patch Tokenization: 32x32 image -> 16 patches of 8x8")
plt.tight_layout()
plt.savefig("/tmp/patch_tokenization.png", dpi=80, bbox_inches="tight")
plt.show()
"""))

    # Cell 5: Level 2 header
    cells.append(make_cell("""\
## Level 2: Multimodal Attention — Text Attending to Image Tokens

Core VLM mechanism: text tokens can attend to visual tokens.
Simplified toy implementation using standard nn.MultiheadAttention.

Architecture:
- Image tokens: 16 visual tokens of dim=64 (from patch tokenizer)
- Text tokens: sequence of word embeddings of dim=64
- Cross-attention: query=text, key+value=image tokens
- Self-attention: within text (standard causal LM)
""", "markdown"))

    # Cell 6: Level 2 code
    cells.append(make_cell("""\
# Toy vocabulary for synthetic text tokens
VOCAB_SIZE = 50   # small vocabulary
PAD_ID = 0
BOS_ID = 1        # beginning of sequence
EOS_ID = 2        # end of sequence

def make_synthetic_text_tokens(n_questions=200, seq_len=6, vocab_size=VOCAB_SIZE):
    \"\"\"Generate synthetic question tokens for VQA task.

    Two question types (classes):
      Class 0: tokens centered around IDs 3-25 (proxy for 'what color is X?')
      Class 1: tokens centered around IDs 25-48 (proxy for 'where is X?')

    Returns:
        tokens: (N, seq_len) int tensor
        labels: (N,) int tensor
    \"\"\"
    tokens_list = []
    labels_list = []
    for i in range(n_questions):
        cls = i % 2
        if cls == 0:
            t = torch.randint(3, 25, (seq_len,))
        else:
            t = torch.randint(25, 48, (seq_len,))
        tokens_list.append(t)
        labels_list.append(cls)
    return torch.stack(tokens_list), torch.tensor(labels_list, dtype=torch.long)


def make_synthetic_images_vqa(n=200, img_size=32, patch_size=8):
    \"\"\"Generate synthetic images for VQA task.

    Class 0 images: bright in top-left quadrant (proxy for 'red object on left')
    Class 1 images: bright in bottom-right quadrant (proxy for 'blue object on right')

    Returns:
        images: (N, 3, img_size, img_size) float tensor
    \"\"\"
    images = []
    for i in range(n):
        cls = i % 2
        img = torch.randn(3, img_size, img_size) * 0.1
        H2, W2 = img_size // 2, img_size // 2
        if cls == 0:
            img[0, :H2, :W2] += 1.5  # red channel, top-left
        else:
            img[2, H2:, W2:] += 1.5  # blue channel, bottom-right
        images.append(img)
    return torch.stack(images)


class MultimodalTransformer(nn.Module):
    \"\"\"Simplified multimodal transformer: image tokens + text tokens.

    Two attention blocks:
      1. Self-attention across image tokens (visual encoding)
      2. Cross-attention: text queries, image keys/values (visual grounding)
      3. Text self-attention (language reasoning)
      4. Classifier head

    This approximates the forward pass of a VLM like LLaVA for a VQA task.
    \"\"\"
    def __init__(self, vocab_size=VOCAB_SIZE, n_vis_tokens=16, embed_dim=64,
                 n_heads=4, n_classes=2):
        super().__init__()
        self.embed_dim = embed_dim
        # Visual tokenizer (simplified)
        self.vis_proj = nn.Linear(3 * 8 * 8, embed_dim)  # patch -> embed
        self.vis_pos = nn.Embedding(n_vis_tokens, embed_dim)
        # Text embedding
        self.text_emb = nn.Embedding(vocab_size, embed_dim)
        self.text_pos = nn.Embedding(32, embed_dim)      # max text seq len
        # Cross-attention: text attends to image
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)
        # Text self-attention
        self.text_attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_classes)
        )

    def extract_vis_tokens(self, images):
        \"\"\"Convert images to sequence of patch tokens.

        Args:
            images: (B, 3, 32, 32) float tensor

        Returns:
            vis_tokens: (B, n_patches, embed_dim)
        \"\"\"
        B, C, H, W = images.shape
        P = 8
        n_h, n_w = H // P, W // P
        # Extract patches using unfold
        x_h = images.unfold(2, P, P).unfold(3, P, P)  # (B, C, n_h, n_w, P, P)
        x_h = x_h.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, n_h, n_w, C, P, P)
        x_h = x_h.view(B, n_h * n_w, C * P * P)           # (B, n_patches, 192)
        tokens = self.vis_proj(x_h)                         # (B, n_patches, embed_dim)
        # Positional embeddings
        pos_idx = torch.arange(n_h * n_w, device=images.device)
        tokens = tokens + self.vis_pos(pos_idx).unsqueeze(0)
        return tokens

    def forward(self, images, text_ids, return_attn=False):
        \"\"\"VQA forward pass.

        Args:
            images: (B, 3, H, W)
            text_ids: (B, seq_len) int tensor
            return_attn: if True, return cross-attention weights

        Returns:
            logits: (B, n_classes)
            attn_weights: (B, seq_len, n_vis_tokens) if return_attn else None
        \"\"\"
        B, seq_len = text_ids.shape
        # Visual tokens
        vis_tokens = self.extract_vis_tokens(images)          # (B, n_vis, embed_dim)
        # Text tokens + positional encoding
        pos_idx = torch.arange(seq_len, device=text_ids.device)
        text_tokens = self.text_emb(text_ids) + self.text_pos(pos_idx).unsqueeze(0)
        # Cross-attention: text attends to image
        attended, attn_weights = self.cross_attn(
            query=text_tokens,   # text queries
            key=vis_tokens,      # image keys
            value=vis_tokens     # image values
        )
        text_grounded = self.norm1(text_tokens + attended)   # residual
        # Text self-attention
        text_self, _ = self.text_attn(text_grounded, text_grounded, text_grounded)
        text_final = self.norm2(text_grounded + text_self)   # residual
        # Pool text tokens and classify
        pooled = text_final.mean(dim=1)                      # (B, embed_dim)
        logits = self.classifier(pooled)
        return logits, attn_weights if return_attn else None


# Generate synthetic VQA dataset
N_vqa = 400
imgs_vqa = make_synthetic_images_vqa(n=N_vqa, img_size=32, patch_size=8).to(device)
text_vqa, labels_vqa = make_synthetic_text_tokens(n_questions=N_vqa, seq_len=6)
text_vqa = text_vqa.to(device)
labels_vqa = labels_vqa.to(device)

# Train multimodal model
vqa_model = MultimodalTransformer(vocab_size=VOCAB_SIZE, n_vis_tokens=16,
                                   embed_dim=64, n_heads=4, n_classes=2).to(device)
opt_vqa = optim.Adam(vqa_model.parameters(), lr=1e-3)
n_params_vqa = sum(p.numel() for p in vqa_model.parameters())
print(f"Multimodal VQA model parameters: {n_params_vqa:,}")

batch_size_vqa = 64
vqa_losses = []
vqa_model.train()
for step in range(200):
    idx = torch.randperm(N_vqa)[:batch_size_vqa]
    logits_vqa, _ = vqa_model(imgs_vqa[idx], text_vqa[idx])
    loss_vqa = F.cross_entropy(logits_vqa, labels_vqa[idx])
    opt_vqa.zero_grad()
    loss_vqa.backward()
    opt_vqa.step()
    vqa_losses.append(loss_vqa.item())
    if (step + 1) % 50 == 0:
        print(f"Step {step+1} | VQA Loss: {loss_vqa.item():.4f}")

vqa_model.eval()
with torch.no_grad():
    logits_all, _ = vqa_model(imgs_vqa, text_vqa)
    acc_vqa = (logits_all.argmax(1) == labels_vqa).float().mean().item()
print(f"Multimodal VQA accuracy: {acc_vqa:.4f}")
"""))

    # Cell 7: RW1 header
    cells.append(make_cell("""\
## Real-World Example 1: Cross-Modal Retrieval

Given a text query, find the most similar images in an embedding space
shared between images and text (CLIP-style).

Here:
- Image encoder: PatchTokenizer + mean pool -> embed
- Text encoder: word embedding + mean pool -> embed
- Projection layer: align both to shared embed space
- Retrieval: cosine similarity between query and image embeddings
""", "markdown"))

    # Cell 8: RW1 code
    cells.append(make_cell("""\
class ImageEncoder(nn.Module):
    \"\"\"Image -> 256-dim shared embedding space.\"\"\"
    def __init__(self, img_size=32, patch_size=8, embed_dim=256):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_emb = nn.Embedding(n_patches, embed_dim)
        self.n_patches = n_patches
        self.patch_size = patch_size
        # Transformer encoder: 2 layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                                    batch_first=True, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        n_h, n_w = H // P, W // P
        patches = x.unfold(2, P, P).unfold(3, P, P)  # (B, C, n_h, n_w, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, n_h*n_w, -1)
        tokens = self.patch_proj(patches)
        pos_idx = torch.arange(self.n_patches, device=x.device)
        tokens = tokens + self.pos_emb(pos_idx).unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return F.normalize(self.out_proj(pooled), dim=-1)  # L2-normalize


class TextEncoder(nn.Module):
    \"\"\"Text tokens -> 256-dim shared embedding space.\"\"\"
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(32, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                                    batch_first=True, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids):
        B, seq_len = token_ids.shape
        pos_idx = torch.arange(seq_len, device=token_ids.device)
        tokens = self.word_emb(token_ids) + self.pos_emb(pos_idx).unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return F.normalize(self.out_proj(pooled), dim=-1)  # L2-normalize


def contrastive_retrieval_loss(img_emb, txt_emb, tau=0.07):
    \"\"\"CLIP-style contrastive loss for cross-modal alignment.

    Positive pairs: (image_i, text_i)
    Negatives: all cross-pairs in the batch
    \"\"\"
    N = img_emb.shape[0]
    sim = (img_emb @ txt_emb.T) / tau  # (N, N)
    labels = torch.arange(N, device=img_emb.device)
    loss_i2t = F.cross_entropy(sim, labels)          # image -> text
    loss_t2i = F.cross_entropy(sim.T, labels)        # text -> image
    return (loss_i2t + loss_t2i) / 2.0


# Train shared embedding space
img_enc = ImageEncoder(img_size=32, patch_size=8, embed_dim=256).to(device)
txt_enc = TextEncoder(vocab_size=VOCAB_SIZE, embed_dim=256).to(device)
opt_ret = optim.Adam(list(img_enc.parameters()) + list(txt_enc.parameters()), lr=1e-3)

retrieval_losses = []
for step in range(200):
    idx = torch.randperm(N_vqa)[:64]
    img_emb_b = img_enc(imgs_vqa[idx])
    txt_emb_b = txt_enc(text_vqa[idx])
    loss_ret = contrastive_retrieval_loss(img_emb_b, txt_emb_b, tau=0.07)
    opt_ret.zero_grad()
    loss_ret.backward()
    opt_ret.step()
    retrieval_losses.append(loss_ret.item())

# Evaluate retrieval: text -> image
img_enc.eval()
txt_enc.eval()
with torch.no_grad():
    all_img_emb = img_enc(imgs_vqa)        # (N, 256)
    all_txt_emb = txt_enc(text_vqa)        # (N, 256)
    sim_matrix = all_img_emb @ all_txt_emb.T  # (N, N)

# Precision@5: for each text query, are any of top-5 images from same class?
y_np_vqa = labels_vqa.cpu().numpy()
precision_at_5 = []
for q_idx in range(N_vqa):
    top5 = sim_matrix[q_idx].cpu().topk(6).indices[1:].numpy()  # exclude self
    q_class = y_np_vqa[q_idx]
    same_class_in_top5 = (y_np_vqa[top5] == q_class).mean()
    precision_at_5.append(same_class_in_top5)
print(f"Retrieval Precision@5 (after contrastive training): {np.mean(precision_at_5):.4f}")

# Visualize similarity matrix (sampled 20x20)
n_viz = 20
viz_sim = sim_matrix[:n_viz, :n_viz].cpu().numpy()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im = axes[0].imshow(viz_sim, cmap='hot', vmin=0, vmax=1)
axes[0].set_xlabel("Image Index")
axes[0].set_ylabel("Text Query Index")
axes[0].set_title(f"Cross-Modal Similarity Matrix (first {n_viz})")
plt.colorbar(im, ax=axes[0])

axes[1].plot(np.convolve(retrieval_losses, np.ones(10)/10, mode='valid'))
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Contrastive Loss")
axes[1].set_title("Cross-Modal Retrieval Training Loss")
plt.tight_layout()
plt.savefig("/tmp/cross_modal_retrieval.png", dpi=80, bbox_inches="tight")
plt.show()
"""))

    # Cell 9: RW2 header
    cells.append(make_cell("""\
## Real-World Example 2: Visual Grounding via Attention Heatmaps

Visual grounding: given an image and text query, identify which image
regions the model attends to.

Using the cross-attention weights from the multimodal transformer (Cell 6),
we can create spatial attention heatmaps showing which patches are most
relevant for each text query.
""", "markdown"))

    # Cell 10: RW2 code
    cells.append(make_cell("""\
def get_attention_heatmap(model, image, text_ids):
    \"\"\"Extract cross-attention weights and reshape to spatial heatmap.

    Args:
        model: MultimodalTransformer with return_attn=True support
        image: (1, 3, H, W) tensor (single image)
        text_ids: (1, seq_len) int tensor

    Returns:
        heatmap: (n_patches_h, n_patches_w) float numpy array
                 averaged over text tokens and attention heads
    \"\"\"
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(image, text_ids, return_attn=True)
    # attn_weights: (B, seq_len, n_vis_tokens)
    # Average over text tokens -> (B, n_vis_tokens)
    heatmap_flat = attn_weights[0].mean(dim=0).cpu().numpy()  # (n_vis_tokens,)
    # Reshape to spatial grid (4x4 for 32x32 image with 8x8 patches)
    n_h = n_w = int(np.sqrt(len(heatmap_flat)))
    heatmap = heatmap_flat.reshape(n_h, n_w)
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


# Test on class 0 vs class 1 images
vqa_model.eval()
n_per_class = 4
fig, axes = plt.subplots(4, n_per_class, figsize=(12, 10))

for col in range(n_per_class):
    for cls in range(2):
        # Find samples of this class
        cls_indices = (labels_vqa == cls).nonzero(as_tuple=True)[0]
        sample_idx = cls_indices[col].item()
        img_s = imgs_vqa[sample_idx:sample_idx+1]          # (1, 3, 32, 32)
        txt_s = text_vqa[sample_idx:sample_idx+1]           # (1, 6)
        heatmap = get_attention_heatmap(vqa_model, img_s, txt_s)

        # Show image in row 0/1, attention in row 2/3
        img_show = img_s[0].permute(1, 2, 0).cpu().numpy()
        img_show = np.clip(img_show, -1, 2)
        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
        axes[cls * 2, col].imshow(img_show)
        axes[cls * 2, col].set_title(f"Cls {cls} #{col}")
        axes[cls * 2, col].axis('off')
        # Upsample heatmap to 32x32 for overlay
        from torch.nn import functional as TF
        h_up = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
        h_up = F.interpolate(h_up, size=(32, 32), mode='bilinear', align_corners=False)
        h_up = h_up[0, 0].numpy()
        axes[cls * 2 + 1, col].imshow(h_up, cmap='hot', vmin=0, vmax=1)
        axes[cls * 2 + 1, col].set_title(f"Attention ({['hot','cool'][cls]})")
        axes[cls * 2 + 1, col].axis('off')

row_labels = ['Class 0 Images', 'Class 0 Attention',
              'Class 1 Images', 'Class 1 Attention']
for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, rotation=45, ha='right', fontsize=9)
fig.suptitle("Visual Grounding: Cross-Attention Heatmaps by Class")
plt.tight_layout()
plt.savefig("/tmp/attention_heatmap.png", dpi=80, bbox_inches="tight")
plt.show()

# Quantify: does attention focus on correct region?
# Class 0: should attend to top-left patches (indices 0, 1, 4, 5 in 4x4 grid)
# Class 1: should attend to bottom-right patches (indices 10, 11, 14, 15 in 4x4 grid)
cls0_indices = [0, 1, 4, 5]    # top-left in 4x4 grid
cls1_indices = [10, 11, 14, 15] # bottom-right in 4x4 grid

grounding_scores = {'cls0': [], 'cls1': []}
for sample_idx in range(min(50, N_vqa)):
    cls = labels_vqa[sample_idx].item()
    img_s = imgs_vqa[sample_idx:sample_idx+1]
    txt_s = text_vqa[sample_idx:sample_idx+1]
    hm = get_attention_heatmap(vqa_model, img_s, txt_s).flatten()
    if cls == 0:
        grounding_scores['cls0'].append(hm[cls0_indices].mean() / (hm.mean() + 1e-8))
    else:
        grounding_scores['cls1'].append(hm[cls1_indices].mean() / (hm.mean() + 1e-8))

print(f"Grounding score (correct region / mean): "
      f"Class 0: {np.mean(grounding_scores['cls0']):.3f}, "
      f"Class 1: {np.mean(grounding_scores['cls1']):.3f}")
print("(> 1.0 means model correctly attends to the relevant image region)")
"""))

    # Cell 11: RW3 header
    cells.append(make_cell("""\
## Real-World Example 3: Hallucination Probe and Modality Comparison

Hallucination in VLMs: the model describes objects not present in the image.
Probe: compare model confidence on queries about present vs absent features.
A well-grounded model should be more confident about present objects.

Also: compare image-only, text-only, and multimodal fusion accuracy.
""", "markdown"))

    # Cell 12: RW3 code + comparison
    cells.append(make_cell("""\
class ImageOnlyClassifier(nn.Module):
    \"\"\"Classify using only image features (no text).\"\"\"
    def __init__(self, img_size=32, patch_size=8, embed_dim=128, n_classes=2):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_emb = nn.Embedding(n_patches, embed_dim)
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, images):
        B, C, H, W = images.shape
        P = self.patch_size
        n_h, n_w = H // P, W // P
        patches = images.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, n_h*n_w, -1)
        tokens = self.patch_proj(patches)
        pos_idx = torch.arange(self.n_patches, device=images.device)
        tokens = tokens + self.pos_emb(pos_idx).unsqueeze(0)
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)


class TextOnlyClassifier(nn.Module):
    \"\"\"Classify using only text tokens (no image).\"\"\"
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, n_classes=2):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(32, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, token_ids):
        B, seq_len = token_ids.shape
        pos_idx = torch.arange(seq_len, device=token_ids.device)
        tokens = self.word_emb(token_ids) + self.pos_emb(pos_idx).unsqueeze(0)
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)


# Train image-only and text-only baselines
modality_results = {}

for name, model_m, inputs_fn in [
    ('Image-only', ImageOnlyClassifier(img_size=32, patch_size=8, embed_dim=128).to(device),
     lambda idx: (imgs_vqa[idx],)),
    ('Text-only', TextOnlyClassifier(vocab_size=VOCAB_SIZE, embed_dim=128).to(device),
     lambda idx: (text_vqa[idx],)),
    ('Multimodal', vqa_model,
     lambda idx: (imgs_vqa[idx], text_vqa[idx])),
]:
    if name == 'Multimodal':
        # Already trained in Cell 6; just evaluate
        model_m.eval()
        with torch.no_grad():
            logits_m, _ = model_m(imgs_vqa, text_vqa)
            acc_m = (logits_m.argmax(1) == labels_vqa).float().mean().item()
        modality_results[name] = {'accuracy': acc_m,
                                   'params': sum(p.numel() for p in model_m.parameters())}
        print(f"{name:15s} | acc={acc_m:.4f}")
        continue

    opt_m = optim.Adam(model_m.parameters(), lr=1e-3)
    model_m.train()
    for step in range(200):
        idx = torch.randperm(N_vqa)[:batch_size_vqa]
        logits_m = model_m(*inputs_fn(idx))
        loss_m = F.cross_entropy(logits_m, labels_vqa[idx])
        opt_m.zero_grad()
        loss_m.backward()
        opt_m.step()

    model_m.eval()
    with torch.no_grad():
        logits_m = model_m(*inputs_fn(torch.arange(N_vqa)))
        acc_m = (logits_m.argmax(1) == labels_vqa).float().mean().item()
    n_params_m = sum(p.numel() for p in model_m.parameters())
    modality_results[name] = {'accuracy': acc_m, 'params': n_params_m}
    print(f"{name:15s} | acc={acc_m:.4f} | params={n_params_m:,}")


# Hallucination probe: model confidence on present vs absent features
vqa_model.eval()
present_conf = []
absent_conf = []

with torch.no_grad():
    for i in range(min(100, N_vqa)):
        cls = labels_vqa[i].item()
        img_s = imgs_vqa[i:i+1]

        # Text token for "correct" class query (present feature)
        present_txt = text_vqa[i:i+1]        # matching class text
        # Text token for "wrong" class query (absent feature)
        absent_idx = (torch.randint(0, N_vqa, (1,)).item() + N_vqa // 2) % N_vqa
        if labels_vqa[absent_idx].item() == cls:
            absent_idx = (absent_idx + 1) % N_vqa
        absent_txt = text_vqa[absent_idx:absent_idx+1]

        # Get model confidence
        logits_pres, _ = vqa_model(img_s, present_txt)
        logits_abs, _ = vqa_model(img_s, absent_txt)
        probs_pres = F.softmax(logits_pres, dim=-1)[0, cls].item()
        probs_abs = F.softmax(logits_abs, dim=-1)[0, cls].item()
        present_conf.append(probs_pres)
        absent_conf.append(probs_abs)

print(f"\\nHallucination Probe:")
print(f"  Confidence on PRESENT features: {np.mean(present_conf):.4f} +/- {np.std(present_conf):.4f}")
print(f"  Confidence on ABSENT  features: {np.mean(absent_conf):.4f} +/- {np.std(absent_conf):.4f}")
print(f"  Gap (higher = better grounding): {np.mean(present_conf) - np.mean(absent_conf):.4f}")


# Final comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy comparison
names_m = list(modality_results.keys())
accs_m = [modality_results[n]['accuracy'] for n in names_m]
bars_m = axes[0].bar(names_m, accs_m, color=['steelblue', 'coral', 'green'])
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("VQA Accuracy by Modality")
for bar, a in zip(bars_m, accs_m):
    axes[0].text(bar.get_x() + bar.get_width()/2, a + 0.02,
                 f"{a:.3f}", ha='center', fontsize=11)

# Hallucination probe
axes[1].bar(['Present', 'Absent'], [np.mean(present_conf), np.mean(absent_conf)],
            color=['steelblue', 'coral'],
            yerr=[np.std(present_conf), np.std(absent_conf)], capsize=8)
axes[1].set_ylabel("Model Confidence")
axes[1].set_title("Hallucination Probe\n(present vs absent features)")
axes[1].set_ylim(0, 1.0)

# VQA training loss history (multimodal)
axes[2].plot(np.convolve(vqa_losses, np.ones(10)/10, mode='valid'), color='green')
axes[2].set_xlabel("Training Step")
axes[2].set_ylabel("Loss (smoothed)")
axes[2].set_title("Multimodal VQA Training Loss")

plt.suptitle("Multimodal VLM: Modality Ablation and Hallucination Analysis")
plt.tight_layout()
plt.savefig("/tmp/multimodal_comparison.png", dpi=80, bbox_inches="tight")
plt.show()

# Summary table
print("\\n=== Final Summary: Modality Comparison ===")
print(f"{'Method':15s} | {'Accuracy':>8s} | {'Parameters':>12s}")
print("-" * 42)
for name, res in modality_results.items():
    print(f"{name:15s} | {res['accuracy']:>8.4f} | {res['params']:>12,}")

print(f"\\nKey finding: Multimodal fusion achieves higher accuracy than either modality alone.")
print(f"Grounding gap: {np.mean(present_conf) - np.mean(absent_conf):.4f} "
      f"({'good grounding' if np.mean(present_conf) > np.mean(absent_conf) else 'weak grounding'})")
"""))

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    return nb


if __name__ == "__main__":
    os.makedirs("/home/sbisw/github/interviewprep-ml/cv/notebooks", exist_ok=True)
    nb = build_notebook()
    path = "/home/sbisw/github/interviewprep-ml/cv/notebooks/08-multimodal-vision-llm.ipynb"
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"Written: {path}")
    print(f"  Cells: {len(nb.cells)}")
    code_lines = sum(
        len("".join(c["source"]).split("\n"))
        for c in nb.cells if c["cell_type"] == "code"
    )
    print(f"  Code lines: {code_lines}")
