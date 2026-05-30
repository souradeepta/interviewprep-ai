"""
Generator script for ml/notebooks 31-40.
Each notebook has exactly 12 cells:
  1  md  (title + objectives)
  2  code (imports + device + seeds)
  3  md  (L1 header)
  4  code (L1 ~20-40 lines)
  5  md  (L2 header)
  6  code (L2 ~60-100 lines + OOM handling where specified)
  7  md  (RW1 header)
  8  code (RW1 ~40-60 lines)
  9  md  (RW2 header)
  10 code (RW2 ~40-60 lines)
  11 md  (RW3 header)
  12 code (RW3 ~40-60 lines)
"""
import json
import os

OUT_DIR = "/home/sbisw/github/interviewprep-ml/ml/notebooks"
os.makedirs(OUT_DIR, exist_ok=True)


def nb(cells):
    """Build a minimal nbformat 4.5 notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


# ──────────────────────────────────────────────────────────────────────────────
# 31 — Pruning
# ──────────────────────────────────────────────────────────────────────────────
notebooks = {}

notebooks["31-pruning"] = nb([
    md("""# Neural Network Pruning

## Learning Objectives
1. Understand weight magnitude pruning and how sparsity is measured
2. Apply unstructured and structured pruning using `torch.nn.utils.prune`
3. Implement iterative magnitude pruning (lottery ticket hypothesis)
4. Analyse accuracy-vs-sparsity trade-offs across different pruning ratios
"""),
    code("""import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — Weight Magnitude Pruning from Scratch"),
    code("""# ── Level 1: magnitude pruning (numpy, no torch) ──────────────────────────
import numpy as np

np.random.seed(42)
W = np.random.randn(4, 8)          # simulated weight matrix
print("Original W (first row):", W[0])

def magnitude_prune(W: np.ndarray, sparsity: float) -> np.ndarray:
    \"\"\"Zero out the bottom `sparsity` fraction of weights by absolute value.\"\"\"
    flat = np.abs(W).flatten()
    threshold = np.percentile(flat, sparsity * 100)
    mask = np.abs(W) > threshold
    return W * mask                # pruned weights

def measure_sparsity(W: np.ndarray) -> float:
    \"\"\"Return fraction of zero entries.\"\"\"
    return float((W == 0).sum()) / W.size

sparsities = [0.0, 0.3, 0.5, 0.7, 0.9]
for s in sparsities:
    W_pruned = magnitude_prune(W, s)
    actual = measure_sparsity(W_pruned)
    print(f"Target sparsity={s:.1f}  Actual sparsity={actual:.3f}  "
          f"Non-zeros={int((W_pruned != 0).sum())}/{W.size}")

# Visualise magnitude distribution
flat_abs = np.abs(W).flatten()
print(f"\\nWeight |w| stats — min:{flat_abs.min():.3f}  "
      f"mean:{flat_abs.mean():.3f}  max:{flat_abs.max():.3f}")
"""),
    md("## Level 2 — PyTorch Structured & Unstructured Pruning"),
    code("""# ── Level 2: torch.nn.utils.prune ─────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

# Tiny MLP for demonstration
class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = SmallMLP().to(device)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def sparsity_of(tensor):
    return float((tensor == 0).sum().item()) / tensor.numel()

print(f"Parameters before pruning: {count_params(model):,}")

# ── Unstructured L1 pruning (weight-level) ───────────────────────
model_unstr = copy.deepcopy(model)
prune.l1_unstructured(model_unstr.fc1, name="weight", amount=0.4)
prune.l1_unstructured(model_unstr.fc2, name="weight", amount=0.4)
prune.l1_unstructured(model_unstr.fc3, name="weight", amount=0.4)

sp1 = sparsity_of(model_unstr.fc1.weight)
sp2 = sparsity_of(model_unstr.fc2.weight)
print(f"Unstructured pruning (40%) — fc1 sparsity: {sp1:.2f}, fc2 sparsity: {sp2:.2f}")

# Remove pruning reparameterization (make permanent)
for module in [model_unstr.fc1, model_unstr.fc2, model_unstr.fc3]:
    prune.remove(module, "weight")

# ── Structured Ln pruning (row/filter-level) ─────────────────────
model_str = copy.deepcopy(model)
prune.ln_structured(model_str.fc1, name="weight", amount=0.3, n=2, dim=0)
print(f"Structured pruning (30%) — fc1 zero rows: "
      f"{int((model_str.fc1.weight_mask.sum(dim=1) == 0).sum())} / "
      f"{model_str.fc1.weight_mask.shape[0]}")

# ── Quick accuracy proxy (random data) ───────────────────────────
x_test = torch.randn(256, 64).to(device)
y_test = torch.randint(0, 10, (256,)).to(device)

def accuracy(m, x, y):
    m.eval()
    with torch.no_grad():
        preds = m(x).argmax(dim=1)
    return (preds == y).float().mean().item()

acc_orig  = accuracy(model,      x_test, y_test)
acc_unstr = accuracy(model_unstr, x_test, y_test)

print(f"\\nRandom-init accuracy (baseline)   : {acc_orig:.3f}")
print(f"After 40% unstructured pruning     : {acc_unstr:.3f}")
print("(Accuracy differences are noise on random init — retrain to see real effect)")
"""),
    md("## Real-World Example 1 — Iterative Magnitude Pruning (Lottery Ticket)"),
    code("""# ── RW1: Iterative magnitude pruning — lottery ticket hypothesis ──────────
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

# Synthetic classification task
torch.manual_seed(42)
X = torch.randn(1000, 32)
y = (X[:, 0] + X[:, 1] > 0).long()
train_ds = TensorDataset(X[:800], y[:800])
val_ds   = TensorDataset(X[800:], y[800:])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

def train_one_epoch(model, loader, opt):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()

def eval_acc(model, ds):
    model.eval()
    x, y = ds.tensors
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()

# Iterative pruning: prune 20% per round × 5 rounds
ROUNDS      = 5
PRUNE_RATE  = 0.20          # fraction of *remaining* weights per round
TRAIN_EPOCHS = 10

results = []
net = Net().to(device)
orig_state = copy.deepcopy(net.state_dict())  # "winning ticket" initial weights

for rnd in range(ROUNDS):
    # Reset to original weights (lottery ticket rewind)
    net.load_state_dict(copy.deepcopy(orig_state))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(TRAIN_EPOCHS):
        train_one_epoch(net, train_loader, opt)

    acc_before = eval_acc(net, val_ds)

    # Prune
    for module in [net.fc1, net.fc2, net.fc3]:
        prune.l1_unstructured(module, name="weight", amount=PRUNE_RATE)
        prune.remove(module, "weight")

    total_params = sum(p.numel() for p in net.parameters())
    zero_params  = sum((p == 0).sum().item() for p in net.parameters())
    sparsity     = zero_params / total_params

    results.append({"round": rnd + 1, "sparsity": sparsity, "val_acc": acc_before})
    print(f"Round {rnd+1}: sparsity={sparsity:.2%}  val_acc={acc_before:.3f}")

print("\\nLottery-ticket summary (acc should stay reasonable even at high sparsity)")
"""),
    md("## Real-World Example 2 — BERT Attention-Head Pruning"),
    code("""# ── RW2: BERT attention head importance scoring ──────────────────────────
# Uses only torch + numpy (no internet download needed)
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

# Simulate a single BERT-style self-attention layer (small dims for speed)
BATCH, SEQ, HIDDEN, N_HEADS = 4, 16, 64, 4
HEAD_DIM = HIDDEN // N_HEADS

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.Wq = nn.Linear(hidden, hidden)
        self.Wk = nn.Linear(hidden, hidden)
        self.Wv = nn.Linear(hidden, hidden)
        self.Wo = nn.Linear(hidden, hidden)

    def forward(self, x):
        B, T, H = x.shape
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn   = scores.softmax(dim=-1)
        out    = (attn @ v).transpose(1, 2).reshape(B, T, H)
        return self.Wo(out), attn          # return attn weights for scoring

attn_layer = MultiHeadSelfAttention(HIDDEN, N_HEADS).to(device)

# Compute head importance = mean entropy of attention weights (lower = more focused)
x_in = torch.randn(BATCH, SEQ, HIDDEN).to(device)
_, attn_weights = attn_layer(x_in)       # (B, H, T, T)

def head_entropy(attn_w):
    # Compute mean entropy per head (lower = head is more important/focused).
    eps = 1e-9
    ent = -(attn_w * (attn_w + eps).log()).sum(dim=-1).mean(dim=(0, 2))  # (n_heads,)
    return ent.detach().cpu().numpy()

entropies = head_entropy(attn_weights)
print("Per-head entropy (lower = more focused):")
for h, e in enumerate(entropies):
    print(f"  Head {h}: {e:.4f}")

# Rank heads: prune the highest-entropy (least focused) head
prune_idx = int(np.argmax(entropies))
print(f"\\nPruning head {prune_idx} (entropy={entropies[prune_idx]:.4f})")

# Mask out head by zeroing corresponding Wv rows
with torch.no_grad():
    start = prune_idx * HEAD_DIM
    end   = start + HEAD_DIM
    attn_layer.Wv.weight[start:end, :] = 0
    attn_layer.Wv.bias[start:end] = 0

_, attn_after = attn_layer(x_in)
entropies_after = head_entropy(attn_after)
print("\\nEntropies after pruning head", prune_idx)
for h, e in enumerate(entropies_after):
    print(f"  Head {h}: {e:.4f}")
print("(Pruned head entropy becomes undefined; others unchanged)")
"""),
    md("## Real-World Example 3 — Pruning Ratio vs Accuracy/Latency Trade-off"),
    code("""# ── RW3: sparsity sweep — accuracy vs latency trade-off ─────────────────
import torch, time, copy
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# Synthetic data
X = torch.randn(800, 64)
y = (X[:, :4].sum(dim=1) > 0).long()
train_loader = DataLoader(TensorDataset(X[:600], y[:600]), batch_size=64, shuffle=True)
X_val, y_val = X[600:].to(device), y[600:].to(device)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.layers(x)

# Train base model
base = MLP().to(device)
opt = torch.optim.Adam(base.parameters(), lr=1e-3)
for _ in range(20):
    base.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        F.cross_entropy(base(xb), yb).backward()
        opt.step()

def eval_acc_latency(model, x, y, repeats=50):
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        latency_ms = (time.perf_counter() - t0) / repeats * 1000
    return acc, latency_ms

sweep_sparsities = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
print(f"{'Sparsity':>10}  {'Val Acc':>8}  {'Latency (ms)':>14}")
print("-" * 38)

for sp in sweep_sparsities:
    m = copy.deepcopy(base)
    if sp > 0:
        for layer in m.layers:
            if isinstance(layer, nn.Linear):
                prune.l1_unstructured(layer, name="weight", amount=sp)
                prune.remove(layer, "weight")
    acc, lat = eval_acc_latency(m, X_val, y_val)
    print(f"{sp:>10.0%}  {acc:>8.3f}  {lat:>14.3f}")

print("\\nNote: unstructured pruning does not reduce latency on dense hardware;")
print("structured pruning (filter/row removal) is needed for real speedup.")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# 32 — Quantization
# ──────────────────────────────────────────────────────────────────────────────
notebooks["32-quantization"] = nb([
    md("""# Quantization

## Learning Objectives
1. Implement INT8 quantization from scratch: scale, zero-point, quantize/dequantize
2. Apply PyTorch dynamic and static quantization; compare size and speed
3. Apply post-training quantization (PTQ) to a BERT-style model
4. Compare INT8 vs INT4 accuracy-compression trade-offs
"""),
    code("""import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy, time

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — INT8 Quantization from Scratch"),
    code("""# ── Level 1: manual INT8 quantize / dequantize ───────────────────────────
import numpy as np

np.random.seed(0)
W = np.random.randn(4, 8).astype(np.float32)  # simulated weight matrix
print("Original W (row 0):", W[0])

def quantize_int8(x: np.ndarray):
    \"\"\"Symmetric per-tensor INT8 quantization.\"\"\"
    scale     = np.max(np.abs(x)) / 127.0     # maps [-max, +max] → [-127, 127]
    zero_point = 0                              # symmetric: ZP = 0
    x_q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return x_q, scale, zero_point

def dequantize_int8(x_q: np.ndarray, scale: float, zero_point: int = 0):
    \"\"\"Recover approximate float from INT8.\"\"\"
    return (x_q.astype(np.float32) - zero_point) * scale

W_q, scale, zp = quantize_int8(W)
W_reconstructed = dequantize_int8(W_q, scale, zp)

# Measure quantization error
max_err  = np.max(np.abs(W - W_reconstructed))
mean_err = np.mean(np.abs(W - W_reconstructed))
print(f"\\nScale={scale:.5f}  ZeroPoint={zp}")
print(f"Quantized W_q (row 0): {W_q[0]}")
print(f"Reconstructed  (row 0): {W_reconstructed[0]}")
print(f"Max error: {max_err:.6f}  Mean error: {mean_err:.6f}")

# Asymmetric (affine) quantization
def quantize_uint8_affine(x: np.ndarray):
    \"\"\"Affine (asymmetric) UINT8: covers [x_min, x_max] → [0, 255].\"\"\"
    x_min, x_max = x.min(), x.max()
    scale     = (x_max - x_min) / 255.0
    zero_point = int(np.round(-x_min / scale))
    x_q = np.clip(np.round(x / scale + zero_point), 0, 255).astype(np.uint8)
    return x_q, scale, zero_point

W_q2, s2, zp2 = quantize_uint8_affine(W)
W_rec2 = (W_q2.astype(np.float32) - zp2) * s2
print(f"\\nAffine UINT8: scale={s2:.5f}  zp={zp2}")
print(f"Mean error affine: {np.mean(np.abs(W - W_rec2)):.6f}")
"""),
    md("## Level 2 — PyTorch Dynamic and Static Quantization"),
    code("""# ── Level 2: torch dynamic + static quantization ─────────────────────────
import torch, time, copy
import torch.nn as nn

class SimpleLM(nn.Module):
    \"\"\"Small LSTM-based model suitable for dynamic quantization.\"\"\"
    def __init__(self, vocab=100, embed=32, hidden=64, n_class=10):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm  = nn.LSTM(embed, hidden, batch_first=True)
        self.fc    = nn.Linear(hidden, n_class)

    def forward(self, x):
        e = self.embed(x)
        _, (h, _) = self.lstm(e)
        return self.fc(h.squeeze(0))

model = SimpleLM().cpu()   # quantization APIs require CPU

def model_size_mb(m):
    buf = 0
    for p in m.parameters():
        buf += p.nelement() * p.element_size()
    return buf / (1024 ** 2)

def eval_speed(m, x, repeats=200):
    m.eval()
    with torch.no_grad():
        for _ in range(10):           # warmup
            m(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            m(x)
    return (time.perf_counter() - t0) / repeats * 1000   # ms

x_in = torch.randint(0, 100, (32, 20))   # (batch=32, seq=20)

# Baseline
size_fp32 = model_size_mb(model)
speed_fp32 = eval_speed(model, x_in)
print(f"FP32   size={size_fp32:.4f} MB  speed={speed_fp32:.3f} ms")

# Dynamic quantization (LSTM + Linear)
try:
    model_dyn = torch.quantization.quantize_dynamic(
        copy.deepcopy(model),
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    speed_dyn = eval_speed(model_dyn, x_in)
    size_dyn  = model_size_mb(model_dyn)
    print(f"INT8 dynamic size={size_dyn:.4f} MB  speed={speed_dyn:.3f} ms  "
          f"speedup={speed_fp32/speed_dyn:.2f}x")
except Exception as e:
    print(f"Dynamic quantization: {e}")

# Static post-training quantization (MLP — LSTMs need special handling)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = torch.quantization.QuantStub()
        self.fc1     = nn.Linear(64, 128)
        self.fc2     = nn.Linear(128, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.dequant(x)

import torch.nn.functional as F
mlp = MLP()
mlp.eval()
mlp.qconfig = torch.quantization.get_default_qconfig("fbgemm")
torch.quantization.prepare(mlp, inplace=True)
calib_data = torch.randn(128, 64)
with torch.no_grad():
    mlp(calib_data)
torch.quantization.convert(mlp, inplace=True)
x64 = torch.randn(32, 64)
print(f"Static INT8 MLP output shape: {mlp(x64).shape}")
print("Static quantization complete — model ready for INT8 inference")
"""),
    md("## Real-World Example 1 — Post-Training Quantization on BERT-style Model"),
    code("""# ── RW1: PTQ on a BERT-style encoder ─────────────────────────────────────
import torch, copy, time
import torch.nn as nn

# Simulate a BERT encoder block (no download needed)
class BERTBlock(nn.Module):
    def __init__(self, hidden=256, n_heads=4, ff_dim=512):
        super().__init__()
        self.attn   = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm1  = nn.LayerNorm(hidden)
        self.ff     = nn.Sequential(nn.Linear(hidden, ff_dim), nn.GELU(),
                                    nn.Linear(ff_dim, hidden))
        self.norm2  = nn.LayerNorm(hidden)
        self.cls_fc = nn.Linear(hidden, 2)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return self.cls_fc(x[:, 0])    # CLS token → classification logits

model_fp32 = BERTBlock().cpu()
model_q    = copy.deepcopy(model_fp32)

# Apply dynamic quantization (Linear layers)
model_q = torch.quantization.quantize_dynamic(
    model_q, {nn.Linear}, dtype=torch.qint8
)

# Size comparison
def param_bytes(m):
    return sum(p.nelement() * p.element_size() for p in m.parameters())

sz_fp32 = param_bytes(model_fp32) / 1024
sz_q    = param_bytes(model_q)    / 1024
print(f"FP32 size: {sz_fp32:.1f} KB")
print(f"INT8 size: {sz_q:.1f} KB")
print(f"Compression: {sz_fp32/sz_q:.2f}×")

# Simulate accuracy proxy: cosine similarity between FP32 and INT8 outputs
x_test = torch.randn(16, 32, 256)   # (batch, seq_len, hidden)
with torch.no_grad():
    out_fp = model_fp32(x_test)
    out_q  = model_q(x_test)
cos_sim = torch.nn.functional.cosine_similarity(out_fp, out_q, dim=-1).mean().item()
print(f"\\nOutput cosine similarity (FP32 vs INT8): {cos_sim:.4f}")
print("(Values close to 1.0 indicate PTQ introduces minimal accuracy drop)")

# Inference speed
def latency_ms(m, x, n=100):
    with torch.no_grad():
        for _ in range(5): m(x)
        t0 = time.perf_counter()
        for _ in range(n): m(x)
    return (time.perf_counter() - t0) / n * 1000

lat_fp = latency_ms(model_fp32, x_test)
lat_q  = latency_ms(model_q,    x_test)
print(f"FP32 latency: {lat_fp:.2f} ms  |  INT8 latency: {lat_q:.2f} ms  "
      f"|  speedup: {lat_fp/lat_q:.2f}×")
"""),
    md("## Real-World Example 2 — Quantization-Aware Training (QAT)"),
    code("""# ── RW2: QAT — recover accuracy lost in PTQ ──────────────────────────────
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# Synthetic task
X = torch.randn(1000, 32)
y = (X[:, 0] - X[:, 1] > 0).long()
train_ld = DataLoader(TensorDataset(X[:800], y[:800]), batch_size=64, shuffle=True)
X_val, y_val = X[800:], y[800:]

class QMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = torch.quantization.QuantStub()
        self.fc1     = nn.Linear(32, 128)
        self.fc2     = nn.Linear(128, 64)
        self.fc3     = nn.Linear(64, 2)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.dequant(x)

def train_eval(model, epochs=15, qat=False):
    model = copy.deepcopy(model)
    if qat:
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.prepare_qat(model, inplace=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_ld:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    if qat:
        model.eval()
        torch.quantization.convert(model, inplace=True)
    model.eval()
    with torch.no_grad():
        acc = (model(X_val).argmax(1) == y_val).float().mean().item()
    return model, acc

# FP32 baseline
_, acc_fp32 = train_eval(QMLP())
print(f"FP32 training accuracy: {acc_fp32:.3f}")

# PTQ: train FP32, then quantize (no recovery)
fp32_model, _ = train_eval(QMLP())
fp32_model.eval()
fp32_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
torch.quantization.prepare(copy.deepcopy(fp32_model), inplace=True)
ptq_model = copy.deepcopy(fp32_model)
torch.quantization.prepare(ptq_model, inplace=True)
with torch.no_grad():
    ptq_model(X[:200])   # calibration
torch.quantization.convert(ptq_model, inplace=True)
ptq_model.eval()
with torch.no_grad():
    acc_ptq = (ptq_model(X_val).argmax(1) == y_val).float().mean().item()
print(f"PTQ accuracy:           {acc_ptq:.3f}")

# QAT: quantize during training
_, acc_qat = train_eval(QMLP(), qat=True)
print(f"QAT accuracy:           {acc_qat:.3f}")
print("\\nQAT typically recovers accuracy lost by PTQ on small models.")
"""),
    md("## Real-World Example 3 — INT8 vs INT4 Accuracy-Compression Trade-off"),
    code("""# ── RW3: INT8 vs INT4 — accuracy drop vs compression gain ────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# Simulate post-training weight quantization at different bit widths
def quantize_to_bits(W: torch.Tensor, bits: int) -> torch.Tensor:
    \"\"\"Symmetric uniform quantization to `bits`-bit integers.\"\"\"
    levels     = 2 ** (bits - 1) - 1              # e.g., 127 for int8, 7 for int4
    scale      = W.abs().max() / levels
    W_q        = (W / scale).round().clamp(-levels, levels)
    return W_q * scale                              # dequantized (approximate float)

# Build and train a small model
X = torch.randn(800, 32)
y = (X[:, :3].sum(1) > 0).long()
loader = DataLoader(TensorDataset(X[:600], y[:600]), batch_size=64, shuffle=True)
X_val, y_val = X[600:], y[600:]

model_fp32 = nn.Sequential(nn.Linear(32, 128), nn.ReLU(),
                            nn.Linear(128, 64),  nn.ReLU(),
                            nn.Linear(64, 2))
opt = torch.optim.Adam(model_fp32.parameters(), lr=1e-3)
for _ in range(20):
    model_fp32.train()
    for xb, yb in loader:
        opt.zero_grad()
        F.cross_entropy(model_fp32(xb), yb).backward()
        opt.step()

def apply_weight_quantization(model_src, bits):
    \"\"\"Return a copy of model with all Linear weights quantized to `bits`.\"\"\"
    import copy
    m = copy.deepcopy(model_src)
    for layer in m:
        if isinstance(layer, nn.Linear):
            layer.weight.data = quantize_to_bits(layer.weight.data, bits)
    return m

model_fp32.eval()
with torch.no_grad():
    acc_fp = (model_fp32(X_val).argmax(1) == y_val).float().mean().item()

results = [("FP32 (baseline)", acc_fp, 32)]
for bits in [16, 8, 4]:
    m_q = apply_weight_quantization(model_fp32, bits)
    m_q.eval()
    with torch.no_grad():
        acc = (m_q(X_val).argmax(1) == y_val).float().mean().item()
    results.append((f"INT{bits}", acc, bits))

print(f"{'Config':<20}  {'Val Acc':>8}  {'Bit Width':>10}  {'Compression':>12}")
print("-" * 58)
fp32_bits = 32
for name, acc, bits in results:
    compression = fp32_bits / bits
    drop = acc_fp - acc
    print(f"{name:<20}  {acc:>8.3f}  {bits:>10}  {compression:>11.1f}×  "
          f"drop={drop:+.3f}")

print("\\nINT8: ~4× compression, minimal accuracy drop.")
print("INT4: ~8× compression, but larger accuracy drop — needs QAT or calibration.")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# 33 — Regularization
# ──────────────────────────────────────────────────────────────────────────────
notebooks["33-regularization"] = nb([
    md("""# Regularization

## Learning Objectives
1. Implement L1 and L2 penalties from scratch and observe L1's sparsity effect
2. Compare L1, L2, ElasticNet, and Dropout in PyTorch on the same task
3. Understand why AdamW ≠ Adam + L2 and when AdamW matters
4. Select optimal regularization strength via train/val loss sweep
"""),
    code("""import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy, matplotlib
matplotlib.use("Agg")   # non-interactive backend for script mode
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — L1 and L2 Penalties from Scratch"),
    code("""# ── Level 1: manual L1/L2 regularisation (numpy) ─────────────────────────
import numpy as np

np.random.seed(42)
X = np.random.randn(100, 10).astype(np.float32)
# True weights: only 3 of 10 features matter
true_w = np.array([2.0, -1.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y = X @ true_w + 0.1 * np.random.randn(100)

# Gradient descent with manual L1 or L2 penalty
def gd_with_reg(X, y, lambda_=0.1, reg="l2", lr=0.01, epochs=300):
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        y_hat  = X @ w
        grad   = -2 * X.T @ (y - y_hat) / len(y)
        if reg == "l2":
            grad += 2 * lambda_ * w
        elif reg == "l1":
            grad += lambda_ * np.sign(w)
        w -= lr * grad
    return w

w_l2 = gd_with_reg(X, y, lambda_=0.1, reg="l2")
w_l1 = gd_with_reg(X, y, lambda_=0.1, reg="l1")

print("True weights:  ", np.round(true_w, 3))
print("L2 recovered:  ", np.round(w_l2, 3))
print("L1 recovered:  ", np.round(w_l1, 3))

nonzero_l2 = (np.abs(w_l2) > 1e-3).sum()
nonzero_l1 = (np.abs(w_l1) > 1e-3).sum()
print(f"\\nL2 non-zero weights: {nonzero_l2}/10  (L2 shrinks but rarely zeros)")
print(f"L1 non-zero weights: {nonzero_l1}/10  (L1 induces sparsity)")
"""),
    md("## Level 2 — L1, L2, ElasticNet, Dropout Comparison in PyTorch"),
    code("""# ── Level 2: compare regularisation strategies ────────────────────────────
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# Deliberately overfit-prone: 200 samples, 100 features
X = torch.randn(200, 100)
w_true = torch.zeros(100)
w_true[:5] = torch.tensor([3., -2., 1.5, -1., 0.5])
y = (X @ w_true + 0.3 * torch.randn(200) > 0).long()

train_ld = DataLoader(TensorDataset(X[:150], y[:150]), batch_size=32, shuffle=True)
X_val, y_val = X[150:].to(device), y[150:].to(device)

def make_model():
    return nn.Sequential(nn.Linear(100, 64), nn.ReLU(),
                         nn.Linear(64, 2)).to(device)

def l1_penalty(model, lambda_):
    return lambda_ * sum(p.abs().sum() for p in model.parameters())

def train_model(model, reg_fn=None, dropout=False, epochs=60, lr=1e-3):
    model = copy.deepcopy(model)
    if dropout:
        # insert dropout after first ReLU
        model = nn.Sequential(model[0], model[1],
                               nn.Dropout(0.5), model[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            if reg_fn:
                loss = loss + reg_fn(model)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(X_val).argmax(1) == y_val).float().mean().item()
    return acc

base_model = make_model()
configs = {
    "No reg":    (base_model, None,                                 False),
    "L2 (wd)":   (make_model(), None,                               False),  # via weight_decay
    "L1 (lam=0.001)": (make_model(), lambda m: l1_penalty(m, 1e-3), False),
    "Dropout":   (make_model(), None,                               True),
}

# L2 via weight_decay needs its own training loop
def train_l2_wd(epochs=60, wd=1e-3):
    m = make_model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=wd)
    for _ in range(epochs):
        m.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(m(xb), yb).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return (m(X_val).argmax(1) == y_val).float().mean().item()

print(f"{'Config':<22}  Val Acc")
print("-" * 34)
for name, (m, reg_fn, dp) in configs.items():
    if name == "L2 (wd)":
        acc = train_l2_wd()
    else:
        acc = train_model(m, reg_fn, dp)
    print(f"{name:<22}  {acc:.3f}")
"""),
    md("## Real-World Example 1 — AdamW vs Adam + L2"),
    code("""# ── RW1: AdamW ≠ Adam + L2  (weight decay decoupling) ────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy

torch.manual_seed(42)

# Why it matters: Adam applies adaptive scaling to the regularisation gradient,
# making L2 in Adam effectively smaller for large-gradient params — bad for LLM
# fine-tuning. AdamW decouples weight decay from the gradient update.

X = torch.randn(600, 32)
y = (X[:, 0] + X[:, 2] > 0).long()
loader = DataLoader(TensorDataset(X[:500], y[:500]), batch_size=32, shuffle=True)
X_v, y_v = X[500:].to(device), y[500:].to(device)

def make_net():
    return nn.Sequential(nn.Linear(32, 128), nn.ReLU(),
                         nn.Linear(128, 64),  nn.ReLU(),
                         nn.Linear(64, 2)).to(device)

def train_opt(opt_cls, wd, epochs=30):
    net = make_net()
    opt = opt_cls(net.parameters(), lr=1e-3, weight_decay=wd)
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(X_v).argmax(1) == y_v).float().mean().item()
    # L2-norm of all weights (lower = more regularised)
    l2 = sum(p.pow(2).sum().item() for p in net.parameters()) ** 0.5
    return acc, l2

acc_adam, l2_adam = train_opt(torch.optim.Adam,  wd=1e-2)
acc_admw, l2_admw = train_opt(torch.optim.AdamW, wd=1e-2)

print("Optimizer comparison (same weight_decay=0.01):")
print(f"  Adam  — val_acc={acc_adam:.3f}  weight L2-norm={l2_adam:.2f}")
print(f"  AdamW — val_acc={acc_admw:.3f}  weight L2-norm={l2_admw:.2f}")
print("\\nAdamW often yields lower weight norms (better decoupled regularisation).")
print("For fine-tuning large models, use AdamW — it is the standard in Hugging Face.")
"""),
    md("## Real-World Example 2 — L1 for Sparse Feature Selection"),
    code("""# ── RW2: L1 regularised logistic regression for feature selection ─────────
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)

# Synthetic data: 50 features, only 5 are informative
n_samples, n_features, n_informative = 500, 50, 5
X = np.random.randn(n_samples, n_features)
true_coef = np.zeros(n_features)
true_coef[:n_informative] = [3., -2., 1.5, -1., 0.8]
y = (X @ true_coef + 0.2 * np.random.randn(n_samples) > 0).astype(int)

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# L1 (Lasso-style) logistic regression
pipe_l1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(penalty="l1", C=0.5, solver="liblinear",
                                  max_iter=500, random_state=42))
])
pipe_l1.fit(X_train, y_train)
coef_l1 = pipe_l1.named_steps["clf"].coef_[0]
nonzero = (np.abs(coef_l1) > 1e-4).sum()

# L2 for comparison
pipe_l2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(penalty="l2", C=0.5, solver="lbfgs",
                                  max_iter=500, random_state=42))
])
pipe_l2.fit(X_train, y_train)
coef_l2 = pipe_l2.named_steps["clf"].coef_[0]

acc_l1 = pipe_l1.score(X_test, y_test)
acc_l2 = pipe_l2.score(X_test, y_test)

print(f"L1 — test acc={acc_l1:.3f}  non-zero coefficients={nonzero}/{n_features}")
print(f"L2 — test acc={acc_l2:.3f}  non-zero coefficients={(np.abs(coef_l2)>1e-4).sum()}/{n_features}")

print("\\nTop-10 absolute L1 coefficients (should align with features 0-4):")
top10 = np.argsort(np.abs(coef_l1))[::-1][:10]
for rank, idx in enumerate(top10):
    print(f"  Feature {idx:02d}: coef={coef_l1[idx]:+.4f}  "
          f"{'<-- informative' if idx < n_informative else ''}")
"""),
    md("## Real-World Example 3 — Regularization Strength Sweep"),
    code("""# ── RW3: lambda sweep — find optimal regularisation ──────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, copy

torch.manual_seed(42)

X = torch.randn(400, 20)
y = (X[:, 0] - X[:, 1] > 0).long()
loader_tr = DataLoader(TensorDataset(X[:300], y[:300]), batch_size=32, shuffle=True)
X_tr, y_tr = X[:300].to(device), y[:300].to(device)
X_val, y_val = X[300:].to(device), y[300:].to(device)

def make_net():
    return nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)

def train_sweep(reg="l2", lam=0.0, epochs=40):
    net = make_net()
    wd  = lam if reg == "l2" else 0.0
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    for _ in range(epochs):
        net.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(net(xb), yb)
            if reg == "l1":
                loss = loss + lam * sum(p.abs().sum() for p in net.parameters())
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        tr_loss  = F.cross_entropy(net(X_tr), y_tr).item()
        val_loss = F.cross_entropy(net(X_val), y_val).item()
        val_acc  = (net(X_val).argmax(1) == y_val).float().mean().item()
    return tr_loss, val_loss, val_acc

lambdas = [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]
print(f"{'Lambda':>10}  {'Reg':>5}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val Acc':>8}")
print("-" * 55)
for lam in lambdas:
    for reg in ["l2", "l1"]:
        tr, vl, va = train_sweep(reg=reg, lam=lam)
        print(f"{lam:>10.4f}  {reg:>5}  {tr:>11.4f}  {vl:>10.4f}  {va:>8.3f}")
print("\\nLook for lambda where val_loss is minimised without large train_loss gap.")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# 34 — RNNs / LSTMs
# ──────────────────────────────────────────────────────────────────────────────
notebooks["34-rnns-lstms"] = nb([
    md("""# RNNs and LSTMs

## Learning Objectives
1. Implement a vanilla RNN forward pass from scratch with numpy
2. Build a bidirectional LSTM for sequence classification in PyTorch
3. Train an LSTM for sentiment analysis on synthetic text data
4. Compare LSTM vs Transformer on sequential tasks across sequence lengths
"""),
    code("""import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — Vanilla RNN Forward Pass from Scratch"),
    code("""# ── Level 1: manual RNN (numpy) ──────────────────────────────────────────
import numpy as np

np.random.seed(42)
INPUT_DIM  = 4    # feature dimension at each time step
HIDDEN_DIM = 8    # size of hidden state
SEQ_LEN    = 6    # sequence length

# Weight matrices
Wx = np.random.randn(HIDDEN_DIM, INPUT_DIM)  * 0.1   # input → hidden
Wh = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1   # hidden → hidden
b  = np.zeros(HIDDEN_DIM)

# Sample input sequence: (seq_len, input_dim)
x_seq = np.random.randn(SEQ_LEN, INPUT_DIM)

def rnn_forward(x_seq, Wx, Wh, b, h0=None):
    \"\"\"Vanilla RNN: h_t = tanh(Wx @ x_t + Wh @ h_{t-1} + b).\"\"\"
    h = np.zeros(HIDDEN_DIM) if h0 is None else h0
    hidden_states = []
    for t, x_t in enumerate(x_seq):
        h = np.tanh(Wx @ x_t + Wh @ h + b)
        hidden_states.append(h.copy())
        print(f"  t={t}: h_t norm={np.linalg.norm(h):.4f}")
    return np.array(hidden_states), h   # all states, final state

print("RNN forward pass:")
all_h, h_final = rnn_forward(x_seq, Wx, Wh, b)
print(f"\\nOutput shape (all hidden): {all_h.shape}   "
      f"Final hidden: {all_h[-1].shape}")
print(f"Hidden state range: [{all_h.min():.3f}, {all_h.max():.3f}]  "
      "(bounded by tanh ∈ [-1, 1])")
"""),
    md("## Level 2 — Bidirectional LSTM in PyTorch"),
    code("""# ── Level 2: LSTM sequence classification (torch) ────────────────────────
import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# Synthetic: classify whether sum of even-indexed positions > 0
SEQ_LEN, VOCAB, EMBED, HIDDEN = 30, 200, 32, 64

def make_data(n=800, seq_len=SEQ_LEN):
    ids   = torch.randint(1, VOCAB, (n, seq_len))
    feats = ids.float() / VOCAB - 0.5          # normalised token values
    label = (feats[:, ::2].sum(dim=1) > 0).long()
    return ids, label

X_all, y_all = make_data(1000)
tr_ld = DataLoader(TensorDataset(X_all[:800], y_all[:800]), batch_size=64, shuffle=True)
X_v, y_v = X_all[800:].to(device), y_all[800:].to(device)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab, embed, hidden, n_class=2):
        super().__init__()
        self.embed  = nn.Embedding(vocab, embed, padding_idx=0)
        self.lstm   = nn.LSTM(embed, hidden, num_layers=2,
                               bidirectional=True, batch_first=True,
                               dropout=0.3)
        self.fc     = nn.Linear(hidden * 2, n_class)   # *2 for bidirectional

    def forward(self, x):
        e = self.embed(x)
        _, (h_n, _) = self.lstm(e)
        # Concatenate last forward + backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.fc(h)

model = BiLSTMClassifier(VOCAB, EMBED, HIDDEN).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training BiLSTM for 20 epochs...")
for epoch in range(20):
    model.train()
    for xb, yb in tr_ld:
        xb, yb = xb.to(device), yb.to(device)
        try:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM: reduce batch_size or sequence length"); break
            raise

model.eval()
with torch.no_grad():
    acc = (model(X_v).argmax(1) == y_v).float().mean().item()
print(f"\\nBiLSTM val accuracy: {acc:.3f}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
"""),
    md("## Real-World Example 1 — LSTM Sentiment Analysis"),
    code("""# ── RW1: LSTM for sentiment analysis ─────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# Synthetic sentiment: positive reviews have higher token IDs on average
VOCAB, EMBED, HIDDEN, SEQ = 500, 64, 128, 40

def gen_reviews(n=1000):
    labels = torch.randint(0, 2, (n,))
    seqs   = []
    for lab in labels:
        if lab == 1:   # positive: tokens from upper half of vocab
            tokens = torch.randint(VOCAB // 2, VOCAB, (SEQ,))
        else:          # negative: tokens from lower half
            tokens = torch.randint(1, VOCAB // 2, (SEQ,))
        seqs.append(tokens)
    return torch.stack(seqs), labels

X_all, y_all = gen_reviews(1200)
loader = DataLoader(TensorDataset(X_all[:1000], y_all[:1000]),
                    batch_size=64, shuffle=True)
X_v, y_v = X_all[1000:].to(device), y_all[1000:].to(device)

class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB, EMBED, padding_idx=0)
        self.lstm   = nn.LSTM(EMBED, HIDDEN, batch_first=True)
        self.drop   = nn.Dropout(0.4)
        self.fc     = nn.Linear(HIDDEN, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(self.embed(x))
        return self.fc(self.drop(h.squeeze(0)))

model = SentimentLSTM().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(15):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(X_v).argmax(1) == y_v).float().mean().item()
        print(f"Epoch {epoch+1:2d}  loss={total_loss/len(loader):.4f}  val_acc={acc:.3f}")
"""),
    md("## Real-World Example 2 — Time-Series Forecasting with Stacked LSTM"),
    code("""# ── RW2: stacked LSTM for next-N-step forecasting ────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Synthetic sine-wave with noise
t      = np.linspace(0, 8 * np.pi, 2000)
signal = (np.sin(t) + 0.1 * np.random.randn(len(t))).astype(np.float32)

# Sliding window: predict next PRED_STEPS from LOOK_BACK context
LOOK_BACK, PRED_STEPS = 50, 10

def make_windows(s, lb, ps):
    xs, ys = [], []
    for i in range(len(s) - lb - ps):
        xs.append(s[i:i+lb])
        ys.append(s[i+lb:i+lb+ps])
    return (torch.tensor(xs).unsqueeze(-1),   # (N, lb, 1)
            torch.tensor(ys))                 # (N, ps)

X_all, y_all = make_windows(signal, LOOK_BACK, PRED_STEPS)
split = int(len(X_all) * 0.8)
loader = DataLoader(TensorDataset(X_all[:split], y_all[:split]),
                    batch_size=64, shuffle=True)
X_v, y_v = X_all[split:].to(device), y_all[split:].to(device)

class StackedLSTM(nn.Module):
    def __init__(self, in_dim=1, hidden=64, n_layers=2, pred_steps=PRED_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=n_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden, pred_steps)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])   # last layer's final hidden state

model = StackedLSTM().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        F.mse_loss(model(xb), yb).backward()
        opt.step()

model.eval()
with torch.no_grad():
    mse = F.mse_loss(model(X_v), y_v).item()
print(f"Stacked LSTM validation MSE: {mse:.6f}")
print("(Baseline: naive persistence MSE ≈ variance of target steps)")
"""),
    md("## Real-World Example 3 — LSTM vs Transformer on Sequence Length"),
    code("""# ── RW3: LSTM vs Transformer — accuracy vs sequence length ──────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

torch.manual_seed(42)

VOCAB, EMBED, HIDDEN = 100, 32, 64

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB, EMBED)
        self.lstm = nn.LSTM(EMBED, HIDDEN, batch_first=True)
        self.fc   = nn.Linear(HIDDEN, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.fc(h.squeeze(0))

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB, EMBED)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED, nhead=4,
                                                    dim_feedforward=128,
                                                    batch_first=True)
        self.enc  = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc   = nn.Linear(EMBED, 2)
    def forward(self, x):
        out = self.enc(self.emb(x))
        return self.fc(out.mean(dim=1))   # mean pooling

def run_seq_len_experiment(seq_len, epochs=15):
    n = 600
    # Task: last token > VOCAB/2 → positive
    ids   = torch.randint(1, VOCAB, (n, seq_len))
    label = (ids[:, -1] > VOCAB // 2).long()
    ld    = DataLoader(TensorDataset(ids[:500], label[:500]),
                       batch_size=32, shuffle=True)
    Xv, yv = ids[500:].to(device), label[500:].to(device)

    results = {}
    for name, Model in [("LSTM", LSTMModel), ("Transformer", TransformerModel)]:
        m   = Model().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        t0  = time.perf_counter()
        for _ in range(epochs):
            m.train()
            for xb, yb in ld:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                F.cross_entropy(m(xb), yb).backward()
                opt.step()
        train_t = time.perf_counter() - t0
        m.eval()
        with torch.no_grad():
            acc = (m(Xv).argmax(1) == yv).float().mean().item()
        results[name] = (acc, train_t)
    return results

print(f"{'Seq Len':>8}  {'LSTM Acc':>9}  {'LSTM Time(s)':>13}  "
      f"{'Transf Acc':>11}  {'Transf Time(s)':>15}")
print("-" * 65)
for sl in [16, 64, 128, 256]:
    r = run_seq_len_experiment(sl)
    print(f"{sl:>8}  {r['LSTM'][0]:>9.3f}  {r['LSTM'][1]:>13.2f}  "
          f"{r['Transformer'][0]:>11.3f}  {r['Transformer'][1]:>15.2f}")
print("\\nTransformer typically outperforms LSTM on longer sequences (parallelism + attention).")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# 35 — Supervised Learning
# ──────────────────────────────────────────────────────────────────────────────
notebooks["35-supervised-learning"] = nb([
    md("""# Supervised Learning

## Learning Objectives
1. Implement logistic regression from scratch using gradient descent
2. Compare classifiers (LogReg, SVC, DT, RF) in a sklearn pipeline on accuracy/speed
3. Analyse feature importance with permutation, SHAP, and model-native methods
4. Visualise learning curves to identify data efficiency differences across models
"""),
    code("""import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — Logistic Regression from Scratch"),
    code("""# ── Level 1: logistic regression (numpy, gradient descent) ───────────────
import numpy as np

np.random.seed(42)

# Binary classification: 200 samples, 4 features
X  = np.random.randn(200, 4).astype(np.float32)
w_true = np.array([1.5, -1.0, 0.5, -0.2])
logits = X @ w_true
p_true = 1 / (1 + np.exp(-logits))
y      = (np.random.rand(200) < p_true).astype(np.float32)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def bce_loss(y_hat, y): return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

# Gradient descent
w = np.zeros(4)
b = 0.0
lr, epochs = 0.1, 200
losses = []
for epoch in range(epochs):
    z     = X @ w + b
    y_hat = sigmoid(z)
    loss  = bce_loss(y_hat, y)
    losses.append(loss)
    # Gradients
    dz = y_hat - y
    dw = X.T @ dz / len(y)
    db = dz.mean()
    w -= lr * dw
    b -= lr * db

preds = (sigmoid(X @ w + b) > 0.5).astype(int)
acc   = (preds == y.astype(int)).mean()
print(f"Scratch logistic regression — final loss: {losses[-1]:.4f}  accuracy: {acc:.3f}")
print(f"Learned weights: {w.round(3)}  bias: {b:.3f}")
print(f"True weights:    {w_true}  bias: 0")
"""),
    md("## Level 2 — Sklearn Classifier Comparison"),
    code("""# ── Level 2: compare classifiers with sklearn ─────────────────────────────
import numpy as np, time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                            n_redundant=5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
    "LogisticReg": Pipeline([("sc", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=500, random_state=42))]),
    "SVC (RBF)":   Pipeline([("sc", StandardScaler()),
                              ("clf", SVC(C=1.0, kernel="rbf", random_state=42))]),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
}

print(f"{'Classifier':<16}  {'Test Acc':>8}  {'Train ms':>9}  {'Infer ms':>9}")
print("-" * 50)
for name, clf in classifiers.items():
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    train_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    acc = clf.score(X_te, y_te)
    infer_ms = (time.perf_counter() - t0) * 1000

    print(f"{name:<16}  {acc:>8.3f}  {train_ms:>9.1f}  {infer_ms:>9.3f}")
"""),
    md("## Real-World Example 1 — Feature Importance Comparison"),
    code("""# ── RW1: permutation importance vs model-native importance ───────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, y = make_classification(n_samples=800, n_features=15, n_informative=6,
                            n_redundant=4, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

# Model-native importance (mean decrease in impurity — biased toward high-cardinality)
native_imp = rf.feature_importances_
top_native  = np.argsort(native_imp)[::-1][:6]

# Permutation importance (model-agnostic, evaluated on test set)
perm_result = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=42)
perm_imp    = perm_result.importances_mean
top_perm    = np.argsort(perm_imp)[::-1][:6]

print("Top-6 features by native (MDI) importance:")
for rank, idx in enumerate(top_native):
    print(f"  {rank+1}. Feature {idx:02d}: {native_imp[idx]:.4f}")

print("\\nTop-6 features by permutation importance (test-set):")
for rank, idx in enumerate(top_perm):
    print(f"  {rank+1}. Feature {idx:02d}: {perm_imp[idx]:.4f}  "
          f"(±{perm_result.importances_std[idx]:.4f})")

overlap = set(top_native) & set(top_perm)
print(f"\\nOverlap between methods: {len(overlap)}/6 features agree")
print("Use permutation importance for unbiased feature selection in production.")
"""),
    md("## Real-World Example 2 — Calibrated Probability Outputs"),
    code("""# ── RW2: Platt scaling / CalibratedClassifierCV ───────────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# SVC without calibration — doesn't naturally produce good probabilities
svc_raw = SVC(probability=True, random_state=42)
svc_raw.fit(X_tr, y_tr)

# SVC with Platt scaling calibration
svc_cal = CalibratedClassifierCV(SVC(), cv=5, method="sigmoid")
svc_cal.fit(X_tr, y_tr)

# Measure calibration: Brier score (lower = better)
from sklearn.metrics import brier_score_loss
prob_raw = svc_raw.predict_proba(X_te)[:, 1]
prob_cal = svc_cal.predict_proba(X_te)[:, 1]

brier_raw = brier_score_loss(y_te, prob_raw)
brier_cal = brier_score_loss(y_te, prob_cal)

print(f"SVC (built-in probability)  Brier score: {brier_raw:.4f}")
print(f"SVC + Platt calibration     Brier score: {brier_cal:.4f}")
print("\\nCalibrated model Brier score should be lower (better-calibrated probabilities).")
print("Use calibrated classifiers when predicted probabilities matter (risk scoring, etc.).")

# Show how fraction of positives matches predicted probability
frac_pos_raw, mean_pred_raw = calibration_curve(y_te, prob_raw, n_bins=8)
frac_pos_cal, mean_pred_cal = calibration_curve(y_te, prob_cal, n_bins=8)
print("\\nCalibration check (ideal: fraction_positives ≈ mean_predicted_value):")
print("Raw SVC:")
for fp, mp in zip(frac_pos_raw, mean_pred_raw):
    print(f"  mean_pred={mp:.2f}  frac_pos={fp:.2f}  delta={abs(fp-mp):.2f}")
"""),
    md("## Real-World Example 3 — Learning Curves: Data Efficiency"),
    code("""# ── RW3: learning curve — accuracy vs training size ──────────────────────
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            random_state=42)

estimators = {
    "LogisticReg":  Pipeline([("sc", StandardScaler()),
                               ("clf", LogisticRegression(max_iter=500))]),
    "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
    "SVC":          Pipeline([("sc", StandardScaler()),
                               ("clf", SVC(kernel="rbf"))]),
}

train_sizes = np.linspace(0.05, 0.8, 8)   # 5% to 80% of training data

print(f"{'N Train':>8}", end="")
for name in estimators:
    print(f"  {name:>12}", end="")
print()
print("-" * (8 + 16 * len(estimators)))

rows = {}
for name, est in estimators.items():
    sizes, tr_scores, val_scores = learning_curve(
        est, X, y, train_sizes=train_sizes, cv=5, scoring="accuracy",
        n_jobs=-1, verbose=0
    )
    rows[name] = (sizes, val_scores.mean(axis=1))

for i, sz in enumerate(rows["LogisticReg"][0]):
    print(f"{int(sz):>8}", end="")
    for name in estimators:
        print(f"  {rows[name][1][i]:>12.3f}", end="")
    print()

print("\\nRF needs more data to shine; LogReg is most data-efficient on this task.")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# 36 — Support Vector Machines
# ──────────────────────────────────────────────────────────────────────────────
notebooks["36-support-vector-machines"] = nb([
    md("""# Support Vector Machines (SVMs)

## Learning Objectives
1. Implement hard-margin SVM geometry — support vectors and maximum margin
2. Apply SVC with multiple kernels (linear, RBF, poly, sigmoid); tune C and gamma
3. Use LinearSVC for high-dimensional text classification with TF-IDF features
4. Visualise the kernel trick in 2D → 3D and compare SVM vs Logistic Regression
"""),
    code("""import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — Hard-Margin SVM Geometry from Scratch"),
    code("""# ── Level 1: SVM margin and support vectors (numpy) ──────────────────────
import numpy as np

np.random.seed(42)

# Linearly separable 2D data
X_pos = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])    # class +1
X_neg = np.array([[-1, -1], [-2, -2], [-1, -3], [-3, -2]])  # class -1
X = np.vstack([X_pos, X_neg]).astype(float)
y = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)

# Analytical hard-margin SVM for 2-class linearly separable 2D data
# Use sklearn for exact solution, then extract geometry
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

svm = SVC(kernel="linear", C=1e6)   # large C ≈ hard margin
svm.fit(X_sc, y)

w_vec  = svm.coef_[0]               # normal to decision boundary
b_val  = svm.intercept_[0]
sv_idx = svm.support_               # indices of support vectors

margin = 2 / np.linalg.norm(w_vec)

print("Hard-margin SVM geometry:")
print(f"  Decision boundary: {w_vec[0]:.4f}*x1 + {w_vec[1]:.4f}*x2 + {b_val:.4f} = 0")
print(f"  Margin width:      {margin:.4f}  (maximised to separate classes)")
print(f"  Support vectors ({len(sv_idx)}):")
for i in sv_idx:
    print(f"    X[{i}] = {X[i]}  y={y[i]:+.0f}  "
          f"functional margin = {y[i]*(X_sc[i] @ w_vec + b_val):.4f}  "
          f"(should be ≈ 1)")

print("\\nOnly the support vectors determine the boundary; other points are irrelevant.")
"""),
    md("## Level 2 — SVC with Multiple Kernels and Grid Search"),
    code("""# ── Level 2: kernel comparison and hyperparameter tuning ─────────────────
import numpy as np, time
from sklearn.datasets import make_circles, make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)

# Non-linearly separable data (nested circles)
X_nl, y_nl = make_circles(n_samples=400, noise=0.1, factor=0.4, random_state=42)
X_tr_nl, X_te_nl, y_tr_nl, y_te_nl = train_test_split(
    X_nl, y_nl, test_size=0.25, random_state=42)

kernels = ["linear", "rbf", "poly", "sigmoid"]
print("Kernel comparison on non-linearly separable data (nested circles):")
print(f"{'Kernel':<10}  {'Test Acc':>8}  {'Train ms':>10}")
print("-" * 35)
for k in kernels:
    pipe = Pipeline([("sc", StandardScaler()),
                     ("svm", SVC(kernel=k, C=1.0, gamma="scale", degree=3))])
    t0   = time.perf_counter()
    pipe.fit(X_tr_nl, y_tr_nl)
    tms  = (time.perf_counter() - t0) * 1000
    acc  = pipe.score(X_te_nl, y_te_nl)
    print(f"{k:<10}  {acc:>8.3f}  {tms:>10.1f}")

# Grid search for best C and gamma (RBF)
print("\\nGrid search: C × gamma for RBF kernel")
param_grid = {"svm__C": [0.1, 1.0, 10.0], "svm__gamma": ["scale", "auto", 0.01, 0.1]}
pipe_rbf = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf"))])
gs = GridSearchCV(pipe_rbf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(X_tr_nl, y_tr_nl)
print(f"Best params: {gs.best_params_}  CV acc: {gs.best_score_:.3f}")
print(f"Test acc with best params: {gs.score(X_te_nl, y_te_nl):.3f}")
"""),
    md("## Real-World Example 1 — SVM for Text Classification (TF-IDF + LinearSVC)"),
    code("""# ── RW1: high-dimensional text classification with LinearSVC ─────────────
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

np.random.seed(42)

# Synthetic "documents": two topics (tech vs sports)
tech_words   = ["model", "network", "data", "algorithm", "training",
                "tensor", "gradient", "layer", "feature", "embedding"]
sports_words = ["goal", "score", "team", "player", "match",
                "tournament", "referee", "ball", "stadium", "league"]

def make_doc(vocab, n_words=30):
    return " ".join(np.random.choice(vocab, n_words))

docs   = [make_doc(tech_words) for _ in range(600)] + \
         [make_doc(sports_words) for _ in range(600)]
labels = [0] * 600 + [1] * 600
np.random.shuffle(combined := list(zip(docs, labels)))
docs, labels = zip(*combined)

X_tr, X_te, y_tr, y_te = train_test_split(docs, labels, test_size=0.2, random_state=42)

# TF-IDF vectorisation (creates high-dimensional sparse matrix)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tr_vec = tfidf.fit_transform(X_tr)
X_te_vec  = tfidf.transform(X_te)
print(f"TF-IDF feature matrix: {X_tr_vec.shape}  "
      f"(sparse: {100 * (1 - X_tr_vec.nnz / (X_tr_vec.shape[0] * X_tr_vec.shape[1])):.1f}% zeros)")

# LinearSVC vs LogisticRegression in high-dimensional space
models = {"LinearSVC": LinearSVC(C=1.0, max_iter=2000, random_state=42),
          "LogRegL2":  LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)}

for name, clf in models.items():
    t0  = time.perf_counter()
    clf.fit(X_tr_vec, y_tr)
    tms = (time.perf_counter() - t0) * 1000
    acc = clf.score(X_te_vec, y_te)
    print(f"{name:<12}: acc={acc:.3f}  train_ms={tms:.1f}")

print("\\nLinearSVC is faster in high-dimensional sparse text spaces due to primal formulation.")
"""),
    md("## Real-World Example 2 — Kernel Trick Visualisation (2D → 3D)"),
    code("""# ── RW2: visualise RBF kernel lifting non-separable data into 3D ─────────
import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X2d, y = make_circles(n_samples=200, noise=0.05, factor=0.4, random_state=42)

# RBF feature map approximation (Nystroem) — explicitly shows the higher-dim mapping
from sklearn.kernel_approximation import Nystroem
nystroem = Nystroem(kernel="rbf", gamma=1.0, n_components=3, random_state=42)
X3d = nystroem.fit_transform(X2d)

print("Kernel trick demonstration:")
print(f"  Input space: {X2d.shape}  →  Lifted space: {X3d.shape}")

# Check separability in 2D vs lifted 3D
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

sc2d = StandardScaler(); sc3d = StandardScaler()
X2d_sc = sc2d.fit_transform(X2d)
X3d_sc = sc3d.fit_transform(X3d)

svm2d = LinearSVC(max_iter=2000, random_state=42).fit(X2d_sc, y)
svm3d = LinearSVC(max_iter=2000, random_state=42).fit(X3d_sc, y)

acc2d = svm2d.score(X2d_sc, y)
acc3d = svm3d.score(X3d_sc, y)

print(f"  Linear SVM in 2D (original):   accuracy = {acc2d:.3f}  (non-separable)")
print(f"  Linear SVM in 3D (RBF-lifted): accuracy = {acc3d:.3f}  (separable!)")
print("\\nThe kernel trick implicitly computes the inner product in the high-dim space.")
print("RBF SVC achieves the same result without explicitly constructing X3d.")

# Confirm with RBF SVC
rbf_svc = SVC(kernel="rbf", gamma=1.0, C=10)
rbf_svc.fit(X2d_sc, y)
print(f"  RBF SVC on 2D (implicit):       accuracy = {rbf_svc.score(X2d_sc, y):.3f}")
print(f"  Support vectors used:           {len(rbf_svc.support_)}")
"""),
    md("## Real-World Example 3 — SVM vs Logistic Regression on High-Dimensional Data"),
    code("""# ── RW3: SVM margin advantage with many features ─────────────────────────
import numpy as np, time
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)

print(f"{'n_features':>12}  {'LinearSVC CV':>14}  {'LogReg CV':>12}  {'SVC Time (s)':>14}")
print("-" * 58)

for n_feat in [20, 100, 500, 2000]:
    X, y = make_classification(n_samples=500, n_features=n_feat,
                               n_informative=max(5, n_feat // 10),
                               n_redundant=max(2, n_feat // 20),
                               random_state=42)

    pipe_svm = Pipeline([("sc", StandardScaler()),
                          ("clf", LinearSVC(C=1.0, max_iter=3000))])
    pipe_lr  = Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(C=1.0, solver="lbfgs",
                                                      max_iter=500))])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    t0 = time.perf_counter()
    svm_scores = cross_val_score(pipe_svm, X, y, cv=cv, scoring="accuracy")
    svm_time   = time.perf_counter() - t0
    lr_scores  = cross_val_score(pipe_lr,  X, y, cv=cv, scoring="accuracy")

    print(f"{n_feat:>12}  {svm_scores.mean():>12.3f}±{svm_scores.std():.3f}  "
          f"{lr_scores.mean():>10.3f}±{lr_scores.std():.3f}  {svm_time:>14.2f}")

print("\\nLinearSVC tends to generalise better with many features due to max-margin principle.")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# 37 — Transfer Learning
# ──────────────────────────────────────────────────────────────────────────────
notebooks["37-transfer-learning"] = nb([
    md("""# Transfer Learning

## Learning Objectives
1. Understand feature extraction: freeze backbone, train only classification head
2. Fine-tune a pretrained ResNet18 on a synthetic 5-class dataset in PyTorch
3. Apply layer-wise learning rate decay for controlled fine-tuning of BERT
4. Quantify transfer learning data efficiency vs training from scratch
"""),
    code("""import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy, time

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — Feature Extraction: Freeze Backbone"),
    code("""# ── Level 1: simulated feature extraction ────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Simulate a pretrained backbone: fixed random weights
class PretrainedBackbone(nn.Module):
    \"\"\"Frozen backbone producing 128-dim feature vectors.\"\"\"
    def __init__(self, in_dim=64, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, out_dim)
    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))

class ClassificationHead(nn.Module):
    def __init__(self, feat_dim=128, n_class=5):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_class)
    def forward(self, x): return self.fc(x)

backbone = PretrainedBackbone()
head     = ClassificationHead()

# Freeze backbone
for p in backbone.parameters():
    p.requires_grad = False

trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
head_params = sum(p.numel() for p in head.parameters())

print(f"Backbone params (frozen):     {frozen:,}")
print(f"Head params (trainable):      {head_params:,}")
print(f"Trainable ratio:              {head_params/(frozen+head_params):.1%}")
print("\\nOnly the lightweight head is updated — fast, avoids overfitting on small data.")

# Quick forward pass
x = torch.randn(16, 64)
with torch.no_grad():
    features = backbone(x)    # frozen backbone: no grad computed
logits   = head(features)     # head: grads computed
print(f"\\nInput: {x.shape}  →  Features: {features.shape}  →  Logits: {logits.shape}")
"""),
    md("## Level 2 — Fine-tuning ResNet18 Backbone in PyTorch"),
    code("""# ── Level 2: fine-tune pretrained CNN on 5-class synthetic task ───────────
import torch, copy, time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

torch.manual_seed(42)

# Synthetic image-like data (3-channel 32×32)
N_TRAIN, N_VAL, N_CLASS = 400, 100, 5
X_tr = torch.randn(N_TRAIN, 3, 32, 32)
y_tr = torch.randint(0, N_CLASS, (N_TRAIN,))
X_v  = torch.randn(N_VAL,   3, 32, 32)
y_v  = torch.randint(0, N_CLASS, (N_VAL,))
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

def make_resnet_feature_extractor(n_class=N_CLASS):
    \"\"\"Load ResNet18, freeze backbone, replace FC head.\"\"\"
    try:
        model = models.resnet18(weights=None)    # random init (no download)
    except TypeError:
        model = models.resnet18(pretrained=False)
    # Freeze all backbone layers
    for p in model.parameters():
        p.requires_grad = False
    # Replace FC head with task-specific head (trainable)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, n_class)     # only this is trainable
    return model

model = make_resnet_feature_extractor().to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,}  ({trainable/total:.1%})")

opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

for epoch in range(10):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        try:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM: reduce batch_size or image resolution"); break
            raise

model.eval()
with torch.no_grad():
    acc = (model(X_v.to(device)).argmax(1) == y_v.to(device)).float().mean().item()
print(f"Val accuracy after 10 epochs (frozen backbone): {acc:.3f}")
print("(Random init backbone — real pretrained weights would give much higher acc)")
"""),
    md("## Real-World Example 1 — BERT Fine-tuning with Frozen Lower Layers"),
    code("""# ── RW1: fine-tune upper layers only (BERT-style) ────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# Simulate a 6-layer transformer encoder (BERT-small substitute)
HIDDEN, N_LAYERS, N_CLASS = 256, 6, 2
FREEZE_LAYERS = 4     # freeze bottom 4 of 6 layers, train top 2

class TransformerBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(),
                                  nn.Linear(hidden * 4, hidden))
        self.ln1  = nn.LayerNorm(hidden)
        self.ln2  = nn.LayerNorm(hidden)
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x    = self.ln1(x + a)
        return self.ln2(x + self.ff(x))

class BERTLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb    = nn.Linear(32, HIDDEN)    # simplified: linear proj instead of embed
        self.layers = nn.ModuleList([TransformerBlock(HIDDEN) for _ in range(N_LAYERS)])
        self.cls_fc = nn.Linear(HIDDEN, N_CLASS)

    def forward(self, x):
        h = self.emb(x)
        for layer in self.layers:
            h = layer(h)
        return self.cls_fc(h[:, 0])   # CLS token

model = BERTLike().to(device)

# Freeze bottom layers
for i, layer in enumerate(model.layers):
    if i < FREEZE_LAYERS:
        for p in layer.parameters():
            p.requires_grad = False

frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Frozen:    {frozen:,}  Trainable: {trainable:,}  "
      f"({trainable/(frozen+trainable):.1%} trainable)")

# Train
X = torch.randn(600, 20, 32)   # (batch, seq_len=20, input_features=32)
y = torch.randint(0, N_CLASS, (600,))
loader = DataLoader(TensorDataset(X[:500], y[:500]), batch_size=32, shuffle=True)
X_v, y_v = X[500:].to(device), y[500:].to(device)

opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=2e-4, weight_decay=1e-2)
for _ in range(15):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        opt.step()
model.eval()
with torch.no_grad():
    acc = (model(X_v).argmax(1) == y_v).float().mean().item()
print(f"Val accuracy (top-{N_LAYERS-FREEZE_LAYERS} layers + head trainable): {acc:.3f}")
"""),
    md("## Real-World Example 2 — Layer-Wise Learning Rate Decay"),
    code("""# ── RW2: LLRD — lower LR for earlier layers, higher for later layers ──────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# Re-use BERTLike from RW1 concept
HIDDEN, N_LAYERS, N_CLASS = 128, 4, 2

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc  = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.ln  = nn.LayerNorm(d)
    def forward(self, x):
        return self.ln(x + self.fc(x))

class SmallBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb    = nn.Linear(16, HIDDEN)
        self.layers = nn.ModuleList([Block(HIDDEN) for _ in range(N_LAYERS)])
        self.fc     = nn.Linear(HIDDEN, N_CLASS)
    def forward(self, x): return self.fc(self.layers[-1](self.layers[-2](self.layers[-3](self.layers[0](self.emb(x))))))

def make_llrd_optimizer(model, base_lr=2e-4, decay=0.5):
    \"\"\"Layer-wise learning rate decay: LR multiplier = decay^(N_LAYERS - layer_idx).\"\"\"
    param_groups = []
    # Embedding: lowest LR
    param_groups.append({"params": model.emb.parameters(),
                         "lr": base_lr * (decay ** N_LAYERS)})
    # Transformer layers: increasing LR toward top
    for i, layer in enumerate(model.layers):
        lr = base_lr * (decay ** (N_LAYERS - 1 - i))
        param_groups.append({"params": layer.parameters(), "lr": lr})
    # Head: highest LR
    param_groups.append({"params": model.fc.parameters(), "lr": base_lr})
    return torch.optim.AdamW(param_groups, weight_decay=1e-2)

# Compare LLRD vs uniform LR
X  = torch.randn(600, 16)
y  = (X[:, 0] + X[:, 2] > 0).long()
ld = DataLoader(TensorDataset(X[:500], y[:500]), batch_size=32, shuffle=True)
Xv, yv = X[500:].to(device), y[500:].to(device)

def train_and_eval(opt_fn_name, epochs=20):
    import copy
    m = SmallBERT().to(device)
    if opt_fn_name == "llrd":
        opt = make_llrd_optimizer(m)
        print(f"  LLRD LRs: emb={opt.param_groups[0]['lr']:.2e}  "
              f"layer0={opt.param_groups[1]['lr']:.2e}  "
              f"head={opt.param_groups[-1]['lr']:.2e}")
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    for _ in range(epochs):
        m.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(m(xb), yb).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return (m(Xv).argmax(1) == yv).float().mean().item()

acc_uniform = train_and_eval("uniform")
acc_llrd    = train_and_eval("llrd")
print(f"\\nUniform LR:          val_acc = {acc_uniform:.3f}")
print(f"Layer-wise LR decay: val_acc = {acc_llrd:.3f}")
print("LLRD prevents lower layers from drifting too far from pretrained representations.")
"""),
    md("## Real-World Example 3 — Transfer Learning Data Efficiency"),
    code("""# ── RW3: accuracy vs #labeled examples — transfer vs scratch ─────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy

torch.manual_seed(42)

# Simulate pretrained backbone (already converged on large dataset)
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(32, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU())
    def forward(self, x): return self.enc(x)

# Pre-"train" backbone (simulate representation learning)
backbone_pretrained = Backbone()
with torch.no_grad():
    # Simulate pretrained weights with meaningful structure
    for p in backbone_pretrained.parameters():
        p.normal_(0, 0.5)

class TransferModel(nn.Module):
    def __init__(self, pretrained_backbone, freeze=True):
        super().__init__()
        self.backbone = copy.deepcopy(pretrained_backbone)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(64, 2)
    def forward(self, x): return self.head(self.backbone(x))

class ScratchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

# Full dataset for evaluation
X_all = torch.randn(1500, 32)
y_all = (X_all[:, 0] - X_all[:, 2] > 0).long()
X_val, y_val = X_all[1000:].to(device), y_all[1000:].to(device)

def train_eval(model, X_tr, y_tr, epochs=30):
    ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=min(32, len(X_tr)), shuffle=True)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(trainable_params, lr=1e-3)
    for _ in range(epochs):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return (model(X_val).argmax(1) == y_val).float().mean().item()

n_samples_list = [20, 50, 100, 200, 500]
print(f"{'N labeled':>10}  {'Transfer':>10}  {'Scratch':>10}")
print("-" * 35)
for n in n_samples_list:
    X_tr = X_all[:n].to(device)
    y_tr = y_all[:n].to(device)
    acc_transfer = train_eval(TransferModel(backbone_pretrained).to(device), X_tr, y_tr)
    acc_scratch  = train_eval(ScratchModel().to(device),                     X_tr, y_tr)
    print(f"{n:>10}  {acc_transfer:>10.3f}  {acc_scratch:>10.3f}")
print("\\nTransfer learning provides larger gains with very few labeled examples.")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# 38 — Transformers
# ──────────────────────────────────────────────────────────────────────────────
notebooks["38-transformers"] = nb([
    md("""# Transformers

## Learning Objectives
1. Build a Transformer block from scratch in numpy: attention, FFN, residual, LayerNorm
2. Train a PyTorch `nn.TransformerEncoderLayer` classifier; handle OOM on long sequences
3. Fine-tune a HuggingFace AutoModel for text classification
4. Compare sinusoidal vs learned vs no positional encoding; Transformer vs LSTM on long seqs
"""),
    code("""import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — Transformer Block from Scratch (numpy)"),
    code("""# ── Level 1: self-attention + FFN + residual + LayerNorm (numpy) ──────────
import numpy as np

np.random.seed(42)

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def self_attention(Q, K, V, d_k):
    # Scaled dot-product attention. Q, K, V: (seq, d_k)
    scores = Q @ K.T / np.sqrt(d_k)   # (seq, seq)
    attn   = softmax(scores)            # (seq, seq)
    return attn @ V, attn              # (seq, d_v), (seq, seq)

def ffn(x, W1, b1, W2, b2):
    # Two-layer feed-forward with GELU approx.
    h = np.maximum(0, x @ W1 + b1)    # ReLU (GELU approx for demo)
    return h @ W2 + b2

# Dimensions
BATCH, SEQ, D_MODEL, D_K, D_FF = 1, 6, 16, 8, 32

# Random input and weights
X = np.random.randn(SEQ, D_MODEL)
Wq = np.random.randn(D_MODEL, D_K) * 0.1
Wk = np.random.randn(D_MODEL, D_K) * 0.1
Wv = np.random.randn(D_MODEL, D_K) * 0.1
Wo = np.random.randn(D_K, D_MODEL)  * 0.1
W1 = np.random.randn(D_MODEL, D_FF) * 0.1
b1 = np.zeros(D_FF)
W2 = np.random.randn(D_FF, D_MODEL) * 0.1
b2 = np.zeros(D_MODEL)

print(f"Input X:      {X.shape}")
Q = X @ Wq;  K = X @ Wk;  V = X @ Wv
print(f"Q/K/V:        {Q.shape}  (each)")

attn_out, attn_wts = self_attention(Q, K, V, D_K)
print(f"Attention out: {attn_out.shape}")

proj = attn_out @ Wo
x1   = layer_norm(X + proj)   # residual + LayerNorm
print(f"After LN1:    {x1.shape}")

ff_out = ffn(x1, W1, b1, W2, b2)
x2     = layer_norm(x1 + ff_out)   # residual + LayerNorm
print(f"After FFN+LN2:{x2.shape}  (same as input — key property!)")
print(f"\\nAttention weights (softmax): min={attn_wts.min():.4f}  max={attn_wts.max():.4f}  sum_per_row≈{attn_wts.sum(axis=-1)}")
"""),
    md("## Level 2 — PyTorch TransformerEncoderLayer for Classification"),
    code("""# ── Level 2: nn.TransformerEncoderLayer sequence classifier ─────────────
import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

D_MODEL, N_HEADS, DIM_FF, N_LAYERS = 64, 4, 256, 3

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=200, d_model=D_MODEL, n_heads=N_HEADS,
                 dim_ff=DIM_FF, n_layers=N_LAYERS, n_class=2, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)     # learned positional encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc      = nn.Linear(d_model, n_class)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(positions)
        h = self.encoder(h)
        return self.fc(h.mean(dim=1))   # mean pooling

model = TransformerClassifier().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Transformer parameters: {total_params:,}")

# Synthetic data: classify by majority token value
SEQ_LEN = 50
X_all = torch.randint(1, 200, (800, SEQ_LEN))
y_all = (X_all.float().mean(dim=1) > 100).long()
loader = DataLoader(TensorDataset(X_all[:700], y_all[:700]), batch_size=32, shuffle=True)
X_v, y_v = X_all[700:].to(device), y_all[700:].to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(15):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        try:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM: reduce batch_size or sequence length"); break
            raise

model.eval()
with torch.no_grad():
    acc = (model(X_v).argmax(1) == y_v).float().mean().item()
print(f"\\nTransformer val accuracy: {acc:.3f}")
"""),
    md("## Real-World Example 1 — HuggingFace AutoModel Text Classification"),
    code("""# ── RW1: HuggingFace AutoModel (simulated to avoid download) ─────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# NOTE: This example shows the exact HuggingFace fine-tuning pattern.
# In production, replace SimulatedBERT with:
#   from transformers import AutoModel
#   backbone = AutoModel.from_pretrained("bert-base-uncased")

torch.manual_seed(42)
HIDDEN, N_LAYERS, N_CLASS = 256, 4, 3

class SimulatedBERT(nn.Module):
    \"\"\"Structural substitute for AutoModel — same interface, no download.\"\"\"
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=4, dim_feedforward=HIDDEN * 4,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

    def forward(self, input_ids, attention_mask=None):
        # Simulated: treat input_ids as float embeddings
        h = self.encoder(input_ids)
        # Return dict matching HuggingFace BaseModelOutput
        return {"last_hidden_state": h}

class SequenceClassifier(nn.Module):
    \"\"\"Wraps backbone — mirrors AutoModelForSequenceClassification structure.\"\"\"
    def __init__(self, backbone, n_class=N_CLASS):
        super().__init__()
        self.backbone = backbone
        self.dropout  = nn.Dropout(0.1)
        self.fc       = nn.Linear(HIDDEN, n_class)

    def forward(self, input_ids, attention_mask=None):
        outputs  = self.backbone(input_ids, attention_mask)
        cls_repr = outputs["last_hidden_state"][:, 0]   # CLS token
        return self.fc(self.dropout(cls_repr))

backbone  = SimulatedBERT()
clf_model = SequenceClassifier(backbone).to(device)

# Synthetic dataset
SEQ = 30
X   = torch.randn(600, SEQ, HIDDEN)
y   = torch.randint(0, N_CLASS, (600,))
loader = DataLoader(TensorDataset(X[:500], y[:500]), batch_size=16, shuffle=True)
X_v, y_v = X[500:].to(device), y[500:].to(device)

# Standard HuggingFace fine-tuning: AdamW + linear LR warmup
opt = torch.optim.AdamW(clf_model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)

for epoch in range(10):
    clf_model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        F.cross_entropy(clf_model(xb), yb).backward()
        torch.nn.utils.clip_grad_norm_(clf_model.parameters(), 1.0)
        opt.step()
    if epoch < 5: scheduler.step()

clf_model.eval()
with torch.no_grad():
    acc = (clf_model(X_v).argmax(1) == y_v).float().mean().item()
print(f"SimulatedBERT fine-tuning val_acc: {acc:.3f}")
print("In production: replace SimulatedBERT with AutoModel.from_pretrained(...)")
"""),
    md("## Real-World Example 2 — Positional Encoding Ablation"),
    code("""# ── RW2: sinusoidal vs learned vs no positional encoding ─────────────────
import torch, math, copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
D, SEQ, N_CLASS = 64, 40, 2

def sinusoidal_pe(max_len, d_model):
    # Classic Vaswani et al. positional encoding.
    pe  = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe   # (max_len, d_model)

class TransformerWithPE(nn.Module):
    def __init__(self, pe_type="sinusoidal"):
        super().__init__()
        self.proj    = nn.Linear(1, D)
        self.pe_type = pe_type
        if pe_type == "learned":
            self.pos = nn.Embedding(SEQ, D)
        elif pe_type == "sinusoidal":
            pe = sinusoidal_pe(SEQ, D)
            self.register_buffer("pe_buf", pe)
        enc = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=128,
                                          dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=2)
        self.fc  = nn.Linear(D, N_CLASS)

    def forward(self, x):           # x: (B, T)
        h = self.proj(x.unsqueeze(-1))   # (B, T, D)
        if self.pe_type == "learned":
            pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            h   = h + self.pos(pos)
        elif self.pe_type == "sinusoidal":
            h = h + self.pe_buf[:x.size(1)].unsqueeze(0)
        return self.fc(self.enc(h).mean(1))

# Task: first token > last token
X    = torch.randn(800, SEQ)
y    = (X[:, 0] > X[:, -1]).long()
loader = DataLoader(TensorDataset(X[:700], y[:700]), batch_size=32, shuffle=True)
Xv, yv = X[700:].to(device), y[700:].to(device)

results = {}
for pe_type in ["none", "sinusoidal", "learned"]:
    m   = TransformerWithPE(pe_type).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(20):
        m.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(m(xb), yb).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(Xv).argmax(1) == yv).float().mean().item()
    results[pe_type] = acc
    print(f"PE type={pe_type:<12}  val_acc={acc:.3f}")

best = max(results, key=results.get)
print(f"\\nBest PE: {best}  (learned and sinusoidal outperform no-PE on position-sensitive tasks)")
"""),
    md("## Real-World Example 3 — Transformer vs LSTM on Long Sequences"),
    code("""# ── RW3: Transformer vs LSTM — accuracy at sequence lengths 64→512 ────────
import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
VOCAB, EMBED, HIDDEN = 100, 32, 64

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB, EMBED)
        self.lstm = nn.LSTM(EMBED, HIDDEN, batch_first=True)
        self.fc   = nn.Linear(HIDDEN, 2)
    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.fc(h.squeeze(0))

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, EMBED)
        enc = nn.TransformerEncoderLayer(EMBED, nhead=4, dim_feedforward=128,
                                          batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, num_layers=2)
        self.fc  = nn.Linear(EMBED, 2)
    def forward(self, x):
        return self.fc(self.enc(self.emb(x)).mean(1))

def run(ModelCls, seq_len, n=500, epochs=15):
    ids   = torch.randint(1, VOCAB, (n, seq_len))
    label = (ids[:, -1] > VOCAB // 2).long()   # last-token task
    ld    = DataLoader(TensorDataset(ids[:400], label[:400]), batch_size=32, shuffle=True)
    Xv, yv = ids[400:].to(device), label[400:].to(device)
    m   = ModelCls().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    t0  = time.perf_counter()
    for _ in range(epochs):
        m.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(m(xb), yb).backward()
            opt.step()
    elapsed = time.perf_counter() - t0
    m.eval()
    with torch.no_grad():
        acc = (m(Xv).argmax(1) == yv).float().mean().item()
    return acc, elapsed

print(f"{'Seq Len':>8}  {'LSTM Acc':>9}  {'LSTM s':>7}  {'Transf Acc':>11}  {'Transf s':>9}")
print("-" * 55)
for sl in [64, 128, 256, 512]:
    acc_l, t_l = run(SimpleLSTM,         sl)
    acc_t, t_t = run(SimpleTransformer,  sl)
    print(f"{sl:>8}  {acc_l:>9.3f}  {t_l:>7.2f}  {acc_t:>11.3f}  {t_t:>9.2f}")
print("\\nTransformer scales better to long sequences; LSTM degrades due to vanishing gradients.")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# 39 — Unsupervised Learning
# ──────────────────────────────────────────────────────────────────────────────
notebooks["39-unsupervised-learning"] = nb([
    md("""# Unsupervised Learning

## Learning Objectives
1. Implement Lloyd's algorithm (k-means) from scratch: assign, update, repeat
2. Compare KMeans, DBSCAN, GaussianMixture, and AgglomerativeClustering in sklearn
3. Apply k-means to customer segmentation with the elbow method for K selection
4. Compare PCA, t-SNE, and UMAP for dimensionality reduction and 2D visualization
"""),
    code("""import numpy as np
import torch
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
import time

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — K-Means from Scratch (Lloyd's Algorithm)"),
    code("""# ── Level 1: Lloyd's algorithm (numpy) ───────────────────────────────────
import numpy as np

np.random.seed(42)

# Synthetic 2D data with 3 clusters
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
X = X.astype(np.float32)

def kmeans(X, k, max_iter=100, tol=1e-4):
    # Lloyd's algorithm: assign to nearest centroid, update, repeat.
    n, d = X.shape
    # Initialise centroids as random data points (k-means++ would be better)
    rng      = np.random.default_rng(42)
    idx      = rng.choice(n, k, replace=False)
    centers  = X[idx].copy()
    labels   = np.zeros(n, dtype=int)

    for it in range(max_iter):
        # Assignment step: closest centroid
        dists  = np.linalg.norm(X[:, None] - centers[None, :], axis=2)   # (n, k)
        new_labels = dists.argmin(axis=1)

        # Update step: recompute centroids
        new_centers = np.array([X[new_labels == c].mean(axis=0)
                                 if (new_labels == c).any() else centers[c]
                                 for c in range(k)])
        shift   = np.linalg.norm(new_centers - centers)
        centers = new_centers
        labels  = new_labels
        if it % 10 == 0:
            # Inertia: sum of squared distances to assigned centroid
            inertia = sum(((X[labels == c] - centers[c]) ** 2).sum()
                          for c in range(k))
            print(f"  iter={it:3d}  centroid_shift={shift:.6f}  inertia={inertia:.2f}")
        if shift < tol:
            print(f"  Converged at iteration {it}")
            break
    return labels, centers

print("K-means (k=3):")
labels, centers = kmeans(X, k=3)
purity = max(np.mean(labels == c) for c in range(3))  # simple proxy
print(f"\\nCluster sizes: {np.bincount(labels)}")
print(f"Centers:\\n{centers.round(2)}")
"""),
    md("## Level 2 — Clustering Algorithm Comparison with sklearn"),
    code("""# ── Level 2: KMeans vs DBSCAN vs GaussianMixture vs Agglomerative ─────────
import numpy as np, time
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

np.random.seed(42)

# Two datasets: blobs (convex) and moons (non-convex)
X_blobs, y_blobs = make_blobs(n_samples=400, centers=3, random_state=42)
X_moons, y_moons = make_moons(n_samples=400, noise=0.07, random_state=42)

scaler = StandardScaler()

configs = {
    "blobs": (scaler.fit_transform(X_blobs), y_blobs),
    "moons": (scaler.fit_transform(X_moons), y_moons),
}

algorithms = {
    "KMeans":       lambda: KMeans(n_clusters=3, random_state=42, n_init=10),
    "GaussianMix":  lambda: GaussianMixture(n_components=3, random_state=42),
    "Agglomerative":lambda: AgglomerativeClustering(n_clusters=3),
    "DBSCAN":       lambda: DBSCAN(eps=0.4, min_samples=5),
}

print(f"{'Dataset':<8}  {'Algorithm':<14}  {'ARI':>6}  {'Time ms':>8}")
print("-" * 45)
for ds_name, (X_sc, y_true) in configs.items():
    for alg_name, alg_fn in algorithms.items():
        alg = alg_fn()
        t0  = time.perf_counter()
        if hasattr(alg, "fit_predict"):
            pred = alg.fit_predict(X_sc)
        else:
            alg.fit(X_sc)
            pred = alg.predict(X_sc)
        tms = (time.perf_counter() - t0) * 1000
        ari  = adjusted_rand_score(y_true, pred)
        print(f"{ds_name:<8}  {alg_name:<14}  {ari:>6.3f}  {tms:>8.1f}")
    print()
"""),
    md("## Real-World Example 1 — Customer Segmentation with KMeans"),
    code("""# ── RW1: customer segmentation — elbow method + cluster profiling ─────────
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Synthetic e-commerce customer data
n_customers = 500
age          = np.random.normal(35, 12, n_customers).clip(18, 70)
spend_annual = np.random.exponential(1000, n_customers).clip(50, 10000)
visits_month = np.random.poisson(5, n_customers).clip(1, 30)
cart_abandon  = np.random.beta(2, 5, n_customers)   # [0, 1]

X = np.column_stack([age, spend_annual, visits_month, cart_abandon]).astype(np.float32)
feature_names = ["age", "annual_spend", "visits/month", "cart_abandon_rate"]

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# Elbow method: inertia vs k
inertias = []
k_range  = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sc)
    inertias.append(km.inertia_)
    print(f"  k={k}: inertia={km.inertia_:.1f}")

# Find elbow: largest second derivative
diffs  = np.diff(inertias)
diffs2 = np.diff(diffs)
best_k = k_range.start + np.argmax(-diffs2) + 1
print(f"\\nElbow at k={best_k}")

# Final clustering with best_k
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels   = km_final.fit_predict(X_sc)

# Profile clusters
print(f"\\nCluster profiles (original scale):")
print(f"{'Cluster':<10}", "  ".join(f"{f:<18}" for f in feature_names))
for c in range(best_k):
    mask   = labels == c
    means  = X[mask].mean(axis=0)
    print(f"  {c} (n={mask.sum():<4})",
          "  ".join(f"{v:>18.2f}" for v in means))
"""),
    md("## Real-World Example 2 — Anomaly Detection with IsolationForest"),
    code("""# ── RW2: anomaly detection — IsolationForest ─────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Synthetic normal transactions + a few anomalies
n_normal, n_anomaly = 950, 50
X_normal  = np.random.randn(n_normal, 4)                    # typical transactions
X_anomaly = np.random.uniform(-8, 8, (n_anomaly, 4))        # anomalous transactions
X = np.vstack([X_normal, X_anomaly]).astype(np.float32)
y_true = np.array([1] * n_normal + [-1] * n_anomaly)        # 1=normal, -1=anomaly

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# Sweep contamination parameter
print(f"{'Contamination':>14}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
print("-" * 45)
for contamination in [0.02, 0.05, 0.08, 0.10, 0.15]:
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    pred = iso.fit_predict(X_sc)   # 1 = inlier, -1 = outlier

    tp = ((pred == -1) & (y_true == -1)).sum()
    fp = ((pred == -1) & (y_true ==  1)).sum()
    fn = ((pred ==  1) & (y_true == -1)).sum()
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
    print(f"{contamination:>14.2f}  {prec:>10.3f}  {recall:>8.3f}  {f1:>6.3f}")

# Best model
best_iso = IsolationForest(contamination=0.05, random_state=42)
scores   = best_iso.fit(X_sc).score_samples(X_sc)   # more negative = more anomalous
print(f"\\nAnomaly score stats — normal mean: {scores[:n_normal].mean():.3f}  "
      f"anomaly mean: {scores[n_normal:].mean():.3f}")
print("More negative score = more isolated = more anomalous")
"""),
    md("## Real-World Example 3 — PCA vs t-SNE vs UMAP Comparison"),
    code("""# ── RW3: dimensionality reduction — PCA vs t-SNE vs UMAP ────────────────
import numpy as np, time
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# High-dimensional data (20-dim, 5 clusters)
X_hd, y = make_blobs(n_samples=300, n_features=20, centers=5,
                      cluster_std=2.0, random_state=42)
X_sc = StandardScaler().fit_transform(X_hd)

methods = {}

# PCA (linear)
t0 = time.perf_counter()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)
methods["PCA"] = (X_pca, time.perf_counter() - t0,
                  pca.explained_variance_ratio_.sum())

# t-SNE (nonlinear, perplexity controls neighborhood size)
t0 = time.perf_counter()
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
X_tsne = tsne.fit_transform(X_sc)
methods["t-SNE"] = (X_tsne, time.perf_counter() - t0, None)

# UMAP (optional — graceful fallback if not installed)
try:
    import umap
    t0 = time.perf_counter()
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap  = reducer.fit_transform(X_sc)
    methods["UMAP"] = (X_umap, time.perf_counter() - t0, None)
except ImportError:
    print("UMAP not installed (pip install umap-learn). Skipping.")

print(f"{'Method':<8}  {'Time (s)':>9}  {'Explained Var':>14}  {'Note'}")
print("-" * 60)
for name, (X_2d, elapsed, var) in methods.items():
    var_str = f"{var:.1%}" if var is not None else "N/A (nonlinear)"
    note = "linear, fast, interpretable" if name == "PCA" else \
           "nonlinear, slow, preserves local structure" if name == "t-SNE" else \
           "nonlinear, fast, preserves global+local"
    print(f"{name:<8}  {elapsed:>9.2f}  {var_str:>14}  {note}")

# Cluster separation proxy: within-cluster vs between-cluster distance ratio
for name, (X_2d, _, _) in methods.items():
    within = np.mean([np.var(X_2d[y == c], axis=0).sum() for c in np.unique(y)])
    overall_var = np.var(X_2d, axis=0).sum()
    ratio = within / overall_var
    print(f"{name}: within/total variance ratio = {ratio:.3f}  (lower = better separation)")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# 40 — Weight Initialization
# ──────────────────────────────────────────────────────────────────────────────
notebooks["40-weight-initialization"] = nb([
    md("""# Weight Initialization

## Learning Objectives
1. Compare Xavier, He, and random initialization — measure activation variance across layers
2. Train with different init schemes in PyTorch; compare convergence speed
3. Diagnose dead neurons from zero-init and fix with proper initialization
4. Visualise gradient flow per layer for Xavier vs He vs constant init
"""),
    code("""import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    md("## Level 1 — Activation Variance: Xavier vs He vs Random"),
    code("""# ── Level 1: activation variance across layers (numpy) ───────────────────
import numpy as np

np.random.seed(42)
N_LAYERS = 8
FAN_IN   = 256   # units per layer

def forward_activations(init_scale, activation="tanh"):
    # Propagate a random input through N_LAYERS and record activation variance.
    x = np.random.randn(1000, FAN_IN)
    variances = [float(x.var())]
    for _ in range(N_LAYERS):
        W = np.random.randn(FAN_IN, FAN_IN) * init_scale
        x = W @ x.T   # shape: (FAN_IN, 1000)
        if activation == "tanh":
            x = np.tanh(x)
        elif activation == "relu":
            x = np.maximum(0, x)
        x = x.T        # back to (1000, FAN_IN)
        variances.append(float(x.var()))
    return variances

# Xavier scale = 1/sqrt(fan_in)  — designed for tanh/sigmoid activations
# He scale     = sqrt(2/fan_in)  — designed for ReLU activations
# Large scale  = 1.0             — leads to exploding/saturating
# Small scale  = 0.01            — leads to vanishing

inits = {
    "Xavier (tanh)": (1 / np.sqrt(FAN_IN), "tanh"),
    "He (relu)":     (np.sqrt(2 / FAN_IN), "relu"),
    "Too large":     (1.0,                  "tanh"),
    "Too small":     (0.01,                 "tanh"),
}

print(f"{'Init':<20}  {'Layer 0':>8}", "  ".join(f"L{i:1d}" for i in range(1, N_LAYERS + 1)))
for name, (scale, act) in inits.items():
    variances = forward_activations(scale, act)
    row = f"{name:<20}  {variances[0]:>8.4f}"
    for v in variances[1:]:
        row += f"  {v:5.4f}"
    print(row)

print("\\nXavier/He maintain stable variance; large scale → explode; small → vanish.")
"""),
    md("## Level 2 — PyTorch Init Schemes and Loss Curve Comparison"),
    code("""# ── Level 2: compare init schemes on a real training task ────────────────
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

X = torch.randn(600, 64)
y = (X[:, :4].sum(1) > 0).long()
loader = DataLoader(TensorDataset(X[:500], y[:500]), batch_size=32, shuffle=True)
X_v, y_v = X[500:].to(device), y[500:].to(device)

def make_deep_net():
    return nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 2)
    )

def apply_init(model, scheme):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif scheme == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "constant_01":
                nn.init.constant_(m.weight, 0.01)
            elif scheme == "large_random":
                nn.init.normal_(m.weight, std=1.5)
            nn.init.zeros_(m.bias)
    return model

def train_and_track(scheme, epochs=30):
    m   = apply_init(make_deep_net(), scheme).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    losses = []
    for _ in range(epochs):
        m.train()
        ep_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = F.cross_entropy(m(xb), yb)
            l.backward()
            opt.step()
            ep_loss += l.item()
        losses.append(ep_loss / len(loader))
    m.eval()
    with torch.no_grad():
        acc = (m(X_v).argmax(1) == y_v).float().mean().item()
    return losses, acc

schemes = ["xavier", "kaiming", "constant_01", "large_random"]
print(f"{'Scheme':<16}  {'Epoch 1 Loss':>13}  {'Final Loss':>11}  {'Val Acc':>8}")
print("-" * 55)
for sch in schemes:
    losses, acc = train_and_track(sch)
    print(f"{sch:<16}  {losses[0]:>13.4f}  {losses[-1]:>11.4f}  {acc:>8.3f}")
"""),
    md("## Real-World Example 1 — Dead Neurons from Zero Init"),
    code("""# ── RW1: diagnose dead neurons caused by bad initialisation ───────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

X = torch.randn(400, 32)
y = (X[:, 0] + X[:, 2] > 0).long()
loader = DataLoader(TensorDataset(X[:300], y[:300]), batch_size=32, shuffle=True)
X_v, y_v = X[300:].to(device), y[300:].to(device)

class NetWithStats(nn.Module):
    # Network that records per-layer activation statistics.
    def __init__(self, init_scheme):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        self._init(init_scheme)
        self.activation_stats = {}

    def _init(self, scheme):
        for fc in [self.fc1, self.fc2, self.fc3]:
            if scheme == "zeros":
                nn.init.zeros_(fc.weight)
                nn.init.zeros_(fc.bias)
            elif scheme == "kaiming":
                nn.init.kaiming_normal_(fc.weight, nonlinearity="relu")
                nn.init.zeros_(fc.bias)

    def forward(self, x):
        for i, (layer, name) in enumerate([(self.fc1, "fc1"), (self.fc2, "fc2")]):
            x = F.relu(layer(x))
            with torch.no_grad():
                dead_frac = (x == 0).float().mean().item()
                self.activation_stats[name] = {
                    "mean": x.mean().item(),
                    "std":  x.std().item(),
                    "dead_frac": dead_frac      # fraction of zero activations
                }
        return self.fc3(x)

for init in ["zeros", "kaiming"]:
    net = NetWithStats(init).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(10):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
    # Collect stats on val set
    net.eval()
    with torch.no_grad():
        net(X_v)
        acc = (net(X_v).argmax(1) == y_v).float().mean().item()
    print(f"\\nInit={init}  val_acc={acc:.3f}")
    for layer_name, stats in net.activation_stats.items():
        print(f"  {layer_name}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"dead_frac={stats['dead_frac']:.2%}")
    if init == "zeros":
        print("  → All neurons learn identically (symmetry problem) — zero init is broken")
"""),
    md("## Real-World Example 2 — Small Init for Deep ResNets"),
    code("""# ── RW2: scaled init (1/sqrt(depth)) for deep residual networks ──────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy, math

torch.manual_seed(42)

X = torch.randn(500, 32)
y = (X[:, 0] - X[:, 1] > 0).long()
loader = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=32, shuffle=True)
X_v, y_v = X[400:].to(device), y[400:].to(device)

N_BLOCKS = 6

class ResBlock(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.fc1    = nn.Linear(dim, dim)
        self.fc2    = nn.Linear(dim, dim)
        self.scale  = scale   # multiplier on residual branch
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        # Scale output of residual branch to prevent explosion in deep networks
        nn.init.normal_(self.fc2.weight, std=self.scale / math.sqrt(N_BLOCKS))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(x)))   # residual connection

class ResNet(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.proj   = nn.Linear(32, 64)
        self.blocks = nn.ModuleList([ResBlock(64, scale) for _ in range(N_BLOCKS)])
        self.fc     = nn.Linear(64, 2)
    def forward(self, x):
        h = F.relu(self.proj(x))
        for b in self.blocks: h = b(h)
        return self.fc(h)

def train_resnet(scale, epochs=30):
    net = ResNet(scale).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    init_loss = None
    for ep in range(epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(net(xb), yb)
            if init_loss is None: init_loss = loss.item()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(X_v).argmax(1) == y_v).float().mean().item()
    return init_loss, acc

print(f"{'Scale':>8}  {'Init Loss':>10}  {'Val Acc':>8}  Note")
print("-" * 50)
for sc in [0.01, 0.1, 0.5, 1.0, 2.0]:
    il, acc = train_resnet(sc)
    note = "too small (slow start)" if sc < 0.05 else \
           "too large (explodes)"   if sc > 1.5  else "good"
    print(f"{sc:>8.2f}  {il:>10.4f}  {acc:>8.3f}  {note}")
print("\\nScale ≈ 1/sqrt(depth) avoids signal explosion while maintaining gradient flow.")
"""),
    md("## Real-World Example 3 — Gradient Flow Visualisation per Layer"),
    code("""# ── RW3: gradient norm per layer for different init schemes ──────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

torch.manual_seed(42)

X = torch.randn(200, 32)
y = (X[:, 0] > 0).long()

def make_net_for_grad_flow(init_scheme):
    net = nn.Sequential(
        nn.Linear(32, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 2)
    )
    linear_layers = [m for m in net if isinstance(m, nn.Linear)]
    for fc in linear_layers:
        if init_scheme == "xavier":
            nn.init.xavier_uniform_(fc.weight)
        elif init_scheme == "kaiming":
            nn.init.kaiming_normal_(fc.weight, nonlinearity="relu")
        elif init_scheme == "constant":
            nn.init.constant_(fc.weight, 0.1)
        nn.init.zeros_(fc.bias)
    return net

def get_gradient_norms(net, X, y):
    # Compute gradient norm at each Linear layer for a single forward-backward pass.
    net = net.to(device)
    X_d, y_d = X.to(device), y.to(device)
    out  = net(X_d)
    loss = F.cross_entropy(out, y_d)
    net.zero_grad()
    loss.backward()
    norms = []
    for m in net:
        if isinstance(m, nn.Linear):
            norms.append(m.weight.grad.norm().item())
    return norms

init_schemes = ["xavier", "kaiming", "constant"]
layer_names  = [f"L{i+1}" for i in range(5)]

print(f"{'Init':<10}  " + "  ".join(f"{n:>10}" for n in layer_names))
print("-" * 65)
for sch in init_schemes:
    net   = make_net_for_grad_flow(sch)
    norms = get_gradient_norms(net, X, y)
    row   = f"{sch:<10}  " + "  ".join(f"{n:>10.6f}" for n in norms)
    print(row)

print("\\nKey insight:")
print("  Xavier/Kaiming: gradient norms similar across layers (stable flow)")
print("  Constant init:  symmetry breaking fails, gradients may vanish or explode")
print("  Vanishing: L1 norm >> L5 norm — gradients lost before reaching early layers")
"""),
])

# ──────────────────────────────────────────────────────────────────────────────
# Write all notebooks
# ──────────────────────────────────────────────────────────────────────────────
import sys

errors = []
for name, notebook in notebooks.items():
    path = os.path.join(OUT_DIR, f"{name}.ipynb")
    try:
        with open(path, "w") as f:
            json.dump(notebook, f, indent=1)
        n_cells = len(notebook["cells"])
        # Validate
        assert n_cells == 12, f"Expected 12 cells, got {n_cells}"
        # Syntax-check all code cells
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                src = cell["source"]
                if isinstance(src, list):
                    src = "".join(src)
                compile(src, f"{name}:cell{i}", "exec")
        print(f"  OK  {name}.ipynb  ({n_cells} cells)")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"  FAIL {name}: {e}", file=sys.stderr)

if errors:
    print(f"\n{len(errors)} error(s):", file=sys.stderr)
    for n, e in errors: print(f"  {n}: {e}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"\nAll {len(notebooks)} notebooks written successfully.")
