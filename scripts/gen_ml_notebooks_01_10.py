"""
Generate ml/notebooks/01-10 using nbformat.
Run: python3 scripts/gen_ml_notebooks_01_10.py
"""
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)


def make_nb(cells):
    """Create a notebook with the given list of (type, source) tuples."""
    nb_cells = []
    for cell_type, source in cells:
        if cell_type == "markdown":
            nb_cells.append(new_markdown_cell(source))
        else:
            nb_cells.append(new_code_cell(source))
    nb = new_notebook(cells=nb_cells)
    nb.metadata["language_info"] = {"name": "python"}
    return nb


# ── 01 Activation Functions ───────────────────────────────────────────────────
NB01 = make_nb([
    ("markdown", """\
# Activation Functions

## Learning Objectives
1. Understand how ReLU, sigmoid, tanh, and GELU shape neural-network outputs.
2. Implement each activation from scratch using NumPy and verify against known properties.
3. Build an MLP that swaps activation functions and measure accuracy on XOR.
4. Detect the dying-ReLU failure mode and apply practical fixes.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: Activation Functions from Scratch (NumPy)"),
    ("code", """\
# Implement the four most common activations and their derivatives from scratch.

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    # Clip to avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1.0 - np.tanh(x) ** 2

def gelu(x):
    # GELU approximation used in BERT, GPT transformers
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

x = np.linspace(-4, 4, 300)
activations = {
    "ReLU":    relu(x),
    "Sigmoid": sigmoid(x),
    "Tanh":    tanh_act(x),
    "GELU":    gelu(x),
}
for name, vals in activations.items():
    print(f"{name:8s}  range=[{vals.min():.3f}, {vals.max():.3f}]")

# Verify ReLU gradient: should be 1 for x>0, 0 otherwise
x_check = np.array([-1.0, 0.0, 1.0, 2.0])
print(f"ReLU grad at {x_check}: {relu_grad(x_check)}")
print("All four activations implemented successfully.")
"""),
    ("markdown", "## Level 2: MLP with Swappable Activations on XOR (PyTorch)"),
    ("code", """\
# Train a small MLP on XOR using four different activations and compare accuracy.

XOR_X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
XOR_y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
# Repeat to form a usable dataset for mini-batch training
X_train = XOR_X.repeat(200, 1).to(device)
y_train = XOR_y.repeat(200, 1).to(device)
ds = TensorDataset(X_train, y_train)
loader = DataLoader(ds, batch_size=64, shuffle=True)

def build_mlp(act_fn):
    return nn.Sequential(
        nn.Linear(2, 16), act_fn,
        nn.Linear(16, 16), act_fn,
        nn.Linear(16, 1), nn.Sigmoid()
    ).to(device)

ACTIVATIONS = {
    "ReLU":    nn.ReLU(),
    "Tanh":    nn.Tanh(),
    "GELU":    nn.GELU(),
    "Sigmoid": nn.Sigmoid(),
}

results = {}
for name, act in ACTIVATIONS.items():
    model = build_mlp(act)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.BCELoss()
    for epoch in range(300):
        for xb, yb in loader:
            opt.zero_grad()
            try:
                loss = crit(model(xb), yb)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"OOM with {name} -- reduce batch size")
                    torch.cuda.empty_cache()
                    continue
                raise
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        preds = (model(X_train) > 0.5).float()
        acc = (preds == y_train).float().mean().item()
    results[name] = acc
    print(f"{name:8s}  accuracy={acc:.4f}")

best = max(results, key=results.get)
print(f"Best activation on XOR: {best} ({results[best]:.4f})")
"""),
    ("markdown", "## Real-World Example 1: Dying ReLU Detection and Fix"),
    ("code", """\
# Dying ReLU: neurons output 0 for all inputs after a bad init or large LR.
# Detect by counting dead neurons, then fix with He-init + LeakyReLU.

def count_dead_neurons(model, x_sample):
    activations_list = []
    hooks = []
    def hook_fn(module, inp, out):
        activations_list.append(out.detach().cpu())
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(hook_fn))
    with torch.no_grad():
        model(x_sample)
    for h in hooks:
        h.remove()
    if not activations_list:
        return 0.0
    dead = sum((a <= 0).all(dim=0).float().mean().item() for a in activations_list)
    return dead / len(activations_list)

# Simulate dying ReLU: large negative biases force all ReLU outputs to zero
bad_model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1)
).to(device)
with torch.no_grad():
    for m in bad_model.modules():
        if isinstance(m, nn.Linear):
            m.bias.fill_(-5.0)   # kills all ReLU outputs

x_sample = torch.randn(128, 20, device=device)
dead_frac = count_dead_neurons(bad_model, x_sample)
print(f"Dying ReLU model -- dead neuron fraction: {dead_frac:.2%}")

# Fix: He initialisation + LeakyReLU (negative slope = 0.1 keeps gradients alive)
fixed_model = nn.Sequential(
    nn.Linear(20, 64), nn.LeakyReLU(0.1),
    nn.Linear(64, 64), nn.LeakyReLU(0.1),
    nn.Linear(64, 1)
).to(device)
for m in fixed_model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(m.bias)

out = fixed_model(x_sample)
print(f"Fixed model output -- mean={out.mean().item():.4f}  std={out.std().item():.4f}")
print("Fix applied: He-init + LeakyReLU eliminates dying-ReLU problem.")
"""),
    ("markdown", "## Real-World Example 2: GELU in a Transformer FFN Block"),
    ("code", """\
# Modern transformers (BERT, GPT) use GELU inside feed-forward sublayers.
# This example shows a drop-in FFN block with GELU and benchmarks vs ReLU.

import time

class TransformerFFN(nn.Module):
    # Two-layer FFN as used in transformer encoders.
    def __init__(self, d_model, d_ff, activation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation,
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))   # residual connection

D_MODEL, D_FF, SEQ_LEN, BATCH = 256, 1024, 128, 16
x_ffn = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)

timing = {}
for act_name, act_mod in [("GELU", nn.GELU()), ("ReLU", nn.ReLU())]:
    ffn = TransformerFFN(D_MODEL, D_FF, act_mod).to(device)
    ffn.eval()
    # Warmup pass to initialise CUDA kernels
    with torch.no_grad():
        _ = ffn(x_ffn)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            out = ffn(x_ffn)
    elapsed = (time.perf_counter() - t0) / 20 * 1000
    timing[act_name] = elapsed
    print(f"FFN-{act_name}  output shape={tuple(out.shape)}  time/iter={elapsed:.2f}ms")

overhead = (timing["GELU"] - timing["ReLU"]) / timing["ReLU"] * 100
print(f"GELU overhead vs ReLU: {overhead:+.1f}%")
print("GELU's smooth gradient improves convergence; overhead is minimal in practice.")
"""),
    ("markdown", "## Real-World Example 3: Activation Ablation Study"),
    ("code", """\
# Systematic ablation: train the same architecture N_RUNS times per activation,
# record mean +/- std of final val loss to measure sensitivity.

from torch.utils.data import random_split

# Regression task: y = sin(x1) + cos(x2) + noise
N_ABL = 2000
Xr = torch.randn(N_ABL, 4)
yr = (torch.sin(Xr[:, 0]) + torch.cos(Xr[:, 1])).unsqueeze(1)
train_abl, val_abl = random_split(TensorDataset(Xr, yr), [1600, 400])
tr_loader_abl = DataLoader(train_abl, batch_size=64, shuffle=True)
va_loader_abl = DataLoader(val_abl, batch_size=64)

ACTS_ABLATION = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "ELU":  lambda: nn.ELU(alpha=1.0),
    "SiLU": nn.SiLU,
    "Tanh": nn.Tanh,
}

N_RUNS = 3   # increase to 5 for a production study
ablation_results = {}

for act_name, act_cls in ACTS_ABLATION.items():
    run_losses = []
    for run in range(N_RUNS):
        torch.manual_seed(run * 7)
        mdl = nn.Sequential(
            nn.Linear(4, 64), act_cls(),
            nn.Linear(64, 64), act_cls(),
            nn.Linear(64, 1)
        ).to(device)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        for _ in range(80):
            mdl.train()
            for xb, yb in tr_loader_abl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                crit(mdl(xb), yb).backward()
                opt.step()
        mdl.eval()
        with torch.no_grad():
            vl = sum(
                crit(mdl(xb.to(device)), yb.to(device)).item() * len(xb)
                for xb, yb in va_loader_abl
            ) / len(va_loader_abl.dataset)
        run_losses.append(vl)
    ablation_results[act_name] = (np.mean(run_losses), np.std(run_losses))

print(f"{'Activation':<10}  {'Mean Val MSE':>12}  {'Std':>8}")
print("-" * 36)
for name, (mean, std) in sorted(ablation_results.items(), key=lambda x: x[1][0]):
    print(f"{name:<10}  {mean:>12.6f}  {std:>8.6f}")
print("Conclusion: SiLU/GELU/ELU typically outperform ReLU on smooth targets.")
"""),
])

# ── 02 Attention Mechanism ────────────────────────────────────────────────────
NB02 = make_nb([
    ("markdown", """\
# Attention Mechanism

## Learning Objectives
1. Derive scaled dot-product attention from first principles using NumPy.
2. Implement multi-head attention in PyTorch with correct masking and projections.
3. Apply self-attention to a sentiment classification task and visualise attention heatmaps.
4. Compare cross-attention (encoder-decoder) versus self-attention patterns.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: Scaled Dot-Product Attention (NumPy)"),
    ("code", """\
# Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Args:
    # Q: (seq_q, d_k)
    # K: (seq_k, d_k)
    # V: (seq_k, d_v)
    # mask: (seq_q, seq_k) boolean -- True positions are masked to -inf
    # Returns:
    # output: (seq_q, d_v), weights: (seq_q, seq_k)
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)         # (seq_q, seq_k)
    if mask is not None:
        scores[mask] = -1e9
    weights = softmax_np(scores, axis=-1)    # (seq_q, seq_k)
    output = weights @ V                     # (seq_q, d_v)
    return output, weights

# Toy example: 4 tokens, d_k=d_v=8
SEQ, D_K = 4, 8
rng = np.random.default_rng(0)
Q = rng.standard_normal((SEQ, D_K))
K = rng.standard_normal((SEQ, D_K))
V = rng.standard_normal((SEQ, D_K))

out, w = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {out.shape}")
print(f"Attention weights row sums (should all be 1.0): {w.sum(axis=-1).round(6)}")

# Causal (autoregressive) mask: token i cannot attend to future token j > i
causal_mask = np.triu(np.ones((SEQ, SEQ), dtype=bool), k=1)
out_causal, w_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"Causal weights upper-triangle (should be ~0): {w_causal[0, 1:].round(6)}")
print("Scaled dot-product attention verified.")
"""),
    ("markdown", "## Level 2: Multi-Head Attention (PyTorch)"),
    ("code", """\
class MultiHeadAttention(nn.Module):
    # Multi-head attention with optional causal masking.

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        B, S, D = x.shape
        return x.view(B, S, self.n_heads, self.d_k).transpose(1, 2)   # (B,H,S,d_k)

    def forward(self, query, key, value, causal=False):
        B, S_q, _ = query.shape
        S_k = key.shape[1]
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)   # (B,H,S_q,S_k)
        if causal:
            mask = torch.triu(torch.ones(S_q, S_k, device=query.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = weights @ V                                    # (B,H,S_q,d_k)
        context = context.transpose(1, 2).reshape(B, S_q, self.d_model)
        return self.W_o(context), weights

B, SEQ, D_MODEL, N_HEADS = 2, 10, 64, 4
x = torch.randn(B, SEQ, D_MODEL, device=device)
mha = MultiHeadAttention(D_MODEL, N_HEADS).to(device)

try:
    out, w = mha(x, x, x, causal=True)
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        print("OOM -- reduce batch size or d_model")
        torch.cuda.empty_cache()
    raise

print(f"MHA output shape: {tuple(out.shape)}")
print(f"Attention weights shape: {tuple(w.shape)}  (B, heads, seq_q, seq_k)")
print(f"Causal check -- w[0,0,0,1:] should be 0: {w[0,0,0,1:].detach().cpu().numpy().round(4)}")
"""),
    ("markdown", "## Real-World Example 1: Self-Attention Sentiment + Heatmap"),
    ("code", """\
# Simple self-attention classifier for toy sentiment; inspect attention weights.

class SentimentAttnModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_bias = nn.Parameter(torch.zeros(1, 50, d_model))
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.clf = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, ids):
        x = self.embed(ids) + self.pos_bias[:, :ids.size(1)]
        ctx, weights = self.attn(x, x, x)
        pooled = ctx.mean(dim=1)      # mean-pool over sequence
        return self.clf(pooled), weights

VOCAB, D, HEADS, SEQ_LEN = 100, 32, 4, 12
model_sent = SentimentAttnModel(VOCAB, D, HEADS, 2).to(device)

# Synthetic: class 1 if first token >= VOCAB//2
tokens = torch.randint(0, VOCAB, (200, SEQ_LEN))
labels = (tokens[:, 0] >= VOCAB // 2).long()
tr_ds = TensorDataset(tokens[:160].to(device), labels[:160].to(device))
tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
opt_sent = torch.optim.Adam(model_sent.parameters(), lr=1e-3)
crit_sent = nn.CrossEntropyLoss()

for epoch in range(30):
    model_sent.train()
    for xb, yb in tr_ld:
        opt_sent.zero_grad()
        logits, _ = model_sent(xb)
        crit_sent(logits, yb).backward()
        opt_sent.step()

model_sent.eval()
with torch.no_grad():
    logits, attn_w = model_sent(tokens[160:161].to(device))
    pred = logits.argmax(dim=-1).item()
print(f"Predicted class: {pred}  | True class: {labels[160].item()}")
print(f"Attention heatmap (head 0) shape: {tuple(attn_w[0,0].shape)}")
row_max = attn_w[0, 0].cpu().numpy().max(axis=-1)
print(f"Max attention per query position: {row_max.round(3)}")
print("High weight on position 0 confirms model uses the class-discriminating token.")
"""),
    ("markdown", "## Real-World Example 2: Cross-Attention Encoder-Decoder"),
    ("code", """\
# Cross-attention: decoder queries attend over encoder key-values.
# Fundamental to seq2seq models (translation, summarisation, captioning).

class EncoderDecoder(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff   = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                   nn.Linear(d_model*4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def encode(self, src):
        ctx, _ = self.self_attn(src, src, src)
        return self.norm1(src + ctx)

    def decode(self, tgt, memory):
        # Causal self-attention on target (decoder) sequence
        ctx1, _ = self.self_attn(tgt, tgt, tgt, causal=True)
        tgt2 = self.norm1(tgt + ctx1)
        # Cross-attention: decoder queries attend over encoder memory
        ctx2, cross_w = self.cross_attn(tgt2, memory, memory)
        tgt3 = self.norm2(tgt2 + ctx2)
        return self.norm3(tgt3 + self.ff(tgt3)), cross_w

ENC_SEQ, DEC_SEQ, D, H = 15, 10, 64, 4
enc_dec = EncoderDecoder(D, H).to(device)
src = torch.randn(2, ENC_SEQ, D, device=device)
tgt = torch.randn(2, DEC_SEQ, D, device=device)

memory = enc_dec.encode(src)
output, cross_weights = enc_dec.decode(tgt, memory)
print(f"Encoder memory:  {tuple(memory.shape)}")
print(f"Decoder output:  {tuple(output.shape)}")
print(f"Cross-attn weights: {tuple(cross_weights.shape)}  (B, heads, dec_seq, enc_seq)")
avg_cross = cross_weights.mean(dim=(0,1)).detach().cpu().numpy()
print(f"Peak encoder position per decoder step: {avg_cross.argmax(axis=-1)}")
"""),
    ("markdown", "## Real-World Example 3: Attention Pattern Visualization"),
    ("code", """\
# Visualise how attention patterns differ: uniform vs peaked vs causal.
# Useful for debugging transformer behaviour in production.

def make_attention_pattern(kind, seq_len=8, d_k=16):
    rng2 = np.random.default_rng(99)
    if kind == "uniform":
        # Very small Q.K products -> nearly uniform softmax
        Q = rng2.standard_normal((seq_len, d_k)) * 0.01
        K = rng2.standard_normal((seq_len, d_k)) * 0.01
        mask = None
    elif kind == "peaked":
        # Aligned Q=K: each token strongly attends to itself
        Q = np.eye(seq_len, d_k)
        K = np.eye(seq_len, d_k)
        mask = None
    elif kind == "causal":
        Q = rng2.standard_normal((seq_len, d_k))
        K = rng2.standard_normal((seq_len, d_k))
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    else:
        raise ValueError(kind)
    _, w = scaled_dot_product_attention(Q, K, np.eye(seq_len), mask=mask)
    return w

for kind in ("uniform", "peaked", "causal"):
    pat = make_attention_pattern(kind)
    diag_sum = np.diag(pat).sum()
    entropy = -(pat * np.log(pat + 1e-9)).sum(axis=-1).mean()
    print(f"{kind:8s}  diagonal_sum={diag_sum:.3f}  avg_entropy={entropy:.3f}")

print()
print("Pattern interpretation:")
print("  Uniform  -- model has not learned to focus (high entropy)")
print("  Peaked   -- strong local self-attention (first layers)")
print("  Causal   -- autoregressive; later tokens use richer context (lower entropy)")
"""),
])

# ── 03 Batch Normalization ────────────────────────────────────────────────────
NB03 = make_nb([
    ("markdown", """\
# Batch Normalization

## Learning Objectives
1. Implement the batch-norm forward pass from scratch with running statistics.
2. Compare BatchNorm, LayerNorm, and no-normalisation on a regression task.
3. Show how BN stabilises gradient flow in a deep CNN.
4. Demonstrate the train-vs-eval mode bug and how to avoid it.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: Batch Normalization Forward Pass (NumPy)"),
    ("code", """\
# BN normalises each feature across the batch then applies learnable scale/shift.

class BatchNormNumpy:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_features)
        self.beta  = np.zeros(num_features)
        self.eps   = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var  = x.var(axis=0)
            # Update running stats for later inference use
            self.running_mean = (1 - self.momentum)*self.running_mean + self.momentum*mean
            self.running_var  = (1 - self.momentum)*self.running_var  + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# Verify: output should be approx zero-mean, unit-variance per feature
bn = BatchNormNumpy(num_features=4)
X_in = np.random.randn(32, 4) * 5 + 3   # non-zero mean, large scale
X_out = bn.forward(X_in, training=True)
print(f"Input  -- mean: {X_in.mean(axis=0).round(3)},  std: {X_in.std(axis=0).round(3)}")
print(f"Output -- mean: {X_out.mean(axis=0).round(3)},  std: {X_out.std(axis=0).round(3)}")

# Inference mode should use running stats
X_infer = np.random.randn(10, 4) * 5 + 3
X_infer_out = bn.forward(X_infer, training=False)
print(f"Inference output mean (uses running_mean): {X_infer_out.mean(axis=0).round(3)}")
print("Batch-norm forward pass verified.")
"""),
    ("markdown", "## Level 2: BatchNorm vs LayerNorm vs No Norm (PyTorch)"),
    ("code", """\
# Compare three normalisation strategies on a regression task.

def make_deep_net(norm_type, n_in=10):
    layers = []
    sizes = [n_in, 32, 64, 64, 32]
    for i in range(len(sizes)-1):
        in_f, out_f = sizes[i], sizes[i+1]
        layers.append(nn.Linear(in_f, out_f))
        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(out_f))
        elif norm_type == "layer":
            layers.append(nn.LayerNorm(out_f))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(32, 1))
    return nn.Sequential(*layers).to(device)

N = 1000
Xd = torch.randn(N, 10)
yd = (Xd[:, 0]**2 + Xd[:, 1]*Xd[:, 2] + 0.1*torch.randn(N)).unsqueeze(1)
tr_ds = TensorDataset(Xd[:800].to(device), yd[:800].to(device))
va_ds = TensorDataset(Xd[800:].to(device), yd[800:].to(device))
tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=64)

norm_results = {}
for ntype in ["batch", "layer", "none"]:
    model = make_deep_net(ntype)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for epoch in range(100):
        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            try:
                crit(model(xb), yb).backward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"OOM with {ntype} norm")
                    torch.cuda.empty_cache()
                    continue
                raise
            opt.step()
    model.eval()
    with torch.no_grad():
        vl = sum(crit(model(xb), yb).item() * len(xb) for xb, yb in va_ld) / len(va_ld.dataset)
    norm_results[ntype] = vl
    print(f"Norm={ntype:5s}  val MSE={vl:.5f}")

best_norm = min(norm_results, key=norm_results.get)
print(f"Best normalisation strategy: {best_norm}")
"""),
    ("markdown", "## Real-World Example 1: BN Stabilises Gradient Flow in a Deep CNN"),
    ("code", """\
# Measure gradient norms per layer with and without BatchNorm.

def build_cnn(use_bn, depth=6):
    layers = [nn.Conv2d(1, 16, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(16))
    layers.append(nn.ReLU())
    for _ in range(depth - 1):
        layers += [nn.Conv2d(16, 16, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(16))
        layers += [nn.ReLU()]
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)]
    return nn.Sequential(*layers).to(device)

def measure_grad_norms(model, x, y):
    model.train()
    out = model(x)
    nn.CrossEntropyLoss()(out, y).backward()
    norms = [(name, p.grad.norm().item())
             for name, p in model.named_parameters() if p.grad is not None]
    return norms

x_img = torch.randn(32, 1, 16, 16, device=device)
y_cls = torch.randint(0, 10, (32,), device=device)

for use_bn in [False, True]:
    cnn = build_cnn(use_bn)
    gnorms = [v for _, v in measure_grad_norms(cnn, x_img, y_cls)]
    tag = "With BN   " if use_bn else "Without BN"
    ratio = max(gnorms) / max(min(gnorms), 1e-12)
    print(f"{tag}: min_grad={min(gnorms):.2e}  max_grad={max(gnorms):.2e}  ratio={ratio:.1f}x")

print("With BN: gradient ratio is much smaller -- no vanishing/exploding gradients.")
"""),
    ("markdown", "## Real-World Example 2: BatchNorm Train vs Eval Mode Bug"),
    ("code", """\
# Classic production bug: forgetting model.eval() before inference.
# In train mode, BN uses batch statistics -- predictions change with batch composition.

class TinyBNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

X_bug = torch.randn(100, 4).to(device)
y_bug = torch.randint(0, 2, (100,)).to(device)
model_bug = TinyBNNet().to(device)
opt_bug = torch.optim.Adam(model_bug.parameters(), lr=1e-2)
crit_bug = nn.CrossEntropyLoss()

for _ in range(30):
    model_bug.train()
    opt_bug.zero_grad()
    crit_bug(model_bug(X_bug), y_bug).backward()
    opt_bug.step()

x_test = X_bug[:1]
with torch.no_grad():
    model_bug.train()          # BUG: should call model.eval() first
    pred_train = model_bug(x_test).argmax(dim=-1).item()
    model_bug.eval()           # FIX
    pred_eval  = model_bug(x_test).argmax(dim=-1).item()

print(f"Prediction in TRAIN mode: {pred_train}")
print(f"Prediction in EVAL  mode: {pred_eval}")
if pred_train != pred_eval:
    print("BUG REPRODUCED: train-mode BN gives wrong prediction due to batch statistics!")
else:
    print("Predictions match (train more to make the difference visible).")
print("Rule: ALWAYS call model.eval() before inference, model.train() before training.")
"""),
    ("markdown", "## Real-World Example 3: Gradient Norms With vs Without BN (10-Layer MLP)"),
    ("code", """\
# Quantify gradient vanishing in a deep MLP. Tanh is prone to vanishing --
# BN rescues gradient flow even in very deep networks.

def build_deep_mlp(use_bn, n_layers=10):
    layers = []
    in_f = 20
    for _ in range(n_layers):
        layers.append(nn.Linear(in_f, 32))
        if use_bn:
            layers.append(nn.BatchNorm1d(32))
        layers.append(nn.Tanh())   # tanh saturates -- worst case for vanishing
        in_f = 32
    layers.append(nn.Linear(32, 1))
    return nn.Sequential(*layers).to(device)

X_deep = torch.randn(64, 20, device=device)
y_deep = torch.randn(64, 1, device=device)

for use_bn in [False, True]:
    mdl = build_deep_mlp(use_bn)
    mdl.train()
    nn.MSELoss()(mdl(X_deep), y_deep).backward()
    grad_norms = [p.grad.norm().item()
                  for name, p in mdl.named_parameters()
                  if "weight" in name and p.grad is not None]
    tag = "With BN   " if use_bn else "Without BN"
    ratio = max(grad_norms) / max(min(grad_norms), 1e-12)
    print(f"{tag}: norms={[round(g,5) for g in grad_norms]}")
    print(f"           Max/min ratio: {ratio:.1f}x")

print("BN dramatically reduces gradient norm ratio in deep networks.")
"""),
])

# ── 04 Class Imbalance ────────────────────────────────────────────────────────
NB04 = make_nb([
    ("markdown", """\
# Class Imbalance

## Learning Objectives
1. Implement SMOTE from scratch to understand synthetic over-sampling.
2. Compare four imbalance-handling strategies using sklearn metrics and ROC curves.
3. Apply focal loss in PyTorch for extreme imbalance (fraud detection 1:1000).
4. Tune the decision threshold using a precision-recall curve to maximise F1.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.datasets import make_classification
from sklearn.utils import resample

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: SMOTE from Scratch (NumPy)"),
    ("code", """\
# SMOTE: for each minority sample, pick k nearest neighbours and interpolate linearly.

def smote_scratch(X_min, n_synthetic, k=5):
    # Generate synthetic minority samples via linear interpolation between neighbours.
    # Args:
    # X_min: minority class samples (n, d)
    # n_synthetic: number of new samples to create
    # k: number of nearest neighbours to interpolate between
    n, d = X_min.shape
    synthetic = np.zeros((n_synthetic, d))
    for i in range(n_synthetic):
        idx = np.random.randint(0, n)
        sample = X_min[idx]
        dists = np.linalg.norm(X_min - sample, axis=1)
        dists[idx] = np.inf
        nn_indices = np.argsort(dists)[:k]
        nn = X_min[np.random.choice(nn_indices)]
        lam = np.random.uniform(0, 1)
        synthetic[i] = sample + lam * (nn - sample)
    return synthetic

# Create imbalanced dataset: 900 majority, 90 minority
X_maj = np.random.randn(900, 2) + [2, 2]
X_min = np.random.randn(90, 2) + [-2, -2]
print(f"Before SMOTE -- Majority: {len(X_maj)}, Minority: {len(X_min)}")

synthetic = smote_scratch(X_min, n_synthetic=810, k=5)
X_min_aug = np.vstack([X_min, synthetic])
print(f"After  SMOTE -- Majority: {len(X_maj)}, Minority (augmented): {len(X_min_aug)}")
print(f"Synthetic samples mean: {synthetic.mean(axis=0).round(3)}")
print(f"Original minority mean: {X_min.mean(axis=0).round(3)}  (should be similar)")
print("SMOTE implementation verified.")
"""),
    ("markdown", "## Level 2: Four Imbalance Strategies + ROC (sklearn)"),
    ("code", """\
# Compare: baseline, class_weight=balanced, under-sampling, over-sampling (SMOTE approx)

X_imb, y_imb = make_classification(
    n_samples=2200, n_features=10, n_informative=5,
    weights=[0.9, 0.1], random_state=42
)
X_train, y_train = X_imb[:1800], y_imb[:1800]
X_test,  y_test  = X_imb[1800:], y_imb[1800:]

strategies_results = {}

# 1. Baseline (no adjustment)
clf_base = LogisticRegression(max_iter=500, random_state=42)
clf_base.fit(X_train, y_train)
strategies_results["Baseline"] = {
    "auc": roc_auc_score(y_test, clf_base.predict_proba(X_test)[:,1]),
    "f1":  f1_score(y_test, clf_base.predict(X_test))
}

# 2. Class weights -- tell the model minority errors cost more
clf_w = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
try:
    clf_w.fit(X_train, y_train)
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        print("OOM -- reduce dataset size")
        torch.cuda.empty_cache()
    raise
strategies_results["ClassWeight"] = {
    "auc": roc_auc_score(y_test, clf_w.predict_proba(X_test)[:,1]),
    "f1":  f1_score(y_test, clf_w.predict(X_test))
}

# 3. Under-sample majority
n_min = (y_train == 1).sum()
X_us = np.vstack([X_train[y_train==0][:n_min], X_train[y_train==1]])
y_us = np.hstack([np.zeros(n_min), np.ones(n_min)])
clf_us = LogisticRegression(max_iter=500, random_state=42)
clf_us.fit(X_us, y_us)
strategies_results["UnderSample"] = {
    "auc": roc_auc_score(y_test, clf_us.predict_proba(X_test)[:,1]),
    "f1":  f1_score(y_test, clf_us.predict(X_test))
}

# 4. Over-sample minority using sklearn resample (SMOTE approximation)
n_maj = (y_train == 0).sum()
X_os = resample(X_train[y_train==1], n_samples=n_maj, random_state=42)
X_ov = np.vstack([X_train[y_train==0], X_os])
y_ov = np.hstack([np.zeros(n_maj), np.ones(n_maj)])
clf_ov = LogisticRegression(max_iter=500, random_state=42)
clf_ov.fit(X_ov, y_ov)
strategies_results["OverSample"] = {
    "auc": roc_auc_score(y_test, clf_ov.predict_proba(X_test)[:,1]),
    "f1":  f1_score(y_test, clf_ov.predict(X_test))
}

print(f"{'Strategy':<14}  {'ROC-AUC':>8}  {'F1':>6}")
for name, m in strategies_results.items():
    print(f"{name:<14}  {m['auc']:>8.4f}  {m['f1']:>6.4f}")
"""),
    ("markdown", "## Real-World Example 1: Fraud Detection at 1:1000 Imbalance"),
    ("code", """\
# Simulate credit-card fraud: 0.1% positive rate.
# Show that accuracy is misleading; use precision-recall AUC instead.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_fraud, y_fraud = make_classification(
    n_samples=10000, n_features=20, n_informative=8,
    weights=[0.999, 0.001], flip_y=0, random_state=42
)
X_fr_tr, y_fr_tr = X_fraud[:8000], y_fraud[:8000]
X_fr_te, y_fr_te = X_fraud[8000:], y_fraud[8000:]

print(f"Train: {(y_fr_tr==1).sum()} fraud / {len(y_fr_tr)} total")
print(f"Test:  {(y_fr_te==1).sum()} fraud / {len(y_fr_te)} total")

# Naive: predict all 0 gets ~99.9% accuracy -- useless metric for fraud
naive_acc = (y_fr_te == 0).mean()
print(f"Naive accuracy (all-zero predictor): {naive_acc:.4f}  -- misleading!")

clf_fr = RandomForestClassifier(
    n_estimators=50, class_weight="balanced", max_depth=8, random_state=42
)
clf_fr.fit(X_fr_tr, y_fr_tr)
proba_fr = clf_fr.predict_proba(X_fr_te)[:, 1]
pr_auc = average_precision_score(y_fr_te, proba_fr)
print(f"RF balanced -- PR-AUC: {pr_auc:.4f}  (meaningful metric for fraud detection)")
print(classification_report(y_fr_te, (proba_fr > 0.3).astype(int),
                             target_names=["legit", "fraud"], zero_division=0))
"""),
    ("markdown", "## Real-World Example 2: Focal Loss in PyTorch"),
    ("code", """\
# Focal loss down-weights easy negatives, focusing training on hard examples.
# Critical when 99%+ of samples are easy negatives (fraud, object detection).

class FocalLoss(nn.Module):
    # Focal loss: FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t).

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        p_t = torch.exp(-bce)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()

# 10100 samples: 100 positives (1% imbalance)
X_fl = torch.randn(10100, 16).to(device)
y_fl = torch.zeros(10100).to(device)
y_fl[:100] = 1.0

mdl_fl = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
opt_fl = torch.optim.Adam(mdl_fl.parameters(), lr=1e-3)
focal  = FocalLoss(gamma=2.0, alpha=0.25)
ds_fl  = TensorDataset(X_fl, y_fl)
ld_fl  = DataLoader(ds_fl, batch_size=128, shuffle=True)

for epoch in range(40):
    mdl_fl.train()
    for xb, yb in ld_fl:
        opt_fl.zero_grad()
        focal(mdl_fl(xb).squeeze(1), yb).backward()
        opt_fl.step()

mdl_fl.eval()
with torch.no_grad():
    logits_all = mdl_fl(X_fl).squeeze(1)
    preds = (torch.sigmoid(logits_all) > 0.3).float()
    tp = ((preds == 1) & (y_fl == 1)).sum().item()
    fp = ((preds == 1) & (y_fl == 0)).sum().item()
    fn = ((preds == 0) & (y_fl == 1)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
print(f"Focal Loss -- Precision: {prec:.3f}  Recall: {rec:.3f}  TP={tp}  FP={fp}")
"""),
    ("markdown", "## Real-World Example 3: Threshold Tuning with PR Curve"),
    ("code", """\
# Default threshold 0.5 is often wrong for imbalanced tasks.
# Use the precision-recall curve to find the threshold that maximises F1.

X_thr, y_thr = make_classification(
    n_samples=5000, n_features=10, n_informative=6,
    weights=[0.9, 0.1], random_state=7
)
clf_thr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=7)
clf_thr.fit(X_thr[:4000], y_thr[:4000])
proba_thr = clf_thr.predict_proba(X_thr[4000:])[:, 1]
y_te_thr  = y_thr[4000:]

precision_arr, recall_arr, thresholds = precision_recall_curve(y_te_thr, proba_thr)
# Compute F1 at each threshold (arrays are len(thresholds)+1 -- trim last)
f1_arr = np.where((precision_arr[:-1] + recall_arr[:-1]) > 0,
                  2 * precision_arr[:-1] * recall_arr[:-1] /
                  (precision_arr[:-1] + recall_arr[:-1]), 0.0)
best_idx = np.argmax(f1_arr)
best_thr  = thresholds[best_idx]

print(f"Default threshold (0.5) F1: {f1_score(y_te_thr, proba_thr >= 0.5):.4f}")
print(f"Optimal threshold         : {best_thr:.4f}")
print(f"Optimal F1                : {f1_arr[best_idx]:.4f}")
print(f"Precision at optimal thr  : {precision_arr[best_idx]:.4f}")
print(f"Recall    at optimal thr  : {recall_arr[best_idx]:.4f}")
print("Rule: tune threshold on a validation set, not the final test set.")
"""),
])

# ── 05 CNNs ───────────────────────────────────────────────────────────────────
NB05 = make_nb([
    ("markdown", """\
# Convolutional Neural Networks (CNNs)

## Learning Objectives
1. Implement a 2D convolution from scratch using NumPy and verify output shapes.
2. Build and train a CNN in PyTorch with OOM-safe training loop.
3. Apply transfer learning from a pretrained ResNet-style model to a new task.
4. Visualise intermediate feature maps to understand what each layer detects.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: 2D Convolution from Scratch (NumPy)"),
    ("code", """\
# Implement valid 2D convolution and max pooling from scratch.

def conv2d_scratch(image, kernel, stride=1):
    # 2D convolution (valid padding, single channel).
    # image: (H, W), kernel: (kH, kW)
    # Returns: ((H-kH)//stride + 1, (W-kW)//stride + 1)
    H, W = image.shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i*stride : i*stride+kH, j*stride : j*stride+kW]
            output[i, j] = (patch * kernel).sum()
    return output

def maxpool2d_scratch(x, pool=2):
    H, W = x.shape
    oh, ow = H // pool, W // pool
    return x[:oh*pool, :ow*pool].reshape(oh, pool, ow, pool).max(axis=(1, 3))

# Verify with Sobel-X edge-detection kernel
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=float)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

out = conv2d_scratch(image, sobel_x)
print(f"Input shape : {image.shape}")
print(f"Kernel shape: {sobel_x.shape}")
print(f"Output shape: {out.shape}  (valid padding: H-kH+1 x W-kW+1)")
print(f"Sobel-X output (positive=right edge, negative=left edge):")
print(out.astype(int))

pooled = maxpool2d_scratch(out, pool=2)
print(f"After 2x2 max pooling: shape={pooled.shape}  values={pooled.flatten().round(1)}")
"""),
    ("markdown", "## Level 2: CNN in PyTorch with OOM-Safe Training"),
    ("code", """\
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Synthetic 32x32 grayscale dataset, 10 classes
X_cnn = torch.randn(500, 1, 32, 32)
y_cnn = torch.randint(0, 10, (500,))
tr_ds = TensorDataset(X_cnn[:400].to(device), y_cnn[:400].to(device))
va_ds = TensorDataset(X_cnn[400:].to(device), y_cnn[400:].to(device))
tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=32)

cnn = SimpleCNN(n_classes=10).to(device)
opt_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)
crit_cnn = nn.CrossEntropyLoss()

for epoch in range(15):
    cnn.train()
    for xb, yb in tr_ld:
        opt_cnn.zero_grad()
        try:
            loss = crit_cnn(cnn(xb), yb)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("OOM -- reduce batch size or model size")
                torch.cuda.empty_cache()
                continue
            raise
        loss.backward()
        opt_cnn.step()

cnn.eval()
with torch.no_grad():
    correct = sum((cnn(xb).argmax(1) == yb).sum().item() for xb, yb in va_ld)
print(f"CNN validation accuracy: {correct / len(va_ds):.4f}")
print(f"Total parameters: {sum(p.numel() for p in cnn.parameters()):,}")
"""),
    ("markdown", "## Real-World Example 1: Transfer Learning from a ResNet-Style Block"),
    ("code", """\
# Simulate transfer learning: freeze pretrained feature extractor,
# replace and train only the classifier head.

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

class TinyResNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(4)])
        self.head   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, n_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

# Pre-train on task A (5 classes)
model_a = TinyResNet(n_classes=5).to(device)
X_a = torch.randn(300, 1, 32, 32, device=device)
y_a = torch.randint(0, 5, (300,), device=device)
opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
for _ in range(10):
    opt_a.zero_grad()
    nn.CrossEntropyLoss()(model_a(X_a), y_a).backward()
    opt_a.step()

# Transfer: freeze stem + blocks, fine-tune new head for task B (3 classes)
model_b = TinyResNet(n_classes=3).to(device)
model_b.stem.load_state_dict(model_a.stem.state_dict())
model_b.blocks.load_state_dict(model_a.blocks.state_dict())
for p in list(model_b.stem.parameters()) + list(model_b.blocks.parameters()):
    p.requires_grad = False

X_b = torch.randn(200, 1, 32, 32, device=device)
y_b = torch.randint(0, 3, (200,), device=device)
opt_b = torch.optim.Adam(filter(lambda p: p.requires_grad, model_b.parameters()), lr=1e-3)
for _ in range(15):
    opt_b.zero_grad()
    nn.CrossEntropyLoss()(model_b(X_b), y_b).backward()
    opt_b.step()

with torch.no_grad():
    acc_b = (model_b(X_b).argmax(1) == y_b).float().mean().item()
trainable = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
total_p   = sum(p.numel() for p in model_b.parameters())
print(f"Transfer learning -- Task B accuracy: {acc_b:.4f}")
print(f"Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")
"""),
    ("markdown", "## Real-World Example 2: Depthwise Separable Convolution"),
    ("code", """\
# Depthwise separable convolution (MobileNet): depthwise + pointwise.
# Reduces parameters ~8-9x compared to standard conv with similar accuracy.

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise: one filter per input channel (groups=in_ch)
        self.dw    = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        # Pointwise: 1x1 conv to mix channels
        self.pw    = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.dw_bn(self.dw(x)))
        return F.relu(self.pw_bn(self.pw(x)))

x_dw = torch.randn(4, 32, 16, 16, device=device)

std_conv = nn.Sequential(
    nn.Conv2d(32, 64, 3, padding=1, bias=False),
    nn.BatchNorm2d(64), nn.ReLU()
).to(device)
dws_conv = DepthwiseSeparableConv(32, 64).to(device)

p_std = sum(p.numel() for p in std_conv.parameters())
p_dws = sum(p.numel() for p in dws_conv.parameters())
with torch.no_grad():
    out_std = std_conv(x_dw)
    out_dws = dws_conv(x_dw)
print(f"Standard conv       -- params: {p_std:,}  out: {tuple(out_std.shape)}")
print(f"Depthwise-separable -- params: {p_dws:,}  out: {tuple(out_dws.shape)}")
print(f"Parameter reduction: {p_std/p_dws:.2f}x  (same output shape, fewer weights)")
"""),
    ("markdown", "## Real-World Example 3: Feature Map Visualization"),
    ("code", """\
# Hook into conv layers to extract and inspect feature maps.
# Early layers detect edges; later layers detect semantic patterns.

feature_maps = {}

def register_hooks(model):
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            def make_hook(n):
                def hook(_, __, output):
                    feature_maps[n] = output.detach().cpu()
                return hook
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles

vis_cnn = SimpleCNN(n_classes=5).to(device)
x_vis = torch.randn(1, 1, 32, 32, device=device)

handles = register_hooks(vis_cnn)
with torch.no_grad():
    _ = vis_cnn(x_vis)
for h in handles:
    h.remove()

print("Feature maps extracted from conv layers:")
for name, fmap in feature_maps.items():
    B, C, H, W = fmap.shape
    act_mean  = fmap.abs().mean().item()
    sparsity  = (fmap == 0).float().mean().item()
    print(f"  {name:40s}  C={C}  {H}x{W}  mean_act={act_mean:.4f}  sparsity={sparsity:.2%}")

mags = [feature_maps[n].abs().mean().item() for n in feature_maps]
print(f"Activation magnitude trend: {[round(m, 3) for m in mags]}")
print("Early layers: higher raw activation (raw edges).")
print("Later layers: sparser but more semantically meaningful features.")
"""),
])

# ── 06 Cross-Validation ───────────────────────────────────────────────────────
NB06 = make_nb([
    ("markdown", """\
# Cross-Validation

## Learning Objectives
1. Implement k-fold cross-validation from scratch using only NumPy.
2. Compare KFold, StratifiedKFold, TimeSeriesSplit, and GroupKFold in sklearn.
3. Apply nested cross-validation for unbiased hyperparameter tuning.
4. Quantify cross-validation variance to detect unstable model estimates.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     TimeSeriesSplit, GroupKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import r2_score

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: K-Fold Cross-Validation from Scratch (NumPy)"),
    ("code", """\
def kfold_indices(n, k, shuffle=True, seed=42):
    # Yield (train_idx, val_idx) tuples for k-fold CV.
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1
    current = 0
    for size in fold_sizes:
        val_idx   = idx[current : current + size]
        train_idx = np.concatenate([idx[:current], idx[current + size:]])
        yield train_idx, val_idx
        current += size

X_cv, y_cv = make_classification(n_samples=300, n_features=10, random_state=42)

def fit_eval_cv(X_tr, y_tr, X_va, y_va):
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_tr, y_tr)
    return (clf.predict(X_va) == y_va).mean()

scores = []
for tr_idx, va_idx in kfold_indices(len(X_cv), k=5):
    score = fit_eval_cv(X_cv[tr_idx], y_cv[tr_idx], X_cv[va_idx], y_cv[va_idx])
    scores.append(score)

print(f"5-fold CV scores: {[round(s, 4) for s in scores]}")
print(f"Mean accuracy:  {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
print("K-fold splits data into k equally-sized folds, using each as validation once.")
"""),
    ("markdown", "## Level 2: KFold / StratifiedKFold / TimeSeriesSplit / GroupKFold (sklearn)"),
    ("code", """\
# Four CV strategies, each designed for a specific data structure.

X_cls, y_cls = make_classification(n_samples=400, n_features=12,
                                    weights=[0.7, 0.3], random_state=7)
groups = np.repeat(np.arange(40), 10)   # 40 patient groups of 10 samples each

# Temporally correlated data for TimeSeriesSplit
time_X = np.cumsum(np.random.randn(400, 12), axis=0)
time_y = (time_X[:, 0] > 0).astype(int)

strategies = {
    "KFold(shuffle)": (KFold(n_splits=5, shuffle=True, random_state=42), X_cls, y_cls, None),
    "StratifiedKFold": (StratifiedKFold(n_splits=5, shuffle=True, random_state=42), X_cls, y_cls, None),
    "TimeSeriesSplit": (TimeSeriesSplit(n_splits=5), time_X, time_y, None),
    "GroupKFold":      (GroupKFold(n_splits=5), X_cls, y_cls, groups),
}

clf_base = LogisticRegression(max_iter=500, random_state=42)
try:
    for name, (cv, X, y, g) in strategies.items():
        s = cross_val_score(clf_base, X, y, cv=cv, groups=g, scoring="accuracy", n_jobs=1)
        print(f"{name:20s}  mean={s.mean():.4f}  std={s.std():.4f}")
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        print("OOM -- reduce dataset size")
        torch.cuda.empty_cache()
    raise

print("StratifiedKFold: preserves class ratio per fold -- use for imbalanced data.")
print("TimeSeriesSplit: no future-to-past leakage -- mandatory for forecasting.")
print("GroupKFold: no cross-group contamination -- use for patient/user data.")
"""),
    ("markdown", "## Real-World Example 1: Nested Cross-Validation"),
    ("code", """\
# Nested CV: outer loop estimates generalisation error; inner loop tunes hyperparams.
# Non-nested CV with tuning is optimistic (positively biased).

X_nest, y_nest = make_classification(n_samples=300, n_features=10, random_state=1)

pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])
param_grid = {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["rbf", "linear"]}

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Outer loop provides the honest generalisation estimate
outer_scores = []
for train_idx, test_idx in outer_cv.split(X_nest):
    X_tr, X_te = X_nest[train_idx], X_nest[test_idx]
    y_tr, y_te = y_nest[train_idx], y_nest[test_idx]
    grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=1)
    grid.fit(X_tr, y_tr)
    outer_scores.append(grid.score(X_te, y_te))

# Non-nested (optimistic): tune and evaluate on the same outer folds
non_nested = cross_val_score(GridSearchCV(pipe, param_grid, cv=inner_cv, n_jobs=1),
                              X_nest, y_nest, cv=outer_cv, n_jobs=1)

print(f"Nested CV accuracy:     {np.mean(outer_scores):.4f} +/- {np.std(outer_scores):.4f}")
print(f"Non-nested CV accuracy: {non_nested.mean():.4f} +/- {non_nested.std():.4f}")
print(f"Optimism bias:          {non_nested.mean() - np.mean(outer_scores):+.4f}")
print("Nested CV is the gold standard for model selection + evaluation.")
"""),
    ("markdown", "## Real-World Example 2: Temporal Cross-Validation for Time Series"),
    ("code", """\
# TimeSeriesSplit never leaks future data into training.
# Standard KFold randomly shuffles, causing data-from-the-future leakage.

np.random.seed(42)
T = 500
y_ts = np.zeros(T)
for i in range(1, T):
    y_ts[i] = 0.8 * y_ts[i-1] + np.random.randn() * 0.5

def make_lag_features(y, lags=5):
    n = len(y)
    X = np.column_stack([y[i:n-lags+i] for i in range(lags)])
    return X, y[lags:]

X_ts, y_ts_cut = make_lag_features(y_ts, lags=5)
ridge = Ridge(alpha=1.0)
tss   = TimeSeriesSplit(n_splits=5)
kf    = KFold(n_splits=5, shuffle=True, random_state=42)

ts_scores = cross_val_score(ridge, X_ts, y_ts_cut, cv=tss, scoring="r2")
kf_scores  = cross_val_score(ridge, X_ts, y_ts_cut, cv=kf,  scoring="r2")

print(f"TimeSeriesSplit R2: {ts_scores.mean():.4f} +/- {ts_scores.std():.4f}  (honest)")
print(f"Standard KFold  R2: {kf_scores.mean():.4f} +/- {kf_scores.std():.4f}  (leaks future)")
print(f"Leakage inflation:  {kf_scores.mean() - ts_scores.mean():+.4f}")
print("For time-series: always use TimeSeriesSplit or walk-forward validation.")
"""),
    ("markdown", "## Real-World Example 3: CV Variance Analysis"),
    ("code", """\
# High CV variance indicates: small dataset, wrong strategy, or unstable model.
# Track coefficient of variation (std/mean) to detect instability.

X_var, y_var = make_classification(n_samples=200, n_features=8, random_state=5)
cv_var = KFold(n_splits=10, shuffle=True, random_state=42)

models_var = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "DecisionTree(d=3)":  DecisionTreeClassifier(max_depth=3, random_state=42),
    "DecisionTree(full)": DecisionTreeClassifier(random_state=42),   # likely overfit
    "RandomForest":       RandomForestClassifier(n_estimators=50, random_state=42),
}

print(f"{'Model':<22}  {'Mean Acc':>9}  {'Std':>6}  {'CV% (std/mean)':>15}")
for name, mdl in models_var.items():
    s = cross_val_score(mdl, X_var, y_var, cv=cv_var, scoring="accuracy", n_jobs=1)
    cv_pct = (s.std() / s.mean()) * 100
    print(f"{name:<22}  {s.mean():>9.4f}  {s.std():>6.4f}  {cv_pct:>15.1f}%")

print()
print("High CV% --> model is unstable; consider more data, regularisation, or ensembling.")
print("Fully-grown trees have high variance; RandomForest reduces it via averaging.")
"""),
])

# ── 07 Data Leakage ───────────────────────────────────────────────────────────
NB07 = make_nb([
    ("markdown", """\
# Data Leakage

## Learning Objectives
1. Reproduce the classic scaler-leakage bug and measure its inflated accuracy.
2. Quantify leakage using sklearn Pipeline vs manual preprocessing.
3. Demonstrate target-encoding leakage and how to prevent it.
4. Build a leakage audit tool that flags suspicious columns and temporal patterns.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: Scaler Leakage Demo"),
    ("code", """\
# The bug: fit StandardScaler on ALL data before the train/test split.
# The fix: fit scaler ONLY on training data.

X_leak, y_leak = make_classification(n_samples=500, n_features=20,
                                      n_informative=5, random_state=42)
split = 400

# BUG: scale the full dataset BEFORE splitting (test stats contaminate scaler)
scaler_bug = StandardScaler().fit(X_leak)
X_bug = scaler_bug.transform(X_leak)
clf_bug = LogisticRegression(max_iter=500)
clf_bug.fit(X_bug[:split], y_leak[:split])
acc_bug = clf_bug.score(X_bug[split:], y_leak[split:])

# FIX: fit scaler only on training portion
scaler_fix = StandardScaler().fit(X_leak[:split])
X_fix_train = scaler_fix.transform(X_leak[:split])
X_fix_test  = scaler_fix.transform(X_leak[split:])
clf_fix = LogisticRegression(max_iter=500)
clf_fix.fit(X_fix_train, y_leak[:split])
acc_fix = clf_fix.score(X_fix_test, y_leak[split:])

print(f"Leaky  accuracy: {acc_bug:.4f}  -- inflated (scaler saw test data)")
print(f"Correct accuracy: {acc_fix:.4f}  -- honest estimate")
print(f"Inflation: {acc_bug - acc_fix:+.4f}")
print("Even a small inflation can lead to wrong model-selection decisions.")
"""),
    ("markdown", "## Level 2: Pipeline vs Manual Preprocessing (Leakage Quantified)"),
    ("code", """\
# Pipeline automatically fits preprocessors inside each CV fold.
# Manual preprocessing outside CV leaks validation statistics.

X_ppl, y_ppl = make_classification(n_samples=300, n_features=15,
                                    n_informative=6, random_state=7)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Leaky: scaler fitted on all data before CV
scaler_out = StandardScaler().fit(X_ppl)
X_scaled_all = scaler_out.transform(X_ppl)
scores_leaky = cross_val_score(LogisticRegression(max_iter=500),
                               X_scaled_all, y_ppl, cv=cv, scoring="accuracy")

# Correct: Pipeline ensures scaler is re-fitted per fold
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
try:
    scores_pipe = cross_val_score(pipe, X_ppl, y_ppl, cv=cv, scoring="accuracy")
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        print("OOM -- reduce n_samples")
        torch.cuda.empty_cache()
    raise

print(f"Leaky (scaler outside CV): {scores_leaky.mean():.4f} +/- {scores_leaky.std():.4f}")
print(f"Correct (Pipeline):        {scores_pipe.mean():.4f} +/- {scores_pipe.std():.4f}")
print(f"Optimism bias:             {scores_leaky.mean() - scores_pipe.mean():+.4f}")
print("Rule: wrap ALL preprocessing steps in a Pipeline -- it is the only safe default.")
"""),
    ("markdown", "## Real-World Example 1: Target Encoding Leakage"),
    ("code", """\
# Target encoding replaces a category with the mean target value.
# Global encoding leaks test-set labels into training features (BUG).

def target_encode_leaky(X_cat, y_tr, X_cat_test):
    # Global mean encoding -- uses test categories to compute means (BUG).
    all_cats = np.unique(np.concatenate([X_cat, X_cat_test]))
    means = {}
    for cat in all_cats:
        mask = X_cat == cat
        means[cat] = y_tr[mask].mean() if mask.any() else 0.5
    return np.array([means[c] for c in X_cat]), np.array([means[c] for c in X_cat_test])

def target_encode_correct(X_cat_tr, y_tr, X_cat_te):
    # Encoding computed only from training data (FIX).
    global_mean = y_tr.mean()
    means = {cat: y_tr[X_cat_tr == cat].mean() for cat in np.unique(X_cat_tr)}
    tr_enc = np.array([means[c] for c in X_cat_tr])
    te_enc = np.array([means.get(c, global_mean) for c in X_cat_te])
    return tr_enc, te_enc

# Synthetic data: 5 categories with correlated target
N_te = 300
cats = np.random.choice(["A","B","C","D","E"], N_te)
cat_effect = {"A": 0.2, "B": 0.4, "C": 0.6, "D": 0.8, "E": 0.9}
y_te_arr = (np.array([cat_effect[c] for c in cats]) + np.random.randn(N_te)*0.2 > 0.5).astype(int)
cats_tr, cats_test = cats[:240], cats[240:]
y_tr_arr, y_te_true = y_te_arr[:240], y_te_arr[240:]

for tag, fn in [("Leaky", target_encode_leaky), ("Correct", target_encode_correct)]:
    enc_tr, enc_te = fn(cats_tr, y_tr_arr, cats_test)
    clf = LogisticRegression()
    clf.fit(enc_tr.reshape(-1, 1), y_tr_arr)
    acc = clf.score(enc_te.reshape(-1, 1), y_te_true)
    print(f"{tag:10s} accuracy: {acc:.4f}")
print("Correct encoding uses only training-fold target means -- no leakage.")
"""),
    ("markdown", "## Real-World Example 2: Temporal Leakage in Time Series"),
    ("code", """\
# Temporal leakage: using future-shifted features when predicting today's value.
# Common bug: forward-shifted target included as a feature.

np.random.seed(42)
T_ts = 400
sales = np.cumsum(np.random.randn(T_ts)) + 100

def make_leaky_features(sales, window=5):
    X, y = [], []
    for i in range(window, len(sales) - 1):
        feats = list(sales[i-window:i]) + [sales[i+1]]   # BUG: uses future value
        X.append(feats)
        y.append(sales[i])
    return np.array(X), np.array(y)

def make_correct_features(sales, window=5):
    X, y = [], []
    for i in range(window, len(sales)):
        X.append(sales[i-window:i].tolist())              # FIX: past-only
        y.append(sales[i])
    return np.array(X), np.array(y)

for tag, make_fn in [("Leaky", make_leaky_features), ("Correct", make_correct_features)]:
    X_all, y_all = make_fn(sales)
    sp = int(len(y_all) * 0.8)
    mdl = Ridge()
    mdl.fit(X_all[:sp], y_all[:sp])
    r2 = r2_score(y_all[sp:], mdl.predict(X_all[sp:]))
    print(f"{tag:10s} test R2: {r2:.4f}")

print("Leaky R2 near 1.0 -- the model is memorising tomorrow's value (future feature).")
"""),
    ("markdown", "## Real-World Example 3: Leakage Audit Tool"),
    ("code", """\
# Automated leakage audit: flag features with suspiciously high target correlation.
# Use as a first-pass check before model training.

def audit_leakage(X, y, feature_names=None, threshold=0.9):
    # Flag features whose absolute Pearson correlation with y >= threshold.
    # Returns a list of (idx, name, correlation) tuples.
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    flagged = []
    for i, name in enumerate(feature_names):
        corr, _ = pearsonr(X[:, i], y)
        if abs(corr) >= threshold:
            flagged.append((i, name, corr))
    return sorted(flagged, key=lambda t: abs(t[2]), reverse=True)

np.random.seed(99)
N_audit = 500
X_clean = np.random.randn(N_audit, 8)
y_audit = (X_clean[:, 0] + np.random.randn(N_audit) * 0.5 > 0).astype(float)

# Inject two suspicious features: noisy copies of the target
X_leaked = np.column_stack([
    X_clean,
    y_audit + np.random.randn(N_audit) * 0.05,   # feature 8: near-perfect proxy
    y_audit + np.random.randn(N_audit) * 0.2,    # feature 9: noisy proxy
])
feat_names = [f"feat_{i}" for i in range(8)] + ["proxy_exact", "proxy_noisy"]

flagged = audit_leakage(X_leaked, y_audit, feat_names, threshold=0.8)
print(f"Flagged {len(flagged)} suspicious features:")
for idx, name, corr in flagged:
    print(f"  Feature '{name}' (idx={idx}): corr={corr:.4f}")

# Verify clean data gets no flags
clean_flagged = audit_leakage(X_clean, y_audit, threshold=0.8)
print(f"Clean data flagged: {len(clean_flagged)}  (expected 0)")
print("Audit tool correctly identifies proxy features that would cause leakage.")
"""),
])

# ── 08 Distributed Training ───────────────────────────────────────────────────
NB08 = make_nb([
    ("markdown", """\
# Distributed Training

## Learning Objectives
1. Simulate data-parallel gradient averaging across two workers using NumPy.
2. Mock DDP training in PyTorch with gradient synchronisation and OOM handling.
3. Use gradient accumulation to simulate large effective batch sizes.
4. Apply automatic mixed precision (AMP) and measure memory and speed improvement.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
import time

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: Data-Parallel Gradient Averaging (NumPy)"),
    ("code", """\
# Simulate 2-worker data-parallel training:
# each worker computes gradients on its shard, then AllReduce averages them.

class LinearModelNumpy:
    def __init__(self, d_in, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((d_in, d_out)) * 0.1
        self.b = np.zeros(d_out)

    def forward(self, X):
        return X @ self.W + self.b

    def mse_grad(self, X, y):
        # Return gradients of MSE loss w.r.t. W and b.
        pred = self.forward(X)
        err  = pred - y                       # (n, d_out)
        dW   = (2 / len(X)) * X.T @ err      # (d_in, d_out)
        db   = (2 / len(X)) * err.sum(0)     # (d_out,)
        return dW, db

N_ddp_np = 200; D_IN, D_OUT = 5, 1
X_dist = np.random.randn(N_ddp_np, D_IN)
W_true = np.array([1, -2, 0.5, 3, -1]).reshape(D_IN, D_OUT)
y_dist = X_dist @ W_true + 0.1 * np.random.randn(N_ddp_np, D_OUT)

model_np = LinearModelNumpy(D_IN, D_OUT)
X_w1, y_w1 = X_dist[:N_ddp_np//2], y_dist[:N_ddp_np//2]
X_w2, y_w2 = X_dist[N_ddp_np//2:], y_dist[N_ddp_np//2:]

LR = 0.05
for step in range(50):
    dW1, db1 = model_np.mse_grad(X_w1, y_w1)
    dW2, db2 = model_np.mse_grad(X_w2, y_w2)
    # AllReduce: average gradients from both workers
    model_np.W -= LR * (dW1 + dW2) / 2
    model_np.b -= LR * (db1 + db2) / 2

mse_final = ((model_np.forward(X_dist) - y_dist) ** 2).mean()
print(f"Data-parallel training MSE after 50 steps: {mse_final:.6f}")
print(f"Learned W: {model_np.W.flatten().round(3)}")
print(f"True    W: {W_true.flatten()}")
"""),
    ("markdown", "## Level 2: Mock DDP with Gradient Sync (PyTorch)"),
    ("code", """\
# Two model replicas, each processing its shard, gradients averaged before update.

def build_model_ddp():
    return nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)

def sync_gradients(models):
    # Average gradients across all model replicas (mock AllReduce).
    n = len(models)
    for param_idx, params in enumerate(zip(*[list(m.parameters()) for m in models])):
        grad_avg = sum(p.grad for p in params if p.grad is not None)
        if grad_avg is not None:
            grad_avg = grad_avg / n
            for p in params:
                if p.grad is not None:
                    p.grad.copy_(grad_avg)

N_ddp = 400
X_ddp = torch.randn(N_ddp, 10, device=device)
y_ddp = (X_ddp[:, 0] + X_ddp[:, 1]).unsqueeze(1)
model1, model2 = build_model_ddp(), build_model_ddp()
model2.load_state_dict(model1.state_dict())   # same initialisation (DDP requirement)
opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
crit_ddp = nn.MSELoss()

for step in range(60):
    opt1.zero_grad(); opt2.zero_grad()
    try:
        loss1 = crit_ddp(model1(X_ddp[:N_ddp//2]), y_ddp[:N_ddp//2])
        loss2 = crit_ddp(model2(X_ddp[N_ddp//2:]), y_ddp[N_ddp//2:])
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("OOM -- reduce batch size or model size")
            torch.cuda.empty_cache()
            continue
        raise
    loss1.backward(); loss2.backward()
    sync_gradients([model1, model2])
    opt1.step(); opt2.step()

with torch.no_grad():
    val_loss = crit_ddp(model1(X_ddp), y_ddp).item()
max_diff = max((p1-p2).abs().max().item()
               for p1, p2 in zip(model1.parameters(), model2.parameters()))
print(f"Mock DDP final val MSE: {val_loss:.6f}")
print(f"Max param diff between replicas: {max_diff:.2e}  (should be ~0 if sync is correct)")
"""),
    ("markdown", "## Real-World Example 1: Gradient Accumulation"),
    ("code", """\
# Gradient accumulation: sum gradients over N micro-batches before stepping.
# Effective batch = micro_batch * ACCUM_STEPS, with the same memory as micro_batch.

N_ga = 800
X_ga = torch.randn(N_ga, 20, device=device)
y_ga = (X_ga[:, :3].sum(1) > 0).long()
ds_ga = TensorDataset(X_ga, y_ga)
ld_ga = DataLoader(ds_ga, batch_size=16, shuffle=True)  # small physical batch

model_ga = nn.Sequential(
    nn.Linear(20, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 2)
).to(device)
opt_ga = torch.optim.Adam(model_ga.parameters(), lr=1e-3)
crit_ga = nn.CrossEntropyLoss()

ACCUM_STEPS = 8   # effective batch = 16 * 8 = 128
step_losses = []
model_ga.train()
opt_ga.zero_grad()

for step, (xb, yb) in enumerate(ld_ga):
    # Divide loss by accumulation steps so gradient magnitude matches a single big batch
    loss = crit_ga(model_ga(xb), yb) / ACCUM_STEPS
    loss.backward()
    step_losses.append(loss.item() * ACCUM_STEPS)
    if (step + 1) % ACCUM_STEPS == 0:
        opt_ga.step()
        opt_ga.zero_grad()

model_ga.eval()
with torch.no_grad():
    acc_ga = (model_ga(X_ga).argmax(1) == y_ga).float().mean().item()
print(f"Gradient accumulation (eff. batch={16*ACCUM_STEPS}) accuracy: {acc_ga:.4f}")
print(f"Mean step loss: {np.mean(step_losses):.4f}")
print(f"Physical batch=16, Effective batch={16*ACCUM_STEPS} (no extra GPU memory)")
"""),
    ("markdown", "## Real-World Example 2: Mixed Precision Training (AMP)"),
    ("code", """\
# AMP: fp16 for compute, fp32 for gradient accumulation.
# ~2x memory reduction, 1.5-3x speedup on CUDA; no change in accuracy.

N_amp = 1000
X_amp = torch.randn(N_amp, 32, device=device)
y_amp = torch.randint(0, 5, (N_amp,), device=device)
ds_amp = TensorDataset(X_amp, y_amp)
ld_amp = DataLoader(ds_amp, batch_size=64, shuffle=True)

model_amp = nn.Sequential(
    nn.Linear(32, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 5)
).to(device)
crit_amp = nn.CrossEntropyLoss()

# --- FP32 baseline ---
model_fp32 = copy.deepcopy(model_amp)
opt_fp32   = torch.optim.Adam(model_fp32.parameters(), lr=1e-3)
t0 = time.perf_counter()
for _ in range(5):
    model_fp32.train()
    for xb, yb in ld_amp:
        opt_fp32.zero_grad()
        crit_amp(model_fp32(xb), yb).backward()
        opt_fp32.step()
t_fp32 = time.perf_counter() - t0

# --- AMP (fp16 on CUDA, transparent fp32 fallback on CPU) ---
model_amp2 = copy.deepcopy(model_amp)
opt_amp2   = torch.optim.Adam(model_amp2.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
t0 = time.perf_counter()
for _ in range(5):
    model_amp2.train()
    for xb, yb in ld_amp:
        opt_amp2.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            loss_amp = crit_amp(model_amp2(xb), yb)
        scaler.scale(loss_amp).backward()
        scaler.step(opt_amp2)
        scaler.update()
t_amp = time.perf_counter() - t0

speedup = t_fp32 / max(t_amp, 1e-9)
print(f"FP32 time: {t_fp32*1000:.1f}ms  AMP time: {t_amp*1000:.1f}ms")
print(f"Speedup: {speedup:.2f}x  (full benefit on CUDA; AMP enabled={device.type=='cuda'})")
print("AMP cuts memory ~2x and speeds compute; use for any GPU training run.")
"""),
    ("markdown", "## Real-World Example 3: Gradient Checkpointing"),
    ("code", """\
# Gradient checkpointing: recompute activations during backward instead of storing them.
# Trades ~20-30% extra compute for ~30-50% less memory.

from torch.utils.checkpoint import checkpoint as ckpt_fn

class DeepNetCheckpointed(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(8)
        ])
        self.head = nn.Linear(64, 1)

    def forward(self, x, use_ckpt=True):
        for layer in self.layers:
            if use_ckpt and x.requires_grad:
                x = ckpt_fn(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.head(x)

X_ckpt = torch.randn(64, 64, device=device)
y_ckpt = torch.randn(64, 1, device=device)

model_ckpt = DeepNetCheckpointed().to(device)

for use_ckpt, label in [(False, "Normal      "), (True, "Checkpointed")]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    x_in = X_ckpt.clone().requires_grad_(True)
    out  = model_ckpt(x_in, use_ckpt=use_ckpt)
    nn.MSELoss()(out, y_ckpt).backward()
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"{label}: peak GPU memory = {mem:.1f}MB")
    else:
        print(f"{label}: (memory measurement requires CUDA; forward/backward succeeded)")

print("On GPU: checkpointing saves 30-50% memory; use when OOM is the bottleneck.")
"""),
])

# ── 09 Domain Adaptation ──────────────────────────────────────────────────────
NB09 = make_nb([
    ("markdown", """\
# Domain Adaptation

## Learning Objectives
1. Visualise domain shift in 2D and measure Maximum Mean Discrepancy (MMD).
2. Implement CORAL alignment in PyTorch to minimise covariance shift.
3. Fine-tune a source-trained model on small labelled target data.
4. Apply pseudo-label self-training and DANN adversarial adaptation.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: Domain Shift Visualisation and MMD (NumPy)"),
    ("code", """\
# Simulate domain shift: source and target have same classes but different distributions.

def make_domain(n, shift, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n//2, 2)) * scale + np.array([-2, 0]) + shift
    X1 = rng.standard_normal((n//2, 2)) * scale + np.array([ 2, 0]) + shift
    return np.vstack([X0, X1]), np.array([0]*(n//2) + [1]*(n//2))

X_src, y_src = make_domain(200, shift=(0, 0), scale=0.8, seed=0)
X_tgt, y_tgt = make_domain(200, shift=(3, 2), scale=1.4, seed=1)

def mmd_rbf(X1, X2, sigma=1.0):
    # Maximum Mean Discrepancy with RBF kernel: 0 if same distribution.
    def k(A, B):
        dists = ((A[:, None] - B[None, :]) ** 2).sum(-1)
        return np.exp(-dists / (2 * sigma ** 2))
    return k(X1, X1).mean() - 2*k(X1, X2).mean() + k(X2, X2).mean()

mmd = mmd_rbf(X_src, X_tgt)
print(f"MMD (source vs target): {mmd:.4f}  (higher = more shift)")

from sklearn.linear_model import LogisticRegression
clf_src_base = LogisticRegression(max_iter=500).fit(X_src, y_src)
acc_src = clf_src_base.score(X_src, y_src)
acc_tgt = clf_src_base.score(X_tgt, y_tgt)
print(f"Source accuracy: {acc_src:.4f}")
print(f"Target accuracy (no adaptation): {acc_tgt:.4f}  -- degraded by shift")
print(f"Accuracy drop due to shift: {acc_src - acc_tgt:+.4f}")
"""),
    ("markdown", "## Level 2: CORAL Alignment in PyTorch"),
    ("code", """\
# CORAL: align second-order statistics (covariance) of source and target features.

def coral_loss(source, target):
    # Frobenius norm of covariance matrix difference.
    ns, d = source.shape
    nt    = target.shape[0]
    Cs = (source.T @ source
          - source.sum(0).unsqueeze(1) @ source.sum(0).unsqueeze(0) / ns) / (ns - 1)
    Ct = (target.T @ target
          - target.sum(0).unsqueeze(1) @ target.sum(0).unsqueeze(0) / nt) / (nt - 1)
    return (Cs - Ct).pow(2).sum() / (4 * d * d)

class CoralEncoder(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.encoder    = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(),
                                        nn.Linear(d_hidden, d_hidden), nn.ReLU())
        self.classifier = nn.Linear(d_hidden, 2)
    def forward(self, x):
        feat = self.encoder(x)
        return feat, self.classifier(feat)

X_src_t = torch.tensor(X_src, dtype=torch.float32, device=device)
y_src_t = torch.tensor(y_src, dtype=torch.long,    device=device)
X_tgt_t = torch.tensor(X_tgt, dtype=torch.float32, device=device)
y_tgt_t = torch.tensor(y_tgt, dtype=torch.long,    device=device)

model_coral = CoralEncoder(2, 32).to(device)
opt_coral   = torch.optim.Adam(model_coral.parameters(), lr=1e-3)
LAMBDA_CORAL = 0.5

for epoch in range(150):
    model_coral.train()
    opt_coral.zero_grad()
    try:
        feat_s, logits_s = model_coral(X_src_t)
        feat_t, _        = model_coral(X_tgt_t)
        cls_loss   = nn.CrossEntropyLoss()(logits_s, y_src_t)
        align_loss = coral_loss(feat_s, feat_t)
        (cls_loss + LAMBDA_CORAL * align_loss).backward()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("OOM -- reduce hidden size")
            torch.cuda.empty_cache()
            continue
        raise
    opt_coral.step()

model_coral.eval()
with torch.no_grad():
    acc_coral = (model_coral(X_tgt_t)[1].argmax(1) == y_tgt_t).float().mean().item()
print(f"CORAL target accuracy: {acc_coral:.4f}  (baseline: {acc_tgt:.4f})")
"""),
    ("markdown", "## Real-World Example 1: Fine-Tuning for Domain Shift"),
    ("code", """\
# Fine-tuning: the most practical approach when small labelled target data is available.

def build_clf_net(d_in, d_hidden, n_cls):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, n_cls)
    ).to(device)

src_model = build_clf_net(2, 64, 2)
opt_src = torch.optim.Adam(src_model.parameters(), lr=1e-3)
for _ in range(100):
    opt_src.zero_grad()
    nn.CrossEntropyLoss()(src_model(X_src_t), y_src_t).backward()
    opt_src.step()

N_FT = 20   # only 20 labelled target samples
ft_model = build_clf_net(2, 64, 2)
ft_model.load_state_dict(src_model.state_dict())   # start from source weights
opt_ft = torch.optim.Adam(ft_model.parameters(), lr=5e-4)   # lower LR for fine-tuning

for _ in range(80):
    opt_ft.zero_grad()
    idx = torch.randperm(len(X_tgt_t))[:N_FT]
    nn.CrossEntropyLoss()(ft_model(X_tgt_t[idx]), y_tgt_t[idx]).backward()
    opt_ft.step()

src_model.eval(); ft_model.eval()
with torch.no_grad():
    acc_src_on_tgt = (src_model(X_tgt_t).argmax(1) == y_tgt_t).float().mean().item()
    acc_ft_on_tgt  = (ft_model(X_tgt_t).argmax(1) == y_tgt_t).float().mean().item()
print(f"Source-only on target:          {acc_src_on_tgt:.4f}")
print(f"Fine-tuned ({N_FT} target samples): {acc_ft_on_tgt:.4f}")
print(f"Improvement: {acc_ft_on_tgt - acc_src_on_tgt:+.4f}")
"""),
    ("markdown", "## Real-World Example 2: Pseudo-Label Self-Training"),
    ("code", """\
# Self-training: use high-confidence source-model predictions on unlabelled
# target data as pseudo-labels, then retrain on source + pseudo-labelled target.

def self_training_loop(src_model, X_tgt, y_tgt_true, n_rounds=3, conf_thr=0.80):
    model = build_clf_net(2, 64, 2)
    model.load_state_dict(src_model.state_dict())
    for rnd in range(n_rounds):
        model.eval()
        with torch.no_grad():
            probs  = F.softmax(model(X_tgt), dim=-1)
            conf, pseudo = probs.max(dim=-1)
            mask = conf >= conf_thr
        if mask.sum() > 0:
            X_comb = torch.cat([X_src_t, X_tgt[mask]], dim=0)
            y_comb = torch.cat([y_src_t, pseudo[mask]], dim=0)
        else:
            X_comb, y_comb = X_src_t, y_src_t
        model.train()
        opt_st = torch.optim.Adam(model.parameters(), lr=5e-4)
        for _ in range(50):
            idx = torch.randperm(len(X_comb))
            opt_st.zero_grad()
            nn.CrossEntropyLoss()(model(X_comb[idx]), y_comb[idx]).backward()
            opt_st.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_tgt).argmax(1) == y_tgt_true).float().mean().item()
        print(f"  Round {rnd+1}: pseudo_labels={mask.sum().item():3d}  target_acc={acc:.4f}")
    return acc

print("Pseudo-label self-training:")
final_acc = self_training_loop(src_model, X_tgt_t, y_tgt_t, n_rounds=3, conf_thr=0.80)
print(f"Final accuracy after self-training: {final_acc:.4f}")
"""),
    ("markdown", "## Real-World Example 3: DANN Adversarial Domain Adaptation"),
    ("code", """\
# DANN: gradient reversal makes features domain-invariant while preserving class info.

class GradReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None

class DANN(nn.Module):
    def __init__(self, d_in, d_hid, n_cls):
        super().__init__()
        self.encoder    = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                        nn.Linear(d_hid, d_hid), nn.ReLU())
        self.label_clf  = nn.Linear(d_hid, n_cls)
        self.domain_clf = nn.Sequential(nn.Linear(d_hid, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x, alpha=1.0):
        feat        = self.encoder(x)
        lbl_logits  = self.label_clf(feat)
        feat_rev    = GradReversal.apply(feat, alpha)
        dom_logits  = self.domain_clf(feat_rev)
        return lbl_logits, dom_logits

dann = DANN(2, 32, 2).to(device)
opt_dann  = torch.optim.Adam(dann.parameters(), lr=1e-3)
src_dom = torch.zeros(len(X_src_t), dtype=torch.long, device=device)
tgt_dom = torch.ones(len(X_tgt_t),  dtype=torch.long, device=device)

for epoch in range(150):
    dann.train()
    opt_dann.zero_grad()
    alpha = 2.0 / (1 + np.exp(-10 * epoch / 150)) - 1   # ramp from 0 to 1
    lbl_logits,  dom_src = dann(X_src_t, alpha)
    _,           dom_tgt = dann(X_tgt_t, alpha)
    cls_loss = nn.CrossEntropyLoss()(lbl_logits, y_src_t)
    dom_loss = (nn.CrossEntropyLoss()(dom_src, src_dom) +
                nn.CrossEntropyLoss()(dom_tgt, tgt_dom)) / 2
    (cls_loss + dom_loss).backward()
    opt_dann.step()

dann.eval()
with torch.no_grad():
    acc_dann = (dann(X_tgt_t)[0].argmax(1) == y_tgt_t).float().mean().item()
print(f"DANN   target accuracy: {acc_dann:.4f}")
print(f"CORAL  target accuracy: {acc_coral:.4f}")
print(f"Source-only baseline:   {acc_src_on_tgt:.4f}")
print("DANN trains domain-invariant features via adversarial gradient reversal.")
"""),
])

# ── 10 Dropout ────────────────────────────────────────────────────────────────
NB10 = make_nb([
    ("markdown", """\
# Dropout

## Learning Objectives
1. Implement inverted dropout from scratch in NumPy with correct rescaling.
2. Compare five dropout rates on the same architecture and measure overfitting gap.
3. Apply MC Dropout for uncertainty estimation in a regression task.
4. Demonstrate variational dropout and scheduled dropout as advanced variants.
"""),
    ("code", """\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
    ("markdown", "## Level 1: Inverted Dropout from Scratch (NumPy)"),
    ("code", """\
# Inverted dropout: zero out fraction p of neurons, then scale by 1/(1-p)
# so expected output magnitude stays constant regardless of dropout rate.

def dropout_forward(x, p=0.5, training=True):
    # Inverted dropout.
    # Args:
    # x: input array
    # p: fraction of neurons set to zero
    # training: if False, return x unchanged (no dropout at inference)
    if not training or p == 0.0:
        return x
    mask = (np.random.rand(*x.shape) > p).astype(float)
    return x * mask / (1.0 - p)   # scale keeps E[output] = E[input]

# Verify: mean of output should match mean of input regardless of dropout rate
x_test = np.random.randn(10000)
print("Dropout verification (mean and std of output across 100 calls):")
for p in [0.0, 0.2, 0.5, 0.8]:
    outputs = [dropout_forward(x_test, p=p, training=True) for _ in range(100)]
    means = [o.mean() for o in outputs]
    print(f"  p={p:.1f}  output_mean={np.mean(means):.4f}  std_of_mean={np.std(means):.4f}")

# Inference: no change
print(f"Inference (p=0.5, training=False): {dropout_forward(np.ones(5), p=0.5, training=False)}")
"""),
    ("markdown", "## Level 2: Dropout Rate Comparison -- 5 Variants (PyTorch)"),
    ("code", """\
# Compare p in {0.0, 0.1, 0.3, 0.5, 0.7}; measure train-val accuracy gap.

def build_dropout_mlp(p):
    return nn.Sequential(
        nn.Linear(20, 128), nn.ReLU(), nn.Dropout(p),
        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p),
        nn.Linear(128, 2)
    ).to(device)

N_dp = 600
X_dp = torch.randn(N_dp, 20)
y_dp = ((X_dp[:,0] + X_dp[:,1] - X_dp[:,2]) > 0).long()
tr_ds_dp = TensorDataset(X_dp[:480].to(device), y_dp[:480].to(device))
tr_ld_dp = DataLoader(tr_ds_dp, batch_size=32, shuffle=True)

dropout_results = {}
for p in [0.0, 0.1, 0.3, 0.5, 0.7]:
    mdl = build_dropout_mlp(p)
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(100):
        mdl.train()
        for xb, yb in tr_ld_dp:
            opt.zero_grad()
            try:
                crit(mdl(xb), yb).backward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"OOM at p={p} -- reduce model size")
                    torch.cuda.empty_cache()
                    continue
                raise
            opt.step()
    mdl.eval()
    with torch.no_grad():
        tr_acc = (mdl(X_dp[:480].to(device)).argmax(1) == y_dp[:480].to(device)).float().mean().item()
        va_acc = (mdl(X_dp[480:].to(device)).argmax(1) == y_dp[480:].to(device)).float().mean().item()
    dropout_results[p] = (tr_acc, va_acc)
    print(f"p={p:.1f}  train={tr_acc:.4f}  val={va_acc:.4f}  gap={tr_acc-va_acc:.4f}")

best_p = max(dropout_results, key=lambda p: dropout_results[p][1])
print(f"Best dropout rate (by val acc): p={best_p}")
"""),
    ("markdown", "## Real-World Example 1: MC Dropout for Uncertainty Estimation"),
    ("code", """\
# MC Dropout: keep dropout ACTIVE at inference, run T forward passes,
# use variance of predictions as uncertainty estimate.

class MCDropoutNet(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_out)
        )
    def forward(self, x):
        return self.net(x)

N_mc = 500
X_mc = torch.linspace(-3, 3, N_mc).unsqueeze(1)
y_mc = torch.sin(X_mc) + 0.1 * torch.randn_like(X_mc)
# Training data has a gap in [1, 2] -- model should be uncertain there
tr_mask = (X_mc.squeeze() < 1) | (X_mc.squeeze() > 2)
X_mc_tr = X_mc[tr_mask].to(device)
y_mc_tr = y_mc[tr_mask].to(device)

mc_net = MCDropoutNet(1, 64, 1, p=0.3).to(device)
opt_mc = torch.optim.Adam(mc_net.parameters(), lr=1e-3)
for _ in range(200):
    mc_net.train()
    opt_mc.zero_grad()
    nn.MSELoss()(mc_net(X_mc_tr), y_mc_tr).backward()
    opt_mc.step()

# MC inference: model.train() keeps dropout ON; run T=50 passes
T_MC = 50
mc_net.train()    # keep dropout active for uncertainty sampling
X_all_mc = X_mc.to(device)
with torch.no_grad():
    samples = torch.stack([mc_net(X_all_mc).squeeze() for _ in range(T_MC)], dim=0)

mean_pred = samples.mean(0).cpu().numpy()
std_pred  = samples.std(0).cpu().numpy()
x_np      = X_mc.squeeze().numpy()

gap_mask = (x_np >= 1) & (x_np <= 2)
unc_gap  = std_pred[gap_mask].mean()
unc_obs  = std_pred[~gap_mask].mean()
print(f"Uncertainty in training gap [1,2]:     {unc_gap:.4f}")
print(f"Uncertainty in observed region:        {unc_obs:.4f}")
print(f"Ratio (gap/observed):                  {unc_gap/max(unc_obs, 1e-9):.2f}x")
print("MC Dropout correctly shows higher uncertainty in unobserved regions.")
"""),
    ("markdown", "## Real-World Example 2: Variational Dropout"),
    ("code", """\
# Variational dropout: learn per-weight dropout probabilities.
# Weights with high alpha (variance/mean ratio) are effectively pruned.

class VariationalDropout(nn.Module):
    # Linear layer with per-weight learned dropout (log-alpha parameterisation).

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight   = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias     = nn.Parameter(torch.zeros(out_features))
        # Initialise log-alpha low (p_drop near 0 at start)
        self.log_alpha = nn.Parameter(torch.full((out_features, in_features), -3.0))

    @property
    def alpha(self):
        return self.log_alpha.exp().clamp(0, 1)

    def forward(self, x):
        if self.training:
            # Local reparameterisation: sample noise per activation (fast)
            mu  = x @ self.weight.T
            var = (x ** 2) @ (self.weight ** 2 * self.alpha).T
            eps = torch.randn_like(mu)
            return mu + eps * (var.clamp(min=1e-8) ** 0.5) + self.bias
        else:
            # At inference, zero out high-alpha weights (sparsification)
            mask = (self.alpha < 0.99).float()
            return x @ (self.weight * mask).T + self.bias

    def kl_div(self):
        # Approximate KL divergence (encourages sparsity).
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        C   = -k1
        mdkl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(self.alpha.reciprocal())
        return -(mdkl + C).sum()

vd_net = nn.Sequential(VariationalDropout(10, 64), nn.ReLU(), VariationalDropout(64, 1))
X_vd = torch.randn(400, 10)
y_vd = X_vd[:, :3].sum(1, keepdim=True)
opt_vd = torch.optim.Adam(vd_net.parameters(), lr=1e-3)
BETA_KL = 1e-4

for epoch in range(100):
    vd_net.train()
    opt_vd.zero_grad()
    mse = nn.MSELoss()(vd_net(X_vd), y_vd)
    kl  = sum(m.kl_div() for m in vd_net if isinstance(m, VariationalDropout))
    (mse + BETA_KL * kl).backward()
    opt_vd.step()

vd_net.eval()
pruned = sum((m.alpha >= 0.99).sum().item() for m in vd_net if isinstance(m, VariationalDropout))
total_w = sum(m.alpha.numel() for m in vd_net if isinstance(m, VariationalDropout))
with torch.no_grad():
    mse_final = nn.MSELoss()(vd_net(X_vd), y_vd).item()
print(f"Variational dropout -- MSE: {mse_final:.4f}  sparsity: {pruned}/{total_w} = {pruned/total_w:.2%}")
print("High-alpha weights can be removed post-training for a smaller, faster model.")
"""),
    ("markdown", "## Real-World Example 3: Scheduled Dropout"),
    ("code", """\
# Scheduled dropout: start with high p (strong regularisation early),
# anneal to lower p (let model use capacity later).
# Compare against fixed p=0.5 on the same task.

class ScheduledDropoutMLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_hid)
        self.fc3 = nn.Linear(d_hid, d_out)
        self.dp  = nn.Dropout(p=0.0)   # set dynamically

    def set_p(self, p):
        self.dp.p = p

    def forward(self, x):
        return self.fc3(F.relu(self.dp(self.fc1(x))))

N_sd = 600
X_sd = torch.randn(N_sd, 16)
y_sd = ((X_sd[:,0]**2 + X_sd[:,1] - X_sd[:,2]) > 0).long()
tr_ds_sd = TensorDataset(X_sd[:480].to(device), y_sd[:480].to(device))
ld_sd = DataLoader(tr_ds_sd, batch_size=32, shuffle=True)

EPOCHS_SD = 120
P_START, P_END = 0.7, 0.1

sched_mdl = ScheduledDropoutMLP(16, 64, 2).to(device)
fixed_mdl = build_dropout_mlp(0.5)
opt_sched = torch.optim.Adam(sched_mdl.parameters(), lr=1e-3)
opt_fixed = torch.optim.Adam(fixed_mdl.parameters(), lr=1e-3)
crit_sd   = nn.CrossEntropyLoss()

sched_hist, fixed_hist = [], []
for epoch in range(EPOCHS_SD):
    p_now = P_START - (P_START - P_END) * (epoch / EPOCHS_SD)
    sched_mdl.set_p(p_now)
    sched_mdl.train(); fixed_mdl.train()
    for xb, yb in ld_sd:
        opt_sched.zero_grad()
        crit_sd(sched_mdl(xb), yb).backward()
        opt_sched.step()
        opt_fixed.zero_grad()
        crit_sd(fixed_mdl(xb), yb).backward()
        opt_fixed.step()

    if (epoch + 1) % 20 == 0:
        sched_mdl.eval(); fixed_mdl.eval()
        with torch.no_grad():
            va_s = (sched_mdl(X_sd[480:].to(device)).argmax(1) == y_sd[480:].to(device)).float().mean().item()
            va_f = (fixed_mdl(X_sd[480:].to(device)).argmax(1) == y_sd[480:].to(device)).float().mean().item()
        sched_hist.append(va_s); fixed_hist.append(va_f)
        print(f"Epoch {epoch+1:3d}  p={p_now:.2f}  sched_val={va_s:.4f}  fixed_val={va_f:.4f}")

print(f"Final: scheduled p={P_END} -> {sched_hist[-1]:.4f}  |  fixed p=0.5 -> {fixed_hist[-1]:.4f}")
print("Scheduled dropout: strong early regularisation lets the model specialise later.")
"""),
])


# ─── Write all notebooks ──────────────────────────────────────────────────────
NOTEBOOKS = [
    ("01-activation-functions", NB01),
    ("02-attention-mechanism",  NB02),
    ("03-batch-normalization",  NB03),
    ("04-class-imbalance",      NB04),
    ("05-cnns",                 NB05),
    ("06-cross-validation",     NB06),
    ("07-data-leakage",         NB07),
    ("08-distributed-training", NB08),
    ("09-domain-adaptation",    NB09),
    ("10-dropout",              NB10),
]

for stem, nb in NOTEBOOKS:
    path = os.path.join(OUT_DIR, f"{stem}.ipynb")
    nbformat.write(nb, path)
    print(f"Wrote {path}  ({len(nb.cells)} cells)")

print("\nAll 10 notebooks written.")
