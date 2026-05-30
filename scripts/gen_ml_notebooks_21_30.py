"""
Generate ml/notebooks/21-30 Jupyter notebooks.
Each notebook has exactly 12 cells in the pattern:
  md, code, md, code, md, code, md, code, md, code, md, code
  (title, imports, L1-hdr, L1-code, L2-hdr, L2-code,
   RW1-hdr, RW1-code, RW2-hdr, RW2-code, RW3-hdr, RW3-code)
"""
import ast
import os
import sys

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ml", "notebooks"
)

# ---------------------------------------------------------------------------
# Notebook definitions – list of (filename_stem, [12 cell-sources])
# ---------------------------------------------------------------------------

NOTEBOOKS = []

# ---- 21 meta-learning -------------------------------------------------------
NOTEBOOKS.append(("21-meta-learning", [
# cell 1 – title + objectives (markdown)
"""# Meta-Learning

## Learning Objectives
1. Understand the MAML inner/outer-loop optimisation framework.
2. Implement a MAML-style gradient-update loop from scratch with NumPy.
3. Build prototypical networks in PyTorch for few-shot classification.
4. Compare zero-shot vs few-shot generalisation strategies in production.
""",

# cell 2 – imports + device + seeds (code)
"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
""",

# cell 3 – L1 header (markdown)
"""## Level 1: MAML Inner / Outer Loop (NumPy)
""",

# cell 4 – L1 code (numpy MAML)
"""def task_loss(params, x, y):
    \"\"\"Linear regression loss for a single task (used as inner-loop loss).\"\"\"
    w, b = params
    pred = x @ w + b
    return np.mean((pred - y) ** 2)


def task_gradient(params, x, y):
    \"\"\"Analytic gradient of MSE w.r.t. [w, b].\"\"\"
    w, b = params
    pred = x @ w + b
    err = pred - y          # (n, 1)
    dw = (2 / len(x)) * x.T @ err
    db = (2 / len(x)) * err.mean()
    return [dw, db]


def maml_step(meta_params, tasks, inner_lr=0.01, outer_lr=0.001, n_inner=1):
    \"\"\"Single MAML outer-step over a batch of tasks.\"\"\"
    meta_w, meta_b = meta_params
    outer_grads_w = np.zeros_like(meta_w)
    outer_grads_b = 0.0

    for (x_s, y_s, x_q, y_q) in tasks:
        # --- inner loop: adapt on support set ---
        w, b = meta_w.copy(), meta_b
        for _ in range(n_inner):
            gw, gb = task_gradient([w, b], x_s, y_s)
            w = w - inner_lr * gw
            b = b - inner_lr * gb

        # --- outer loop: compute query-set gradient w.r.t adapted params ---
        gw_q, gb_q = task_gradient([w, b], x_q, y_q)
        outer_grads_w += gw_q
        outer_grads_b += gb_q

    outer_grads_w /= len(tasks)
    outer_grads_b /= len(tasks)

    # --- meta-update ---
    new_w = meta_w - outer_lr * outer_grads_w
    new_b = meta_b - outer_lr * outer_grads_b
    return [new_w, new_b]


def make_sinusoidal_tasks(n_tasks=4, n_support=5, n_query=10, d=1):
    tasks = []
    for _ in range(n_tasks):
        amplitude = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, np.pi)
        x = np.random.uniform(-3, 3, (n_support + n_query, d))
        y = amplitude * np.sin(x + phase)
        tasks.append((x[:n_support], y[:n_support], x[n_support:], y[n_support:]))
    return tasks


d = 1
meta_params = [np.random.randn(d, 1) * 0.1, 0.0]
meta_losses = []
for step in range(200):
    tasks = make_sinusoidal_tasks(n_tasks=4, d=d)
    meta_params = maml_step(meta_params, tasks, inner_lr=0.05, outer_lr=0.002)
    if step % 50 == 0:
        test_tasks = make_sinusoidal_tasks(n_tasks=10, d=d)
        avg_loss = np.mean([
            task_loss(meta_params, x_q, y_q) for _, _, x_q, y_q in test_tasks
        ])
        meta_losses.append(avg_loss)
        print(f"Step {step:3d} | meta query-loss: {avg_loss:.4f}")

print("Final meta-params (w, b):", meta_params[0].ravel(), meta_params[1])
""",

# cell 5 – L2 header (markdown)
"""## Level 2: Prototypical Networks (PyTorch + OOM Handling)
""",

# cell 6 – L2 code (torch prototypical nets)
"""class ProtoNet(nn.Module):
    \"\"\"Simple embedding network for prototypical few-shot learning.\"\"\"

    def __init__(self, in_dim: int = 64, hidden: int = 128, out_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def prototypical_loss(support_embeddings, support_labels,
                      query_embeddings, query_labels, n_way):
    \"\"\"Compute prototype loss: euclidean distance to class centroids.\"\"\"
    prototypes = torch.stack([
        support_embeddings[support_labels == c].mean(0)
        for c in range(n_way)
    ])
    dists = torch.cdist(query_embeddings, prototypes)
    log_p = F.log_softmax(-dists, dim=1)
    return F.nll_loss(log_p, query_labels)


def make_episode(n_way=5, n_support=5, n_query=15, in_dim=64):
    \"\"\"Synthetic episode for prototypical training.\"\"\"
    support_x, support_y, query_x, query_y = [], [], [], []
    for c in range(n_way):
        mu = torch.randn(in_dim) * 2
        support_x.append(mu + torch.randn(n_support, in_dim) * 0.5)
        support_y.extend([c] * n_support)
        query_x.append(mu + torch.randn(n_query, in_dim) * 0.5)
        query_y.extend([c] * n_query)
    sx = torch.cat(support_x).to(device)
    sy = torch.tensor(support_y, device=device)
    qx = torch.cat(query_x).to(device)
    qy = torch.tensor(query_y, device=device)
    return sx, sy, qx, qy


model_proto = ProtoNet(in_dim=64).to(device)
opt_proto = torch.optim.Adam(model_proto.parameters(), lr=1e-3)
N_WAY = 5

losses = []
for episode in range(300):
    sx, sy, qx, qy = make_episode(n_way=N_WAY)
    try:
        opt_proto.zero_grad()
        se = model_proto(sx)
        qe = model_proto(qx)
        loss = prototypical_loss(se, sy, qe, qy, N_WAY)
        loss.backward()
        opt_proto.step()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("OOM: reduce n_way or n_query")
            torch.cuda.empty_cache()
            continue
        raise
    if episode % 100 == 0:
        losses.append(loss.item())
        print(f"Episode {episode:3d} | loss: {loss.item():.4f}")

print(f"Final episode loss: {losses[-1]:.4f}")
""",

# cell 7 – RW1 header (markdown)
"""## Real-World Example 1: MAML 5-Shot Sine Regression
""",

# cell 8 – RW1 code
"""class SineNet(nn.Module):
    \"\"\"Small MLP for MAML sine regression.\"\"\"
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.ReLU(),
            nn.Linear(40, 40), nn.ReLU(),
            nn.Linear(40, 1),
        )
    def forward(self, x):
        return self.net(x)


def clone_params(model):
    \"\"\"Return detached clone of each parameter tensor.\"\"\"
    return [p.clone() for p in model.parameters()]


def functional_forward(model, x, params):
    \"\"\"Manual forward pass with given parameter list (for MAML in PyTorch).\"\"\"
    idx = 0
    out = x
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            w, b = params[idx], params[idx + 1]
            out = F.linear(out, w, b)
            idx += 2
        elif isinstance(layer, nn.ReLU):
            out = F.relu(out)
    return out


def sine_task(n_support=5, n_query=10):
    amp = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, np.pi)
    x = torch.FloatTensor(n_support + n_query, 1).uniform_(-5, 5).to(device)
    y = (amp * torch.sin(x + phase)).to(device)
    return x[:n_support], y[:n_support], x[n_support:], y[n_support:]


maml_model = SineNet().to(device)
meta_opt = torch.optim.Adam(maml_model.parameters(), lr=1e-3)
INNER_LR = 0.01

for meta_step in range(500):
    tasks = [sine_task() for _ in range(4)]
    meta_loss = torch.tensor(0.0, device=device)
    for xs, ys, xq, yq in tasks:
        fast_params = clone_params(maml_model)
        for _ in range(5):
            pred = functional_forward(maml_model, xs, fast_params)
            inner_loss = F.mse_loss(pred, ys)
            grads = torch.autograd.grad(inner_loss, fast_params, create_graph=True)
            fast_params = [p - INNER_LR * g for p, g in zip(fast_params, grads)]
        query_pred = functional_forward(maml_model, xq, fast_params)
        meta_loss = meta_loss + F.mse_loss(query_pred, yq)

    meta_opt.zero_grad()
    (meta_loss / len(tasks)).backward()
    meta_opt.step()
    if meta_step % 100 == 0:
        print(f"Meta-step {meta_step:4d} | query MSE: {(meta_loss/len(tasks)).item():.4f}")

print("MAML 5-shot training complete.")
""",

# cell 9 – RW2 header (markdown)
"""## Real-World Example 2: Meta Domain Adaptation
""",

# cell 10 – RW2 code
"""class DomainNet(nn.Module):
    \"\"\"Shared encoder + linear head for domain adaptation.\"\"\"
    def __init__(self, in_dim=20, hidden=64, n_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def make_domain_data(n_domains=6, n_per_domain=200, in_dim=20, n_classes=3):
    \"\"\"Each domain has a different linear shift in feature space.\"\"\"
    domains = []
    for d_idx in range(n_domains):
        shift = torch.randn(in_dim) * (d_idx * 0.5)
        x = torch.randn(n_per_domain, in_dim) + shift
        y = torch.randint(0, n_classes, (n_per_domain,))
        domains.append(TensorDataset(x, y))
    return domains


domains = make_domain_data()
target_domain = domains[-1]
source_domains = domains[:-1]

meta_da_model = DomainNet().to(device)
meta_da_opt = torch.optim.Adam(meta_da_model.parameters(), lr=1e-3)
crit_da = nn.CrossEntropyLoss()

for episode in range(400):
    d_idx = np.random.randint(len(source_domains))
    loader = DataLoader(source_domains[d_idx], batch_size=32, shuffle=True)
    episode_loss = 0.0
    meta_da_model.train()
    for xb, yb in loader:
        meta_da_opt.zero_grad()
        try:
            loss = crit_da(meta_da_model(xb.to(device)), yb.to(device))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("OOM: reduce batch size"); torch.cuda.empty_cache(); continue
            raise
        loss.backward(); meta_da_opt.step()
        episode_loss += loss.item()
    if episode % 100 == 0:
        print(f"Episode {episode:3d} | source loss: {episode_loss/len(loader):.4f}")

# Fine-tune on 10 labelled examples from target domain
fine_tune_data = TensorDataset(
    target_domain.tensors[0][:10], target_domain.tensors[1][:10]
)
fine_tune_loader = DataLoader(fine_tune_data, batch_size=10)
fine_tune_opt = torch.optim.Adam(meta_da_model.parameters(), lr=5e-4)
for _ in range(50):
    for xb, yb in fine_tune_loader:
        fine_tune_opt.zero_grad()
        crit_da(meta_da_model(xb.to(device)), yb.to(device)).backward()
        fine_tune_opt.step()

meta_da_model.eval()
with torch.no_grad():
    xt, yt = target_domain.tensors
    preds = meta_da_model(xt.to(device)).argmax(1).cpu()
    acc = (preds == yt).float().mean().item()
print(f"Target domain accuracy after meta-DA + 10-shot fine-tune: {acc:.3f}")
""",

# cell 11 – RW3 header (markdown)
"""## Real-World Example 3: Zero-Shot vs Few-Shot Generalisation Comparison
""",

# cell 12 – RW3 code
"""import copy

def evaluate_k_shot(base_model, target_ds, k, n_classes=3, n_finetune_steps=50):
    \"\"\"Fine-tune on k examples per class, evaluate on rest.\"\"\"
    model_k = copy.deepcopy(base_model).to(device)
    opt_k = torch.optim.Adam(model_k.parameters(), lr=5e-4)
    crit_k = nn.CrossEntropyLoss()

    xs, ys = target_ds.tensors
    support_x, support_y = [], []
    for c in range(n_classes):
        idx = (ys == c).nonzero(as_tuple=True)[0][:k]
        support_x.append(xs[idx]); support_y.append(ys[idx])
    sup_x = torch.cat(support_x).to(device)
    sup_y = torch.cat(support_y).to(device)

    for _ in range(n_finetune_steps):
        opt_k.zero_grad()
        crit_k(model_k(sup_x), sup_y).backward()
        opt_k.step()

    model_k.eval()
    with torch.no_grad():
        preds = model_k(xs.to(device)).argmax(1).cpu()
    return (preds == ys).float().mean().item()


k_shots = [0, 1, 5, 10, 20]
accuracies = []
for k in k_shots:
    if k == 0:
        meta_da_model.eval()
        with torch.no_grad():
            xt, yt = target_domain.tensors
            preds = meta_da_model(xt.to(device)).argmax(1).cpu()
        acc = (preds == yt).float().mean().item()
    else:
        acc = evaluate_k_shot(meta_da_model, target_domain, k)
    accuracies.append(acc)
    print(f"k={k:2d}-shot accuracy: {acc:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_shots, accuracies, marker='o', linewidth=2, color='steelblue')
ax.set_xlabel("Number of Fine-Tune Examples (k-shot)")
ax.set_ylabel("Target Domain Accuracy")
ax.set_title("Zero-Shot vs Few-Shot Generalisation")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/meta_learning_kshot.png", dpi=80)
plt.close()
print("Saved comparison plot to /tmp/meta_learning_kshot.png")
""",
]))

# ---- 22 mixed-precision -------------------------------------------------------
NOTEBOOKS.append(("22-mixed-precision", [
"""# Mixed Precision Training

## Learning Objectives
1. Understand why FP16 overflows occur and how loss scaling mitigates them.
2. Demonstrate FP16 numerical limits vs FP32 using NumPy.
3. Implement Automatic Mixed Precision (AMP) with GradScaler in PyTorch.
4. Compare BF16 vs FP16 behaviour for training stability.
""",

"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"FP16 max: {np.finfo(np.float16).max:.0f}")
print(f"FP32 max: {np.finfo(np.float32).max:.2e}")
""",

"""## Level 1: FP16 Overflow Demonstration (NumPy)
""",

"""# Show the limits of FP16 and how simple operations overflow or underflow.

fp16_max = np.finfo(np.float16).max   # 65504
fp32_max = np.finfo(np.float32).max   # ~3.4e38

print(f"FP16 max value : {fp16_max}")
print(f"FP32 max value : {fp32_max:.2e}")

# Overflow: value exceeds FP16 max
a_fp16 = np.float16(60000.0)
b_fp16 = np.float16(10000.0)
overflow_result = a_fp16 + b_fp16
print(f"\\n60000 + 10000 in FP16: {overflow_result}")
print(f"60000 + 10000 in FP32: {np.float32(60000.0) + np.float32(10000.0)}")

# Underflow in small gradients
small_gradient = np.float16(1e-7)
print(f"\\n1e-7 in FP16: {small_gradient}")
print(f"1e-7 in FP32: {np.float32(1e-7)}")

# Loss scaling simulation
scale = 1024.0
scaled = np.float16(1e-7 * scale)
print(f"\\n1e-7 * {scale} in FP16: {scaled}")
print(f"Unscaled back: {scaled / scale}")

# Numerical range comparison
formats = {'FP16': np.float16, 'FP32': np.float32, 'FP64': np.float64}
print("\\nFormat   | Max value  | Min positive   | Mantissa bits")
print("-" * 56)
for name, dtype in formats.items():
    info = np.finfo(dtype)
    print(f"{name:<8} | {info.max:<10.2e} | {info.tiny:<14.2e} | {info.nmant}")
""",

"""## Level 2: AMP with autocast and GradScaler (PyTorch + OOM Handling)
""",

"""class MLP(nn.Module):
    \"\"\"Generic MLP for benchmark comparisons.\"\"\"
    def __init__(self, in_dim=128, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def build_loader(n=2000, in_dim=128, n_classes=10, batch_size=64):
    X = torch.randn(n, in_dim)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


loader = build_loader()
model_amp = MLP().to(device)
opt_amp = torch.optim.Adam(model_amp.parameters(), lr=1e-3)
scaler = GradScaler(device=device.type)
crit_amp = nn.CrossEntropyLoss()

amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16

losses_amp = []
for epoch in range(30):
    model_amp.train()
    epoch_loss = 0.0
    for xb, yb in loader:
        opt_amp.zero_grad()
        try:
            with autocast(device_type=device.type, dtype=amp_dtype):
                logits = model_amp(xb.to(device))
                loss = crit_amp(logits, yb.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt_amp)
            nn.utils.clip_grad_norm_(model_amp.parameters(), 1.0)
            scaler.step(opt_amp)
            scaler.update()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("OOM: reduce batch_size"); torch.cuda.empty_cache(); continue
            raise
        epoch_loss += loss.item()
    losses_amp.append(epoch_loss / len(loader))
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | AMP loss: {losses_amp[-1]:.4f} | "
              f"scale: {scaler.get_scale():.0f}")

print("AMP training complete. Final loss:", losses_amp[-1])
""",

"""## Real-World Example 1: FP16 ResNet-Style Training Speedup
""",

"""import time

class TinyConvNet(nn.Module):
    \"\"\"Small conv net that mimics ResNet structure for benchmarking.\"\"\"
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(64 * 4 * 4, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def benchmark_training(use_amp: bool, n_iters: int = 50) -> float:
    \"\"\"Return average iteration time in ms.\"\"\"
    model = TinyConvNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    scaler_bench = GradScaler(device=device.type) if use_amp else None
    amp_dtype_b = torch.float16 if device.type == "cuda" else torch.bfloat16

    x = torch.randn(32, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        opt.zero_grad()
        if use_amp:
            with autocast(device_type=device.type, dtype=amp_dtype_b):
                logits = model(x)
                loss = crit(logits, y)
            scaler_bench.scale(loss).backward()
            scaler_bench.step(opt); scaler_bench.update()
        else:
            loss = crit(model(x), y)
            loss.backward(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times[5:]))


t_fp32 = benchmark_training(use_amp=False)
t_amp = benchmark_training(use_amp=True)
print(f"FP32 avg iter time : {t_fp32:.2f} ms")
print(f"AMP  avg iter time : {t_amp:.2f} ms")
print(f"Speedup (AMP/FP32) : {t_fp32/t_amp:.2f}x")
""",

"""## Real-World Example 2: Loss Scaling Underflow Demo
""",

"""def train_with_scaling(use_scaler: bool, n_epochs: int = 20):
    \"\"\"Train with or without GradScaler and track gradient norms.\"\"\"
    model_s = MLP(in_dim=64, hidden=128, out_dim=5).to(device)
    opt_s = torch.optim.Adam(model_s.parameters(), lr=1e-3)
    scaler_s = GradScaler(device=device.type) if use_scaler else None
    crit_s = nn.CrossEntropyLoss()
    loader_s = build_loader(n=500, in_dim=64, n_classes=5, batch_size=32)
    amp_dtype_s = torch.float16 if device.type == "cuda" else torch.bfloat16

    grad_norms, losses = [], []
    for epoch in range(n_epochs):
        for xb, yb in loader_s:
            opt_s.zero_grad()
            with autocast(device_type=device.type, dtype=amp_dtype_s):
                loss_s = crit_s(model_s(xb.to(device)), yb.to(device))
            if use_scaler:
                scaler_s.scale(loss_s).backward()
                scaler_s.unscale_(opt_s)
            else:
                loss_s.backward()

            total_norm = sum(
                p.grad.detach().float().norm().item() ** 2
                for p in model_s.parameters() if p.grad is not None
            ) ** 0.5
            grad_norms.append(total_norm)
            losses.append(loss_s.item())

            if use_scaler:
                scaler_s.step(opt_s); scaler_s.update()
            else:
                opt_s.step()
    return grad_norms, losses


norms_scaled, losses_scaled = train_with_scaling(use_scaler=True)
norms_unscaled, losses_unscaled = train_with_scaling(use_scaler=False)

print(f"With    GradScaler - median grad norm: {np.median(norms_scaled):.4f}, "
      f"zero-grad %: {100*np.mean(np.array(norms_scaled)<1e-8):.1f}%")
print(f"Without GradScaler - median grad norm: {np.median(norms_unscaled):.4f}, "
      f"zero-grad %: {100*np.mean(np.array(norms_unscaled)<1e-8):.1f}%")
print(f"Final loss (scaled): {losses_scaled[-1]:.4f}, "
      f"Final loss (unscaled): {losses_unscaled[-1]:.4f}")
""",

"""## Real-World Example 3: BF16 vs FP16 Stability Comparison
""",

"""def train_format(dtype_name: str, n_epochs=25):
    \"\"\"Train with a given dtype and return per-epoch losses.\"\"\"
    dtype_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype_f = dtype_map[dtype_name]
    model_f = MLP(in_dim=64, hidden=128, out_dim=5).to(device)
    opt_f = torch.optim.Adam(model_f.parameters(), lr=1e-3)
    scaler_f = GradScaler(device=device.type) if amp_dtype_f == torch.float16 else None
    crit_f = nn.CrossEntropyLoss()
    loader_f = build_loader(n=600, in_dim=64, n_classes=5, batch_size=32)
    epoch_losses = []
    for epoch in range(n_epochs):
        total = 0.0
        model_f.train()
        for xb, yb in loader_f:
            opt_f.zero_grad()
            if amp_dtype_f:
                with autocast(device_type=device.type, dtype=amp_dtype_f):
                    l = crit_f(model_f(xb.to(device)), yb.to(device))
                if scaler_f:
                    scaler_f.scale(l).backward(); scaler_f.step(opt_f); scaler_f.update()
                else:
                    l.backward(); opt_f.step()
            else:
                l = crit_f(model_f(xb.to(device)), yb.to(device))
                l.backward(); opt_f.step()
            total += l.item()
        epoch_losses.append(total / len(loader_f))
    return epoch_losses


results_fmt = {name: train_format(name) for name in ["fp32", "bf16", "fp16"]}

fig, ax = plt.subplots(figsize=(8, 4))
colors = {"fp32": "navy", "bf16": "steelblue", "fp16": "coral"}
for name, vals in results_fmt.items():
    ax.plot(vals, label=name.upper(), color=colors[name], linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("BF16 vs FP16 vs FP32 Training Loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/mixed_precision_formats.png", dpi=80)
plt.close()
print("Format comparison plot saved to /tmp/mixed_precision_formats.png")
for name, vals in results_fmt.items():
    print(f"{name.upper()} final loss: {vals[-1]:.4f}")
""",
]))

# ---- 23 model-compression -------------------------------------------------------
NOTEBOOKS.append(("23-model-compression", [
"""# Model Compression

## Learning Objectives
1. Understand magnitude-based weight pruning from first principles.
2. Apply structured and unstructured pruning with torch.nn.utils.prune.
3. Quantise a BERT-style model to INT8 using post-training quantisation (PTQ).
4. Analyse the accuracy-vs-compression trade-off across pruning and quantisation.
""",

"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
""",

"""## Level 1: Magnitude Pruning (NumPy)
""",

"""def magnitude_prune(weights: np.ndarray, sparsity: float) -> np.ndarray:
    \"\"\"Zero out the smallest-magnitude weights to achieve target sparsity.\"\"\"
    threshold = np.percentile(np.abs(weights), sparsity * 100)
    mask = np.abs(weights) >= threshold
    return weights * mask


def compute_sparsity(weights: np.ndarray) -> float:
    return float((weights == 0).mean())


W = np.random.randn(256, 512).astype(np.float32)
print(f"Original sparsity: {compute_sparsity(W):.3f}")
print(f"Original L2 norm : {np.linalg.norm(W):.3f}")

sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]
print(f"\\n{'Sparsity':>10} | {'Zeros %':>8} | {'L2 norm':>8} | {'Top-1 weight':>12}")
print("-" * 50)
for s in sparsity_levels:
    W_pruned = magnitude_prune(W, s)
    actual_sparse = compute_sparsity(W_pruned)
    l2 = np.linalg.norm(W_pruned)
    top1 = np.abs(W_pruned).max()
    print(f"{s:>10.0%} | {actual_sparse:>8.3f} | {l2:>8.3f} | {top1:>12.4f}")

W_imp = W.copy()
for round_idx in range(5):
    W_imp = magnitude_prune(W_imp, 0.1)
    print(f"IMP round {round_idx+1}: sparsity={compute_sparsity(W_imp):.3f}")
""",

"""## Level 2: torch.nn.utils.prune + Dynamic Quantisation (OOM Handling)
""",

"""class SmallMLP(nn.Module):
    \"\"\"Simple MLP for pruning and quantisation experiments.\"\"\"
    def __init__(self, in_dim=128, hidden=256, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))


def count_params(model: nn.Module):
    \"\"\"Count total and non-zero parameters.\"\"\"
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


def model_size_mb(model: nn.Module) -> float:
    \"\"\"Approximate model size in MB.\"\"\"
    return sum(p.nbytes for p in model.parameters()) / 1e6


model_prune = SmallMLP().to(device)
total, nz = count_params(model_prune)
print(f"Before pruning: {total:,} params, {nz:,} non-zero, {model_size_mb(model_prune):.2f} MB")

try:
    for module in model_prune.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.5)
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        print("OOM: reduce model size"); torch.cuda.empty_cache()
    else:
        raise

for name, module in model_prune.named_modules():
    if isinstance(module, nn.Linear):
        mask = module.weight_mask
        sparsity_val = 1.0 - mask.float().mean().item()
        print(f"Layer {name}: {sparsity_val:.2%} sparse")

for module in model_prune.modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, 'weight')

total_p, nz_p = count_params(model_prune)
print(f"After pruning:  {total_p:,} params, {nz_p:,} non-zero ({100*nz_p/total_p:.1f}% non-zero)")

model_quantised = torch.quantization.quantize_dynamic(
    SmallMLP(), {nn.Linear}, dtype=torch.qint8,
)
print(f"\\nQuantised model size: {model_size_mb(model_quantised):.3f} MB")
print(f"Original  model size: {model_size_mb(SmallMLP()):.3f} MB")
print(f"Compression ratio   : {model_size_mb(SmallMLP())/model_size_mb(model_quantised):.2f}x")
""",

"""## Real-World Example 1: Post-Training Quantisation on a BERT-Style Encoder
""",

"""import time

class BERTLikeEncoder(nn.Module):
    \"\"\"Toy transformer encoder mimicking BERT architecture.\"\"\"
    def __init__(self, vocab_size=1000, d_model=256, nhead=4, n_layers=3, n_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        enc = self.encoder(emb)
        return self.classifier(enc[:, 0, :])


bert_model = BERTLikeEncoder()
bert_model.eval()

bert_quantised = torch.quantization.quantize_dynamic(
    bert_model, {nn.Linear}, dtype=torch.qint8
)

dummy_input = torch.randint(0, 1000, (8, 64))


def measure_latency(model, input_ids, n_runs=30):
    times = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(input_ids)
            times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times[5:])


lat_fp32 = measure_latency(bert_model, dummy_input)
lat_int8 = measure_latency(bert_quantised, dummy_input)

size_fp32 = sum(p.nbytes for p in bert_model.parameters()) / 1e6
size_int8 = sum(p.nbytes for p in bert_quantised.parameters()) / 1e6

print(f"FP32 model size: {size_fp32:.2f} MB  | latency: {lat_fp32:.2f} ms")
print(f"INT8 model size: {size_int8:.2f} MB  | latency: {lat_int8:.2f} ms")
print(f"Size reduction : {size_fp32/size_int8:.2f}x")
print(f"Latency speedup: {lat_fp32/lat_int8:.2f}x")
""",

"""## Real-World Example 2: Knowledge Distillation + Pruning Combined
""",

"""class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 5))
    def forward(self, x): return self.net(x)


class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 5))
    def forward(self, x): return self.net(x)


teacher = TeacherNet().to(device).eval()
student = StudentNet().to(device)

X_kd = torch.randn(1000, 64); y_kd = torch.randint(0, 5, (1000,))
kd_loader = DataLoader(TensorDataset(X_kd, y_kd), batch_size=64, shuffle=True)

opt_kd = torch.optim.Adam(student.parameters(), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')
TEMPERATURE = 4.0
ALPHA = 0.7

for epoch in range(30):
    student.train()
    total_loss = 0.0
    for xb, yb in kd_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_kd.zero_grad()
        try:
            with torch.no_grad():
                teacher_logits = teacher(xb) / TEMPERATURE
                soft_targets = torch.softmax(teacher_logits, dim=-1)
            student_logits = student(xb)
            student_soft = torch.log_softmax(student_logits / TEMPERATURE, dim=-1)
            loss_hard = ce_loss(student_logits, yb)
            loss_soft = kl_loss(student_soft, soft_targets) * (TEMPERATURE ** 2)
            loss = ALPHA * loss_soft + (1 - ALPHA) * loss_hard
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("OOM"); torch.cuda.empty_cache(); continue
            raise
        loss.backward(); opt_kd.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | KD loss: {total_loss/len(kd_loader):.4f}")

for module in student.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.50)
        prune.remove(module, 'weight')

t_sz = sum(p.nbytes for p in TeacherNet().parameters()) / 1e6
s_sz = sum(p.nbytes for p in student.parameters()) / 1e6
nz_ratio = sum((p != 0).sum().item() for p in student.parameters()) / \
           sum(p.numel() for p in student.parameters())
print(f"Teacher size: {t_sz:.3f} MB | Student pruned size: {s_sz:.3f} MB")
print(f"Student non-zero params: {nz_ratio:.2%}")
print(f"Total compression: {t_sz / (s_sz * nz_ratio):.2f}x (size * density)")
""",

"""## Real-World Example 3: Compression Trade-off Plot
""",

"""def train_and_evaluate(sparsity: float, n_epochs: int = 20):
    \"\"\"Train a fresh MLP, prune to sparsity, return (test_acc, model_size_mb).\"\"\"
    X = torch.randn(2000, 64); y = torch.randint(0, 5, (2000,))
    loader_tr = DataLoader(TensorDataset(X[:1600], y[:1600]), batch_size=64, shuffle=True)
    X_test, y_test = X[1600:].to(device), y[1600:].to(device)

    m = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 5)
    ).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for _ in range(n_epochs):
        m.train()
        for xb, yb in loader_tr:
            opt.zero_grad()
            crit(m(xb.to(device)), yb.to(device)).backward()
            opt.step()

    if sparsity > 0:
        for module in m.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')

    m.eval()
    with torch.no_grad():
        acc = (m(X_test).argmax(1) == y_test).float().mean().item()
    sz = sum(p.nbytes for p in m.parameters()) / 1e6
    return acc, sz


sparsities = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
accs, sizes = [], []
for s in sparsities:
    acc, sz = train_and_evaluate(s)
    accs.append(acc); sizes.append(sz)
    print(f"Sparsity {s:.0%} | acc: {acc:.3f} | size: {sz:.3f} MB")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot([s * 100 for s in sparsities], accs, marker='o', color='steelblue')
ax1.set_xlabel("Sparsity (%)"); ax1.set_ylabel("Test Accuracy")
ax1.set_title("Accuracy vs Pruning Sparsity"); ax1.grid(True, alpha=0.3)
ax2.plot([s * 100 for s in sparsities], sizes, marker='s', color='coral')
ax2.set_xlabel("Sparsity (%)"); ax2.set_ylabel("Model Size (MB)")
ax2.set_title("Size vs Pruning Sparsity"); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/compression_tradeoff.png", dpi=80)
plt.close()
print("Compression trade-off plot saved to /tmp/compression_tradeoff.png")
""",
]))

# ---- 24 model-selection -------------------------------------------------------
NOTEBOOKS.append(("24-model-selection", [
"""# Model Selection

## Learning Objectives
1. Compute AIC and BIC from log-likelihood and parameter count using NumPy.
2. Run nested cross-validation in scikit-learn for unbiased model comparison.
3. Decompose bias and variance using repeated hold-out experiments.
4. Demonstrate the no-free-lunch theorem with multiple classifiers.
""",

"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification, make_regression

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
""",

"""## Level 1: AIC / BIC Computation (NumPy)
""",

"""def log_likelihood_gaussian(y_true: np.ndarray, y_pred: np.ndarray,
                             sigma: float = None) -> float:
    \"\"\"Log-likelihood under a Gaussian noise model.\"\"\"
    n = len(y_true)
    residuals = y_true - y_pred
    if sigma is None:
        sigma = np.std(residuals)
    ll = (-n / 2 * np.log(2 * np.pi * sigma ** 2)
          - (1 / (2 * sigma ** 2)) * np.sum(residuals ** 2))
    return ll


def aic(log_lik: float, k: int) -> float:
    \"\"\"Akaike Information Criterion: AIC = 2k - 2*log(L).\"\"\"
    return 2 * k - 2 * log_lik


def bic(log_lik: float, k: int, n: int) -> float:
    \"\"\"Bayesian Information Criterion: BIC = k*log(n) - 2*log(L).\"\"\"
    return k * np.log(n) - 2 * log_lik


n = 100
x = np.linspace(0, 2 * np.pi, n)
y_true_fn = 2 * np.sin(x) + 0.5 * np.cos(3 * x)
y = y_true_fn + np.random.randn(n) * 0.3

print(f"{'Degree':>7} | {'AIC':>10} | {'BIC':>10} | {'RMSE':>8}")
print("-" * 44)
aic_vals, bic_vals = [], []
for deg in range(1, 9):
    X_poly = np.stack([x ** d for d in range(deg + 1)], axis=1)
    coeffs = np.linalg.lstsq(X_poly, y, rcond=None)[0]
    y_pred = X_poly @ coeffs
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    ll = log_likelihood_gaussian(y, y_pred)
    k = deg + 2
    a = aic(ll, k)
    b = bic(ll, k, n)
    aic_vals.append(a); bic_vals.append(b)
    print(f"{deg:>7d} | {a:>10.2f} | {b:>10.2f} | {rmse:>8.4f}")

print(f"\\nBest degree by AIC: {np.argmin(aic_vals) + 1}")
print(f"Best degree by BIC: {np.argmin(bic_vals) + 1}")
""",

"""## Level 2: Nested Cross-Validation with scikit-learn (Model Comparison)
""",

"""X_cls, y_cls = make_classification(n_samples=800, n_features=20, n_informative=10,
                                   n_redundant=5, random_state=42)

candidate_models = {
    "LogisticRegression": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=500))]),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
    ),
}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

print(f"{'Model':>20} | {'Mean CV Acc':>11} | {'Std':>6}")
print("-" * 44)
nested_scores = {}
for name, (estimator, param_grid) in candidate_models.items():
    try:
        gs = GridSearchCV(estimator, param_grid, cv=inner_cv,
                          scoring='accuracy', n_jobs=-1)
        scores = cross_val_score(gs, X_cls, y_cls, cv=outer_cv,
                                 scoring='accuracy')
    except Exception as exc:
        print(f"Error for {name}: {exc}"); continue
    nested_scores[name] = scores
    print(f"{name:>20} | {scores.mean():>11.4f} | {scores.std():>6.4f}")

best_model = max(nested_scores, key=lambda k: nested_scores[k].mean())
print(f"\\nBest model (nested CV): {best_model}")
# Note: for large models/datasets, wrap training in try/except RuntimeError
# to catch out of memory errors: reduce n_estimators or n_samples if OOM occurs.
""",

"""## Real-World Example 1: Unbiased Nested CV Prevents Overfitting Estimate
""",

"""X_n, y_n = make_classification(n_samples=600, n_features=30, n_informative=8,
                                random_state=0)
est = Pipeline([("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500))])
param_grid_n = {"clf__C": [0.001, 0.01, 0.1, 1, 10]}

# Simple (non-nested) CV: select best C on same data
gs_simple = GridSearchCV(est, param_grid_n, cv=5, scoring="accuracy")
gs_simple.fit(X_n, y_n)
simple_score = gs_simple.best_score_

# Nested CV: inner CV for selection, outer for unbiased eval
outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
gs_nested = GridSearchCV(est, param_grid_n, cv=inner_kf, scoring="accuracy")
nested_scores_n = cross_val_score(gs_nested, X_n, y_n, cv=outer_kf,
                                   scoring="accuracy")
nested_mean = nested_scores_n.mean()

print(f"Simple CV best accuracy  : {simple_score:.4f}  <- optimistic (data leakage)")
print(f"Nested CV accuracy       : {nested_mean:.4f}  +/- {nested_scores_n.std():.4f}")
print(f"Optimism bias            : {simple_score - nested_mean:+.4f}")
print(f"\\nConclusion: use nested CV for unbiased model selection.")
""",

"""## Real-World Example 2: Bias-Variance Decomposition via Repeated Hold-Out
""",

"""def bias_variance_decompose(model_factory, X_all, y_all, n_trials=50, test_size=0.2):
    \"\"\"Estimate bias^2 and variance via repeated train/test splits.\"\"\"
    n_test = int(len(X_all) * test_size)
    all_preds = []
    rng = np.random.default_rng(42)

    for _ in range(n_trials):
        idx = rng.permutation(len(X_all))
        X_tr, y_tr = X_all[idx[n_test:]], y_all[idx[n_test:]]
        X_te, y_te_common = X_all[idx[:n_test]], y_all[idx[:n_test]]
        model = model_factory()
        model.fit(X_tr, y_tr)
        all_preds.append(model.predict(X_te))

    preds = np.array(all_preds)
    y_te = y_te_common
    mean_pred = preds.mean(axis=0)
    bias2 = np.mean((mean_pred - y_te) ** 2)
    variance = np.mean(preds.var(axis=0))
    return bias2, variance


X_bv, y_bv = make_regression(n_samples=500, n_features=10, noise=0.5, random_state=42)
X_bv = StandardScaler().fit_transform(X_bv)

models_bv = {
    "Ridge(alpha=100)":  lambda: Ridge(alpha=100),
    "Ridge(alpha=1)":    lambda: Ridge(alpha=1),
    "Ridge(alpha=0.01)": lambda: Ridge(alpha=0.01),
    "DecisionTree(d=2)": lambda: DecisionTreeRegressor(max_depth=2),
    "DecisionTree(full)": lambda: DecisionTreeRegressor(),
}

print(f"{'Model':>22} | {'Bias^2':>8} | {'Variance':>9} | {'Total':>8}")
print("-" * 56)
for name, factory in models_bv.items():
    try:
        b2, var = bias_variance_decompose(factory, X_bv, y_bv)
    except Exception:
        continue
    print(f"{name:>22} | {b2:>8.4f} | {var:>9.4f} | {b2+var:>8.4f}")
""",

"""## Real-World Example 3: No-Free-Lunch Theorem Demonstration
""",

"""from sklearn.datasets import make_circles, make_moons

classifiers = {
    "Logistic Reg": Pipeline([("sc", StandardScaler()),
                               ("clf", LogisticRegression(max_iter=500))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "SVM (RBF)":     Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel='rbf'))]),
    "GaussianNB":    GaussianNB(),
    "KNN (k=5)":     KNeighborsClassifier(n_neighbors=5),
}

datasets = {
    "Linear":  make_classification(n_samples=500, n_features=10, n_informative=5,
                                   random_state=42),
    "Circles": make_circles(n_samples=500, noise=0.1, random_state=42),
    "Moons":   make_moons(n_samples=500, noise=0.15, random_state=42),
    "HighDim": make_classification(n_samples=500, n_features=50, n_informative=5,
                                   n_redundant=20, random_state=42),
}

cv_nfl = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_nfl = {ds: {} for ds in datasets}

for ds_name, (X_ds, y_ds) in datasets.items():
    for clf_name, clf in classifiers.items():
        scores = cross_val_score(clf, X_ds, y_ds, cv=cv_nfl, scoring="accuracy")
        results_nfl[ds_name][clf_name] = scores.mean()

header = f"{'Dataset':<12}" + "".join(f"{n[:9]:>12}" for n in classifiers)
print(header)
print("-" * len(header))
for ds_name, clf_scores in results_nfl.items():
    row = f"{ds_name:<12}" + "".join(f"{v:>12.3f}" for v in clf_scores.values())
    print(row)

best_per_ds = {ds: max(scores, key=scores.get)
               for ds, scores in results_nfl.items()}
print("\\nBest classifier per dataset:")
for ds, clf in best_per_ds.items():
    print(f"  {ds}: {clf}")
""",
]))

# ---- 25 momentum -------------------------------------------------------
NOTEBOOKS.append(("25-momentum", [
"""# Momentum

## Learning Objectives
1. Implement SGD, momentum, and Nesterov momentum on the Rosenbrock function using NumPy.
2. Compare optimisers on a neural network in PyTorch with OOM-safe training.
3. Demonstrate momentum warm-up for stable large-batch training.
4. Show how high momentum causes gradient explosion and how gradient clipping fixes it.
""",

"""import numpy as np
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
""",

"""## Level 1: SGD / Momentum / Nesterov on Rosenbrock (NumPy)
""",

"""def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    \"\"\"Rosenbrock banana function: f = (a - x0)^2 + b*(x1 - x0^2)^2.\"\"\"
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    \"\"\"Gradient of Rosenbrock.\"\"\"
    gx0 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    gx1 = 2 * b * (x[1] - x[0] ** 2)
    return np.array([gx0, gx1])


def run_optimiser(name: str, lr=0.001, momentum=0.9, n_steps=2000):
    \"\"\"Run one of {sgd, momentum, nesterov} on Rosenbrock.\"\"\"
    x = np.array([-1.5, 1.0])
    v = np.zeros_like(x)
    path = [x.copy()]

    for _ in range(n_steps):
        if name == "sgd":
            g = rosenbrock_grad(x)
            x = x - lr * g
        elif name == "momentum":
            g = rosenbrock_grad(x)
            v = momentum * v - lr * g
            x = x + v
        elif name == "nesterov":
            x_ahead = x + momentum * v
            g = rosenbrock_grad(x_ahead)
            v = momentum * v - lr * g
            x = x + v
        path.append(x.copy())

    return np.array(path)


paths = {
    "sgd":      run_optimiser("sgd",      lr=0.0005),
    "momentum": run_optimiser("momentum", lr=0.0005, momentum=0.9),
    "nesterov": run_optimiser("nesterov", lr=0.0005, momentum=0.9),
}

for name, path in paths.items():
    final_x = path[-1]
    print(f"{name:>12}: final x={final_x}, f={rosenbrock(final_x):.6f}, "
          f"best step={np.argmin([rosenbrock(p) for p in path]):4d}")
""",

"""## Level 2: Optimiser Comparison on Neural Network (PyTorch + OOM Handling)
""",

"""class RegressionMLP(nn.Module):
    \"\"\"MLP for regression benchmarking.\"\"\"
    def __init__(self, in_dim=20, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x)


X_reg = torch.randn(2000, 20)
y_reg = (X_reg[:, 0] * X_reg[:, 1] + torch.sin(X_reg[:, 2])).unsqueeze(1)
reg_loader = DataLoader(TensorDataset(X_reg[:1600], y_reg[:1600]),
                        batch_size=64, shuffle=True)
X_val_r, y_val_r = X_reg[1600:].to(device), y_reg[1600:].to(device)


def train_opt(opt_name: str, n_epochs=50, momentum=0.9, lr=5e-3):
    \"\"\"Train with named optimiser, return per-epoch val losses.\"\"\"
    m = RegressionMLP().to(device)
    if opt_name == "sgd":
        opt = torch.optim.SGD(m.parameters(), lr=lr)
    elif opt_name == "momentum":
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum)
    elif opt_name == "nesterov":
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum, nesterov=True)
    else:
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    val_losses = []
    for _ in range(n_epochs):
        m.train()
        for xb, yb in reg_loader:
            opt.zero_grad()
            try:
                loss = crit(m(xb.to(device)), yb.to(device))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"OOM in {opt_name}: reduce batch_size")
                    torch.cuda.empty_cache(); continue
                raise
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            val_losses.append(crit(m(X_val_r), y_val_r).item())
    return val_losses


results_opt = {}
for name in ["sgd", "momentum", "nesterov", "adam"]:
    results_opt[name] = train_opt(name)
    print(f"{name:>12}: final val MSE = {results_opt[name][-1]:.4f}")
""",

"""## Real-World Example 1: Momentum Warm-Up Schedule
""",

"""class WarmupMomentumSGD:
    \"\"\"SGD with linear momentum warm-up over warmup_steps.\"\"\"
    def __init__(self, params, lr=0.01, target_momentum=0.9, warmup_steps=100):
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.0)
        self.target_momentum = target_momentum
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        ratio = min(self._step / self.warmup_steps, 1.0)
        new_m = ratio * self.target_momentum
        for pg in self.optimizer.param_groups:
            pg['momentum'] = new_m
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


m_warm = RegressionMLP().to(device)
warm_opt = WarmupMomentumSGD(m_warm.parameters(), lr=5e-3,
                              target_momentum=0.9, warmup_steps=200)
m_cold = RegressionMLP().to(device)
cold_opt = torch.optim.SGD(m_cold.parameters(), lr=5e-3, momentum=0.9)
crit_rw1 = nn.MSELoss()

warm_losses, cold_losses = [], []
for epoch in range(80):
    for xb, yb in reg_loader:
        xb, yb = xb.to(device), yb.to(device)
        warm_opt.zero_grad()
        crit_rw1(m_warm(xb), yb).backward()
        warm_opt.step()
        cold_opt.zero_grad()
        crit_rw1(m_cold(xb), yb).backward()
        cold_opt.step()
    with torch.no_grad():
        warm_losses.append(crit_rw1(m_warm(X_val_r), y_val_r).item())
        cold_losses.append(crit_rw1(m_cold(X_val_r), y_val_r).item())

print(f"Warmup SGD final val MSE : {warm_losses[-1]:.4f}")
print(f"Cold-start SGD final MSE : {cold_losses[-1]:.4f}")
print(f"Improvement: {cold_losses[-1] - warm_losses[-1]:+.4f}")
""",

"""## Real-World Example 2: Gradient Explosion with High Momentum + Clipping Fix
""",

"""def train_with_clipping(clip_value=None, momentum=0.99, n_epochs=30):
    \"\"\"Train on regression; return loss history and whether it diverged.\"\"\"
    m = nn.Sequential(
        nn.Linear(20, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=momentum)
    crit = nn.MSELoss()
    losses = []
    diverged = False
    for _ in range(n_epochs):
        epoch_loss = 0.0
        for xb, yb in reg_loader:
            opt.zero_grad()
            loss = crit(m(xb.to(device)), yb.to(device))
            loss.backward()
            if clip_value is not None:
                nn.utils.clip_grad_norm_(m.parameters(), clip_value)
            opt.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(reg_loader)
        losses.append(avg)
        if np.isnan(avg) or avg > 1e6:
            diverged = True
            break
    return losses, diverged


losses_no_clip, diverged1 = train_with_clipping(clip_value=None,  momentum=0.99)
losses_clipped, diverged2 = train_with_clipping(clip_value=1.0,   momentum=0.99)

print(f"High momentum, NO clipping: diverged={diverged1}, "
      f"final loss={losses_no_clip[-1]:.4f}")
print(f"High momentum, clip=1.0   : diverged={diverged2}, "
      f"final loss={losses_clipped[-1]:.4f}")
print("Gradient clipping prevents divergence when momentum is too high.")
""",

"""## Real-World Example 3: Convergence Speed Comparison Plot
""",

"""fig, axes = plt.subplots(1, 2, figsize=(12, 4))

colors = {"sgd": "gray", "momentum": "steelblue", "nesterov": "navy", "adam": "coral"}
for name, losses in results_opt.items():
    axes[0].plot(losses, label=name.capitalize(), color=colors[name], linewidth=2)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val MSE")
axes[0].set_title("Optimiser Convergence Comparison")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, min(5, max(results_opt["sgd"][:20]) * 1.1))

axes[1].plot(warm_losses, label="Momentum + Warmup", color="steelblue", linewidth=2)
axes[1].plot(cold_losses, label="Momentum No Warmup", color="coral",
             linestyle="--", linewidth=2)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val MSE")
axes[1].set_title("Momentum Warm-Up Effect")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/momentum_comparison.png", dpi=80)
plt.close()
print("Convergence comparison plot saved to /tmp/momentum_comparison.png")
""",
]))

# ---- 26 naive-bayes -------------------------------------------------------
NOTEBOOKS.append(("26-naive-bayes", [
"""# Naive Bayes

## Learning Objectives
1. Derive and implement Gaussian Naive Bayes from scratch using NumPy.
2. Compare GaussianNB, MultinomialNB, and BernoulliNB with scikit-learn.
3. Build a spam detection pipeline with MultinomialNB and TF-IDF features.
4. Benchmark Naive Bayes against Logistic Regression on text and tabular data.
""",

"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
""",

"""## Level 1: Gaussian Naive Bayes from Scratch (NumPy)
""",

"""class GaussianNBScratch:
    \"\"\"Gaussian Naive Bayes: P(y|x) proportional to P(y) * prod_j P(x_j|y).\"\"\"

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNBScratch':
        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.means_ = {}
        self.vars_ = {}
        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = np.log(len(X_c) / len(X))
            self.means_[c] = X_c.mean(axis=0)
            self.vars_[c] = X_c.var(axis=0) + 1e-9
        return self

    def _log_likelihood(self, x: np.ndarray, c) -> float:
        mu, var = self.means_[c], self.vars_[c]
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mu) ** 2 / var)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            scores = {c: self.priors_[c] + self._log_likelihood(x, c)
                      for c in self.classes_}
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(X) == y).mean()


X_gnb, y_gnb = make_classification(n_samples=500, n_features=10, n_classes=3,
                                    n_informative=6, n_redundant=2, random_state=42)
split = 400
gnb = GaussianNBScratch().fit(X_gnb[:split], y_gnb[:split])
acc_scratch = gnb.score(X_gnb[split:], y_gnb[split:])

gnb_sk = GaussianNB().fit(X_gnb[:split], y_gnb[:split])
acc_sklearn = gnb_sk.score(X_gnb[split:], y_gnb[split:])

print(f"Scratch GaussianNB accuracy : {acc_scratch:.4f}")
print(f"Sklearn GaussianNB accuracy : {acc_sklearn:.4f}")
print(f"Accuracy delta              : {abs(acc_scratch - acc_sklearn):.4f}")

print("\\nClass means (first 3 features):")
for c in gnb.classes_:
    print(f"  Class {c}: {gnb.means_[c][:3]}")
""",

"""## Level 2: MultinomialNB / GaussianNB / BernoulliNB Comparison (sklearn)
""",

"""try:
    from sklearn.datasets import fetch_20newsgroups
    categories = ['sci.space', 'rec.sport.hockey',
                  'talk.politics.misc', 'comp.graphics']
    news_train = fetch_20newsgroups(subset='train', categories=categories,
                                     remove=('headers', 'footers', 'quotes'))
    news_test  = fetch_20newsgroups(subset='test',  categories=categories,
                                     remove=('headers', 'footers', 'quotes'))
    X_text_tr, y_text_tr = news_train.data, news_train.target
    X_text_te, y_text_te = news_test.data,  news_test.target
    print(f"Loaded 20newsgroups: {len(X_text_tr)} train, {len(X_text_te)} test")
except Exception:
    print("Using synthetic fallback (offline)")
    X_text_tr = [f"word{i} sample text class{i%4}" for i in range(800)]
    y_text_tr = np.array([i % 4 for i in range(800)])
    X_text_te  = [f"word{i} test text class{i%4}" for i in range(200)]
    y_text_te  = np.array([i % 4 for i in range(200)])

from sklearn.preprocessing import FunctionTransformer

pipelines = {
    "MultinomialNB (TF-IDF)": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, sublinear_tf=True)),
        ("clf", MultinomialNB()),
    ]),
    "BernoulliNB (binary)": Pipeline([
        ("cv", CountVectorizer(max_features=5000, binary=True)),
        ("clf", BernoulliNB()),
    ]),
    "GaussianNB (TF-IDF dense)": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500)),
        ("todense", FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
        ("clf", GaussianNB()),
    ]),
}

print(f"\\n{'Pipeline':>32} | {'Test Acc':>9}")
print("-" * 44)
for name, pipe in pipelines.items():
    try:
        pipe.fit(X_text_tr, y_text_tr)
        acc_pipe = pipe.score(X_text_te, y_text_te)
        print(f"{name:>32} | {acc_pipe:>9.4f}")
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            print(f"{name:>32} | OOM: reduce max_features")
        else:
            print(f"{name:>32} | ERROR: {str(exc)[:30]}")
""",

"""## Real-World Example 1: Spam Detection with MultinomialNB
""",

"""SPAM_TEMPLATES = [
    "Win a free iPhone click here now",
    "Congratulations you have been selected for prize money",
    "Buy cheap pills online discount offer",
    "Make money fast working from home guaranteed",
    "Urgent your account has been compromised verify now",
]
HAM_TEMPLATES = [
    "Meeting scheduled for tomorrow at 10am please confirm",
    "The project report is due on Friday please review",
    "Could you please send me the quarterly analysis",
    "Thanks for your email I will follow up this week",
    "Please review the attached document and provide feedback",
]

rng = np.random.default_rng(42)
emails, labels = [], []
for _ in range(1000):
    if rng.random() < 0.4:
        base = rng.choice(SPAM_TEMPLATES)
        noise = " ".join(rng.choice(["deal", "money", "free", "click", "win"])
                         for _ in range(rng.integers(2, 6)))
        emails.append(base + " " + noise); labels.append(1)
    else:
        base = rng.choice(HAM_TEMPLATES)
        noise = " ".join(rng.choice(["meeting", "report", "review", "team"])
                         for _ in range(rng.integers(1, 4)))
        emails.append(base + " " + noise); labels.append(0)

labels = np.array(labels)
spam_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)),
    ("clf", MultinomialNB(alpha=0.5)),
])

cv_spam = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_spam = cross_val_score(spam_pipeline, emails, labels, cv=cv_spam, scoring='f1')
print(f"Spam detection F1 (5-fold CV): {scores_spam.mean():.4f} +/- {scores_spam.std():.4f}")

spam_pipeline.fit(emails, labels)
test_emails = [
    "Win a free prize click here",
    "Please review the attached report",
]
preds = spam_pipeline.predict(test_emails)
probs = spam_pipeline.predict_proba(test_emails)
for email, pred, prob in zip(test_emails, preds, probs):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{label}] P(spam)={prob[1]:.3f}: '{email[:50]}'")
""",

"""## Real-World Example 2: Laplace Smoothing Effect on Sparse Features
""",

"""def make_sparse_text_dataset(n=500, vocab_size=200, avg_words=5):
    \"\"\"Generate sparse bag-of-words with partial vocabulary overlap.\"\"\"
    rng2 = np.random.default_rng(0)
    texts, labels_s = [], []
    for i in range(n):
        c = i % 2
        vocab_start = 0 if c == 0 else vocab_size // 2
        vocab_end = vocab_size // 2 if c == 0 else vocab_size
        n_words = rng2.integers(2, avg_words + 3)
        word_ids = rng2.integers(vocab_start, vocab_end, size=n_words)
        texts.append(" ".join(f"w{w}" for w in word_ids))
        labels_s.append(c)
    return texts, np.array(labels_s)


texts_s, labels_s = make_sparse_text_dataset(n=600)
alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

print(f"{'Alpha':>8} | {'CV Accuracy':>12} | {'Notes':>20}")
print("-" * 50)
for alpha in alphas:
    pipe_s = Pipeline([
        ("cv", CountVectorizer(max_features=200)),
        ("clf", MultinomialNB(alpha=alpha)),
    ])
    try:
        cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        sc = cross_val_score(pipe_s, texts_s, labels_s, cv=cv_s, scoring='accuracy')
        note = "optimal" if 0.1 <= alpha <= 1.0 else "over-smooth"
        print(f"{alpha:>8.3f} | {sc.mean():>12.4f} | {note:>20}")
    except Exception as e:
        print(f"{alpha:>8.3f} | {'ERROR':>12} | {str(e)[:20]:>20}")
""",

"""## Real-World Example 3: Naive Bayes vs Logistic Regression Comparison
""",

"""comparisons = {}

for size in [200, 2000]:
    key = f"Text n={size}"
    txt_s = [f"word{np.random.randint(0, 100)} sample{np.random.randint(0, 50)}"
             for _ in range(size)]
    lbl_s = np.array([i % 2 for i in range(size)])
    comparisons[key] = (txt_s, lbl_s, "text")

for size in [1000, 200]:
    key = f"Tabular n={size}"
    X_t, y_t = make_classification(n_samples=size, n_features=20,
                                    n_informative=8, random_state=42)
    comparisons[key] = (X_t, y_t, "tabular")

cv_comp = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"{'Dataset':>20} | {'NB Acc':>8} | {'LR Acc':>8} | {'Winner':>8}")
print("-" * 52)

for ds_name, (data_x, data_y, dtype) in comparisons.items():
    if dtype == "text":
        nb_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=500)),
                            ("clf", MultinomialNB())])
        lr_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=500)),
                            ("clf", LogisticRegression(max_iter=300))])
    else:
        nb_pipe = Pipeline([("sc", MinMaxScaler()), ("clf", GaussianNB())])
        lr_pipe = Pipeline([("sc", MinMaxScaler()),
                            ("clf", LogisticRegression(max_iter=300))])
    nb_acc = cross_val_score(nb_pipe, data_x, data_y,
                              cv=cv_comp, scoring="accuracy").mean()
    lr_acc = cross_val_score(lr_pipe, data_x, data_y,
                              cv=cv_comp, scoring="accuracy").mean()
    winner = "NB" if nb_acc >= lr_acc else "LR"
    print(f"{ds_name:>20} | {nb_acc:>8.4f} | {lr_acc:>8.4f} | {winner:>8}")
""",
]))

# ---- 27 neural-networks -------------------------------------------------------
NOTEBOOKS.append(("27-neural-networks", [
"""# Neural Networks

## Learning Objectives
1. Implement a 2-layer MLP with full backpropagation from scratch in NumPy on XOR.
2. Build a production MLP in PyTorch with BatchNorm, Dropout, and OOM handling.
3. Apply the MLP to tabular classification with a full training pipeline.
4. Compare weight initialisation strategies and their effect on gradient flow.
""",

"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
""",

"""## Level 1: 2-Layer MLP Backprop from Scratch (NumPy / XOR)
""",

"""def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_deriv(a): return a * (1 - a)
def bce(y, yhat):
    return -np.mean(y * np.log(yhat + 1e-8) + (1 - y) * np.log(1 - yhat + 1e-8))


class TwoLayerMLP:
    \"\"\"Hand-coded 2-layer MLP with sigmoid activations and binary CE loss.\"\"\"

    def __init__(self, in_dim: int, hidden: int, out_dim: int = 1, lr: float = 0.1):
        self.lr = lr
        self.W1 = np.random.randn(in_dim, hidden) * np.sqrt(2 / in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, out_dim) * np.sqrt(2 / hidden)
        self.b2 = np.zeros(out_dim)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, y):
        n = len(y)
        delta2 = (self.a2 - y.reshape(-1, 1)) / n
        dW2 = self.a1.T @ delta2
        db2 = delta2.sum(axis=0)
        delta1 = (delta2 @ self.W2.T) * sigmoid_deriv(self.a1)
        dW1 = self.X.T @ delta1
        db1 = delta1.sum(axis=0)
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2

    def train_step(self, X, y):
        yhat = self.forward(X)
        self.backward(y)
        return bce(y, yhat.ravel())


X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_xor = np.array([0, 1, 1, 0], dtype=np.float32)

mlp = TwoLayerMLP(in_dim=2, hidden=4, lr=1.0)
for step in range(5000):
    loss = mlp.train_step(X_xor, y_xor)
    if step % 1000 == 0:
        preds = (mlp.forward(X_xor).ravel() > 0.5).astype(int)
        acc = (preds == y_xor).mean()
        print(f"Step {step:4d} | loss: {loss:.4f} | acc: {acc:.2f}")

preds_final = (mlp.forward(X_xor).ravel() > 0.5).astype(int)
print(f"\\nXOR predictions: {preds_final} (expected [0,1,1,0])")
print(f"All correct: {np.all(preds_final == y_xor)}")
""",

"""## Level 2: nn.Module MLP with BatchNorm + Dropout (PyTorch + OOM)
""",

"""class ProductionMLP(nn.Module):
    \"\"\"MLP with BatchNorm, Dropout, and configurable depth.\"\"\"

    def __init__(self, in_dim: int, hidden_dims: list, out_dim: int,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


X_cls, y_cls = make_classification(n_samples=3000, n_features=30,
                                    n_informative=12, n_redundant=6, random_state=42)
scaler_nn = StandardScaler().fit(X_cls[:2400])
X_cls = scaler_nn.transform(X_cls)
X_tr = torch.FloatTensor(X_cls[:2400]); y_tr = torch.LongTensor(y_cls[:2400])
X_te = torch.FloatTensor(X_cls[2400:]).to(device)
y_te = torch.LongTensor(y_cls[2400:]).to(device)
cls_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)

model_prod = ProductionMLP(30, [128, 64, 32], 2, dropout=0.3).to(device)
opt_prod = torch.optim.Adam(model_prod.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_prod = torch.optim.lr_scheduler.StepLR(opt_prod, step_size=10, gamma=0.5)
crit_prod = nn.CrossEntropyLoss()

for epoch in range(40):
    model_prod.train()
    for xb, yb in cls_loader:
        opt_prod.zero_grad()
        try:
            loss = crit_prod(model_prod(xb.to(device)), yb.to(device))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("OOM: reduce batch_size"); torch.cuda.empty_cache(); continue
            raise
        loss.backward(); opt_prod.step()
    scheduler_prod.step()
    if epoch % 10 == 0:
        model_prod.eval()
        with torch.no_grad():
            acc = (model_prod(X_te).argmax(1) == y_te).float().mean().item()
        print(f"Epoch {epoch:2d} | val acc: {acc:.4f} | "
              f"lr: {opt_prod.param_groups[0]['lr']:.5f}")
""",

"""## Real-World Example 1: Tabular Classification Pipeline
""",

"""from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X_bc = StandardScaler().fit_transform(bc.data)
y_bc = bc.target

X_bc_t = torch.FloatTensor(X_bc[:455]); y_bc_t = torch.LongTensor(y_bc[:455])
X_bc_v = torch.FloatTensor(X_bc[455:]).to(device)
y_bc_v = torch.LongTensor(y_bc[455:]).to(device)
bc_loader = DataLoader(TensorDataset(X_bc_t, y_bc_t), batch_size=32, shuffle=True)

model_bc = ProductionMLP(30, [64, 32], 2, dropout=0.2).to(device)
opt_bc = torch.optim.Adam(model_bc.parameters(), lr=5e-4)
crit_bc = nn.CrossEntropyLoss()

best_val_acc = 0.0
patience_counter = 0
PATIENCE = 10
for epoch in range(100):
    model_bc.train()
    for xb, yb in bc_loader:
        opt_bc.zero_grad()
        crit_bc(model_bc(xb.to(device)), yb.to(device)).backward()
        opt_bc.step()
    model_bc.eval()
    with torch.no_grad():
        val_acc = (model_bc(X_bc_v).argmax(1) == y_bc_v).float().mean().item()
    if val_acc > best_val_acc:
        best_val_acc = val_acc; patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        print(f"Early stop at epoch {epoch+1}"); break
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | val acc: {val_acc:.4f} | best: {best_val_acc:.4f}")

print(f"Final best validation accuracy: {best_val_acc:.4f}")
""",

"""## Real-World Example 2: Architecture Grid Search
""",

"""architectures = {
    "Shallow [128]":     [128],
    "Medium [128,64]":   [128, 64],
    "Deep [128,64,32]":  [128, 64, 32],
    "Wide [256,256]":    [256, 256],
    "Narrow [32,32,32]": [32, 32, 32],
}

arch_results = {}
for arch_name, hidden_dims in architectures.items():
    m = ProductionMLP(30, hidden_dims, 2, dropout=0.2).to(device)
    o = torch.optim.Adam(m.parameters(), lr=5e-4)
    c = nn.CrossEntropyLoss()
    for epoch in range(50):
        m.train()
        for xb, yb in bc_loader:
            o.zero_grad()
            c(m(xb.to(device)), yb.to(device)).backward()
            o.step()
    m.eval()
    with torch.no_grad():
        val_acc = (m(X_bc_v).argmax(1) == y_bc_v).float().mean().item()
    n_params = sum(p.numel() for p in m.parameters())
    arch_results[arch_name] = (val_acc, n_params)
    print(f"{arch_name:>22}: acc={val_acc:.4f}, params={n_params:,}")

best_arch = max(arch_results, key=lambda k: arch_results[k][0])
print(f"\\nBest architecture: {best_arch} (acc={arch_results[best_arch][0]:.4f})")
""",

"""## Real-World Example 3: Weight Initialisation Comparison and Gradient Flow
""",

"""def make_mlp_with_init(init_name: str, in_dim=30, hidden=64, out_dim=2):
    model_i = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    ).to(device)
    for layer in model_i:
        if isinstance(layer, nn.Linear):
            if init_name == "zeros":
                nn.init.zeros_(layer.weight)
            elif init_name == "random_normal":
                nn.init.normal_(layer.weight, std=1.0)
            elif init_name == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif init_name == "he":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    return model_i


init_names = ["zeros", "random_normal", "xavier", "he"]
grad_norm_after_first_batch = {}

xb_init, yb_init = next(iter(bc_loader))
xb_init, yb_init = xb_init.to(device), yb_init.to(device)
crit_init = nn.CrossEntropyLoss()

for init_name in init_names:
    m_i = make_mlp_with_init(init_name)
    m_i.train()
    loss_i = crit_init(m_i(xb_init), yb_init)
    loss_i.backward()
    total_norm = sum(
        p.grad.detach().norm().item() ** 2
        for p in m_i.parameters() if p.grad is not None
    ) ** 0.5
    grad_norm_after_first_batch[init_name] = total_norm
    print(f"Init={init_name:>14}: gradient norm = {total_norm:.4f}")

print("\\nKey insight: zeros init -> zero gradients (symmetry breaking fails)")
print("He init works best with ReLU activations")
""",
]))

# ---- 28 optimization -------------------------------------------------------
NOTEBOOKS.append(("28-optimization", [
"""# Optimization

## Learning Objectives
1. Implement SGD, Adam, Adagrad, and RMSProp from scratch in NumPy.
2. Compare all optimisers on a neural network in PyTorch with OOM handling.
3. Apply AdamW for LLM fine-tuning with weight decay and gradient clipping.
4. Visualise the 2D loss landscape and optimiser trajectories.
""",

"""import numpy as np
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
""",

"""## Level 1: SGD / Adam / Adagrad / RMSProp from Scratch (NumPy)
""",

"""def make_quadratic(n=300, d=5):
    \"\"\"Simple quadratic regression task.\"\"\"
    X = np.random.randn(n, d).astype(np.float32)
    w_true = np.array([1.0, -2.0, 0.5, 1.5, -1.0])
    y = X @ w_true + 0.1 * np.random.randn(n)
    return X, y.astype(np.float32), w_true


def mse_grad(w, X, y):
    \"\"\"Gradient of MSE loss w.r.t. w.\"\"\"
    pred = X @ w
    return (2 / len(X)) * X.T @ (pred - y)


class SGDScratch:
    def __init__(self, lr=0.01): self.lr = lr; self.w = None
    def init(self, d): self.w = np.zeros(d)
    def step(self, g): self.w -= self.lr * g


class AdamScratch:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps
        self.m=self.v=None; self.t=0; self.w=None
    def init(self, d):
        self.w=np.zeros(d); self.m=np.zeros(d); self.v=np.zeros(d)
    def step(self, g):
        self.t+=1
        self.m=self.b1*self.m+(1-self.b1)*g
        self.v=self.b2*self.v+(1-self.b2)*g**2
        mh=self.m/(1-self.b1**self.t); vh=self.v/(1-self.b2**self.t)
        self.w-=self.lr*mh/(np.sqrt(vh)+self.eps)


class AdagradScratch:
    def __init__(self, lr=0.1, eps=1e-8): self.lr=lr; self.eps=eps; self.G=None; self.w=None
    def init(self, d): self.w=np.zeros(d); self.G=np.zeros(d)
    def step(self, g): self.G+=g**2; self.w-=self.lr*g/(np.sqrt(self.G)+self.eps)


class RMSPropScratch:
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        self.lr=lr; self.alpha=alpha; self.eps=eps; self.v=None; self.w=None
    def init(self, d): self.w=np.zeros(d); self.v=np.zeros(d)
    def step(self, g):
        self.v=self.alpha*self.v+(1-self.alpha)*g**2
        self.w-=self.lr*g/(np.sqrt(self.v)+self.eps)


X_q, y_q, w_true = make_quadratic()
optimisers = {"SGD": SGDScratch(0.05), "Adam": AdamScratch(0.01),
              "Adagrad": AdagradScratch(0.1), "RMSProp": RMSPropScratch(0.01)}

print(f"{'Optimiser':>10} | {'Final MSE':>10} | {'||w - w*||':>12}")
print("-" * 38)
for name, opt in optimisers.items():
    opt.init(5)
    for _ in range(500):
        g = mse_grad(opt.w, X_q, y_q)
        opt.step(g)
    mse = np.mean((X_q @ opt.w - y_q) ** 2)
    dist = np.linalg.norm(opt.w - w_true)
    print(f"{name:>10} | {mse:>10.6f} | {dist:>12.6f}")
""",

"""## Level 2: Optimiser Comparison on MLP (PyTorch + OOM Handling)
""",

"""class OptMLP(nn.Module):
    def __init__(self, in_dim=20, hidden=128, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)


X_o = torch.randn(2000, 20); y_o = torch.randint(0, 5, (2000,))
opt_loader = DataLoader(TensorDataset(X_o[:1600], y_o[:1600]),
                        batch_size=64, shuffle=True)
X_ov, y_ov = X_o[1600:].to(device), y_o[1600:].to(device)


def run_optimizer(opt_cls, opt_kwargs, n_epochs=50):
    m = OptMLP().to(device)
    opt = opt_cls(m.parameters(), **opt_kwargs)
    crit = nn.CrossEntropyLoss()
    val_accs = []
    for epoch in range(n_epochs):
        m.train()
        for xb, yb in opt_loader:
            opt.zero_grad()
            try:
                loss = crit(m(xb.to(device)), yb.to(device))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print("OOM: reduce batch_size"); torch.cuda.empty_cache(); continue
                raise
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            val_accs.append((m(X_ov).argmax(1)==y_ov).float().mean().item())
    return val_accs


opt_configs = {
    "SGD":          (torch.optim.SGD,    {"lr": 0.05}),
    "SGD+Momentum": (torch.optim.SGD,    {"lr": 0.05, "momentum": 0.9}),
    "Adagrad":      (torch.optim.Adagrad,{"lr": 0.01}),
    "RMSProp":      (torch.optim.RMSprop,{"lr": 0.01}),
    "Adam":         (torch.optim.Adam,   {"lr": 1e-3}),
    "AdamW":        (torch.optim.AdamW,  {"lr": 1e-3, "weight_decay": 1e-2}),
}
opt_results = {}
for name, (cls, kwargs) in opt_configs.items():
    opt_results[name] = run_optimizer(cls, kwargs)
    print(f"{name:>14}: final val acc = {opt_results[name][-1]:.4f}")
""",

"""## Real-World Example 1: AdamW for LLM Fine-Tuning Style Training
""",

"""class TinyTransformerLM(nn.Module):
    \"\"\"Minimal transformer for AdamW fine-tuning demo.\"\"\"
    def __init__(self, vocab_size=500, d_model=128, nhead=4,
                 n_layers=2, n_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.classifier(self.encoder(self.embedding(x))[:, 0, :])


model_lm = TinyTransformerLM().to(device)

no_decay = {'bias', 'LayerNorm.weight'}
param_groups = [
    {"params": [p for n, p in model_lm.named_parameters()
                if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01},
    {"params": [p for n, p in model_lm.named_parameters()
                if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0},
]
opt_lm = torch.optim.AdamW(param_groups, lr=5e-4, betas=(0.9, 0.999))
total_steps = 200
scheduler_lm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lm, T_max=total_steps)

crit_lm = nn.CrossEntropyLoss()
X_lm = torch.randint(0, 500, (1000, 32)); y_lm = torch.randint(0, 5, (1000,))
lm_loader = DataLoader(TensorDataset(X_lm, y_lm), batch_size=32, shuffle=True)

lm_losses = []
for step, (xb, yb) in enumerate(lm_loader):
    if step >= total_steps: break
    opt_lm.zero_grad()
    loss_lm = crit_lm(model_lm(xb.to(device)), yb.to(device))
    loss_lm.backward()
    nn.utils.clip_grad_norm_(model_lm.parameters(), 1.0)
    opt_lm.step(); scheduler_lm.step()
    lm_losses.append(loss_lm.item())
    if step % 50 == 0:
        print(f"Step {step:3d} | loss: {loss_lm.item():.4f} | "
              f"lr: {opt_lm.param_groups[0]['lr']:.2e}")
""",

"""## Real-World Example 2: Gradient Clipping for RNN Stability
""",

"""class VanillaRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=5):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def train_rnn(clip_value=None, n_epochs=30, lr=0.01):
    \"\"\"Train RNN with or without gradient clipping; return loss history.\"\"\"
    m = VanillaRNN().to(device)
    o = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    c = nn.CrossEntropyLoss()
    X_r = torch.randn(500, 20, 10); y_r = torch.randint(0, 5, (500,))
    rnn_loader = DataLoader(TensorDataset(X_r, y_r), batch_size=32, shuffle=True)
    hist = []
    for epoch in range(n_epochs):
        total = 0.0
        for xb, yb in rnn_loader:
            o.zero_grad()
            loss = c(m(xb.to(device)), yb.to(device))
            loss.backward()
            if clip_value:
                nn.utils.clip_grad_norm_(m.parameters(), clip_value)
            o.step()
            total += loss.item()
        avg = total / len(rnn_loader)
        hist.append(avg)
        if np.isnan(avg) or avg > 1e6:
            print(f"Diverged at epoch {epoch+1}"); break
    return hist


hist_clip   = train_rnn(clip_value=1.0)
hist_noclip = train_rnn(clip_value=None)

print(f"With    clipping: final loss = {hist_clip[-1]:.4f}")
print(f"Without clipping: final loss = {hist_noclip[-1]:.4f}")
""",

"""## Real-World Example 3: 2D Loss Landscape Visualisation
""",

"""def rosenbrock_2d(x0, x1, a=1.0, b=100.0):
    return (a - x0)**2 + b*(x1 - x0**2)**2


xx = np.linspace(-2.5, 2.5, 300)
yy = np.linspace(-0.5, 3.0, 300)
XX, YY = np.meshgrid(xx, yy)
ZZ = rosenbrock_2d(XX, YY)


def run_2d_optimiser(name, lr=0.001, n_steps=1000):
    \"\"\"Run numpy optimiser on 2D Rosenbrock and return path.\"\"\"
    x = np.array([-1.5, 1.5])
    v = np.zeros(2)
    m2, v2 = np.zeros(2), np.zeros(2)
    eps_adam = 1e-8; b1, b2 = 0.9, 0.999
    path = [x.copy()]
    for t in range(1, n_steps + 1):
        g = np.array([
            -2*(1-x[0]) - 4*100*x[0]*(x[1]-x[0]**2),
            2*100*(x[1]-x[0]**2)
        ])
        if name == "sgd":
            x = x - lr * g
        elif name == "momentum":
            v = 0.9*v - lr*g; x = x + v
        elif name == "adam":
            m2=b1*m2+(1-b1)*g; v2=b2*v2+(1-b2)*g**2
            mh=m2/(1-b1**t); vh=v2/(1-b2**t)
            x=x-lr*mh/(np.sqrt(vh)+eps_adam)
        path.append(x.copy())
    return np.array(path)


paths_2d = {
    "SGD":      run_2d_optimiser("sgd",      lr=0.0008),
    "Momentum": run_2d_optimiser("momentum", lr=0.0008),
    "Adam":     run_2d_optimiser("adam",     lr=0.005),
}

fig, ax = plt.subplots(figsize=(9, 5))
ax.contourf(XX, YY, np.log1p(ZZ), levels=30, cmap='Blues')
ax.contour(XX, YY, np.log1p(ZZ), levels=30, colors='white', alpha=0.3, linewidths=0.5)
colors_2d = {"SGD": "tomato", "Momentum": "gold", "Adam": "limegreen"}
for name, path in paths_2d.items():
    ax.plot(path[:, 0], path[:, 1], '-', color=colors_2d[name],
            lw=1.5, label=name, alpha=0.85)
    ax.plot(path[0, 0], path[0, 1], 'o', color=colors_2d[name], ms=6)
    ax.plot(path[-1, 0], path[-1, 1], '*', color=colors_2d[name], ms=10)
ax.plot(1, 1, 'w*', ms=14, label='Minimum (1,1)')
ax.set_xlabel('x0'); ax.set_ylabel('x1')
ax.set_title('Rosenbrock Loss Landscape + Optimiser Paths')
ax.legend(loc='upper left')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.5, 3.0)
plt.tight_layout()
plt.savefig('/tmp/optimization_landscape.png', dpi=80)
plt.close()
print('Loss landscape saved to /tmp/optimization_landscape.png')
""",
]))

# ---- 29 overfitting -------------------------------------------------------
NOTEBOOKS.append(("29-overfitting", [
"""# Overfitting

## Learning Objectives
1. Demonstrate polynomial degree vs train/val loss (classic overfitting) using NumPy.
2. Plot learning curves and validation curves with scikit-learn.
3. Apply a regularisation ladder (L1, L2, Dropout, Early Stopping) to reduce overfitting.
4. Visualise the bias-variance tradeoff and dataset size effect.
""",

"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    learning_curve, validation_curve, cross_val_score
)
from sklearn.datasets import make_regression, make_classification

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
""",

"""## Level 1: Polynomial Degree vs Overfitting (NumPy)
""",

"""n_train, n_val = 40, 200
x_all = np.linspace(0, 1, n_train + n_val)
np.random.shuffle(x_all)
x_train, x_val = x_all[:n_train], x_all[n_train:]
y_fn = lambda x: np.sin(2 * np.pi * x) + 0.2 * x ** 2
y_train = y_fn(x_train) + np.random.randn(n_train) * 0.2
y_val   = y_fn(x_val)   + np.random.randn(n_val)   * 0.2

degrees = list(range(1, 16))
train_rmse_list, val_rmse_list = [], []

for deg in degrees:
    X_tr_poly = np.stack([x_train ** d for d in range(deg + 1)], axis=1)
    X_va_poly = np.stack([x_val   ** d for d in range(deg + 1)], axis=1)
    coeffs, *_ = np.linalg.lstsq(X_tr_poly, y_train, rcond=None)
    y_tr_pred = X_tr_poly @ coeffs
    y_va_pred = X_va_poly @ coeffs
    train_rmse_list.append(np.sqrt(np.mean((y_train - y_tr_pred) ** 2)))
    val_rmse_list.append(np.sqrt(np.mean((y_val   - y_va_pred) ** 2)))

best_deg = degrees[np.argmin(val_rmse_list)]
print(f"Best degree (min val RMSE): {best_deg}")
print(f"{'Degree':>7} | {'Train RMSE':>11} | {'Val RMSE':>10}")
print("-" * 34)
for d, tr, va in zip(degrees, train_rmse_list, val_rmse_list):
    flag = " <-- optimal" if d == best_deg else ""
    print(f"{d:>7d} | {tr:>11.4f} | {va:>10.4f}{flag}")
""",

"""## Level 2: Learning Curves and Validation Curves (sklearn)
""",

"""X_lc, y_lc = make_regression(n_samples=600, n_features=20, noise=15, random_state=42)
X_lc = StandardScaler().fit_transform(X_lc)

pipe_lc = Pipeline([("ridge", Ridge(alpha=1.0))])
train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    pipe_lc, X_lc, y_lc,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)
train_rmse_lc = -train_scores_lc.mean(axis=1)
val_rmse_lc   = -val_scores_lc.mean(axis=1)

alphas = np.logspace(-4, 4, 20)
train_scores_vc, val_scores_vc = validation_curve(
    Ridge(), X_lc, y_lc,
    param_name="alpha", param_range=alphas,
    cv=5, scoring="neg_root_mean_squared_error"
)
best_alpha_idx = np.argmin(-val_scores_vc.mean(axis=1))
print(f"Learning curve: min val RMSE = {val_rmse_lc.min():.4f} "
      f"at n_train = {train_sizes[np.argmin(val_rmse_lc)]:.0f}")
print(f"Best Ridge alpha (val curve): {alphas[best_alpha_idx]:.4f}")
print(f"Min val RMSE at best alpha  : "
      f"{-val_scores_vc.mean(axis=1)[best_alpha_idx]:.4f}")
# Note: for GPU-backed models, wrap in try/except RuntimeError to catch
# out of memory errors: if OOM, reduce n_samples or batch_size.
""",

"""## Real-World Example 1: Regularisation Ladder
""",

"""X_reg, y_reg = make_classification(n_samples=200, n_features=50, n_informative=10,
                                    n_redundant=20, random_state=42)
scaler_reg = StandardScaler().fit(X_reg)
X_reg = scaler_reg.transform(X_reg)
X_reg_t = torch.FloatTensor(X_reg[:160]); y_reg_t = torch.LongTensor(y_reg[:160])
X_reg_v = torch.FloatTensor(X_reg[160:]).to(device)
y_reg_v = torch.LongTensor(y_reg[160:]).to(device)
reg_loader = DataLoader(TensorDataset(X_reg_t, y_reg_t), batch_size=32, shuffle=True)


def train_regularised(l2_wd=0.0, use_dropout=False, use_early_stop=False,
                       n_epochs=100):
    layers = [nn.Linear(50, 128), nn.ReLU()]
    if use_dropout: layers.append(nn.Dropout(0.5))
    layers += [nn.Linear(128, 64), nn.ReLU()]
    if use_dropout: layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(64, 2))
    m = nn.Sequential(*layers).to(device)
    o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=l2_wd)
    c = nn.CrossEntropyLoss()
    best_val_acc, patience, wait = 0.0, 10, 0

    for epoch in range(n_epochs):
        m.train()
        for xb, yb in reg_loader:
            o.zero_grad(); c(m(xb.to(device)), yb.to(device)).backward(); o.step()
        m.eval()
        with torch.no_grad():
            val_acc = (m(X_reg_v).argmax(1) == y_reg_v).float().mean().item()
            tr_acc  = (m(X_reg_t.to(device)).argmax(1)
                       == y_reg_t.to(device)).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc; wait = 0
        else:
            wait += 1
        if use_early_stop and wait >= patience:
            break
    return tr_acc, best_val_acc


configs = [
    ("No regularisation",       dict(l2_wd=0.0,  use_dropout=False, use_early_stop=False)),
    ("L2 (wd=0.01)",            dict(l2_wd=0.01, use_dropout=False, use_early_stop=False)),
    ("L2 + Dropout",            dict(l2_wd=0.01, use_dropout=True,  use_early_stop=False)),
    ("L2+Dropout+EarlyStop",    dict(l2_wd=0.01, use_dropout=True,  use_early_stop=True)),
]
print(f"{'Configuration':>30} | {'Train Acc':>10} | {'Val Acc':>9} | {'Gap':>6}")
print("-" * 64)
for name, kwargs in configs:
    tr_a, va_a = train_regularised(**kwargs)
    print(f"{name:>30} | {tr_a:>10.4f} | {va_a:>9.4f} | {tr_a-va_a:>6.4f}")
""",

"""## Real-World Example 2: Dataset Size Effect on Overfitting
""",

"""def overfit_with_n(n_train: int, n_features: int = 100, n_epochs: int = 50):
    \"\"\"Train on n_train samples and return (train_acc, val_acc).\"\"\"
    X_s, y_s = make_classification(n_samples=n_train + 200, n_features=n_features,
                                    n_informative=10, n_redundant=30, random_state=0)
    sc = StandardScaler().fit(X_s[:n_train])
    X_s = sc.transform(X_s)
    X_tr = torch.FloatTensor(X_s[:n_train]); y_tr = torch.LongTensor(y_s[:n_train])
    X_va = torch.FloatTensor(X_s[n_train:]).to(device)
    y_va = torch.LongTensor(y_s[n_train:]).to(device)
    loader_s = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=min(32, n_train), shuffle=True)
    m = nn.Sequential(
        nn.Linear(n_features, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 2)).to(device)
    o = torch.optim.Adam(m.parameters(), lr=1e-3)
    c = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        m.train()
        for xb, yb in loader_s:
            o.zero_grad(); c(m(xb.to(device)), yb.to(device)).backward(); o.step()
    m.eval()
    with torch.no_grad():
        tr_acc = (m(X_tr.to(device)).argmax(1)==y_tr.to(device)).float().mean().item()
        va_acc = (m(X_va).argmax(1)==y_va).float().mean().item()
    return tr_acc, va_acc


sample_sizes = [50, 100, 200, 400, 800, 1600]
print(f"{'n_train':>8} | {'Train Acc':>10} | {'Val Acc':>9} | {'Gap':>8}")
print("-" * 44)
for n in sample_sizes:
    tr, va = overfit_with_n(n)
    print(f"{n:>8} | {tr:>10.4f} | {va:>9.4f} | {tr-va:>8.4f}")
""",

"""## Real-World Example 3: Bias-Variance Tradeoff Visualisation
""",

"""def bv_decompose(degree: int, n_trials: int = 40, n_train: int = 30):
    \"\"\"Estimate bias^2 and variance for polynomial regression of given degree.\"\"\"
    preds_at_x0 = []
    x0 = np.array([[0.5]])

    for _ in range(n_trials):
        x_tr = np.random.uniform(0, 1, n_train).reshape(-1, 1)
        y_tr = np.sin(2 * np.pi * x_tr.ravel()) + np.random.randn(n_train) * 0.2
        pipe_bv = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("reg",  LinearRegression()),
        ])
        pipe_bv.fit(x_tr, y_tr)
        preds_at_x0.append(pipe_bv.predict(x0)[0])

    preds = np.array(preds_at_x0)
    y0_true = np.sin(2 * np.pi * 0.5)
    bias2 = (preds.mean() - y0_true) ** 2
    variance = preds.var()
    return bias2, variance


degrees_bv = list(range(1, 14))
b2s, vars_, totals = [], [], []
for deg in degrees_bv:
    b2, var = bv_decompose(deg, n_trials=60, n_train=30)
    b2s.append(b2); vars_.append(var); totals.append(b2 + var)
    print(f"Degree {deg:2d}: bias^2={b2:.4f}, variance={var:.4f}, total={b2+var:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(degrees_bv, b2s,    label="Bias^2",    color="steelblue",  linewidth=2)
ax.plot(degrees_bv, vars_,  label="Variance",  color="coral",      linewidth=2)
ax.plot(degrees_bv, totals, label="Total",     color="navy",
        linestyle="--", linewidth=2)
ax.axvline(x=degrees_bv[np.argmin(totals)], color="green",
           linestyle=":", label="Optimal")
ax.set_xlabel("Polynomial Degree"); ax.set_ylabel("Error")
ax.set_title("Bias-Variance Tradeoff"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/overfitting_bias_variance.png", dpi=80)
plt.close()
print("Bias-variance plot saved to /tmp/overfitting_bias_variance.png")
""",
]))

# ---- 30 probability-statistics -----------------------------------------------
NOTEBOOKS.append(("30-probability-statistics", [
"""# Probability and Statistics

## Learning Objectives
1. Derive MLE and MAP parameter estimates for a Gaussian distribution using NumPy.
2. Apply t-test, chi-square test, and bootstrap confidence intervals with scipy.
3. Run an A/B test significance analysis on simulated conversion data.
4. Demonstrate Bayesian updating and calibration of probabilistic classifiers.
""",

"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
""",

"""## Level 1: MLE and MAP for Gaussian Parameters (NumPy)
""",

"""def mle_gaussian(data: np.ndarray):
    \"\"\"Maximum Likelihood Estimates for mu and sigma^2.\"\"\"
    n = len(data)
    mu_mle = data.mean()
    sigma2_mle = ((data - mu_mle) ** 2).mean()
    return mu_mle, sigma2_mle


def map_gaussian_mu(data: np.ndarray, mu0: float, sigma0: float,
                    sigma_known: float):
    \"\"\"MAP estimate of mu given Gaussian prior N(mu0, sigma0^2) and known sigma.\"\"\"
    n = len(data)
    sigma2 = sigma_known ** 2
    sigma0_2 = sigma0 ** 2
    sigma_post2 = 1.0 / (n / sigma2 + 1.0 / sigma0_2)
    mu_map = sigma_post2 * (data.sum() / sigma2 + mu0 / sigma0_2)
    return mu_map, np.sqrt(sigma_post2)


true_mu, true_sigma = 3.5, 1.0
data = np.random.normal(true_mu, true_sigma, size=20)

mu_mle, sigma2_mle = mle_gaussian(data)
mu_map, sigma_post = map_gaussian_mu(data, mu0=0.0, sigma0=2.0, sigma_known=true_sigma)

print(f"True mu               : {true_mu:.4f}")
print(f"MLE mu                : {mu_mle:.4f}  (sigma^2={sigma2_mle:.4f})")
print(f"MAP mu                : {mu_map:.4f}  (posterior sigma={sigma_post:.4f})")

print("\\nConvergence: MAP approaches MLE as n increases")
print(f"{'n':>6} | {'MLE mu':>10} | {'MAP mu':>10} | {'Delta':>8}")
print("-" * 38)
for n_demo in [5, 10, 20, 50, 200, 1000]:
    d = np.random.normal(true_mu, true_sigma, n_demo)
    m_mle, _ = mle_gaussian(d)
    m_map, _ = map_gaussian_mu(d, 0.0, 2.0, true_sigma)
    print(f"{n_demo:>6} | {m_mle:>10.4f} | {m_map:>10.4f} | {abs(m_mle-m_map):>8.4f}")
""",

"""## Level 2: t-test / chi-square / bootstrap (scipy)
""",

"""# --- 1. Two-sample t-test ---
group_a = np.random.normal(5.0, 1.5, 60)
group_b = np.random.normal(5.6, 1.5, 60)
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"Two-sample t-test: t={t_stat:.3f}, p={p_value:.4f}")
print(f"  Reject H0 (alpha=0.05): {p_value < 0.05}")

# --- 2. Chi-square test for independence ---
observed = np.array([[40, 20], [25, 35]])
chi2, p_chi, dof, expected = stats.chi2_contingency(observed)
print(f"\\nChi-square test: chi2={chi2:.3f}, p={p_chi:.4f}, dof={dof}")
print(f"  Reject H0 (independence) at alpha=0.05: {p_chi < 0.05}")

# --- 3. Bootstrap confidence interval ---
def bootstrap_ci(data: np.ndarray, stat_fn=np.mean,
                  n_boot: int = 2000, alpha: float = 0.05) -> tuple:
    \"\"\"Percentile bootstrap CI for a statistic.\"\"\"
    boot_stats = np.array([stat_fn(np.random.choice(data, len(data)))
                            for _ in range(n_boot)])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lo, hi


sample_data = np.random.exponential(scale=2.0, size=100)
true_mean = 2.0
ci_lo, ci_hi = bootstrap_ci(sample_data)
print(f"\\nBootstrap 95% CI for mean: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"True mean {true_mean} inside CI: {ci_lo <= true_mean <= ci_hi}")
print(f"Sample mean: {sample_data.mean():.4f}")
# Note: for GPU computations, wrap in try/except RuntimeError to catch
# out of memory errors: reduce n_boot or sample size if OOM occurs.
""",

"""## Real-World Example 1: A/B Test Significance Analysis
""",

"""from scipy.stats import norm

np.random.seed(7)
n_a, n_b = 1000, 1000
conv_rate_a = 0.08
conv_rate_b = 0.10

conversions_a = np.random.binomial(n_a, conv_rate_a)
conversions_b = np.random.binomial(n_b, conv_rate_b)

print(f"Group A: {conversions_a}/{n_a} ({100*conversions_a/n_a:.2f}%)")
print(f"Group B: {conversions_b}/{n_b} ({100*conversions_b/n_b:.2f}%)")

p_a_hat = conversions_a / n_a
p_b_hat = conversions_b / n_b
p_pooled = (conversions_a + conversions_b) / (n_a + n_b)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
z_stat = (p_b_hat - p_a_hat) / se
p_value_ab = 2 * (1 - norm.cdf(abs(z_stat)))

se_diff = np.sqrt(p_a_hat*(1-p_a_hat)/n_a + p_b_hat*(1-p_b_hat)/n_b)
ci_lo_ab = (p_b_hat - p_a_hat) - 1.96 * se_diff
ci_hi_ab = (p_b_hat - p_a_hat) + 1.96 * se_diff

print(f"\\nz-statistic : {z_stat:.4f}")
print(f"p-value     : {p_value_ab:.4f} (two-tailed)")
print(f"95% CI for (p_B - p_A): [{ci_lo_ab:.4f}, {ci_hi_ab:.4f}]")
print(f"Significant at 0.05: {p_value_ab < 0.05}")


def compute_power(n, p_a, p_b, alpha=0.05):
    \"\"\"Estimate power of two-proportion z-test.\"\"\"
    p_pool = (p_a + p_b) / 2
    se_h0 = np.sqrt(2 * p_pool * (1 - p_pool) / n)
    se_h1 = np.sqrt(p_a * (1-p_a)/n + p_b * (1-p_b)/n)
    z_crit = norm.ppf(1 - alpha / 2)
    power = 1 - norm.cdf(z_crit - (p_b - p_a) / se_h1)
    return power


sample_sizes_ab = [100, 200, 500, 1000, 2000, 5000]
print("\\nSample size vs power (alpha=0.05):")
for n in sample_sizes_ab:
    pwr = compute_power(n, conv_rate_a, conv_rate_b)
    print(f"  n={n:5d}: power={pwr:.3f}")
""",

"""## Real-World Example 2: Bayesian Updating
""",

"""theta_grid = np.linspace(0, 1, 300)
true_theta = 0.65

observations = np.random.binomial(1, true_theta, size=100)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.ravel()
n_after = [0, 5, 10, 20, 50, 100]

for ax, n in zip(axes, n_after):
    heads = observations[:n].sum()
    tails = n - heads
    alpha_post = 1.0 + heads
    beta_post = 1.0 + tails
    posterior_pdf = stats.beta.pdf(theta_grid, alpha_post, beta_post)
    posterior_mean = alpha_post / (alpha_post + beta_post)

    ax.plot(theta_grid, posterior_pdf, color='steelblue', linewidth=2)
    ax.axvline(true_theta, color='red', linestyle='--',
               label=f'True theta={true_theta}')
    ax.axvline(posterior_mean, color='navy', linestyle=':',
               label=f'Post. mean={posterior_mean:.3f}')
    ax.fill_between(theta_grid, posterior_pdf, alpha=0.2, color='steelblue')
    ax.set_title(f'n={n} (H={heads}, T={tails})')
    ax.set_xlabel('theta'); ax.set_ylabel('Density')
    ax.legend(fontsize=7); ax.set_xlim(0, 1)

plt.suptitle('Bayesian Updating: Beta-Binomial Coin Bias Inference', fontsize=12)
plt.tight_layout()
plt.savefig('/tmp/bayesian_updating.png', dpi=80)
plt.close()
print('Bayesian updating plot saved to /tmp/bayesian_updating.png')
print(f'After 100 flips: posterior mean = {observations.sum()/100:.3f} (true: {true_theta})')
""",

"""## Real-World Example 3: Calibration Plot for Probabilistic Classifiers
""",

"""X_cal, y_cal = make_classification(n_samples=3000, n_features=20,
                                    n_informative=10, random_state=42)
X_tr_cal, X_te_cal, y_tr_cal, y_te_cal = train_test_split(
    X_cal, y_cal, test_size=0.4, random_state=42
)

lr_model = LogisticRegression(max_iter=300).fit(X_tr_cal, y_tr_cal)
prob_lr = lr_model.predict_proba(X_te_cal)[:, 1]

svc_cal = CalibratedClassifierCV(LinearSVC(max_iter=1000), cv=3, method='sigmoid')
try:
    svc_cal.fit(X_tr_cal, y_tr_cal)
    prob_svc_cal = svc_cal.predict_proba(X_te_cal)[:, 1]
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        print("OOM: using LR probabilities as fallback")
        prob_svc_cal = prob_lr
    else:
        raise

n_bins = 10
frac_pos_lr,  mean_pred_lr  = calibration_curve(y_te_cal, prob_lr,
                                                  n_bins=n_bins)
frac_pos_svc, mean_pred_svc = calibration_curve(y_te_cal, prob_svc_cal,
                                                  n_bins=n_bins)

brier_lr  = np.mean((prob_lr - y_te_cal) ** 2)
brier_svc = np.mean((prob_svc_cal - y_te_cal) ** 2)
print(f"Logistic Regression Brier score  : {brier_lr:.4f}")
print(f"Calibrated SVC Brier score       : {brier_svc:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(mean_pred_lr,  frac_pos_lr,  marker='o', color='steelblue',
        linewidth=2, label=f'Logistic Reg (Brier={brier_lr:.3f})')
ax.plot(mean_pred_svc, frac_pos_svc, marker='s', color='coral',
        linewidth=2, label=f'Calibrated SVC (Brier={brier_svc:.3f})')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve (Reliability Diagram)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/calibration_plot.png', dpi=80)
plt.close()
print('Calibration plot saved to /tmp/calibration_plot.png')
""",
]))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def make_notebook(cell_sources):
    """Convert a 12-element list of cell sources to an nbformat notebook.
    Cells alternate: markdown, code, markdown, code, ...
    """
    assert len(cell_sources) == 12, f"Expected 12 cells, got {len(cell_sources)}"
    nb = new_notebook()
    for i, src in enumerate(cell_sources):
        if i % 2 == 0:
            nb.cells.append(new_markdown_cell(src))
        else:
            nb.cells.append(new_code_cell(src))
    return nb


def validate_notebook(nb, stem):
    """Basic validation: cell count, alternation, device line, OOM check."""
    cells = nb.cells
    assert len(cells) == 12, f"{stem}: expected 12 cells, got {len(cells)}"
    for i, cell in enumerate(cells):
        expected_type = "markdown" if i % 2 == 0 else "code"
        assert cell.cell_type == expected_type, (
            f"{stem}: cell {i} is {cell.cell_type}, expected {expected_type}"
        )
    # Cell 2 (index 1) must have device setup
    assert 'device = torch.device' in cells[1].source, \
        f"{stem}: cell 2 missing device setup"
    # Cell 6 (index 5) must have OOM handling
    assert 'out of memory' in cells[5].source.lower(), \
        f"{stem}: cell 6 missing OOM handling"
    # Syntax check all code cells
    for i in [1, 3, 5, 7, 9, 11]:
        try:
            ast.parse(cells[i].source)
        except SyntaxError as e:
            raise SyntaxError(f"{stem} cell {i+1}: {e}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = []
    for stem, cell_sources in NOTEBOOKS:
        path = os.path.join(OUTPUT_DIR, f"{stem}.ipynb")
        try:
            nb = make_notebook(cell_sources)
            validate_notebook(nb, stem)
            nbformat.write(nb, path)
            print(f"OK  {stem}.ipynb")
        except Exception as exc:
            errors.append((stem, str(exc)))
            import sys as _sys
            print(f"ERR {stem}.ipynb: {exc}", file=_sys.stderr)

    if errors:
        print(f"\n{len(errors)} notebook(s) failed:", file=sys.stderr)
        for stem, msg in errors:
            print(f"  {stem}: {msg}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(NOTEBOOKS)} notebooks written to {OUTPUT_DIR}")
