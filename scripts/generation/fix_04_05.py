#!/usr/bin/env python3
import json, os

os.chdir("/home/sbisw/github/interviewprep-ml/ai/notebooks")

NB = {
    "04-optimization-algorithms": {
        2: """import numpy as np
import matplotlib.pyplot as plt

class SGDOptimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, w, grad):
        return w - self.lr * grad

class MomentumOptimizer:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.v = None
    def update(self, w, grad):
        if self.v is None: self.v = np.zeros_like(w)
        self.v = self.momentum * self.v - self.lr * grad
        return w + self.v

class AdamOptimizer:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0
        self.m = self.v = None
    def update(self, w, grad):
        if self.m is None:
            self.m, self.v = np.zeros_like(w), np.zeros_like(w)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

np.random.seed(42)
X = np.random.randn(100, 5)
y = X[:, :2].sum(axis=1) + np.random.randn(100) * 0.1

optimizers = {'SGD': SGDOptimizer(0.1), 'Momentum': MomentumOptimizer(0.1), 'Adam': AdamOptimizer(0.1)}
all_losses = {}
for name, opt in optimizers.items():
    theta = np.random.randn(5) * 0.01
    losses = []
    for _ in range(60):
        pred = X @ theta
        grad = (2/len(y)) * X.T @ (pred - y)
        theta = opt.update(theta, grad)
        losses.append(np.mean((X @ theta - y)**2))
    all_losses[name] = losses

plt.figure(figsize=(10, 4))
for name, losses in all_losses.items():
    plt.plot(losses, label=name)
plt.xlabel('Iteration'), plt.ylabel('MSE')
plt.title('SGD vs Momentum vs Adam'), plt.legend(), plt.show()
print(f"Final MSE — SGD: {all_losses['SGD'][-1]:.4f}, Momentum: {all_losses['Momentum'][-1]:.4f}, Adam: {all_losses['Adam'][-1]:.4f}")""",
        11: """## Key Takeaways

**When to Use Each Optimizer:**
- **SGD**: Large datasets, simple models — very predictable, well-studied generalization behavior
- **SGD+Momentum**: Computer vision training — faster convergence than vanilla SGD with minimal overhead
- **Adam**: Default for most deep learning tasks (NLP, transformers) — adaptive learning rates handle sparse gradients
- **AdamW**: When L2 regularization matters — decouples weight decay from gradient scaling
- **RMSprop**: RNNs and non-stationary objectives — handles varying gradient scales well

**Key Hyperparameters:**
- `lr` (learning rate): most important; try 1e-3 (Adam) or 0.1 (SGD)
- `beta1=0.9, beta2=0.999`: Adam defaults work for most tasks
- `weight_decay`: use with AdamW for regularization; typical values 1e-4 to 1e-2

**Production Tips:**
- Always normalize inputs — unnormalized features cause optimizer instability
- Use gradient clipping (`torch.nn.utils.clip_grad_norm_`) for RNNs and transformers
- Log gradient norms to detect exploding/vanishing gradients early
- Adam often needs lower learning rate than SGD

**Related Concepts:**
- [Gradient Descent](./01-gradient-descent.ipynb)
- [Learning Rate Scheduling](./05-learning-rate-scheduling.ipynb)
- [Backpropagation](./02-backpropagation.ipynb)"""
    },
    "05-learning-rate-scheduling": {
        2: """import numpy as np
import matplotlib.pyplot as plt

def step_decay(epoch, lr0=0.1, drop=0.5, every=20):
    return lr0 * (drop ** (epoch // every))

def exponential_decay(epoch, lr0=0.1, gamma=0.95):
    return lr0 * (gamma ** epoch)

def cosine_annealing(epoch, T_max=100, lr_max=0.1, lr_min=1e-5):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T_max))

def warmup_cosine(epoch, warmup=10, T_max=100, lr_max=0.1):
    if epoch < warmup:
        return lr_max * (epoch / warmup)
    return cosine_annealing(epoch - warmup, T_max - warmup, lr_max)

epochs = np.arange(100)
schedules = {
    'Constant (0.1)': np.full(100, 0.1),
    'Step Decay':     [step_decay(e) for e in epochs],
    'Exponential':    [exponential_decay(e) for e in epochs],
    'Cosine':         [cosine_annealing(e) for e in epochs],
    'Warmup+Cosine':  [warmup_cosine(e) for e in epochs],
}

plt.figure(figsize=(12, 5))
for name, lrs in schedules.items():
    plt.plot(epochs, lrs, label=name, linewidth=2)
plt.xlabel('Epoch'), plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules Comparison')
plt.legend(), plt.show()

# Simulate training with each schedule
np.random.seed(42)
X = np.random.randn(100, 5)
y = X[:, :2].sum(axis=1) + np.random.randn(100) * 0.1

for name, lrs in schedules.items():
    theta = np.random.randn(5) * 0.01
    for epoch, lr in enumerate(lrs):
        pred = X @ theta
        grad = (2/len(y)) * X.T @ (pred - y)
        theta -= lr * grad
    final_mse = np.mean((X @ theta - y)**2)
    print(f"{name:25s}: final MSE={final_mse:.4f}")""",
        11: """## Key Takeaways

**When to Use Each Schedule:**
- **Constant LR**: Prototyping and debugging — use first to confirm training is working
- **Step Decay**: Image classification (ResNet, VGG) — well-understood, drop by 10x every 30-50 epochs
- **Cosine Annealing**: Most deep learning tasks — smooth decay, compatible with warm restarts (SGDR)
- **Warmup + Cosine**: Transformers and large language models — warmup prevents early instability
- **Cyclical LR**: When you want automatic exploration without restarts — CLR paper shows 3x–5x speedup
- **OneCycle Policy**: FastAI/PyTorch default — combined warmup + superconvergence

**Key Hyperparameters:**
- `T_max`: total epochs for cosine cycle
- `warmup_steps`: typically 4–10% of total steps for transformers (e.g., 4000 for BERT)
- Initial LR range for cyclical: use LR range test (train 1–5 epochs, log LR vs loss)

**Production Tips:**
- Log LR each step — sudden LR changes reveal schedule bugs
- For fine-tuning pretrained models: use very small LR (1e-5 to 1e-4), no warmup needed if short run
- PyTorch: `torch.optim.lr_scheduler.CosineAnnealingLR`, `OneCycleLR`, `ReduceLROnPlateau`
- Combine with early stopping: reduce LR on plateau before stopping

**Related Concepts:**
- [Optimization Algorithms](./04-optimization-algorithms.ipynb)
- [Gradient Descent](./01-gradient-descent.ipynb)
- [Batch Normalization](./17-batch-normalization.ipynb)"""
    }
}

for slug, cells in NB.items():
    path = f"{slug}.ipynb"
    with open(path, 'r') as f:
        nb = json.load(f)

    for cell_idx, source in cells.items():
        cell_type = 'code' if not source.startswith('#') and not source.startswith('##') else 'markdown'
        # Determine actual type from content
        if source.lstrip().startswith('import') or source.lstrip().startswith('def ') or source.lstrip().startswith('class ') or source.lstrip().startswith('np.') or source.lstrip().startswith('X '):
            cell_type = 'code'
        elif source.startswith('##'):
            cell_type = 'markdown'

        nb['cells'][cell_idx]['cell_type'] = cell_type
        nb['cells'][cell_idx]['source'] = source
        if cell_type == 'code':
            nb['cells'][cell_idx]['outputs'] = []
            nb['cells'][cell_idx]['execution_count'] = None
        else:
            nb['cells'][cell_idx].pop('outputs', None)
            nb['cells'][cell_idx].pop('execution_count', None)

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"✓ {path}")

print("✅ Done")
