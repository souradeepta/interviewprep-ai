# Python ML Snippets Quick Reference

Copy-paste ready code for common ML tasks.

---

## Sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# Example: mixed numeric + categorical features
X = pd.DataFrame({
    'age': [25, 32, 47, 18, 61],
    'income': [50000, 72000, 120000, 25000, 95000],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA'],
    'product': ['A', 'B', 'A', 'C', 'B'],
})
y = np.array([1, 0, 1, 0, 1])

numeric_features = ['age', 'income']
categorical_features = ['city', 'product']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)),
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_val)
print(classification_report(y_val, preds, zero_division=0))
```

---

## PyTorch Training Loop

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# --- Model definition ---
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- Data ---
X = torch.randn(1000, 32)
y = (X[:, 0] + X[:, 1] > 0).long()  # synthetic binary label
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# --- Training ---
model = MLP(in_dim=32, hidden_dim=128, out_dim=2).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
```

---

## PyTorch Evaluation

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate model on a DataLoader; returns accuracy, F1, AUC."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float('nan'),
    }

val_loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=False)
metrics = evaluate(model, val_loader, device)
print(f"Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f} | AUC: {metrics['auc']:.3f}")
```

---

## HuggingFace Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

texts = [
    "This product is absolutely amazing!",
    "Terrible experience, would not recommend.",
    "It's okay, nothing special.",
]

def batch_predict(texts: list[str], batch_size: int = 32) -> list[dict]:
    """Predict sentiment in batches."""
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        for text, prob in zip(batch, probs):
            label_idx = prob.argmax()
            all_results.append({
                'text': text[:50],
                'label': model.config.id2label[label_idx],
                'confidence': float(prob[label_idx]),
            })
    return all_results

for result in batch_predict(texts):
    print(f"{result['label']:10s} ({result['confidence']:.2%}): {result['text']}")
```

---

## NumPy Softmax (Numerically Stable)

```python
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - x.max(axis=axis, keepdims=True)  # subtract max for stability
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

logits = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
probs = softmax(logits)
print("Probabilities:", probs)  # rows sum to 1.0
```

---

## Confusion Matrix (Plotted)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Plot labeled confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=axes[0], colorbar=False)
    axes[0].set_title(f"{title} (Counts)")

    # Normalized (percentages)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
        ax=axes[1], colorbar=False, values_format=".1%"
    )
    axes[1].set_title(f"{title} (Row %)")
    plt.tight_layout()
    plt.show()
    return cm

y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0, 1, 0]
cm = plot_confusion_matrix(y_true, y_pred, labels=[0, 1])
```

---

## Cross-Validation with Timing

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

def timed_cross_val(model, X, y, cv=5, scoring='roc_auc', random_state=42):
    """Run stratified K-fold CV and report mean, std, and wall time."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    t0 = time.perf_counter()
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    elapsed = time.perf_counter() - t0
    print(f"CV {scoring}: {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"Folds: {scores.tolist()}")
    print(f"Wall time: {elapsed:.1f}s total ({elapsed/cv:.1f}s/fold avg)")
    return scores

X = np.random.randn(1000, 20)
y = (X[:, 0] > 0).astype(int)
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
scores = timed_cross_val(rf, X, y, cv=5)
```

---

## Gradient Accumulation

```python
import torch
from torch import nn
from torch.optim import AdamW

# Useful when GPU memory limits true batch size
# Effective batch size = batch_size * accumulation_steps

model = MLP(in_dim=32, hidden_dim=128, out_dim=2).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

accumulation_steps = 4  # effective batch = 64 * 4 = 256

optimizer.zero_grad()
for step, (X_batch, y_batch) in enumerate(loader):
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    logits = model(X_batch)
    # Divide loss by accumulation steps to keep gradient scale correct
    loss = criterion(logits, y_batch) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step+1}: effective batch update applied")
```

---

## Mixed Precision Training

```python
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(in_dim=32, hidden_dim=128, out_dim=2).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler(enabled=torch.cuda.is_available())  # disabled on CPU

for X_batch, y_batch in loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    # Forward pass in fp16 (or bf16 on newer GPUs)
    with autocast(device_type=device.type, dtype=torch.float16):
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
    # Scale gradients to prevent fp16 underflow
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## Early Stopping

```python
class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.stop = False
        self.best_state = None

    def step(self, metric: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        improved = (metric < self.best - self.min_delta) if self.mode == 'min' \
                   else (metric > self.best + self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
            # Save best model state
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

# Usage
stopper = EarlyStopping(patience=5, mode='min')
for epoch in range(100):
    val_loss = 0.0  # compute actual validation loss here
    if stopper.step(val_loss, model):
        print(f"Early stop at epoch {epoch+1}. Best loss: {stopper.best:.4f}")
        model.load_state_dict(stopper.best_state)
        break
```

---

## LRU Cache for Embeddings

```python
from functools import lru_cache
import hashlib
import numpy as np

# For a fixed embedding model: cache results keyed by text hash
# lru_cache requires hashable arguments

@lru_cache(maxsize=10_000)
def _embed_cached(text_hash: str, text: str) -> tuple:
    """Cache embedding by text. Returns tuple (hashable) for lru_cache."""
    # Replace with actual model call
    embedding = np.random.randn(384).tolist()  # dummy
    return tuple(embedding)

def embed_text(text: str) -> np.ndarray:
    """Get embedding with LRU caching."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return np.array(_embed_cached(text_hash, text))

# For production-scale: use Redis with TTL instead of in-process cache
import json

def embed_with_redis_cache(text: str, model, redis_client, ttl: int = 3600) -> np.ndarray:
    """Get embedding from Redis cache or compute and store."""
    cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
    cached = redis_client.get(cache_key)
    if cached:
        return np.array(json.loads(cached))
    embedding = model.encode(text)  # sentence-transformers style
    redis_client.setex(cache_key, ttl, json.dumps(embedding.tolist()))
    return embedding
```
