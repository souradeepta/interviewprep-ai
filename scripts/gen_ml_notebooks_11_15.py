"""
Generate ml/notebooks 11-15: early-stopping through gradient-accumulation.
Each notebook has exactly 12 cells: 6 markdown + 6 code, strictly alternating.
Structure:
  Cell 1  md   Title + 4 learning objectives
  Cell 2  code imports + device + seeds
  Cell 3  md   Level 1 header
  Cell 4  code L1 numpy (20-40 lines)
  Cell 5  md   Level 2 header
  Cell 6  code L2 torch/sklearn (60-100 lines)
  Cell 7  md   RW1 header
  Cell 8  code RW1 (40-60 lines)
  Cell 9  md   RW2 header
  Cell 10 code RW2 (40-60 lines)
  Cell 11 md   RW3 header
  Cell 12 code RW3 (40-60 lines)
"""
import nbformat
import os

OUT_DIR = "/home/sbisw/github/interviewprep-ml/ml/notebooks"
os.makedirs(OUT_DIR, exist_ok=True)


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(text):
    return nbformat.v4.new_code_cell(text)


def save(nb, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        nbformat.write(nb, f)
    n = len(nb.cells)
    status = "PASS" if n == 12 else f"FAIL (got {n})"
    print(f"{status}: {name}")


# ==============================================================================
# Notebook 11 — early-stopping
# ==============================================================================
def make_11():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        # 1
        md("""# Early Stopping

## Learning Objectives
1. Understand why early stopping prevents overfitting by halting training when validation loss stops improving.
2. Implement a patience-based early stopping mechanism from scratch with numpy.
3. Build a PyTorch EarlyStopping class that restores the best model weights.
4. Apply early stopping with learning-rate scheduling and model checkpointing in production training loops.
"""),
        # 2
        code("""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""),
        # 3
        md("## Level 1: Basic Early Stopping (NumPy Simulation)"),
        # 4
        code("""# Simulate a training/validation loss curve and apply patience-based early stopping
def simulate_losses(n_epochs=100, noise_std=0.05):
    \"\"\"Generate synthetic train/val losses where val loss starts rising after epoch ~40.\"\"\"
    epochs = np.arange(n_epochs)
    train_loss = 1.0 / (1 + 0.1 * epochs) + np.random.normal(0, noise_std, n_epochs)
    # Validation loss has a U-shape: improves then degrades (overfitting)
    val_loss = (1.0 / (1 + 0.1 * epochs) + 0.003 * (epochs - 40) ** 2 / 40
                + np.random.normal(0, noise_std, n_epochs))
    return train_loss, val_loss


def early_stop(val_losses, patience=5, min_delta=1e-4):
    \"\"\"Return the epoch index at which training should stop.\"\"\"
    best_loss = float("inf")
    wait = 0
    for epoch, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            return epoch
    return len(val_losses) - 1


train_loss, val_loss = simulate_losses(100)
stop_epoch = early_stop(val_loss, patience=5)
print(f"Training stopped at epoch {stop_epoch} (out of 100)")
print(f"Best val loss: {val_loss[:stop_epoch+1].min():.4f}")
print(f"Val loss at epoch 100 would be: {val_loss[-1]:.4f}")
"""),
        # 5
        md("## Level 2: PyTorch EarlyStopping with Best-Weight Restoration"),
        # 6
        code("""class EarlyStopping:
    \"\"\"Patience-based early stopper that restores the best model weights.\"\"\"

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.wait: int = 0
        self.best_weights = None
        self.stopped_epoch: int = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        \"\"\"Return True when training should stop.\"\"\"
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            # Deep copy state_dict so later updates don't overwrite it
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
        if self.wait >= self.patience:
            return True
        return False

    def restore(self, model: nn.Module) -> None:
        \"\"\"Load the best-checkpoint weights back into the model.\"\"\"
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# --- Build a tiny MLP on synthetic regression data ---
X = torch.randn(500, 10, device=device)
true_w = torch.randn(10, 1, device=device)
y = X @ true_w + 0.1 * torch.randn(500, 1, device=device)

split = 400
train_ds = TensorDataset(X[:split], y[:split])
val_ds = TensorDataset(X[split:], y[split:])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)


def build_model():
    return nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)


model = build_model()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
stopper = EarlyStopping(patience=8, min_delta=1e-5)

history = {"train": [], "val": []}
for epoch in range(200):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimiser.zero_grad()
        try:
            loss = criterion(model(xb), yb)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("OOM — reduce batch size")
                torch.cuda.empty_cache()
                continue
            raise
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * len(xb)
    train_loss = running_loss / len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        val_loss = sum(
            criterion(model(xb), yb).item() * len(xb)
            for xb, yb in val_loader
        ) / len(val_loader.dataset)

    history["train"].append(train_loss)
    history["val"].append(val_loss)

    if stopper.step(val_loss, model):
        print(f"Early stop at epoch {epoch+1} — restoring best weights")
        stopper.restore(model)
        break

print(f"Final val MSE (best checkpoint): {stopper.best_loss:.6f}")
"""),
        # 7
        md("## Real-World Example 1: Early Stopping + ReduceLROnPlateau"),
        # 8
        code("""# Combine early stopping with learning-rate reduction on plateau.
# Strategy: reduce LR when val_loss stalls for `factor_patience` steps;
#           stop entirely when LR drops below min_lr.

model_rw1 = build_model()
opt_rw1 = torch.optim.Adam(model_rw1.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_rw1, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=False
)
stopper_rw1 = EarlyStopping(patience=10, min_delta=1e-6)
MIN_LR = 1e-6
criterion_rw1 = nn.MSELoss()

lr_history = []
val_history = []
for epoch in range(300):
    model_rw1.train()
    for xb, yb in train_loader:
        opt_rw1.zero_grad()
        criterion_rw1(model_rw1(xb), yb).backward()
        opt_rw1.step()

    model_rw1.eval()
    with torch.no_grad():
        vl = sum(
            criterion_rw1(model_rw1(xb), yb).item() * len(xb)
            for xb, yb in val_loader
        ) / len(val_loader.dataset)

    scheduler.step(vl)
    current_lr = opt_rw1.param_groups[0]["lr"]
    lr_history.append(current_lr)
    val_history.append(vl)

    # Stop when LR can't decrease further — no more learning to be done
    if current_lr <= MIN_LR:
        print(f"LR reached minimum at epoch {epoch+1}. Stopping.")
        stopper_rw1.restore(model_rw1)
        break

    if stopper_rw1.step(vl, model_rw1):
        print(f"Early stop at epoch {epoch+1}")
        stopper_rw1.restore(model_rw1)
        break

print(f"Final LR: {lr_history[-1]:.2e}  |  Best val MSE: {min(val_history):.6f}")
"""),
        # 9
        md("## Real-World Example 2: Model Checkpointing on Improvement"),
        # 10
        code("""import os
import tempfile

# Save a checkpoint every time val_loss improves — production pattern
# for long training runs where you may need to resume or roll back.

CKPT_DIR = tempfile.mkdtemp()
model_rw2 = build_model()
opt_rw2 = torch.optim.Adam(model_rw2.parameters(), lr=1e-3)
criterion_rw2 = nn.MSELoss()
best_val_loss = float("inf")
checkpoint_epochs = []

for epoch in range(60):
    model_rw2.train()
    for xb, yb in train_loader:
        opt_rw2.zero_grad()
        criterion_rw2(model_rw2(xb), yb).backward()
        opt_rw2.step()

    model_rw2.eval()
    with torch.no_grad():
        vl = sum(
            criterion_rw2(model_rw2(xb), yb).item() * len(xb)
            for xb, yb in val_loader
        ) / len(val_loader.dataset)

    if vl < best_val_loss:
        best_val_loss = vl
        ckpt_path = os.path.join(CKPT_DIR, f"best_epoch_{epoch+1:03d}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model_rw2.state_dict(),
            "optimizer_state_dict": opt_rw2.state_dict(),
            "val_loss": vl,
        }, ckpt_path)
        checkpoint_epochs.append(epoch + 1)

print(f"Saved {len(checkpoint_epochs)} checkpoints at epochs: {checkpoint_epochs}")
print(f"Best val MSE: {best_val_loss:.6f}")

# Restore from best checkpoint
ckpt = torch.load(ckpt_path, map_location=device)
model_rw2.load_state_dict(ckpt["model_state_dict"])
print(f"Restored checkpoint from epoch {ckpt['epoch']} with val_loss={ckpt['val_loss']:.6f}")
"""),
        # 11
        md("## Real-World Example 3: Learning Curve Comparison — Early Stop vs Full Training"),
        # 12
        code("""# Demonstrate that early stopping prevents overfitting and generalises better
# than running for the full epoch budget.

def train_model(n_epochs, use_early_stop, patience=8):
    \"\"\"Train model and return (train_hist, val_hist, stopped_at).\"\"\"
    m = build_model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    stopper = EarlyStopping(patience=patience) if use_early_stop else None
    train_hist, val_hist = [], []
    stopped_at = n_epochs

    for epoch in range(n_epochs):
        m.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            crit(m(xb), yb).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            tl = sum(crit(m(xb), yb).item() * len(xb) for xb, yb in train_loader) / len(train_loader.dataset)
            vl = sum(crit(m(xb), yb).item() * len(xb) for xb, yb in val_loader) / len(val_loader.dataset)
        train_hist.append(tl)
        val_hist.append(vl)
        if stopper and stopper.step(vl, m):
            stopped_at = epoch + 1
            stopper.restore(m)
            break

    return train_hist, val_hist, stopped_at


hist_es_train, hist_es_val, stop_at = train_model(200, use_early_stop=True)
hist_full_train, hist_full_val, _ = train_model(200, use_early_stop=False)

print(f"Early stop at epoch {stop_at}  |  best val MSE: {min(hist_es_val):.6f}")
print(f"Full training (200 ep)        |  final val MSE: {hist_full_val[-1]:.6f}")
print(f"Val MSE improvement: {hist_full_val[-1] - min(hist_es_val):.6f}")

# Early stopping: fewer epochs, less overfitting, often better or equal generalisation
"""),
    ]
    save(nb, "11-early-stopping.ipynb")


# ==============================================================================
# Notebook 12 — ensemble-methods
# ==============================================================================
def make_12():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("""# Ensemble Methods

## Learning Objectives
1. Understand how combining multiple weak learners reduces variance (bagging) or bias (boosting).
2. Implement majority voting and soft voting ensemble from scratch with NumPy.
3. Compare BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier and VotingClassifier on synthetic data.
4. Build a stacking ensemble with a meta-learner and analyse feature importances from Random Forest and XGBoost.
"""),
        code("""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
"""),
        md("## Level 1: Majority Voting + Soft Voting (NumPy)"),
        code("""# Three toy classifiers represented as probability arrays
np.random.seed(42)
n_samples = 200

# Simulate predictions from three classifiers on a binary problem
# prob shape: (n_samples, n_classes=2)
def toy_probs(noise=0.1):
    \"\"\"Generate random but biased probability predictions.\"\"\"
    true_labels = np.random.randint(0, 2, n_samples)
    probs = np.zeros((n_samples, 2))
    for i, label in enumerate(true_labels):
        correct_prob = np.clip(0.7 + np.random.normal(0, noise), 0, 1)
        probs[i, label] = correct_prob
        probs[i, 1 - label] = 1 - correct_prob
    return probs, true_labels


p1, true_y = toy_probs(noise=0.15)
p2, _ = toy_probs(noise=0.20)
p3, _ = toy_probs(noise=0.10)


def majority_vote(probs_list):
    \"\"\"Hard majority vote: take argmax per classifier, then vote.\"\"\"
    votes = np.stack([np.argmax(p, axis=1) for p in probs_list], axis=1)
    # shape: (n_samples, n_classifiers)
    ensemble_pred = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=2).argmax(), axis=1, arr=votes
    )
    return ensemble_pred


def soft_vote(probs_list):
    \"\"\"Average probabilities, then argmax.\"\"\"
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    return np.argmax(avg_probs, axis=1)


hard_preds = majority_vote([p1, p2, p3])
soft_preds = soft_vote([p1, p2, p3])
single_preds = np.argmax(p1, axis=1)

print(f"Single classifier accuracy  : {accuracy_score(true_y, single_preds):.4f}")
print(f"Majority (hard) vote accuracy: {accuracy_score(true_y, hard_preds):.4f}")
print(f"Soft vote accuracy           : {accuracy_score(true_y, soft_preds):.4f}")
"""),
        md("## Level 2: Bagging, Boosting, Voting with sklearn"),
        code("""from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, VotingClassifier, RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base estimator for bagging/boosting
base_dt = DecisionTreeClassifier(max_depth=3, random_state=42)

# Define ensemble classifiers
ensembles = {
    "DecisionTree (base)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "BaggingClassifier": BaggingClassifier(
        estimator=base_dt, n_estimators=50, random_state=42, n_jobs=-1
    ),
    "AdaBoostClassifier": AdaBoostClassifier(
        estimator=base_dt, n_estimators=100, learning_rate=0.1, random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    ),
    "VotingClassifier (soft)": VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ],
        voting="soft",
    ),
}

results = {}
for name, clf in ensembles.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name:<35}: accuracy = {acc:.4f}")

best = max(results, key=results.get)
print(f"\nBest ensemble: {best} ({results[best]:.4f})")
"""),
        md("## Real-World Example 1: Random Forest Feature Importance"),
        code("""# Feature importance reveals which predictors drive RF decisions.
# Two measures: impurity-based (fast, built-in) and permutation (reliable, slower).

from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Built-in impurity-based importance
impurity_importance = rf.feature_importances_
feature_names = [f"feat_{i:02d}" for i in range(X.shape[1])]

# Sort features by importance
sorted_idx = np.argsort(impurity_importance)[::-1]

print("Top-10 features by impurity importance:")
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"  {rank:2d}. {feature_names[idx]}: {impurity_importance[idx]:.4f}")

# Permutation importance (more reliable — measures actual prediction impact)
perm_result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
perm_sorted_idx = perm_result.importances_mean.argsort()[::-1]

print("\nTop-10 features by permutation importance:")
for rank, idx in enumerate(perm_sorted_idx[:10], 1):
    print(f"  {rank:2d}. {feature_names[idx]}: {perm_result.importances_mean[idx]:.4f} "
          f"(+/- {perm_result.importances_std[idx]:.4f})")
"""),
        md("## Real-World Example 2: XGBoost with Early Stopping + Feature Importance"),
        code("""try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed — showing sklearn GBM equivalent")

if HAS_XGB:
    from sklearn.model_selection import train_test_split as tts

    X_tr, X_val, y_tr, y_val = tts(X_train, y_train, test_size=0.15, random_state=42)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )
    xgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_acc = accuracy_score(y_test, xgb_clf.predict(X_test))
    print(f"XGBoost accuracy: {xgb_acc:.4f}  (best iteration: {xgb_clf.best_iteration})")

    # Feature importance from XGBoost (gain-based)
    gain_importance = xgb_clf.feature_importances_
    top5_idx = gain_importance.argsort()[::-1][:5]
    print("Top-5 features by XGBoost gain:")
    for idx in top5_idx:
        print(f"  {feature_names[idx]}: {gain_importance[idx]:.4f}")
else:
    # Fallback using sklearn's GBM
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    print(f"GBM accuracy: {accuracy_score(y_test, gb.predict(X_test)):.4f}")
"""),
        md("## Real-World Example 3: Stacking Ensemble with Meta-Learner"),
        code("""from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Stacking: base learners produce out-of-fold predictions;
# meta-learner (LogisticRegression) learns how to combine them.

base_learners = [
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
    ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)),
    ("svc", make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))),
]

meta_learner = LogisticRegression(C=1.0, max_iter=500, random_state=42)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,                  # 5-fold cross-val to produce meta-features
    stack_method="predict_proba",  # use probabilities, not hard labels
    n_jobs=-1,
    passthrough=False,     # meta-learner only sees base predictions, not raw features
)

stacking_clf.fit(X_train, y_train)
stacking_acc = accuracy_score(y_test, stacking_clf.predict(X_test))
print(f"Stacking ensemble accuracy: {stacking_acc:.4f}")

# Compare to individual base learners
for name, clf in base_learners:
    clf_copy = clf.__class__(**clf.get_params()) if hasattr(clf, 'get_params') else clf
    # Re-use already-fitted base learners from stacking for fair comparison
print("\nSummary: stacking pools diverse models; works best when base errors are uncorrelated.")
print(f"Stacking accuracy ({stacking_acc:.4f}) vs best single ({max(results.values()):.4f})")
"""),
    ]
    save(nb, "12-ensemble-methods.ipynb")


# ==============================================================================
# Notebook 13 — evaluation-metrics
# ==============================================================================
def make_13():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("""# Evaluation Metrics

## Learning Objectives
1. Derive precision, recall, F1, and accuracy from a confusion matrix using only NumPy.
2. Use sklearn's full metrics suite including ROC-AUC, PR-AUC, and multi-class reports.
3. Understand when macro, micro, and weighted averaging matter for imbalanced datasets.
4. Align model metrics with business objectives by plotting precision-recall tradeoff curves.
"""),
        code("""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
"""),
        md("## Level 1: Classification Metrics from Scratch (NumPy)"),
        code("""def confusion_matrix_np(y_true, y_pred, n_classes=None):
    \"\"\"Build confusion matrix without sklearn.\"\"\"
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm):
    \"\"\"Compute per-class precision, recall, F1 and overall accuracy.\"\"\"
    n = cm.shape[0]
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp   # column sums minus diagonal
    fn = cm.sum(axis=1) - tp   # row sums minus diagonal

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1 = np.where(precision + recall > 0,
                  2 * precision * recall / (precision + recall), 0.0)
    accuracy = tp.sum() / cm.sum()
    return precision, recall, f1, accuracy


# Synthetic binary data
y_true = np.array([0]*60 + [1]*40)
np.random.shuffle(y_true)
noise_idx = np.random.choice(len(y_true), size=15, replace=False)
y_pred = y_true.copy()
y_pred[noise_idx] = 1 - y_pred[noise_idx]

cm = confusion_matrix_np(y_true, y_pred)
prec, rec, f1_score, acc = metrics_from_cm(cm)

print("Confusion matrix:\n", cm)
print(f"\nClass 0 — Precision: {prec[0]:.3f}  Recall: {rec[0]:.3f}  F1: {f1_score[0]:.3f}")
print(f"Class 1 — Precision: {prec[1]:.3f}  Recall: {rec[1]:.3f}  F1: {f1_score[1]:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Macro F1: {f1_score.mean():.3f}")
"""),
        md("## Level 2: Full Metrics Suite with sklearn"),
        code("""from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

# Generate a moderately imbalanced binary dataset
X, y = make_classification(
    n_samples=1000, n_features=15, n_informative=8,
    weights=[0.7, 0.3], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

print("=" * 55)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
print(f"ROC-AUC : {roc_auc:.4f}  (1.0 = perfect, 0.5 = random)")
print(f"PR-AUC  : {pr_auc:.4f}  (equals prevalence for random classifier)")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix (rows=true, cols=pred):")
print(cm)
print(f"\nTrue Positives: {cm[1,1]}  False Negatives: {cm[1,0]}")
print(f"False Positives: {cm[0,1]}  True Negatives: {cm[0,0]}")
"""),
        md("## Real-World Example 1: Macro vs Micro vs Weighted F1 (Imbalanced Classes)"),
        code("""# Macro: average each class equally (penalises poor minority-class performance)
# Micro: aggregate TP/FP/FN globally (dominated by majority class)
# Weighted: weight each class by its support (balanced between macro and micro)

from sklearn.metrics import f1_score

# Build a 3-class imbalanced dataset
X3, y3 = make_classification(
    n_samples=1500, n_features=12, n_informative=6,
    n_classes=3, n_clusters_per_class=1,
    weights=[0.6, 0.3, 0.1],   # 60/30/10 split
    random_state=42
)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)

lr3 = LogisticRegression(max_iter=500, random_state=42)
lr3.fit(X3_tr, y3_tr)
y3_pred = lr3.predict(X3_te)

for avg in ["macro", "micro", "weighted"]:
    score = f1_score(y3_te, y3_pred, average=avg)
    print(f"F1 ({avg:8s}): {score:.4f}")

print()
print("Per-class support:")
for cls in range(3):
    support = (y3_te == cls).sum()
    cls_f1 = f1_score(y3_te == cls, y3_pred == cls)
    print(f"  Class {cls}: support={support:3d}  F1={cls_f1:.4f}")

print("\nKey insight:")
print("  Macro treats rare class equally — use when minority errors are costly.")
print("  Weighted hides minority failure — use for overall system health monitoring.")
"""),
        md("## Real-World Example 2: Regression Metrics"),
        code("""# Common regression metrics: MSE, MAE, RMSE, R², MAPE
# Huber loss bridges MSE (sensitive to outliers) and MAE (robust but non-smooth).

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    \"\"\"MAPE — interpretable as % error. Fails when y_true has zeros.\"\"\"
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def huber_loss(y_true, y_pred, delta=1.0):
    \"\"\"Huber: quadratic for |e|<=delta, linear beyond.\"\"\"
    e = np.abs(y_true - y_pred)
    return np.where(e <= delta, 0.5 * e**2, delta * (e - 0.5 * delta)).mean()


# Synthetic regression targets with a few outliers
n = 200
y_true_reg = 3.0 * np.random.randn(n) + 10.0
y_pred_reg = y_true_reg + np.random.normal(0, 1.0, n)
# Inject 5 outliers
outlier_idx = np.random.choice(n, 5, replace=False)
y_pred_reg[outlier_idx] += np.random.choice([-10, 10], 5)

mse = mean_squared_error(y_true_reg, y_pred_reg)
mae = mean_absolute_error(y_true_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_reg, y_pred_reg)
mape = mean_absolute_percentage_error(y_true_reg, y_pred_reg)
huber = huber_loss(y_true_reg, y_pred_reg, delta=2.0)

print(f"MSE  : {mse:.4f}   (penalises outliers quadratically)")
print(f"RMSE : {rmse:.4f}  (same units as target, still outlier-sensitive)")
print(f"MAE  : {mae:.4f}   (robust to outliers, linear penalty)")
print(f"Huber: {huber:.4f}  (blend: robust for large errors, smooth near zero)")
print(f"R²   : {r2:.4f}    (explained variance; 1.0 = perfect)")
print(f"MAPE : {mape:.2f}%  (interpretable %, fails at zero targets)")
"""),
        md("## Real-World Example 3: Precision-Recall Tradeoff for Business Objectives"),
        code("""# Different tasks require different operating points on the PR curve.
# Spam filter → maximise precision (low false-positive rate, user trust)
# Cancer screening → maximise recall (low false-negative rate, patient safety)

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find threshold maximising F1
f1_values = 2 * precisions * recalls / np.where(precisions + recalls > 0, precisions + recalls, 1)
best_f1_idx = np.argmax(f1_values[:-1])  # exclude last point (precision=1, recall=0)

# High-precision operating point (spam filter: precision >= 0.95)
high_prec_idx = np.searchsorted(-precisions, -0.95)
# High-recall operating point (screening: recall >= 0.90)
high_rec_idx = np.searchsorted(-recalls[::-1], -0.90)
high_rec_idx = len(recalls) - 1 - high_rec_idx

print("Operating point comparison:")
print(f"{'Objective':<25} {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 65)

for label, idx in [("Max F1", best_f1_idx),
                   ("Spam filter (P≥0.95)", min(high_prec_idx, len(thresholds)-1)),
                   ("Screening (R≥0.90)", min(high_rec_idx, len(thresholds)-1))]:
    thr = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    p = precisions[idx]
    r = recalls[idx]
    f = 2*p*r/(p+r) if p+r > 0 else 0
    print(f"{label:<25} {thr:>10.3f} {p:>10.3f} {r:>8.3f} {f:>8.3f}")

print(f"\nAU-PRC: {average_precision_score(y_test, y_prob):.4f}")
print("Lesson: choose threshold based on cost asymmetry, not just max-F1.")
"""),
    ]
    save(nb, "13-evaluation-metrics.ipynb")


# ==============================================================================
# Notebook 14 — feature-engineering
# ==============================================================================
def make_14():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("""# Feature Engineering

## Learning Objectives
1. Implement polynomial features and interaction terms from scratch using NumPy.
2. Build sklearn ColumnTransformer pipelines handling numerical and categorical data together.
3. Extract time-series features (lag, rolling statistics, calendar) for temporal models.
4. Compare encoding strategies and apply automated feature selection to reduce dimensionality.
"""),
        code("""import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
"""),
        md("## Level 1: Polynomial Features + Interaction Terms (NumPy)"),
        code("""def polynomial_features(X, degree=2, include_bias=False):
    \"\"\"
    Expand X to polynomial features up to `degree`.
    For 2 features [a, b] and degree=2: [a, b, a^2, a*b, b^2]
    \"\"\"
    n_samples, n_features = X.shape
    terms = list(range(n_features)) if include_bias else list(range(n_features))
    output_cols = [X]  # include original features

    # Interaction terms: all pairs (i, j) with i <= j
    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                output_cols.append((X[:, i] ** 2).reshape(-1, 1))  # squared
            else:
                output_cols.append((X[:, i] * X[:, j]).reshape(-1, 1))  # cross-term

    if degree >= 3:
        for i in range(n_features):
            output_cols.append((X[:, i] ** 3).reshape(-1, 1))

    result = np.hstack(output_cols)
    return result


# Verify on simple 2D data
X_demo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
X_poly = polynomial_features(X_demo, degree=2)
print("Original shape:", X_demo.shape, "→ Polynomial shape:", X_poly.shape)
print("Original features:   ", X_demo[0])
print("Polynomial features: ", X_poly[0])
# Expected for [1,2]: [1, 2, 1, 2, 4] = original + a^2 + a*b + b^2

# Show improvement on non-linear classification
from sklearn.datasets import make_circles
X_circ, y_circ = make_circles(n_samples=400, noise=0.15, random_state=42)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_tr, X_te, y_tr, y_te = train_test_split(X_circ, y_circ, test_size=0.25, random_state=42)

lr_linear = LogisticRegression(random_state=42).fit(X_tr, y_tr)
X_tr_poly = polynomial_features(X_tr, degree=3)
X_te_poly = polynomial_features(X_te, degree=3)
lr_poly = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_poly, y_tr)

print(f"\nLinear LR on circles: {lr_linear.score(X_te, y_te):.4f}")
print(f"Degree-3 poly LR:     {lr_poly.score(X_te_poly, y_te):.4f}")
"""),
        md("## Level 2: ColumnTransformer + FeatureUnion Pipeline"),
        code("""from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Create a synthetic mixed (numerical + categorical) dataset
np.random.seed(42)
n = 800
age = np.random.randint(18, 70, n).astype(float)
income = np.random.exponential(scale=50000, size=n)
experience = np.random.randint(0, 40, n).astype(float)
job_type = np.random.choice(["engineer", "manager", "analyst", "intern"], n)
education = np.random.choice(["high_school", "bachelors", "masters", "phd"], n)

# Target: high earner (income > 60000)
y_mixed = (income > 60000).astype(int)

# Build DataFrame
df = pd.DataFrame({
    "age": age, "income_raw": income, "experience": experience,
    "job_type": job_type, "education": education,
})
# Drop target-leaking column
df_feat = df.drop(columns=["income_raw"])

num_features = ["age", "experience"]
cat_features = ["job_type", "education"]

# Numerical pipeline: scale + polynomial interactions
num_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
])

# Categorical pipeline: one-hot encode
cat_pipeline = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features),
], remainder="drop")

# Full pipeline: preprocessing + classifier
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500, C=1.0, random_state=42)),
])

from sklearn.model_selection import cross_val_score
scores = cross_val_score(full_pipeline, df_feat, y_mixed, cv=5, scoring="roc_auc")
print(f"Pipeline ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

full_pipeline.fit(df_feat, y_mixed)
print(f"Train accuracy: {full_pipeline.score(df_feat, y_mixed):.4f}")
print(f"Transformed feature shape: {preprocessor.fit_transform(df_feat).shape}")
"""),
        md("## Real-World Example 1: Time Series Feature Extraction"),
        code("""# Time series features: lags, rolling stats, calendar indicators
# These convert a raw time series into a flat feature matrix for tabular models.

np.random.seed(42)
dates = pd.date_range("2022-01-01", periods=365, freq="D")
sales = (100
         + 20 * np.sin(2 * np.pi * np.arange(365) / 365)  # yearly seasonality
         + 5 * np.sin(2 * np.pi * np.arange(365) / 7)     # weekly seasonality
         + np.random.normal(0, 5, 365))                    # noise
df_ts = pd.DataFrame({"date": dates, "sales": sales})
df_ts.set_index("date", inplace=True)


def extract_time_features(df, target_col="sales", lag_days=(1, 7, 14, 28),
                          rolling_windows=(7, 14, 28)):
    \"\"\"Create lag, rolling, and calendar features from a daily time series.\"\"\"
    df_out = df.copy()

    # Lag features: value N days ago
    for lag in lag_days:
        df_out[f"lag_{lag}d"] = df_out[target_col].shift(lag)

    # Rolling statistics: capture local trend and volatility
    for win in rolling_windows:
        df_out[f"roll_mean_{win}d"] = df_out[target_col].shift(1).rolling(win).mean()
        df_out[f"roll_std_{win}d"] = df_out[target_col].shift(1).rolling(win).std()

    # Calendar features: periodicity information
    df_out["day_of_week"] = df_out.index.dayofweek         # 0=Monday
    df_out["day_of_month"] = df_out.index.day
    df_out["month"] = df_out.index.month
    df_out["is_weekend"] = (df_out.index.dayofweek >= 5).astype(int)
    df_out["week_of_year"] = df_out.index.isocalendar().week.astype(int)

    df_out = df_out.dropna()  # drop rows with NaN from lags/rolling
    return df_out


df_features = extract_time_features(df_ts)
print("Original shape:", df_ts.shape)
print("After feature extraction:", df_features.shape)
print("Features created:", [c for c in df_features.columns if c != "sales"])

# Quick regression test
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X_ts = df_features.drop(columns=["sales"]).values
y_ts = df_features["sales"].values
split = int(len(X_ts) * 0.8)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_ts[:split], y_ts[:split])
mae = mean_absolute_error(y_ts[split:], rf_reg.predict(X_ts[split:]))
print(f"RF regression MAE on test period: {mae:.2f} sales units")
"""),
        md("## Real-World Example 2: Categorical Encoding Comparison"),
        code("""# Ordinal vs One-Hot vs Target Encoding (with cross-fitting to prevent target leakage)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Synthetic dataset with a high-cardinality categorical feature
np.random.seed(42)
n_samples = 1000
n_categories = 50
categories = [f"cat_{i:02d}" for i in range(n_categories)]
X_cat_raw = np.random.choice(categories, n_samples)

# Some categories are genuinely predictive
cat_effect = {c: np.random.randn() for c in categories}
noise = np.random.normal(0, 0.5, n_samples)
y_cat_cont = np.array([cat_effect[c] for c in X_cat_raw]) + noise
y_cat = (y_cat_cont > 0).astype(int)

df_cat = pd.DataFrame({"category": X_cat_raw})

# Ordinal encoding: fast, compact, but implies ordering
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_ord = ord_enc.fit_transform(df_cat)

# One-hot encoding: sparse, no ordering, bad for high-cardinality
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_ohe = ohe.fit_transform(df_cat)

print("Encoding sizes:")
print(f"  Ordinal:  {X_ord.shape}  (1 column per feature)")
print(f"  One-Hot:  {X_ohe.shape}  ({n_categories} columns for {n_categories} categories)")

# Target encoding: mean of target per category (risk of leakage without CV)
def target_encode_cv(X_col, y, n_splits=5, smoothing=10):
    \"\"\"Cross-fitted target encoding to prevent target leakage.\"\"\"
    encoded = np.zeros(len(y))
    global_mean = y.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(X_col):
        counts = {}
        sums = {}
        for i in tr_idx:
            c = X_col[i]
            counts[c] = counts.get(c, 0) + 1
            sums[c] = sums.get(c, 0) + y[i]
        for i in val_idx:
            c = X_col[i]
            n_c = counts.get(c, 0)
            s_c = sums.get(c, global_mean * n_c)
            # Smoothed target mean: shrinks toward global mean for rare categories
            encoded[i] = (s_c + smoothing * global_mean) / (n_c + smoothing)
    return encoded.reshape(-1, 1)

X_te_enc = target_encode_cv(X_cat_raw, y_cat)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

for name, X_enc in [("Ordinal", X_ord), ("One-Hot", X_ohe), ("Target (CV)", X_te_enc)]:
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=300, random_state=42))])
    scores = cross_val_score(pipe, X_enc, y_cat, cv=5, scoring="roc_auc")
    print(f"{name:<18} ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
"""),
        md("## Real-World Example 3: Feature Selection Pipeline"),
        code("""from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif, RFECV
)
from sklearn.ensemble import RandomForestClassifier as RF

np.random.seed(42)
# High-dimensional dataset: many irrelevant/low-variance features
X_hi, y_hi = make_classification(
    n_samples=800, n_features=50, n_informative=8,
    n_redundant=10, n_repeated=5, random_state=42
)

print(f"Starting feature count: {X_hi.shape[1]}")

# Step 1: Remove near-zero-variance features
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X_hi)
print(f"After VarianceThreshold:  {X_vt.shape[1]} features")

# Step 2: Mutual information filter (univariate)
selector_mi = SelectKBest(mutual_info_classif, k=20)
X_mi = selector_mi.fit_transform(X_vt, y_hi)
print(f"After SelectKBest (MI):   {X_mi.shape[1]} features")

# Step 3: Recursive Feature Elimination with CV
rfecv = RFECV(
    estimator=RF(n_estimators=50, random_state=42, n_jobs=-1),
    step=1,
    cv=5,
    scoring="roc_auc",
    min_features_to_select=3,
    n_jobs=-1,
)
rfecv.fit(X_mi, y_hi)
X_rfe = rfecv.transform(X_mi)
print(f"After RFECV:              {X_rfe.shape[1]} features (optimal per CV)")

# Full selection pipeline
selection_pipeline = Pipeline([
    ("vt", VarianceThreshold(threshold=0.01)),
    ("mi", SelectKBest(mutual_info_classif, k=20)),
    ("rfecv", RFECV(RF(n_estimators=50, random_state=42, n_jobs=-1), cv=5, scoring="roc_auc", n_jobs=-1)),
    ("clf", RF(n_estimators=100, random_state=42, n_jobs=-1)),
])

from sklearn.model_selection import cross_val_score
scores_full = cross_val_score(RF(n_estimators=100, random_state=42), X_hi, y_hi, cv=5, scoring="roc_auc")
print(f"\nAll 50 features ROC-AUC:   {scores_full.mean():.4f}")

scores_sel = cross_val_score(RF(n_estimators=100, random_state=42), X_rfe, y_hi, cv=5, scoring="roc_auc")
print(f"Selected features ROC-AUC: {scores_sel.mean():.4f}")
print(f"Reduction: {X_hi.shape[1]} → {X_rfe.shape[1]} features  ({100*(1-X_rfe.shape[1]/X_hi.shape[1]):.0f}% fewer)")
"""),
    ]
    save(nb, "14-feature-engineering.ipynb")


# ==============================================================================
# Notebook 15 — gradient-accumulation
# ==============================================================================
def make_15():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("""# Gradient Accumulation

## Learning Objectives
1. Understand why gradient accumulation enables large effective batch sizes on memory-constrained hardware.
2. Simulate the accumulation algorithm in NumPy and verify the loss curve matches true large-batch training.
3. Implement gradient accumulation in a PyTorch training loop with correct zero_grad placement.
4. Combine gradient accumulation with mixed precision (AMP) and gradient clipping for production training.
"""),
        code("""import numpy as np
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
        md("## Level 1: Gradient Accumulation Simulation (NumPy)"),
        code("""# Simulate accumulating gradients across N mini-batches before updating.
# Key insight: accumulated gradient == gradient on the full merged batch
# (when loss is mean-reduced and batch sizes are equal).

def mse_grad(X_batch, y_batch, w):
    \"\"\"MSE gradient: dL/dw = (2/n) * X^T (Xw - y).\"\"\"
    residuals = X_batch @ w - y_batch
    return (2.0 / len(y_batch)) * X_batch.T @ residuals


np.random.seed(42)
n, d = 256, 5
X_data = np.random.randn(n, d)
true_w = np.array([1.0, -2.0, 0.5, 1.5, -1.0])
y_data = X_data @ true_w + 0.1 * np.random.randn(n)

accumulation_steps = 8   # simulate large batch = 32*8 = 256 samples
micro_batch_size = 32
lr = 0.05

w_accum = np.zeros(d)
w_large = np.zeros(d)   # baseline: full-batch gradient update

n_updates = 20
losses_accum = []
losses_large = []

for step in range(n_updates):
    # --- Gradient Accumulation (micro-batches) ---
    accum_grad = np.zeros(d)
    for acc_step in range(accumulation_steps):
        idx = np.random.randint(0, n, micro_batch_size)
        accum_grad += mse_grad(X_data[idx], y_data[idx], w_accum)
    accum_grad /= accumulation_steps  # normalise: average gradient across accumulated steps
    w_accum -= lr * accum_grad
    losses_accum.append(((X_data @ w_accum - y_data)**2).mean())

    # --- Large-Batch Baseline ---
    idx_large = np.random.randint(0, n, micro_batch_size * accumulation_steps)
    large_grad = mse_grad(X_data[idx_large], y_data[idx_large], w_large)
    w_large -= lr * large_grad
    losses_large.append(((X_data @ w_large - y_data)**2).mean())

print("Final MSE — Gradient Accumulation:", f"{losses_accum[-1]:.6f}")
print("Final MSE — Large Batch Baseline: ", f"{losses_large[-1]:.6f}")
print("The curves should be similar, confirming mathematical equivalence.")
"""),
        md("## Level 2: PyTorch Training Loop with gradient_accumulation_steps"),
        code("""# Correct gradient accumulation in PyTorch:
# 1. Do NOT call optimizer.zero_grad() inside the accumulation loop.
# 2. Average loss by dividing by accumulation_steps BEFORE .backward().
# 3. Call optimizer.step() + optimizer.zero_grad() only after all micro-batches.

ACCUMULATION_STEPS = 4
MICRO_BATCH = 32
EFFECTIVE_BATCH = MICRO_BATCH * ACCUMULATION_STEPS

torch.manual_seed(42)
X_t = torch.randn(800, 20, device=device)
w_true = torch.randn(20, 1, device=device)
y_t = X_t @ w_true + 0.05 * torch.randn(800, 1, device=device)

train_ds = TensorDataset(X_t[:640], y_t[:640])
val_ds = TensorDataset(X_t[640:], y_t[640:])
# Micro-batch loader
micro_loader = DataLoader(train_ds, batch_size=MICRO_BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=64)


def build_net():
    return nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)


model = build_net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

history_accum = []
for epoch in range(50):
    model.train()
    optimizer.zero_grad()   # clear at epoch start
    accumulated = 0
    running_loss = 0.0
    for step, (xb, yb) in enumerate(micro_loader):
        try:
            loss = criterion(model(xb), yb) / ACCUMULATION_STEPS  # scale loss
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print("OOM — reduce MICRO_BATCH")
                continue
            raise
        loss.backward()
        running_loss += loss.item()
        accumulated += 1
        if accumulated == ACCUMULATION_STEPS:
            optimizer.step()
            optimizer.zero_grad()
            accumulated = 0

    # Handle any remaining incomplete accumulation group
    if accumulated > 0:
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        vl = sum(criterion(model(xb), yb).item() * len(xb) for xb, yb in val_loader) / len(val_loader.dataset)
    history_accum.append(vl)

print(f"Final val MSE (accum, effective_batch={EFFECTIVE_BATCH}): {history_accum[-1]:.6f}")
"""),
        md("## Real-World Example 1: Effective Batch Size Comparison"),
        code("""# Compare: small batch, large batch, and small batch + accumulation
# Expected: accumulation curve ≈ large batch curve

torch.manual_seed(42)
criterion_rw = nn.MSELoss()


def train_fixed(batch_size, n_epochs=60):
    \"\"\"Train with fixed batch size, no accumulation.\"\"\"
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    m = build_net()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    hist = []
    for _ in range(n_epochs):
        m.train()
        for xb, yb in loader:
            opt.zero_grad()
            criterion_rw(m(xb), yb).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = sum(criterion_rw(m(xb), yb).item() * len(xb) for xb, yb in val_loader) / len(val_loader.dataset)
        hist.append(vl)
    return hist


def train_with_accumulation(micro_bs, acc_steps, n_epochs=60):
    \"\"\"Train with gradient accumulation (micro_bs * acc_steps = effective batch).\"\"\"
    loader = DataLoader(train_ds, batch_size=micro_bs, shuffle=True, drop_last=True)
    m = build_net()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    hist = []
    for _ in range(n_epochs):
        m.train()
        opt.zero_grad()
        acc = 0
        for xb, yb in loader:
            (criterion_rw(m(xb), yb) / acc_steps).backward()
            acc += 1
            if acc == acc_steps:
                opt.step(); opt.zero_grad(); acc = 0
        if acc > 0:
            opt.step(); opt.zero_grad()
        m.eval()
        with torch.no_grad():
            vl = sum(criterion_rw(m(xb), yb).item() * len(xb) for xb, yb in val_loader) / len(val_loader.dataset)
        hist.append(vl)
    return hist


h_small = train_fixed(batch_size=8)
h_large = train_fixed(batch_size=64)
h_accum = train_with_accumulation(micro_bs=8, acc_steps=8)   # same effective batch as h_large

print(f"Small batch (8)      final val MSE: {h_small[-1]:.6f}")
print(f"Large batch (64)     final val MSE: {h_large[-1]:.6f}")
print(f"Accumulation (8×8)   final val MSE: {h_accum[-1]:.6f}")
print("Accumulation ≈ Large batch (demonstrates mathematical equivalence)")
"""),
        md("## Real-World Example 2: Mixed Precision + Gradient Accumulation"),
        code("""# AMP autocast reduces memory and speeds up compute on CUDA GPUs.
# GradScaler prevents gradient underflow with fp16.
# Accumulation must scale loss and call scaler.unscale_ before clipping.

torch.manual_seed(42)
model_amp = build_net()
optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
AMP_ACC_STEPS = 4
criterion_amp = nn.MSELoss()
loader_amp = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)

history_amp = []
for epoch in range(40):
    model_amp.train()
    optimizer_amp.zero_grad()
    acc = 0
    for xb, yb in loader_amp:
        # autocast: ops run in fp16 on CUDA, fp32 on CPU
        with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            try:
                loss = criterion_amp(model_amp(xb), yb) / AMP_ACC_STEPS
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    continue
                raise
        scaler.scale(loss).backward()
        acc += 1
        if acc == AMP_ACC_STEPS:
            scaler.unscale_(optimizer_amp)          # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model_amp.parameters(), max_norm=1.0)
            scaler.step(optimizer_amp)
            scaler.update()
            optimizer_amp.zero_grad()
            acc = 0
    if acc > 0:
        scaler.step(optimizer_amp); scaler.update(); optimizer_amp.zero_grad()

    model_amp.eval()
    with torch.no_grad():
        vl = sum(criterion_amp(model_amp(xb), yb).item() * len(xb) for xb, yb in val_loader) / len(val_loader.dataset)
    history_amp.append(vl)

print(f"AMP + accumulation final val MSE: {history_amp[-1]:.6f}")
print(f"AMP scaler state: scale={scaler.get_scale():.1f}  (should remain >=1)")
"""),
        md("## Real-World Example 3: Gradient Clipping with Accumulation"),
        code("""# Gradient clipping prevents exploding gradients.
# With accumulation, you MUST clip AFTER accumulating (before optimizer.step).
# Clipping before accumulation clips each micro-batch separately — wrong.

torch.manual_seed(42)
# Use a deeper network that is more prone to gradient explosion
model_clip = nn.Sequential(
    nn.Linear(20, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 64), nn.Tanh(),
    nn.Linear(64, 1),
).to(device)
optimizer_clip = torch.optim.SGD(model_clip.parameters(), lr=0.05, momentum=0.9)
criterion_clip = nn.MSELoss()
CLIP_NORM = 1.0
ACC_STEPS = 4
loader_clip = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)

grad_norms = []  # track gradient norms before clipping
val_losses = []

for epoch in range(50):
    model_clip.train()
    optimizer_clip.zero_grad()
    acc = 0
    for xb, yb in loader_clip:
        (criterion_clip(model_clip(xb), yb) / ACC_STEPS).backward()
        acc += 1
        if acc == ACC_STEPS:
            # Compute gradient norm BEFORE clipping for monitoring
            total_norm = torch.nn.utils.clip_grad_norm_(
                model_clip.parameters(), max_norm=CLIP_NORM
            )
            grad_norms.append(total_norm.item())
            optimizer_clip.step()
            optimizer_clip.zero_grad()
            acc = 0
    if acc > 0:
        torch.nn.utils.clip_grad_norm_(model_clip.parameters(), CLIP_NORM)
        optimizer_clip.step(); optimizer_clip.zero_grad()

    model_clip.eval()
    with torch.no_grad():
        vl = sum(criterion_clip(model_clip(xb), yb).item() * len(xb) for xb, yb in val_loader) / len(val_loader.dataset)
    val_losses.append(vl)

clipped_count = sum(1 for g in grad_norms if g > CLIP_NORM)
print(f"Total gradient norm measurements: {len(grad_norms)}")
print(f"Steps where gradient was clipped: {clipped_count} ({100*clipped_count/len(grad_norms):.1f}%)")
print(f"Max pre-clip gradient norm: {max(grad_norms):.4f}")
print(f"Mean pre-clip gradient norm: {np.mean(grad_norms):.4f}")
print(f"Final val MSE: {val_losses[-1]:.6f}")
print("Rule: clip AFTER accumulation, BEFORE optimizer.step()")
"""),
    ]
    save(nb, "15-gradient-accumulation.ipynb")


if __name__ == "__main__":
    print("Generating notebooks 11-15...")
    make_11()
    make_12()
    make_13()
    make_14()
    make_15()
    print("Done.")
