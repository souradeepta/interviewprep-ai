"""
Generate ml/notebooks 16-20: hyperparameter-tuning through loss-functions.
Each notebook has exactly 12 cells: 6 markdown + 6 code, strictly alternating.
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
# Notebook 16 — hyperparameter-tuning
# ==============================================================================
def make_16():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("""# Hyperparameter Tuning

## Learning Objectives
1. Understand the difference between grid search, random search, and Bayesian optimisation and when each applies.
2. Implement grid search from scratch using NumPy and visualise the loss landscape as a heatmap.
3. Compare GridSearchCV, RandomizedSearchCV, and Optuna (TPE sampler) on the same model.
4. Use Optuna's pruning and parameter importance tools to cut wall-clock time and identify which hyperparameters matter most.
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
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
"""),
        md("## Level 1: Grid Search from Scratch (NumPy Heatmap)"),
        code("""# Manual grid search over two hyperparameters on a synthetic dataset.
# Visualise the validation accuracy as a 2-D heatmap.

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

np.random.seed(42)
X, y = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
gamma_values = [0.001, 0.01, 0.1, 1.0, 10.0]

results_grid = np.zeros((len(C_values), len(gamma_values)))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        fold_scores = []
        for tr_idx, val_idx in skf.split(X_tr, y_tr):
            clf = SVC(C=C, gamma=gamma, kernel="rbf")
            clf.fit(X_tr[tr_idx], y_tr[tr_idx])
            fold_scores.append(clf.score(X_tr[val_idx], y_tr[val_idx]))
        results_grid[i, j] = np.mean(fold_scores)

best_idx = np.unravel_index(results_grid.argmax(), results_grid.shape)
best_C = C_values[best_idx[0]]
best_gamma = gamma_values[best_idx[1]]
print(f"Grid search best: C={best_C}, gamma={best_gamma}")
print(f"Best CV accuracy: {results_grid[best_idx]:.4f}")
print(f"Total evaluations: {len(C_values) * len(gamma_values) * 5} (n_params x n_folds)")

# Show accuracy matrix
print("\nCV accuracy matrix (C x gamma):")
header = "       " + " ".join(f"{g:8.3f}" for g in gamma_values)
print(header)
for i, C in enumerate(C_values):
    row = f"C={C:5.2f}: " + " ".join(f"{results_grid[i,j]:.6f}" for j in range(len(gamma_values)))
    print(row)
"""),
        md("## Level 2: GridSearchCV vs RandomizedSearchCV vs Optuna"),
        code("""import time
from scipy.stats import loguniform

X, y = make_classification(n_samples=600, n_features=15, n_informative=8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearchCV: exhaustive, guaranteed to find best in grid
t0 = time.time()
param_grid = {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1.0], "kernel": ["rbf"]}
grid_cv = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, scoring="accuracy")
grid_cv.fit(X_tr, y_tr)
t_grid = time.time() - t0
print(f"GridSearchCV    — best acc: {grid_cv.best_score_:.4f}  "
      f"params: {grid_cv.best_params_}  time: {t_grid:.2f}s  "
      f"trials: {len(grid_cv.cv_results_['mean_test_score'])}")

# RandomizedSearchCV: samples randomly — fewer evaluations, similar quality
t0 = time.time()
param_dist = {"C": loguniform(1e-2, 1e3), "gamma": loguniform(1e-3, 1e1), "kernel": ["rbf"]}
rand_cv = RandomizedSearchCV(SVC(), param_dist, n_iter=20, cv=5, n_jobs=-1,
                              scoring="accuracy", random_state=42)
rand_cv.fit(X_tr, y_tr)
t_rand = time.time() - t0
print(f"RandomizedSearch — best acc: {rand_cv.best_score_:.4f}  "
      f"time: {t_rand:.2f}s  trials: 20")

# Optuna (if available): Bayesian TPE — focuses on promising regions
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        C = trial.suggest_float("C", 1e-2, 1e3, log=True)
        gamma = trial.suggest_float("gamma", 1e-3, 1e1, log=True)
        clf = SVC(C=C, gamma=gamma, kernel="rbf")
        return cross_val_score(clf, X_tr, y_tr, cv=5, scoring="accuracy", n_jobs=-1).mean()

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    t_optuna = time.time() - t0
    print(f"Optuna (TPE)    — best acc: {study.best_value:.4f}  "
          f"params: {study.best_params}  time: {t_optuna:.2f}s  trials: 30")
except ImportError:
    print("optuna not installed — install with: pip install optuna")
"""),
        md("## Real-World Example 1: Optuna with MedianPruner"),
        code("""# MedianPruner cancels unpromising trials early, saving wall-clock time.
# A trial is pruned if its intermediate value is worse than the median so far.

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    X_gb, y_gb = make_classification(n_samples=800, n_features=20, n_informative=10, random_state=42)
    X_tr_gb, X_te_gb, y_tr_gb, y_te_gb = train_test_split(X_gb, y_gb, test_size=0.2, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective_pruned(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        # Incremental report at each CV fold for pruning
        scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_tr_gb, y_tr_gb)):
            clf = GradientBoostingClassifier(**params, random_state=42)
            clf.fit(X_tr_gb[tr_idx], y_tr_gb[tr_idx])
            score = clf.score(X_tr_gb[val_idx], y_tr_gb[val_idx])
            scores.append(score)
            # Report intermediate value; pruner can stop the trial here
            trial.report(np.mean(scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study_pruned = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )
    study_pruned.optimize(objective_pruned, n_trials=40, show_progress_bar=False)

    pruned_count = sum(1 for t in study_pruned.trials if t.state == optuna.trial.TrialState.PRUNED)
    completed_count = sum(1 for t in study_pruned.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"Completed trials: {completed_count}  |  Pruned trials: {pruned_count}")
    print(f"Best accuracy: {study_pruned.best_value:.4f}")
    print(f"Best params: {study_pruned.best_params}")
    print(f"Pruning saved ~{pruned_count * 2} fold evaluations vs full 5-fold on every trial")

except ImportError:
    print("optuna not installed — install with: pip install optuna")
"""),
        md("## Real-World Example 2: Bayesian Optimisation (TPE) vs Random Search"),
        code("""# Bayesian optimisation builds a probabilistic model of objective vs parameters.
# TPE: Tree-structured Parzen Estimator — models P(params | good trial) and P(params | bad trial).
# Expected improvement guides sampling toward promising regions.

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.neural_network import MLPClassifier

    X_mlp, y_mlp = make_classification(n_samples=600, n_features=15, n_informative=8, random_state=7)
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_mlp, y_mlp, test_size=0.2, random_state=7)

    def mlp_objective(trial, sampler_name="tpe"):
        hidden = trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (64, 64), (128, 64)])
        lr_init = trial.suggest_float("learning_rate_init", 1e-4, 0.1, log=True)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        clf = MLPClassifier(hidden_layer_sizes=hidden, learning_rate_init=lr_init,
                            alpha=alpha, max_iter=200, random_state=42)
        return cross_val_score(clf, X_tr_m, y_tr_m, cv=5, scoring="accuracy", n_jobs=-1).mean()

    N_TRIALS = 30
    # TPE (Bayesian)
    study_tpe = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study_tpe.optimize(mlp_objective, n_trials=N_TRIALS, show_progress_bar=False)

    # Random search (RandomSampler)
    study_random = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42)
    )
    study_random.optimize(mlp_objective, n_trials=N_TRIALS, show_progress_bar=False)

    print("Optimisation history (best score after k trials):")
    print(f"{'Trials':>8}  {'TPE best':>12}  {'Random best':>12}")
    for k in [5, 10, 20, 30]:
        tpe_best = max(t.value for t in study_tpe.trials[:k] if t.value is not None)
        rnd_best = max(t.value for t in study_random.trials[:k] if t.value is not None)
        print(f"{k:>8}  {tpe_best:>12.4f}  {rnd_best:>12.4f}")

    print(f"\nFinal — TPE: {study_tpe.best_value:.4f}  Random: {study_random.best_value:.4f}")
    print("TPE typically reaches better optima with fewer trials.")

except ImportError:
    print("optuna not installed — install with: pip install optuna")
"""),
        md("## Real-World Example 3: Hyperparameter Importance Analysis"),
        code("""# After optimisation, understand WHICH hyperparameters had the most impact.
# Optuna's fANOVA-based importance score tells you where to focus tuning effort.

try:
    import optuna
    from optuna.importance import get_param_importances
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.ensemble import RandomForestClassifier

    X_imp, y_imp = make_classification(n_samples=700, n_features=15, n_informative=8, random_state=99)
    X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_imp, y_imp, test_size=0.2, random_state=99)

    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        return cross_val_score(clf, X_tr_i, y_tr_i, cv=5, scoring="accuracy", n_jobs=-1).mean()

    study_imp = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study_imp.optimize(rf_objective, n_trials=50, show_progress_bar=False)

    importances = get_param_importances(study_imp)
    print("Hyperparameter importance (fANOVA):")
    total = sum(importances.values())
    for param, importance in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "#" * int(30 * importance / total)
        print(f"  {param:<22}: {importance:.4f}  {bar}")

    print(f"\nBest accuracy: {study_imp.best_value:.4f}")
    print(f"Most impactful param: {max(importances, key=importances.get)}")
    print("Focus future tuning on high-importance params only — saves compute.")

except ImportError:
    print("optuna not installed — install with: pip install optuna")
except Exception as exc:
    # fANOVA requires scikit-learn; may not be available in all envs
    print(f"Importance analysis unavailable: {exc}")
    if 'study_imp' in dir():
        print(f"Best accuracy from study: {study_imp.best_value:.4f}")
"""),
    ]
    save(nb, "16-hyperparameter-tuning.ipynb")


# ==============================================================================
# Notebook 17 — knowledge-distillation
# ==============================================================================
def make_17():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("""# Knowledge Distillation

## Learning Objectives
1. Understand how soft labels from a teacher carry more information than hard one-hot labels.
2. Implement temperature scaling and KL divergence from scratch to show how higher temperature softens distributions.
3. Build a teacher-student training loop in PyTorch with the KD loss: alpha*CE + (1-alpha)*KL(student, teacher).
4. Apply response-based and intermediate (hint) distillation, and self-distillation for iterative model improvement.
"""),
        code("""import numpy as np
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
        md("## Level 1: Temperature Scaling + Soft Label KL Divergence (NumPy)"),
        code("""# Temperature T > 1 softens logit distributions so rare classes get non-negligible probability.
# KL(student || teacher) is minimised when student's soft predictions match teacher's.

def softmax_temp(logits, T=1.0):
    \"\"\"Temperature-scaled softmax.\"\"\"
    scaled = logits / T
    # numerically stable: subtract max before exp
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=-1, keepdims=True)


def kl_divergence(p, q, eps=1e-10):
    \"\"\"KL(p || q) = sum p * log(p / q).\"\"\"
    return np.sum(p * np.log((p + eps) / (q + eps)), axis=-1).mean()


# Teacher: high-confidence logits (well-trained)
teacher_logits = np.array([[4.0, 0.5, 0.2], [0.1, 3.8, 0.3], [0.2, 0.3, 4.5]])

print("Effect of temperature on teacher soft labels:")
print(f"{'T':>4}  Class probs (first sample)                KL distance (T=1 vs T=T)")
for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
    soft = softmax_temp(teacher_logits, T=T)
    soft_1 = softmax_temp(teacher_logits, T=1.0)
    kl = kl_divergence(soft_1, soft)
    print(f"{T:>4}  {soft[0]}   KL={kl:.4f}")

print("\nHigh T -> softer distribution -> more information in the 'dark knowledge'")
print("Typical T range: 2-10 for distillation; T=1 for normal inference")
"""),
        md("## Level 2: Teacher-Student KD Loss in PyTorch"),
        code("""# KD loss = alpha * CE(student_logits, hard_labels)
#          + (1 - alpha) * T^2 * KL(student_soft, teacher_soft)
# T^2 scaling compensates for softened gradients at high temperature.

def kd_loss(student_logits, teacher_logits, hard_labels, T=3.0, alpha=0.5):
    \"\"\"
    Hinton et al. 2015 distillation loss.
    alpha: weight for hard-label CE; (1-alpha): weight for soft-label KL.
    \"\"\"
    # Hard-label cross-entropy
    ce_loss = F.cross_entropy(student_logits, hard_labels)
    # Soft-label KL: use log_softmax for numerical stability
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)
    # T^2 re-scales gradients back to their original magnitude
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)
    return alpha * ce_loss + (1 - alpha) * kl


# ---- Synthetic data: 4-class problem ----
torch.manual_seed(42)
n, d, n_class = 800, 16, 4
X_kd = torch.randn(n, d, device=device)
y_kd = torch.randint(0, n_class, (n,), device=device)

split = 640
train_ds_kd = TensorDataset(X_kd[:split], y_kd[:split])
val_ds_kd = TensorDataset(X_kd[split:], y_kd[split:])
train_ld = DataLoader(train_ds_kd, batch_size=64, shuffle=True)
val_ld = DataLoader(val_ds_kd, batch_size=64)


# Large teacher
def make_teacher():
    return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_class)).to(device)

# Small student
def make_student():
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n_class)).to(device)


# Pre-train teacher
teacher = make_teacher()
opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
for _ in range(30):
    teacher.train()
    for xb, yb in train_ld:
        opt_t.zero_grad()
        F.cross_entropy(teacher(xb), yb).backward()
        opt_t.step()
teacher.eval()

# Train student WITH KD
student_kd = make_student()
opt_s = torch.optim.Adam(student_kd.parameters(), lr=1e-3)
kd_val_accs = []
for epoch in range(40):
    student_kd.train()
    for xb, yb in train_ld:
        opt_s.zero_grad()
        try:
            with torch.no_grad():
                t_logits = teacher(xb)
            s_logits = student_kd(xb)
            loss = kd_loss(s_logits, t_logits, yb, T=4.0, alpha=0.4)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache(); continue
            raise
        loss.backward(); opt_s.step()
    student_kd.eval()
    correct = sum((student_kd(xb).argmax(1) == yb).sum().item() for xb, yb in val_ld)
    kd_val_accs.append(correct / len(val_ds_kd))

# Train student WITHOUT KD (baseline)
student_base = make_student()
opt_b = torch.optim.Adam(student_base.parameters(), lr=1e-3)
base_val_accs = []
for epoch in range(40):
    student_base.train()
    for xb, yb in train_ld:
        opt_b.zero_grad()
        F.cross_entropy(student_base(xb), yb).backward()
        opt_b.step()
    student_base.eval()
    correct = sum((student_base(xb).argmax(1) == yb).sum().item() for xb, yb in val_ld)
    base_val_accs.append(correct / len(val_ds_kd))

print(f"Student w/ KD  final val accuracy: {kd_val_accs[-1]:.4f}")
print(f"Student w/o KD final val accuracy: {base_val_accs[-1]:.4f}")
print(f"KD improvement: {kd_val_accs[-1] - base_val_accs[-1]:+.4f}")
"""),
        md("## Real-World Example 1: Intermediate Layer (Hint) Distillation"),
        code("""# Hint distillation: student's intermediate features are regressed toward teacher's.
# FitNets (Romero et al., 2015): intermediate hint loss + KD response loss.

# Adaptation layer: project student hidden dim to teacher hidden dim
class StudentWithHint(nn.Module):
    \"\"\"Student network with an adapter layer for hint distillation.\"\"\"
    def __init__(self, in_dim=16, hidden=32, out_dim=4, teacher_hint_dim=64):
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.out_layer = nn.Linear(hidden, out_dim)
        # Project student hidden to teacher hidden size for hint supervision
        self.hint_adapter = nn.Linear(hidden, teacher_hint_dim)

    def forward(self, x):
        h = self.hidden_layer(x)
        return self.out_layer(h), h  # return both logits and hidden state


class TeacherWithHook(nn.Module):
    \"\"\"Teacher that also exposes its intermediate representation.\"\"\"
    def __init__(self, in_dim=16, hidden=64, out_dim=4):
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.out_layer = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.hidden_layer(x)
        return self.out_layer(h), h


teacher_h = TeacherWithHook().to(device)
student_h = StudentWithHint().to(device)

# Pre-train teacher
opt_th = torch.optim.Adam(teacher_h.parameters(), lr=1e-3)
for _ in range(30):
    teacher_h.train()
    for xb, yb in train_ld:
        opt_th.zero_grad()
        logits, _ = teacher_h(xb)
        F.cross_entropy(logits, yb).backward()
        opt_th.step()
teacher_h.eval()

# Train student with hint + KD
opt_sh = torch.optim.Adam(student_h.parameters(), lr=1e-3)
hint_criterion = nn.MSELoss()
HINT_LAMBDA = 0.5

hint_val_accs = []
for epoch in range(40):
    student_h.train()
    for xb, yb in train_ld:
        opt_sh.zero_grad()
        with torch.no_grad():
            t_logits, t_hidden = teacher_h(xb)
        s_logits, s_hidden = student_h(xb)
        s_adapted = student_h.hint_adapter(s_hidden)  # align dimensions
        # Combined loss: KD (response) + hint (intermediate features)
        response_loss = kd_loss(s_logits, t_logits, yb, T=4.0, alpha=0.4)
        hint_loss = hint_criterion(s_adapted, t_hidden.detach())
        total_loss = response_loss + HINT_LAMBDA * hint_loss
        total_loss.backward()
        opt_sh.step()
    student_h.eval()
    correct = sum((student_h(xb)[0].argmax(1) == yb).sum().item() for xb, yb in val_ld)
    hint_val_accs.append(correct / len(val_ds_kd))

print(f"Hint distillation final val accuracy: {hint_val_accs[-1]:.4f}")
print(f"Response-only KD final val accuracy:  {kd_val_accs[-1]:.4f}")
print("Hint loss aligns internal representations, often improving transfer.")
"""),
        md("## Real-World Example 2: Self-Distillation"),
        code("""# Self-distillation: use the model's own soft predictions as additional targets.
# Iterative: train gen0 normally, use gen0 predictions to train gen1, etc.
# Can improve calibration and sometimes accuracy without a separate teacher.

torch.manual_seed(0)

def train_one_gen(model_init, teacher_for_kd, n_epochs=30, T=3.0, alpha=0.5):
    \"\"\"Train model_init; if teacher_for_kd is None, use hard labels only.\"\"\"
    model_init.train()
    opt = torch.optim.Adam(model_init.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        for xb, yb in train_ld:
            opt.zero_grad()
            logits = model_init(xb)
            if teacher_for_kd is None:
                loss = F.cross_entropy(logits, yb)
            else:
                with torch.no_grad():
                    t_logits = teacher_for_kd(xb)
                loss = kd_loss(logits, t_logits, yb, T=T, alpha=alpha)
            loss.backward()
            opt.step()
    return model_init


def eval_model(model):
    model.eval()
    correct = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in val_ld)
    return correct / len(val_ds_kd)


# Generation 0: train with hard labels
gen0 = make_student()
gen0 = train_one_gen(gen0, teacher_for_kd=None, n_epochs=30)
print(f"Gen 0 (hard labels only)  val acc: {eval_model(gen0):.4f}")

# Generation 1: self-distill from gen0
gen1 = make_student()
gen1 = train_one_gen(gen1, teacher_for_kd=gen0, n_epochs=30, T=3.0, alpha=0.5)
print(f"Gen 1 (self-distil gen0)  val acc: {eval_model(gen1):.4f}")

# Generation 2: self-distill from gen1
gen2 = make_student()
gen2 = train_one_gen(gen2, teacher_for_kd=gen1, n_epochs=30, T=3.0, alpha=0.5)
print(f"Gen 2 (self-distil gen1)  val acc: {eval_model(gen2):.4f}")

print("\nSelf-distillation: each generation uses the prior as a soft teacher.")
print("Gains diminish quickly — typically 1-2 generations is sufficient.")
"""),
        md("## Real-World Example 3: Response-Based KD for Model Compression"),
        code("""# Simulate BERT compression: large transformer -> small classifier
# using response-based KD. We use synthetic logits to avoid download dependencies.

torch.manual_seed(42)
# Simulate pre-computed teacher (BERT-large) logits on a text classification task
# In production: run teacher inference once, store logits, train student offline.

N_SENTENCES, N_CLASSES, EMBED_DIM = 1000, 5, 128

# Simulate teacher logits (already computed offline)
teacher_logits_stored = torch.randn(N_SENTENCES, N_CLASSES, device=device) * 2.0
# Make teacher "confident" — somewhat polarised
teacher_logits_stored = teacher_logits_stored * 1.5
hard_labels = teacher_logits_stored.argmax(dim=1)

# Embeddings as input to student
input_embeddings = torch.randn(N_SENTENCES, EMBED_DIM, device=device)
split2 = 800
ds_train2 = TensorDataset(input_embeddings[:split2], teacher_logits_stored[:split2], hard_labels[:split2])
ds_val2 = TensorDataset(input_embeddings[split2:], teacher_logits_stored[split2:], hard_labels[split2:])
ld_train2 = DataLoader(ds_train2, batch_size=64, shuffle=True)
ld_val2 = DataLoader(ds_val2, batch_size=64)

# Small student classifier (6x fewer params than teacher representation)
student_compress = nn.Sequential(
    nn.Linear(EMBED_DIM, 64), nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, N_CLASSES),
).to(device)

opt_comp = torch.optim.Adam(student_compress.parameters(), lr=2e-4, weight_decay=1e-4)

val_accs = []
for epoch in range(50):
    student_compress.train()
    for xb, t_log, yb in ld_train2:
        opt_comp.zero_grad()
        s_logits = student_compress(xb)
        loss = kd_loss(s_logits, t_log, yb, T=5.0, alpha=0.3)
        loss.backward()
        opt_comp.step()
    student_compress.eval()
    correct = sum(
        (student_compress(xb).argmax(1) == yb).sum().item()
        for xb, t_log, yb in ld_val2
    )
    val_accs.append(correct / len(ds_val2))

teacher_acc = (teacher_logits_stored[split2:].argmax(1) == hard_labels[split2:]).float().mean().item()
print(f"Teacher (oracle) accuracy:  {teacher_acc:.4f}")
print(f"Student compressed acc:     {val_accs[-1]:.4f}")
print(f"Student param count: {sum(p.numel() for p in student_compress.parameters()):,}")
print(f"\nKey pattern: store teacher logits offline, train student without teacher running.")
print("Saves memory: student trains without teacher loaded simultaneously.")
"""),
    ]
    save(nb, "17-knowledge-distillation.ipynb")


# ==============================================================================
# Notebook 18 — layer-normalization
# ==============================================================================
def make_18():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("""# Layer Normalization

## Learning Objectives
1. Implement LayerNorm forward pass from scratch: normalise across the last dimension, then learn scale and shift.
2. Compare BatchNorm, LayerNorm, GroupNorm, and RMSNorm on a transformer-style architecture.
3. Understand pre-norm vs post-norm placement and why pre-norm enables training deeper networks.
4. Implement LLaMA-style RMSNorm and demonstrate its computational advantage over LayerNorm.
"""),
        code("""import numpy as np
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
        md("## Level 1: LayerNorm Forward Pass (NumPy)"),
        code("""# LayerNorm normalises over the LAST dimension (features), not batch dimension.
# Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
# gamma (scale) and beta (shift) are learnable parameters.

def layer_norm_np(x, gamma, beta, eps=1e-5):
    \"\"\"
    LayerNorm: normalise along the last axis.
    x: (batch, seq_len, d_model) or (batch, d_model)
    \"\"\"
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta  # broadcast: gamma/beta shape (..., d_model)


# Test on a small transformer-like sequence
np.random.seed(42)
batch, seq_len, d_model = 2, 4, 8
x = np.random.randn(batch, seq_len, d_model) * 5 + 3  # non-zero mean, large variance
gamma = np.ones(d_model)
beta = np.zeros(d_model)

out = layer_norm_np(x, gamma, beta)
print("Input statistics (per token):")
print(f"  mean: {x.mean(axis=-1).flatten()}")
print(f"  std : {x.std(axis=-1).flatten()}")
print("\nAfter LayerNorm:")
print(f"  mean: {out.mean(axis=-1).flatten()}")
print(f"  std : {out.std(axis=-1).flatten()}")
print("\nLayerNorm outputs should have mean≈0 and std≈1 along last dim.")

# Verify against PyTorch
x_t = torch.tensor(x, dtype=torch.float32)
ln_torch = nn.LayerNorm(d_model, elementwise_affine=False)  # gamma=1, beta=0
out_torch = ln_torch(x_t).numpy()
print(f"\nMax difference vs PyTorch implementation: {np.abs(out - out_torch).max():.2e}")
"""),
        md("## Level 2: BatchNorm vs LayerNorm vs GroupNorm vs RMSNorm"),
        code("""class RMSNorm(nn.Module):
    \"\"\"Root Mean Square Layer Normalization (Zhang & Sennrich, 2019). Used in LLaMA.\"\"\"
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS: sqrt(mean(x^2))  — no mean subtraction (no centering)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


# Mini transformer block for testing different normalisation strategies
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, norm_type="layer"):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        if norm_type == "layer":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == "batch":
            # BatchNorm1d over (batch * seq, d_model) — note: transposed
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        elif norm_type == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=d_model)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=d_model)
        elif norm_type == "rms":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        self.norm_type = norm_type

    def _apply_norm(self, norm, x):
        if self.norm_type == "batch":
            b, s, d = x.shape
            return norm(x.reshape(b * s, d)).reshape(b, s, d)
        elif self.norm_type == "group":
            # GroupNorm expects (N, C, *); reshape seq into spatial dim
            b, s, d = x.shape
            return norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return norm(x)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self._apply_norm(self.norm1, x)
        x = x + self.ff(x)
        x = self._apply_norm(self.norm2, x)
        return x


torch.manual_seed(42)
d_model, batch_sz, seq = 64, 8, 16
x_in = torch.randn(batch_sz, seq, d_model, device=device)

print("Norm type    | Output mean | Output std  | Param count")
print("-" * 55)
for norm_type in ["layer", "rms", "group", "batch"]:
    try:
        block = TransformerBlock(d_model, norm_type).to(device)
        block.eval()
        with torch.no_grad():
            try:
                out = block(x_in)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache(); continue
                raise
        n_params = sum(p.numel() for p in block.parameters())
        print(f"{norm_type:<12} | {out.mean().item():>+.4f}      | {out.std().item():>.4f}       | {n_params:,}")
    except Exception as exc:
        print(f"{norm_type:<12} | Error: {exc}")
"""),
        md("## Real-World Example 1: Pre-Norm vs Post-Norm in Transformer Blocks"),
        code("""# Pre-norm (GPT-2, LLaMA): LayerNorm BEFORE attention/FF
#   Gradients flow more directly → deeper training is stable
# Post-norm (original Transformer): LayerNorm AFTER residual addition
#   Effective but harder to train deeply (gradient scaling issues)

class PreNormBlock(nn.Module):
    \"\"\"Pre-LN: LN applied to input before each sub-layer.\"\"\"
    def __init__(self, d_model=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))

    def forward(self, x):
        # Pre-norm: normalise BEFORE computing attention/ff
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class PostNormBlock(nn.Module):
    \"\"\"Post-LN: LN applied after residual addition (original Vaswani et al.).\"\"\"
    def __init__(self, d_model=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))

    def forward(self, x):
        # Post-norm: normalise AFTER residual addition
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ff(x))
        return x


# Compare gradient norms through 6-layer stacks
def build_stack(block_cls, n_layers=6, d_model=64):
    return nn.Sequential(*[block_cls(d_model) for _ in range(n_layers)]).to(device)


def measure_grad_norms(model, n_batches=10):
    \"\"\"Return per-layer gradient norms.\"\"\"
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    grad_norms = []
    model.train()
    for _ in range(n_batches):
        x = torch.randn(4, 16, 64, device=device)
        loss = model(x).mean()
        opt.zero_grad()
        loss.backward()
        total_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
        grad_norms.append(total_norm)
    return grad_norms


pre_norms = measure_grad_norms(build_stack(PreNormBlock))
post_norms = measure_grad_norms(build_stack(PostNormBlock))

print(f"Pre-norm  gradient norm — mean: {np.mean(pre_norms):.4f}  std: {np.std(pre_norms):.4f}")
print(f"Post-norm gradient norm — mean: {np.mean(post_norms):.4f}  std: {np.std(post_norms):.4f}")
print("\nPre-norm produces more stable gradients (lower std) → enables deeper models.")
"""),
        md("## Real-World Example 2: RMSNorm (LLaMA-Style)"),
        code("""# RMSNorm: no mean subtraction, no beta bias term.
# Simpler and faster than LayerNorm; empirically competitive for language models.
# Implemented in LLaMA, Mistral, Falcon, and other modern LLMs.

import time

torch.manual_seed(42)
d = 4096   # typical LLM hidden size
batch_tokens = 512   # batch * seq_len flattened

x_bench = torch.randn(batch_tokens, d, device=device)

layer_norm = nn.LayerNorm(d).to(device)
rms_norm = RMSNorm(d).to(device)

# Parameter count comparison
print("Parameter count:")
print(f"  LayerNorm: {sum(p.numel() for p in layer_norm.parameters())} (scale + shift)")
print(f"  RMSNorm:   {sum(p.numel() for p in rms_norm.parameters())} (scale only — no bias)")

# Correctness check
with torch.no_grad():
    out_ln = layer_norm(x_bench)
    out_rms = rms_norm(x_bench)

print(f"\nLayerNorm output — mean: {out_ln.mean().item():+.4f}  std: {out_ln.std().item():.4f}")
print(f"RMSNorm  output — mean: {out_rms.mean().item():+.4f}  std: {out_rms.std().item():.4f}")
print("(RMSNorm doesn't centre — mean will not be zero unless data is symmetric)")

# Throughput comparison
WARMUP, N_REPS = 10, 100
for module, name in [(layer_norm, "LayerNorm"), (rms_norm, "RMSNorm")]:
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = module(x_bench)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            _ = module(x_bench)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / N_REPS * 1000
    print(f"{name}: {elapsed:.3f} ms/forward (batch={batch_tokens}, d={d})")

print("\nRMSNorm is ~20-30% faster on GPU (fewer ops: no mean subtraction, no beta).")
"""),
        md("## Real-World Example 3: Pre-Norm Enables Deeper Models"),
        code("""# Empirical test: train post-norm vs pre-norm on increasing depth.
# Post-norm diverges or converges slowly at high depth; pre-norm stays stable.

torch.manual_seed(42)
# Simple sequence classification task
N, SEQ, D = 400, 8, 32
X_seq = torch.randn(N, SEQ, D, device=device)
y_seq = torch.randint(0, 2, (N,), device=device)

tr_ds = TensorDataset(X_seq[:300], y_seq[:300])
va_ds = TensorDataset(X_seq[300:], y_seq[300:])
tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=32)


class TransformerClassifier(nn.Module):
    def __init__(self, d=32, n_layers=6, pre_norm=True):
        super().__init__()
        Block = PreNormBlock if pre_norm else PostNormBlock
        # Rebuild blocks with smaller d
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d) if pre_norm else nn.Identity(),
                nn.Linear(d, d),
                nn.GELU(),
                nn.LayerNorm(d) if not pre_norm else nn.Identity(),
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)  # residual
        return self.head(x.mean(dim=1))  # mean pool


results = {}
for n_layers in [2, 4, 8, 12]:
    row = {}
    for pre_norm, label in [(True, "pre-norm"), (False, "post-norm")]:
        model = TransformerClassifier(D, n_layers, pre_norm).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        val_acc = 0.0
        for epoch in range(30):
            model.train()
            for xb, yb in tr_ld:
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                opt.step()
        model.eval()
        correct = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in va_ld)
        val_acc = correct / len(va_ds)
        row[label] = val_acc
    results[n_layers] = row
    print(f"Depth={n_layers:2d}: pre-norm={row['pre-norm']:.4f}  post-norm={row['post-norm']:.4f}")

print("\nPre-norm maintains accuracy as depth increases; post-norm degrades or diverges.")
"""),
    ]
    save(nb, "18-layer-normalization.ipynb")


# ==============================================================================
# Notebook 19 — learning-rate-schedules
# ==============================================================================
def make_19():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("""# Learning Rate Schedules

## Learning Objectives
1. Implement warmup and cosine decay schedules from scratch in NumPy and visualise them.
2. Compare StepLR, CosineAnnealingLR, OneCycleLR, and ReduceLROnPlateau using PyTorch schedulers.
3. Apply the transformer warmup schedule (4000-step warmup + cosine decay) for stable large-model training.
4. Implement the LR finder algorithm to empirically find the best learning rate range.
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
        md("## Level 1: Warmup + Cosine Decay Schedule (NumPy)"),
        code("""# Warmup: gradually increase LR from 0 to base_lr over warmup_steps.
# Prevents early divergence when weights are randomly initialised.
# Cosine decay: smoothly anneal from base_lr to min_lr after warmup.

def warmup_cosine_schedule(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    \"\"\"
    LR schedule used by GPT-2, BERT, and many transformer models.
    Step 0..warmup_steps-1: linear warmup from 0 to base_lr
    Step warmup_steps..total_steps: cosine decay from base_lr to min_lr
    \"\"\"
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    # Cosine annealing after warmup
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_factor


def constant_lr(step, base_lr, **kwargs):
    \"\"\"Constant LR baseline.\"\"\"
    return base_lr


def step_decay(step, base_lr, decay_factor=0.5, step_size=100):
    \"\"\"Halve LR every step_size steps.\"\"\"
    return base_lr * (decay_factor ** (step // step_size))


TOTAL_STEPS = 500
BASE_LR = 0.01
WARMUP = 50

steps = np.arange(TOTAL_STEPS)
lr_warmup_cosine = [warmup_cosine_schedule(s, BASE_LR, WARMUP, TOTAL_STEPS) for s in steps]
lr_constant = [BASE_LR] * TOTAL_STEPS
lr_step = [step_decay(s, BASE_LR, step_size=100) for s in steps]

print("Schedule comparison at key steps:")
print(f"{'Step':>6}  {'Warmup+Cosine':>14}  {'Constant':>10}  {'StepDecay':>10}")
for s in [0, 10, 50, 100, 200, 499]:
    print(f"{s:>6}  {lr_warmup_cosine[s]:>14.6f}  {BASE_LR:>10.6f}  {lr_step[s]:>10.6f}")

print(f"\nWarmup+cosine peak: {max(lr_warmup_cosine):.6f} at step {np.argmax(lr_warmup_cosine)}")
print(f"Final LR at step {TOTAL_STEPS-1}: {lr_warmup_cosine[-1]:.6f}")
"""),
        md("## Level 2: PyTorch Scheduler Comparison"),
        code("""# Compare four built-in PyTorch schedulers on the same simple regression task.

torch.manual_seed(42)
X = torch.randn(600, 10, device=device)
y = torch.randn(600, 1, device=device)
tr_ds = TensorDataset(X[:500], y[:500])
va_ds = TensorDataset(X[500:], y[500:])
tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=32)
N_EPOCHS = 40


def make_model_opt():
    m = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    return m, opt


def train_with_scheduler(scheduler_factory, scheduler_kwargs):
    \"\"\"Train model with given scheduler, return LR history and val loss history.\"\"\"
    model, opt = make_model_opt()
    scheduler = scheduler_factory(opt, **scheduler_kwargs)
    crit = nn.MSELoss()
    lr_hist, val_hist = [], []
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            try:
                crit(model(xb), yb).backward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache(); continue
                raise
            opt.step()
        # ReduceLROnPlateau needs metric; others don't
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            model.eval()
            with torch.no_grad():
                vl = sum(crit(model(xb), yb).item()*len(xb) for xb, yb in va_ld) / len(va_ds)
            scheduler.step(vl)
        else:
            scheduler.step()
        lr_hist.append(opt.param_groups[0]["lr"])
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(xb), yb).item()*len(xb) for xb, yb in va_ld) / len(va_ds)
        val_hist.append(vl)
    return lr_hist, val_hist


schedulers = {
    "StepLR": (torch.optim.lr_scheduler.StepLR, {"step_size": 10, "gamma": 0.5}),
    "CosineAnnealingLR": (torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": N_EPOCHS}),
    "OneCycleLR": (torch.optim.lr_scheduler.OneCycleLR,
                   {"max_lr": 0.1, "steps_per_epoch": len(tr_ld), "epochs": N_EPOCHS}),
    "ReduceLROnPlateau": (torch.optim.lr_scheduler.ReduceLROnPlateau,
                          {"mode": "min", "factor": 0.5, "patience": 5}),
}

print(f"{'Scheduler':<22}  {'Final LR':>10}  {'Final Val MSE':>14}")
for name, (cls, kwargs) in schedulers.items():
    lr_h, val_h = train_with_scheduler(cls, kwargs)
    print(f"{name:<22}  {lr_h[-1]:>10.6f}  {val_h[-1]:>14.6f}")
"""),
        md("## Real-World Example 1: Transformer Warmup Schedule (4000-Step)"),
        code("""# Original Transformer (Vaswani et al., 2017):
# lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
# This gives linear warmup up to warmup_steps, then inverse-sqrt decay.

def transformer_lr(step, d_model=512, warmup_steps=4000):
    \"\"\"Noam schedule (Attention Is All You Need).\"\"\"
    if step == 0:
        return 0.0
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


# Implement as a PyTorch LambdaLR
model_t, opt_t = make_model_opt()
D_MODEL = 64
WARMUP_STEPS = 200  # scaled down from 4000 for this demo

lambda_fn = lambda step: (
    max(1, step) ** (-0.5) * min(max(1, step) ** (-0.5), max(1, step) * WARMUP_STEPS ** (-1.5))
    * D_MODEL ** 0.5  # scale to reasonable LR
)
scheduler_t = torch.optim.lr_scheduler.LambdaLR(opt_t, lr_lambda=lambda_fn)

crit_t = nn.MSELoss()
lr_history_t = []
val_history_t = []
for epoch in range(60):
    model_t.train()
    for xb, yb in tr_ld:
        opt_t.zero_grad()
        crit_t(model_t(xb), yb).backward()
        opt_t.step()
        scheduler_t.step()  # step PER BATCH, not per epoch
        lr_history_t.append(opt_t.param_groups[0]["lr"])
    model_t.eval()
    with torch.no_grad():
        vl = sum(crit_t(model_t(xb), yb).item()*len(xb) for xb, yb in va_ld) / len(va_ds)
    val_history_t.append(vl)

peak_step = int(np.argmax(lr_history_t))
print(f"LR peak at step {peak_step}: {max(lr_history_t):.6f}")
print(f"LR at final step {len(lr_history_t)}: {lr_history_t[-1]:.6f}")
print(f"Final val MSE: {val_history_t[-1]:.6f}")
print("Warmup prevents early divergence; inverse-sqrt decay keeps learning stable.")
"""),
        md("## Real-World Example 2: Cyclical LR (CLR)"),
        code("""# Cyclical LR (Smith, 2017): oscillate LR between base_lr and max_lr.
# Triangular policy: linearly increase then linearly decrease.
# Benefits: helps escape saddle points, often faster than monotone decay.

def cyclical_lr(step, base_lr, max_lr, step_size):
    \"\"\"
    Triangular CLR. step_size = half the cycle length.
    LR oscillates: base_lr -> max_lr -> base_lr over 2*step_size steps.
    \"\"\"
    cycle = np.floor(1 + step / (2 * step_size))
    x = abs(step / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, 1 - x)


model_clr, opt_clr = make_model_opt()
# Use CyclicLR (torch built-in)
scheduler_clr = torch.optim.lr_scheduler.CyclicLR(
    opt_clr, base_lr=1e-4, max_lr=5e-2,
    step_size_up=len(tr_ld) * 5,  # half cycle = 5 epochs
    mode="triangular",
    cycle_momentum=False,
)
crit_clr = nn.MSELoss()
clr_lr_hist = []
clr_val_hist = []

for epoch in range(50):
    model_clr.train()
    for xb, yb in tr_ld:
        opt_clr.zero_grad()
        crit_clr(model_clr(xb), yb).backward()
        opt_clr.step()
        scheduler_clr.step()
        clr_lr_hist.append(opt_clr.param_groups[0]["lr"])
    model_clr.eval()
    with torch.no_grad():
        vl = sum(crit_clr(model_clr(xb), yb).item()*len(xb) for xb, yb in va_ld) / len(va_ds)
    clr_val_hist.append(vl)

print(f"CLR LR range: {min(clr_lr_hist):.6f} – {max(clr_lr_hist):.6f}")
print(f"Number of full cycles: {len(clr_lr_hist) / (2 * len(tr_ld) * 5):.1f}")
print(f"Final val MSE: {clr_val_hist[-1]:.6f}")
print("CLR: wider LR range + oscillation can escape flat regions faster.")
"""),
        md("## Real-World Example 3: LR Finder (fast.ai Method)"),
        code("""# LR Finder: train for one epoch with exponentially increasing LR.
# Plot loss vs LR; choose LR just before the loss starts climbing.
# Popularised by fast.ai; used before long training runs.

def lr_finder(model_fn, train_loader, init_lr=1e-7, final_lr=10.0,
              n_iter=None, smooth_factor=0.05):
    \"\"\"
    Run LR range test: exponentially increase LR each step, record loss.
    Returns (lrs, smoothed_losses).
    \"\"\"
    if n_iter is None:
        n_iter = len(train_loader)

    model = model_fn()
    opt = torch.optim.SGD(model.parameters(), lr=init_lr)
    crit = nn.MSELoss()
    mult = (final_lr / init_lr) ** (1.0 / n_iter)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=mult)

    lrs, losses, smoothed, best_loss = [], [], [], float("inf")
    avg_loss = 0.0

    for step, (xb, yb) in enumerate(train_loader):
        if step >= n_iter:
            break
        model.train()
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        scheduler.step()

        current_lr = opt.param_groups[0]["lr"]
        current_loss = loss.item()
        avg_loss = smooth_factor * current_loss + (1 - smooth_factor) * (avg_loss if step > 0 else current_loss)
        smoothed_loss = avg_loss / (1 - (1 - smooth_factor) ** (step + 1))  # bias correction

        lrs.append(current_lr)
        losses.append(current_loss)
        smoothed.append(smoothed_loss)

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        if step > 10 and smoothed_loss > 4 * best_loss:
            break  # diverging — stop

    return lrs, smoothed


# Use a loader with more steps for a smoother LR-loss curve
finder_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
lrs, smooth_losses = lr_finder(lambda: make_model_opt()[0], finder_loader, n_iter=200)

# Find suggested LR: where gradient of loss is steepest (most negative)
if len(lrs) > 5:
    loss_arr = np.array(smooth_losses)
    grad = np.gradient(loss_arr)
    best_idx = max(1, np.argmin(grad))  # steepest decline
    suggested_lr = lrs[best_idx]
else:
    suggested_lr = lrs[len(lrs)//3] if lrs else 1e-3

print(f"LR finder tested range: {lrs[0]:.2e} — {lrs[-1]:.2e}")
print(f"Suggested LR (max descent): {suggested_lr:.2e}")
print(f"Loss at suggested LR: {smooth_losses[lrs.index(suggested_lr)]:.4f}")
print("\nRule of thumb: pick LR 10x smaller than where loss starts rising.")
print(f"Conservative pick: {suggested_lr / 10:.2e}")
"""),
    ]
    save(nb, "19-learning-rate-schedules.ipynb")


# ==============================================================================
# Notebook 20 — loss-functions
# ==============================================================================
def make_20():
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("""# Loss Functions

## Learning Objectives
1. Implement MSE, MAE, Huber, BCE, and Cross-Entropy from scratch using NumPy.
2. Compare regression and classification losses in PyTorch and understand their behaviour on outliers.
3. Build Focal Loss for severe class imbalance and compare it to standard Cross-Entropy.
4. Implement contrastive loss for metric learning and a compound multi-task loss with balanced gradients.
"""),
        code("""import numpy as np
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
        md("## Level 1: Loss Functions from Scratch (NumPy)"),
        code("""# Implement the six most common loss functions without any ML library.
# Understanding from scratch reveals WHY each behaves differently.

def mse(y_true, y_pred):
    \"\"\"Mean Squared Error: quadratic penalty — large errors penalised heavily.\"\"\"
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    \"\"\"Mean Absolute Error: linear penalty — robust to outliers.\"\"\"
    return np.mean(np.abs(y_true - y_pred))


def huber(y_true, y_pred, delta=1.0):
    \"\"\"Huber loss: MSE for |e|<=delta, linear (MAE-like) beyond.\"\"\"
    e = np.abs(y_true - y_pred)
    return np.where(e <= delta, 0.5 * e**2, delta * (e - 0.5 * delta)).mean()


def binary_cross_entropy(y_true, y_prob, eps=1e-7):
    \"\"\"BCE: -[y*log(p) + (1-y)*log(1-p)]. Undefined at p=0 or 1 — clip with eps.\"\"\"
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def cross_entropy(y_true_indices, logits, eps=1e-7):
    \"\"\"Categorical CE: -sum y_i * log(softmax(logits)_i).\"\"\"
    # Stable softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    n = len(y_true_indices)
    log_probs = -np.log(probs[np.arange(n), y_true_indices] + eps)
    return log_probs.mean()


def kl_divergence(p, q, eps=1e-7):
    \"\"\"KL(p||q): measures how much p diverges from q.\"\"\"
    return np.sum(p * np.log((p + eps) / (q + eps)), axis=-1).mean()


# Test on synthetic data
np.random.seed(42)
y_true_reg = np.array([1.0, 2.0, 3.0, 10.0])  # last value is an outlier
y_pred_reg = np.array([1.1, 1.9, 3.2, 5.0])    # prediction is far from outlier

print("Regression losses (with outlier at index 3):")
print(f"  MSE  : {mse(y_true_reg, y_pred_reg):.4f}  (sensitive to outlier)")
print(f"  MAE  : {mae(y_true_reg, y_pred_reg):.4f}  (less sensitive)")
print(f"  Huber: {huber(y_true_reg, y_pred_reg, delta=1.0):.4f}  (compromise)")

y_true_cls = np.array([1, 0, 1, 1])
y_prob_cls = np.array([0.9, 0.1, 0.8, 0.3])
print(f"\nBCE: {binary_cross_entropy(y_true_cls, y_prob_cls):.4f}")

logits = np.array([[2.0, 0.5, -1.0], [0.1, 3.0, 0.2], [-1.0, 0.1, 2.5]])
labels = np.array([0, 1, 2])
print(f"CE : {cross_entropy(labels, logits):.4f}")
"""),
        md("## Level 2: Regression vs Classification Losses in PyTorch"),
        code("""# Compare: how MSE, MAE, Huber behave on regression with outliers.
# Compare: how CE, BCE, NLL behave on classification with hard vs soft labels.

torch.manual_seed(42)

# --- Regression losses on data with outliers ---
y_true_r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 20.0], device=device)  # 20.0 is outlier
y_pred_r = torch.tensor([1.1, 2.2, 2.9, 4.1, 5.2, 10.0], device=device)

print("Regression losses (outlier at y=20, pred=10):")
print(f"  MSELoss  : {F.mse_loss(y_pred_r, y_true_r):.4f}  (outlier dominates)")
print(f"  L1Loss   : {F.l1_loss(y_pred_r, y_true_r):.4f}   (robust)")
print(f"  HuberLoss: {F.huber_loss(y_pred_r, y_true_r, delta=1.0):.4f}  (blend)")
print(f"  SmoothL1 : {F.smooth_l1_loss(y_pred_r, y_true_r):.4f} (SmoothL1=Huber with delta=1)")

# Effect of delta on Huber
print("\nHuber sensitivity to delta:")
for delta in [0.5, 1.0, 2.0, 5.0]:
    print(f"  delta={delta}: {F.huber_loss(y_pred_r, y_true_r, delta=delta):.4f}")

# --- Classification losses ---
logits_c = torch.tensor([[2.0, 0.5, -0.3], [0.1, 3.2, 0.5], [-0.5, 0.2, 2.8]], device=device)
targets_c = torch.tensor([0, 1, 2], device=device)

print("\nClassification losses (correct labels):")
print(f"  CrossEntropyLoss : {F.cross_entropy(logits_c, targets_c):.4f}")
# NLLLoss expects log-softmax input
log_probs = F.log_softmax(logits_c, dim=-1)
print(f"  NLLLoss          : {F.nll_loss(log_probs, targets_c):.4f}  (= CE)")

# BCE on binary classification
try:
    logits_b = torch.tensor([[0.8], [-0.7], [1.2], [-0.3]], device=device)
    targets_b = torch.tensor([[1.0], [0.0], [1.0], [0.0]], device=device)
    print(f"  BCEWithLogitsLoss: {F.binary_cross_entropy_with_logits(logits_b, targets_b):.4f}")
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        torch.cuda.empty_cache()
        print("OOM during BCE computation — reduce batch size")
    else:
        raise
"""),
        md("## Real-World Example 1: Focal Loss for Class Imbalance"),
        code("""# Focal Loss (Lin et al., 2017 — RetinaNet):
# FL(p_t) = -(1 - p_t)^gamma * log(p_t)
# When the model is confident (p_t close to 1), (1-p_t)^gamma -> 0 -> less loss weight.
# Hard examples (low p_t) get higher weight -> focuses learning on hard samples.
# Critical for object detection and any task with severe class imbalance.

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # weight for positive class
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        \"\"\"
        logits: (N,) raw logits for binary classification
        targets: (N,) float 0/1 labels
        \"\"\"
        p = torch.sigmoid(logits)
        # p_t = p for positives, 1-p for negatives
        p_t = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(p, self.alpha),
                              torch.full_like(p, 1 - self.alpha))
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss


# Create a severely imbalanced dataset: 1:100 ratio
torch.manual_seed(42)
n_pos, n_neg = 50, 4950
X_pos = torch.randn(n_pos, 10, device=device) + 1.0  # positive class shifted
X_neg = torch.randn(n_neg, 10, device=device)
X_imb = torch.cat([X_pos, X_neg])
y_imb = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]).to(device)

ds_imb = TensorDataset(X_imb, y_imb)
ld_imb = DataLoader(ds_imb, batch_size=128, shuffle=True)

def train_binary(loss_fn_name, n_epochs=20):
    torch.manual_seed(42)
    m = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Flatten(0)).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    fl = FocalLoss(gamma=2.0, alpha=0.25)
    for _ in range(n_epochs):
        m.train()
        for xb, yb in ld_imb:
            opt.zero_grad()
            logits = m(xb)
            if loss_fn_name == "focal":
                loss = fl(logits, yb)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        logits_all = m(X_imb)
        preds = (torch.sigmoid(logits_all) > 0.5).float()
    tp = ((preds == 1) & (y_imb == 1)).sum().item()
    fp = ((preds == 1) & (y_imb == 0)).sum().item()
    fn = ((preds == 0) & (y_imb == 1)).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1

print(f"{'Loss':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
for loss_name in ["bce", "focal"]:
    p, r, f = train_binary(loss_name)
    print(f"{loss_name:>10}  {p:>10.4f}  {r:>8.4f}  {f:>8.4f}")

print("\nFocal loss improves recall on the minority class by down-weighting easy negatives.")
"""),
        md("## Real-World Example 2: Contrastive Loss for Metric Learning"),
        code("""# Contrastive loss (Hadsell et al., 2006):
# For SAME class pairs: L = d^2
# For DIFFERENT class pairs: L = max(0, margin - d)^2
# Pulls same-class embeddings together, pushes different-class apart.

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        \"\"\"
        emb1, emb2: (N, D) embedding pairs
        labels: (N,) — 1 = same class, 0 = different class
        \"\"\"
        dist = F.pairwise_distance(emb1, emb2)
        # Same-class: minimise distance
        same_loss = labels * dist.pow(2)
        # Different-class: push distance beyond margin
        diff_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return (same_loss + diff_loss).mean()


# Encoder: maps raw features to embedding space
class EmbeddingNet(nn.Module):
    def __init__(self, in_dim=10, embed_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)  # L2-normalise to unit sphere


torch.manual_seed(42)
N_CLASSES = 4
# Synthetic: 4 well-separated Gaussian clusters
X_cls = torch.cat([torch.randn(100, 10) + i * 2.0 for i in range(N_CLASSES)])
y_cls = torch.cat([torch.full((100,), i) for i in range(N_CLASSES)])

# Generate pairs: 50% same-class, 50% different
def generate_pairs(X, y, n_pairs=500):
    n = len(X)
    idx1 = torch.randint(0, n, (n_pairs,))
    idx2 = torch.randint(0, n, (n_pairs,))
    labels = (y[idx1] == y[idx2]).float()
    return X[idx1], X[idx2], labels

X1, X2, pair_labels = generate_pairs(X_cls, y_cls, n_pairs=2000)
ds_pairs = TensorDataset(X1.to(device), X2.to(device), pair_labels.to(device))
ld_pairs = DataLoader(ds_pairs, batch_size=64, shuffle=True)

encoder = EmbeddingNet().to(device)
opt_enc = torch.optim.Adam(encoder.parameters(), lr=1e-3)
contrastive = ContrastiveLoss(margin=1.0)

for epoch in range(30):
    encoder.train()
    for a, b, lbl in ld_pairs:
        opt_enc.zero_grad()
        contrastive(encoder(a), encoder(b), lbl).backward()
        opt_enc.step()

# Evaluate: intra-class vs inter-class distances
encoder.eval()
with torch.no_grad():
    embeddings = encoder(X_cls.to(device))

intra_dists, inter_dists = [], []
for i in range(len(X_cls)):
    for j in range(i+1, min(i+20, len(X_cls))):
        d = F.pairwise_distance(embeddings[i:i+1], embeddings[j:j+1]).item()
        if y_cls[i] == y_cls[j]:
            intra_dists.append(d)
        else:
            inter_dists.append(d)

print(f"Intra-class distance (same class):  mean={np.mean(intra_dists):.4f}")
print(f"Inter-class distance (diff class):  mean={np.mean(inter_dists):.4f}")
print(f"Separability ratio: {np.mean(inter_dists)/np.mean(intra_dists):.2f}x  (>1 = good separation)")
"""),
        md("## Real-World Example 3: Compound Multi-Task Loss"),
        code("""# Multi-task learning: predict both a class (CE) and a regression value (MSE).
# L_total = alpha * L_task + beta * L_reg + gamma * L_aux
# Key challenge: loss scales differ — normalise or tune alpha/beta/gamma.

torch.manual_seed(42)
N_MT = 600
X_mt = torch.randn(N_MT, 15, device=device)
# Task 1: binary classification
y_class = (X_mt[:, 0] + X_mt[:, 1] > 0).long()
# Task 2: regression
y_reg = X_mt[:, 0] * 2 + X_mt[:, 2] + 0.1 * torch.randn(N_MT, device=device)
# Auxiliary: auxiliary regression (e.g., uncertainty estimate)
y_aux = (X_mt[:, 3] + X_mt[:, 4]).abs()

ds_mt = TensorDataset(X_mt[:480], y_class[:480], y_reg[:480], y_aux[:480])
va_mt = TensorDataset(X_mt[480:], y_class[480:], y_reg[480:], y_aux[480:])
ld_mt = DataLoader(ds_mt, batch_size=32, shuffle=True)
ld_va = DataLoader(va_mt, batch_size=32)


class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(15, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.cls_head = nn.Linear(32, 2)    # classification
        self.reg_head = nn.Linear(32, 1)    # regression
        self.aux_head = nn.Linear(32, 1)    # auxiliary

    def forward(self, x):
        h = self.shared(x)
        return self.cls_head(h), self.reg_head(h).squeeze(-1), self.aux_head(h).squeeze(-1)


def compound_loss(cls_logits, reg_pred, aux_pred,
                  y_class, y_reg, y_aux,
                  alpha=1.0, beta=0.5, gamma=0.1):
    \"\"\"
    L_total = alpha * CE(classification)
              + beta * MSE(regression)  / scale_reg
              + gamma * MSE(auxiliary)  / scale_aux
    Dividing by scale normalises each loss to similar magnitude.
    \"\"\"
    l_task = F.cross_entropy(cls_logits, y_class)
    # Normalise regression loss by target variance to keep it ~O(1)
    scale_reg = y_reg.var().detach() + 1e-6
    l_reg = F.mse_loss(reg_pred, y_reg) / scale_reg
    scale_aux = y_aux.var().detach() + 1e-6
    l_aux = F.mse_loss(aux_pred, y_aux) / scale_aux
    return alpha * l_task + beta * l_reg + gamma * l_aux


model_mt = MultiTaskModel().to(device)
opt_mt = torch.optim.Adam(model_mt.parameters(), lr=1e-3)

for epoch in range(50):
    model_mt.train()
    for xb, yc, yr, ya in ld_mt:
        opt_mt.zero_grad()
        try:
            cls_out, reg_out, aux_out = model_mt(xb)
            loss = compound_loss(cls_out, reg_out, aux_out, yc, yr, ya,
                                 alpha=1.0, beta=0.5, gamma=0.1)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache(); continue
            raise
        loss.backward()
        opt_mt.step()

model_mt.eval()
correct, total, reg_mse_sum = 0, 0, 0.0
with torch.no_grad():
    for xb, yc, yr, ya in ld_va:
        cls_out, reg_out, _ = model_mt(xb)
        correct += (cls_out.argmax(1) == yc).sum().item()
        total += len(yc)
        reg_mse_sum += F.mse_loss(reg_out, yr).item() * len(yr)

print(f"Multi-task model — classification accuracy: {correct/total:.4f}")
print(f"Multi-task model — regression MSE:          {reg_mse_sum/len(va_mt):.4f}")
print("\nCompound loss: alpha=1.0 (classification) + beta=0.5 (regression) + gamma=0.1 (auxiliary)")
print("Normalising by target variance keeps each sub-loss at O(1), preventing one task from dominating.")
"""),
    ]
    save(nb, "20-loss-functions.ipynb")


if __name__ == "__main__":
    print("Generating notebooks 16-20...")
    make_16()
    make_17()
    make_18()
    make_19()
    make_20()
    print("Done.")
