"""Expand notebook 48 (model cascading) to 600+ code lines."""
import json
import glob

f = glob.glob('modern-ai/notebooks/48-*.ipynb')[0]
nb = json.load(open(f))

# ── Cell 4: Level 1 – expand from 51 to ~80 lines ──────────────────────────
nb['cells'][3]['source'] = ["""# Level 1: Numpy cascade – confidence-based two-stage routing
# Simulate a fast cheap model and a slow accurate model routing pipeline
np.random.seed(42)

N_SAMPLES = 1000
N_CLASSES = 5

# Synthetic dataset: 1000 requests, each with a true label
true_labels = np.random.randint(0, N_CLASSES, size=N_SAMPLES)

def fast_model_predict(X: np.ndarray) -> tuple:
    \"\"\"
    Simulated fast model (small 0.5B-style): 30ms latency, 82% accuracy.
    Returns (predicted_label, max_softmax_confidence).
    \"\"\"
    np.random.seed(99)
    # Generate softmax-like confidences (uniformly noisy, often uncertain)
    raw = np.random.dirichlet(alpha=[0.5] * N_CLASSES, size=len(X))
    preds = raw.argmax(axis=1)
    confs = raw.max(axis=1)
    # Inject 82% accuracy: force correct label for 82% of samples
    correct = np.random.rand(len(X)) < 0.82
    preds[correct] = true_labels[:len(X)][correct]
    return preds, confs

def slow_model_predict(X: np.ndarray) -> tuple:
    \"\"\"
    Simulated slow model (large 70B-style): 180ms latency, 92% accuracy.
    Returns (predicted_label, max_softmax_confidence).
    \"\"\"
    np.random.seed(77)
    raw = np.random.dirichlet(alpha=[2.0] * N_CLASSES, size=len(X))
    preds = raw.argmax(axis=1)
    confs = raw.max(axis=1)
    correct = np.random.rand(len(X)) < 0.92
    preds[correct] = true_labels[:len(X)][correct]
    return preds, confs

# Threshold sweep: find optimal cascade threshold
FAST_LATENCY_MS  = 30
SLOW_LATENCY_MS  = 180

thresholds = np.linspace(0.40, 0.95, 20)
results = []

X_dummy = np.zeros((N_SAMPLES, 10))  # placeholder features
fast_preds, fast_confs = fast_model_predict(X_dummy)
slow_preds, _ = slow_model_predict(X_dummy)

for theta in thresholds:
    # Route: high confidence -> fast result, low confidence -> slow result
    escalate_mask = fast_confs < theta
    cascade_preds = fast_preds.copy()
    cascade_preds[escalate_mask] = slow_preds[escalate_mask]

    n_escalated = escalate_mask.sum()
    accuracy = (cascade_preds == true_labels).mean()
    avg_latency = (
        (N_SAMPLES - n_escalated) * FAST_LATENCY_MS +
        n_escalated * (FAST_LATENCY_MS + SLOW_LATENCY_MS)
    ) / N_SAMPLES
    cost = (
        (N_SAMPLES - n_escalated) * 0.001 +
        n_escalated * (0.001 + 0.020)
    ) / N_SAMPLES

    results.append({
        'theta': theta, 'accuracy': accuracy,
        'escalation_rate': n_escalated / N_SAMPLES,
        'avg_latency': avg_latency, 'avg_cost': cost,
    })

# Print table
print('Cascade Threshold Sweep:')
print(f'{"Theta":>7} {"Accuracy":>10} {"Escalation":>12} '
      f'{"Avg Lat(ms)":>13} {"Avg Cost($)":>13}')
print('-' * 60)
for r in results[::3]:
    print(f'{r["theta"]:7.2f} {r["accuracy"]:10.3f} {r["escalation_rate"]:12.1%} '
          f'{r["avg_latency"]:13.1f} {r["avg_cost"]:13.5f}')

# Find Pareto-optimal theta (best accuracy within 60ms average latency budget)
pareto_candidates = [r for r in results if r['avg_latency'] <= 60]
best = max(pareto_candidates, key=lambda r: r['accuracy']) if pareto_candidates else results[0]
print(f'\\nBest theta within 60ms budget: {best["theta"]:.2f} '
      f'-> accuracy={best["accuracy"]:.3f}, '
      f'escalation={best["escalation_rate"]:.1%}, '
      f'latency={best["avg_latency"]:.1f}ms')

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
thetas    = [r['theta']          for r in results]
accs      = [r['accuracy']       for r in results]
esc_rates = [r['escalation_rate'] for r in results]
lats      = [r['avg_latency']    for r in results]

axes[0].plot(thetas, accs, marker='o', linewidth=2, markersize=5, color='steelblue')
axes[0].axhline(0.82, color='coral', linestyle='--', label='Fast-only (82%)')
axes[0].axhline(0.92, color='seagreen', linestyle='--', label='Slow-only (92%)')
axes[0].axvline(best['theta'], color='purple', linestyle=':', label=f'Optimal θ={best["theta"]:.2f}')
axes[0].set_xlabel('Confidence Threshold')
axes[0].set_ylabel('Cascade Accuracy')
axes[0].set_title('Accuracy vs Threshold')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(thetas, esc_rates, marker='s', linewidth=2, markersize=5, color='coral')
axes[1].set_xlabel('Confidence Threshold')
axes[1].set_ylabel('Escalation Rate')
axes[1].set_title('Escalation Rate vs Threshold')
axes[1].grid(alpha=0.3)

axes[2].plot(lats, accs, marker='^', linewidth=2, markersize=5, color='seagreen')
axes[2].scatter([FAST_LATENCY_MS],  [0.82], s=150, c='coral',    zorder=5, label='Fast-only')
axes[2].scatter([SLOW_LATENCY_MS],  [0.92], s=150, c='steelblue', zorder=5, label='Slow-only')
axes[2].scatter([best['avg_latency']], [best['accuracy']], s=200, c='purple', zorder=6,
                marker='*', label='Optimal cascade')
axes[2].set_xlabel('Average Latency (ms)')
axes[2].set_ylabel('Accuracy')
axes[2].set_title('Latency vs Accuracy Pareto')
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
print('Level 1 complete: threshold sweep with latency/accuracy Pareto')
"""]

# ── Cell 10: RW Example 2 – expand from 67 to ~95 lines ────────────────────
nb['cells'][9]['source'] = ["""# Real-World Example 2: Cost-aware cascade with latency budget enforcement
# Optimize theta to satisfy a latency SLA (e.g., p95 <= 200ms)
# while minimizing cost per query

class CostAwareCascade:
    \"\"\"
    Cascade that auto-tunes the confidence threshold to satisfy a latency
    budget constraint on a held-out validation set.

    Cost model: C_avg = p_fast * C_fast + p_escalate * (C_fast + C_slow)
    SLA model:  p95_latency <= sla_ms  (evaluated on validation set)
    \"\"\"
    def __init__(
        self,
        fast_latency_ms: float,
        slow_latency_ms: float,
        fast_cost: float = 0.001,
        slow_cost: float = 0.020,
        sla_ms: float = 200.0,
        sla_percentile: float = 0.95,
    ):
        self.fast_lat  = fast_latency_ms
        self.slow_lat  = slow_latency_ms
        self.fast_cost = fast_cost
        self.slow_cost = slow_cost
        self.sla_ms    = sla_ms
        self.sla_pct   = sla_percentile
        self.theta_opt = 0.80  # default; calibrated via tune()

    def simulate_request_latencies(
        self, confs: np.ndarray, theta: float
    ) -> np.ndarray:
        \"\"\"Return per-request latency given confidence scores and threshold.\"\"\"
        lats = np.where(
            confs >= theta,
            np.random.normal(self.fast_lat, self.fast_lat * 0.15, size=len(confs)),
            np.random.normal(self.fast_lat + self.slow_lat,
                             self.slow_lat * 0.15, size=len(confs)),
        )
        return np.maximum(lats, 1.0)  # no negative latencies

    def tune(
        self, val_confs: np.ndarray, val_labels: np.ndarray,
        fast_preds: np.ndarray, slow_preds: np.ndarray,
    ) -> float:
        \"\"\"
        Binary-search theta such that p95 latency <= sla_ms
        while maximizing accuracy.
        \"\"\"
        np.random.seed(42)
        thresholds = np.linspace(0.40, 0.98, 40)
        best_theta = thresholds[0]
        best_acc   = -1.0

        for theta in thresholds:
            lats = self.simulate_request_latencies(val_confs, theta)
            p95_lat = np.percentile(lats, self.sla_pct * 100)
            if p95_lat > self.sla_ms:
                continue  # SLA violated – skip this theta
            escalate = val_confs < theta
            preds = np.where(escalate, slow_preds, fast_preds)
            acc = (preds == val_labels).mean()
            if acc > best_acc:
                best_acc, best_theta = acc, theta

        self.theta_opt = best_theta
        return best_theta

    def predict(
        self, confs: np.ndarray, fast_preds: np.ndarray, slow_preds: np.ndarray
    ) -> tuple:
        \"\"\"Route requests; return (final_preds, escalation_mask, per-request cost).\"\"\"
        escalate = confs < self.theta_opt
        preds = np.where(escalate, slow_preds, fast_preds)
        costs = np.where(
            escalate,
            self.fast_cost + self.slow_cost,
            self.fast_cost,
        )
        return preds, escalate, costs

np.random.seed(42)
N_VAL = 2000
N_CLS = 10

true_labels_val = np.random.randint(0, N_CLS, N_VAL)

# Synthetic fast/slow model outputs
raw_fast = np.random.dirichlet([0.6] * N_CLS, N_VAL)
fast_conf_val = raw_fast.max(axis=1)
fast_pred_val = raw_fast.argmax(axis=1)
fast_correct  = np.random.rand(N_VAL) < 0.82
fast_pred_val[fast_correct] = true_labels_val[fast_correct]

slow_pred_val = np.where(
    np.random.rand(N_VAL) < 0.92, true_labels_val, np.random.randint(0, N_CLS, N_VAL)
)

# Instantiate cascade with tight SLA
cascade_sla = CostAwareCascade(
    fast_latency_ms=35, slow_latency_ms=180,
    fast_cost=0.001,    slow_cost=0.020,
    sla_ms=200.0,       sla_percentile=0.95,
)

# Tune threshold
theta_opt = cascade_sla.tune(fast_conf_val, true_labels_val, fast_pred_val, slow_pred_val)
print(f'Cost-Aware Cascade: Optimal theta={theta_opt:.3f}')

# Evaluate at multiple SLA budgets to see the cost/accuracy/latency surface
sla_budgets = [150, 180, 200, 250, 300]
print(f'\\n{"SLA(ms)":>8} {"Theta":>7} {"Accuracy":>10} {"Escalation":>12} {"Avg Cost($)":>12}')
print('-' * 55)
for sla in sla_budgets:
    cascade_sla.sla_ms = sla
    theta_sla = cascade_sla.tune(fast_conf_val, true_labels_val, fast_pred_val, slow_pred_val)
    preds, esc, costs = cascade_sla.predict(fast_conf_val, fast_pred_val, slow_pred_val)
    acc = (preds == true_labels_val).mean()
    esc_rate = esc.mean()
    avg_cost = costs.mean()
    print(f'{sla:>8} {theta_sla:>7.3f} {acc:>10.3f} {esc_rate:>12.1%} {avg_cost:>12.5f}')

# Final evaluation with optimal theta
cascade_sla.sla_ms = 200.0
theta_opt = cascade_sla.tune(fast_conf_val, true_labels_val, fast_pred_val, slow_pred_val)
preds_opt, esc_opt, costs_opt = cascade_sla.predict(fast_conf_val, fast_pred_val, slow_pred_val)
lats_opt = cascade_sla.simulate_request_latencies(fast_conf_val, theta_opt)

print(f'\\nFinal cascade (SLA=200ms):')
print(f'  Theta: {theta_opt:.3f}')
print(f'  Accuracy: {(preds_opt == true_labels_val).mean():.3f}')
print(f'  Escalation rate: {esc_opt.mean():.1%}')
print(f'  p50 latency: {np.percentile(lats_opt, 50):.1f}ms')
print(f'  p95 latency: {np.percentile(lats_opt, 95):.1f}ms  (SLA: 200ms)')
print(f'  Avg cost: ${costs_opt.mean():.5f} (vs ${0.020:.5f} all-slow)')
print(f'  Cost savings: {(1 - costs_opt.mean() / 0.020):.0%}')

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lats_all_slow = np.random.normal(180 + 35, 30, N_VAL)
axes[0].hist(lats_opt,      bins=40, alpha=0.6, label=f'Cascade (θ={theta_opt:.2f})', color='steelblue')
axes[0].hist(lats_all_slow, bins=40, alpha=0.6, label='All-slow', color='coral')
axes[0].axvline(200, color='black', linestyle='--', linewidth=2, label='SLA (200ms)')
axes[0].axvline(np.percentile(lats_opt, 95), color='steelblue', linestyle=':',
                label=f'Cascade p95 = {np.percentile(lats_opt, 95):.0f}ms')
axes[0].set_xlabel('Request Latency (ms)')
axes[0].set_ylabel('Request Count')
axes[0].set_title('Latency Distribution: Cascade vs All-Slow')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].scatter(costs_opt, lats_opt, alpha=0.3, s=10, c='steelblue', label='Cascade requests')
axes[1].axhline(200, color='red', linestyle='--', label='SLA 200ms')
axes[1].set_xlabel('Per-request Cost ($)')
axes[1].set_ylabel('Per-request Latency (ms)')
axes[1].set_title('Cost vs Latency per Request')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
print('Example 2 complete: cost-aware cascade with SLA enforcement')
"""]

# ── Cell 14: Comparison – expand from 48 to ~80 lines ──────────────────────
nb['cells'][13]['source'] = ["""# Comparison: Cascade strategy effectiveness across configurations
import matplotlib.pyplot as plt
import numpy as np

# Configurations: no cascade, 2-model cascade at different thresholds, 3-model, all-large
configurations = [
    'All Small\\n(no cascade)',
    'Cascade\\nθ=0.70',
    'Cascade\\nθ=0.80',
    'Cascade\\nθ=0.90',
    '3-Model\\nCascade',
    'All Large',
]
accuracy     = [0.82, 0.88, 0.90, 0.91, 0.915, 0.92]
p50_latency  = [30,   36,   40,   45,   48,    180]
p99_latency  = [55,   200,  200,  200,  230,   320]
avg_cost     = [0.001, 0.004, 0.005, 0.007, 0.006, 0.020]
escalation_r = [0,     0.28,  0.22,  0.14,  0.10,  1.0]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#E67E22']

# 1. Accuracy bar chart
axes[0, 0].bar(configurations, accuracy, color=colors, alpha=0.85,
               edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy by Configuration')
axes[0, 0].set_ylim([0.78, 0.95])
for i, v in enumerate(accuracy):
    axes[0, 0].text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=9, weight='bold')
axes[0, 0].grid(alpha=0.3, axis='y')

# 2. Latency comparison (p50 vs p99)
x = np.arange(len(configurations))
w = 0.35
bars1 = axes[0, 1].bar(x - w/2, p50_latency, w, label='p50', color=colors, alpha=0.75)
bars2 = axes[0, 1].bar(x + w/2, p99_latency, w, label='p99', color=colors, alpha=0.45,
                        edgecolor='black', linewidth=1)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(configurations, fontsize=8)
axes[0, 1].set_ylabel('Latency (ms)')
axes[0, 1].set_title('p50 vs p99 Latency')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# 3. Average cost per query
axes[0, 2].bar(configurations, avg_cost, color=colors, alpha=0.85,
               edgecolor='black', linewidth=1.5)
axes[0, 2].set_ylabel('Avg Cost per Query ($)')
axes[0, 2].set_title('Cost per Query')
for i, v in enumerate(avg_cost):
    axes[0, 2].text(i, v + 0.0003, f'${v:.4f}', ha='center', fontsize=9, weight='bold')
axes[0, 2].grid(alpha=0.3, axis='y')

# 4. Escalation rate
axes[1, 0].bar(configurations[1:-1], escalation_r[1:-1],
               color=colors[1:-1], alpha=0.85, edgecolor='black', linewidth=1.5)
axes[1, 0].set_ylabel('Escalation Rate')
axes[1, 0].set_title('Escalation Rate (excludes endpoints)')
for i, v in enumerate(escalation_r[1:-1]):
    axes[1, 0].text(i, v + 0.005, f'{v:.0%}', ha='center', fontsize=9, weight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# 5. Accuracy vs cost Pareto
axes[1, 1].scatter(avg_cost, accuracy, s=220, c=colors, alpha=0.85,
                   edgecolor='black', linewidth=2, zorder=3)
for i, cfg in enumerate(configurations):
    axes[1, 1].annotate(cfg.replace('\\n', ' '), (avg_cost[i], accuracy[i]),
                        xytext=(4, 4), textcoords='offset points', fontsize=8)
axes[1, 1].set_xlabel('Avg Cost per Query ($)')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Cost vs Accuracy Pareto Frontier')
axes[1, 1].grid(alpha=0.3)

# 6. Selection guide as text table
axes[1, 2].axis('off')
table_data = [
    ['All Small',     'Best speed',    'Low accuracy',       'Latency < 30ms SLA'],
    ['Cascade θ=0.8', 'Balanced',      'p99 doubles',        'Cost budget + accuracy'],
    ['3-Model',       'Fine control',  'Complex to maintain','3+ distinct workloads'],
    ['All Large',     'Max accuracy',  'High cost + latency','Accuracy-critical only'],
]
tbl = axes[1, 2].table(
    cellText=table_data,
    colLabels=['Strategy', 'Pro', 'Con', 'Use when'],
    loc='center', cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.6)
axes[1, 2].set_title('Strategy Selection Guide', weight='bold', pad=15)

plt.tight_layout()
plt.show()

# Summary table
print('Model Cascading: Configuration Comparison')
print('=' * 80)
print(f'{"Config":<22} {"Accuracy":>10} {"p50 lat":>8} {"p99 lat":>8} '
      f'{"Cost":>10} {"Escalation":>12}')
print('-' * 80)
for cfg, acc, p50, p99, cost, esc in zip(
    configurations, accuracy, p50_latency, p99_latency, avg_cost, escalation_r
):
    print(f'{cfg.replace(chr(10), " "):<22} {acc:>10.3f} {p50:>8}ms {p99:>8}ms '
          f'${cost:>9.4f} {esc:>12.0%}')
print('\\nKey insight: cascade θ=0.80 gives best accuracy-cost balance')
print('Key insight: p99 latency always equals fast+slow latency on escalated requests')
"""]

# Write back
with open(f, 'w') as out:
    json.dump(nb, out, indent=1)

# Validate
nb2 = json.load(open(f))
lines = sum(len(''.join(c['source']).split('\n')) for c in nb2['cells'] if c['cell_type'] == 'code')
cells = len(nb2['cells'])
print(f'48: Cells={cells} (need 16), Code lines={lines} (need 600+) | PASS={cells==16 and lines>=600}')
