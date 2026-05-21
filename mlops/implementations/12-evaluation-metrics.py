"""
Auto-generated from 12-evaluation-metrics.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Evaluation Metrics: Choosing the Right Measures of Success
# ## Learning Objectives
# - Implement classification, regression, ranking metrics
# - Handle imbalanced data correctly
# ======================================================================

# ======================================================================
# ## Basic Implementation: Computing Classification Metrics
# ======================================================================

import numpy as np
from typing import Dict

class ClassificationMetrics:
    """Compute classification metrics"""
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute confusion matrix"""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute all classification metrics"""
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        tp, fp, fn, tn = cm['tp'], cm['fp'], cm['fn'], cm['tn']
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }

# Example: Fraud detection (imbalanced data)
print("CLASSIFICATION METRICS ON IMBALANCED DATA\n")
print("Fraud rate: 1% (99% legitimate, 1% fraud)\n")

np.random.seed(42)
n_samples = 10000
fraud_rate = 0.01

# Ground truth
y_true = np.random.binomial(1, fraud_rate, n_samples)

# Predictions: naive model (always predict 0 = no fraud)
y_pred_naive = np.zeros(n_samples)
metrics_naive = ClassificationMetrics.compute_metrics(y_true, y_pred_naive)

print("Naive Model (always predict 'no fraud'):")
print(f"  Accuracy: {metrics_naive['accuracy']:.4f} (99% - seems good!)")
print(f"  Precision: {metrics_naive['precision']:.4f} (undefined - never predicts fraud)")
print(f"  Recall: {metrics_naive['recall']:.4f} (0% - catches no fraud!)")
print(f"  F1: {metrics_naive['f1']:.4f}")
print()

print("Problem: 99% accuracy is USELESS for fraud detection.")
print("Lesson: Never use accuracy for imbalanced data.")
print()

# Better model
y_pred_good = np.random.binomial(1, 0.1, n_samples)  # 10% predict fraud
metrics_good = ClassificationMetrics.compute_metrics(y_true, y_pred_good)

print("\nBetter Model (random 10% fraud prediction):")
print(f"  Accuracy: {metrics_good['accuracy']:.4f} (worse than naive!)")
print(f"  Precision: {metrics_good['precision']:.4f} (of fraud I flag, 10% accurate)")
print(f"  Recall: {metrics_good['recall']:.4f} (catch ~100% of fraud)")
print(f"  F1: {metrics_good['f1']:.4f}")
print()
print("Metrics show true trade-off: high recall, low precision.")


# ======================================================================
# ## Advanced Implementation: ROC-AUC and PR-AUC for Imbalanced Data
# ======================================================================

from scipy.stats import rankdata

class ImbalancedMetrics:
    """Metrics for imbalanced classification"""
    
    @staticmethod
    def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Simplified ROC-AUC: does model rank positives higher than negatives?"""
        # Count: positives ranked higher than negatives
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # For each positive, count how many negatives rank lower
        correct_pairs = 0
        for i in range(len(y_true)):
            if y_true[i] == 1:
                correct_pairs += np.sum(y_scores[i] > y_scores[y_true == 0])
        
        auc = correct_pairs / (n_pos * n_neg)
        return auc
    
    @staticmethod
    def compute_pr_curve(y_true: np.ndarray, y_scores: np.ndarray, thresholds: int = 10) -> Dict:
        """Compute precision-recall at different thresholds"""
        results = []
        
        for threshold in np.linspace(0, 1, thresholds):
            y_pred = (y_scores >= threshold).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            results.append({'threshold': threshold, 'precision': precision, 'recall': recall})
        
        return results

# Example: ROC-AUC on imbalanced fraud data
print("ROC-AUC FOR IMBALANCED DATA\n")

np.random.seed(42)
n = 10000
y_true = np.random.binomial(1, 0.01, n)  # 1% fraud

# Model A: decent fraud detector
y_scores_a = np.random.rand(n)
y_scores_a[y_true == 1] = np.random.uniform(0.6, 1.0, np.sum(y_true == 1))  # Fraud scores higher
y_scores_a[y_true == 0] = np.random.uniform(0.0, 0.4, np.sum(y_true == 0))  # Legitimate scores lower

auc_a = ImbalancedMetrics.roc_auc(y_true, y_scores_a)
print(f"Model A (decent): ROC-AUC = {auc_a:.3f} (excellent, >0.8)")
print()

# Model B: random
y_scores_b = np.random.rand(n)
auc_b = ImbalancedMetrics.roc_auc(y_true, y_scores_b)
print(f"Model B (random): ROC-AUC = {auc_b:.3f} (random, ~0.5)")
print()

print("ROC-AUC advantages on imbalanced data:")
print("- Threshold-invariant (don't need to pick cutoff)")
print("- Measures ranking quality (does model rank fraud higher?)")
print("- Works on imbalanced data (0.5 = random, 1.0 = perfect)")


# ======================================================================
# ## Real-World Example 1: Stripe Fraud Detection Metrics
# ======================================================================

import pandas as pd
import numpy as np

def netflix_ranking_metrics():
    """Evaluate ranking model with multiple metrics"""

    print("NETFLIX: Ranking Model Evaluation")
    print("=" * 60)

    np.random.seed(42)

    # Metrics
    metrics = {
        'NDCG@10': 0.823,  # Normalized Discounted Cumulative Gain
        'HitRate@10': 0.78,  # % of users with >= 1 relevant item in top 10
        'Diversity': 0.65,  # % unique content in top 10
        'Coverage': 0.42,   # % of catalog recommended to >= 1 user
        'Novelty': 0.58,    # % of long-tail content recommended
    }

    print("\nMODEL METRICS:")
    for metric, value in metrics.items():
        print(f"  {metric:20s}: {value:.3f}")

    print("\n\nCOMPARE TO BASELINE:")
    baseline_metrics = {
        'NDCG@10': 0.795,
        'HitRate@10': 0.75,
        'Diversity': 0.62,
        'Coverage': 0.38,
        'Novelty': 0.54,
    }

    for metric in metrics:
        improvement = (metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric] * 100
        status = '✓' if improvement > 0 else '✗'
        print(f"  {status} {metric:20s}: {improvement:+.1f}%")

    print("\n\nBUSINESS ALIGNMENT:")
    print("  Primary: NDCG@10 (ranking quality)")
    print("  Secondary: Diversity (user satisfaction)")
    print("  Tertiary: Coverage (catalog utilization)")

    print("\nDECISION:")
    print("  ✓ All metrics improved")
    print("  ✓ Deploy to 1% traffic (canary)")

netflix_ranking_metrics()



# ======================================================================
# ## Real-World Examples 2-3: Netflix & Uber
# ======================================================================

def netflix_metrics():
    print("NETFLIX: Multi-Metric Recommendation Design\n")
    print("Primary metric: watch_hours (business goal)")
    print("  - Baseline: 300 min/month")
    print("  - Target: +1% (3 min/month)")
    print()
    print("Guardrail metrics:")
    print("  - Engagement rate: % of recommendations clicked (≥95% baseline)")
    print("  - Diversity: % from outside top 100 (≥90% baseline)")
    print("  - Latency: p99 response time (<500ms SLO)")
    print()
    print("Decision logic:")
    print("  If watch_hours +1% AND all guardrails pass → SHIP")
    print("  If watch_hours +1% but engagement -5% → ITERATE (fix engagement)")
    print("  If watch_hours flat → REJECT (no improvement)")

def uber_metrics():
    print("\nUBER: ETA Prediction Metrics\n")
    print("Primary metric: MAE (mean absolute error)")
    print("  - Baseline: 5 minute error")
    print("  - Target: 4.5 minute (10% improvement)")
    print()
    print("Guardrail metrics:")
    print("  - Latency p99: <150ms (must infer within latency SLO)")
    print("  - Error distribution: no systematic bias (not always too high)")
    print()
    print("Why MAE not RMSE?")
    print("  - MAE: 5 minute average error")
    print("  - RMSE: penalizes large errors more (one 30-min error hurts)")
    print("  - For ETA: want balanced performance, not punishing outliers")
    print("  - Use MAE for robustness, but monitor max error")

netflix_metrics()
uber_metrics()


# ======================================================================
# ## Interview Case Study: Credit Scoring Metrics
# ======================================================================

print("CASE STUDY: CREDIT SCORING EVALUATION METRICS")
print()
print("SCENARIO:")
print("  Build credit risk model: predict 'will customer default?'")
print("  5% of customers default (imbalanced data)")
print("  Business constraint: don't discriminate by protected attributes")
print()

print("SOLUTION: Design comprehensive metric strategy")
print()

print("1. METRIC SELECTION")
print()
print("   Data is imbalanced (5% default), so:")
print("   ✗ Don't use accuracy (95% without model)")
print("   ✓ Use ROC-AUC (threshold-invariant, handles imbalance)")
print("   ✓ Use PR-AUC (precision-recall, better for imbalanced)")
print()

print("2. FAIRNESS METRICS")
print()
print("   Must measure performance by group (protected attributes):")
print("   - Gender: ROC-AUC for males, females (should be similar)")
print("   - Age: ROC-AUC for young/middle/senior (should be similar)")
print("   - Fairness constraint: max ROC-AUC gap <3% between groups")
print()

print("3. BUSINESS METRIC")
print()
print("   Loss = (missed_defaults * avg_loss) + (denied_eligible * opportunity_cost)")
print("   - Each missed default costs bank $500")
print("   - Each denied eligible customer costs $10 (lost interest)")
print("   - Optimize threshold to minimize loss")
print()

print("4. THRESHOLD SELECTION")
print()
print("   ROC curve shows precision/recall at each threshold:")
print("   Threshold 0.3: recall 95%, precision 30% (many false denials)")
print("   Threshold 0.5: recall 80%, precision 60% (balanced)")
print("   Threshold 0.7: recall 60%, precision 80% (few denials)")
print()
print("   Choose threshold that minimizes business loss:")
print("   Loss(0.3) = (5% * 500) + (70% * 10) = 25 + 7 = $32")
print("   Loss(0.5) = (20% * 500) + (40% * 10) = 100 + 4 = $104")
print("   Loss(0.7) = (40% * 500) + (20% * 10) = 200 + 2 = $202")
print()
print("   Choose threshold 0.3 (minimizes business loss)")
print()

print("5. FINAL EVALUATION")
print()
print("   Metrics:")
print("   - ROC-AUC: 0.85 (good model)")
print("   - PR-AUC: 0.72 (solid on imbalanced)")
print("   - Fairness: ROC-AUC gap <3% across all groups ✓")
print("   - Business loss: $32 (optimized threshold)")
print()

print("STRONG ANSWER:")
print()
print("'For imbalanced classification (5% default), use ROC-AUC/PR-AUC not accuracy.")
print("Stratify by protected attributes to measure fairness (max gap <3%).")
print("Design business metric: loss = (missed_defaults * cost) + (denied_eligible * cost).")
print("Choose threshold that minimizes business loss.")
print("Monitor both technical metrics (ROC-AUC) and fairness metrics (group parity).'")


# ======================================================================
# ## Key Takeaways
# **Metric selection drives optimization.** Choose wrong metric = optimize wrong thing.
# **For imbalanced data:** ROC-AUC or PR-AUC, not accuracy.
# ======================================================================
