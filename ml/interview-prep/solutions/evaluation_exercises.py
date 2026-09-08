"""Pure evaluation helpers used by the interview drills."""

from __future__ import annotations

import math
import numpy as np


def binary_metrics(labels, scores, threshold=0.5):
    y = np.asarray(labels, dtype=int); s = np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.ndim != 1 or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("labels and scores must be aligned binary vectors")
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


def expected_cost(labels, scores, threshold, *, false_positive=1.0, false_negative=1.0):
    m = binary_metrics(labels, scores, threshold)
    return false_positive * m["fp"] + false_negative * m["fn"]


def ndcg_at_k(relevances, k):
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")
    values = np.asarray(relevances, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("relevances must be finite, non-negative, and one-dimensional")
    if len(values) == 0:
        return 0.0
    displayed = values[:k]

    def dcg(ranked):
        gains = np.exp2(ranked) - 1.0
        discounts = np.log2(np.arange(2, len(ranked) + 2))
        return float(np.sum(gains / discounts))

    denominator = dcg(np.sort(values)[::-1][:k])
    return float(dcg(displayed) / denominator) if denominator else 0.0


def calibration_error(labels, probabilities, bins=10):
    y = np.asarray(labels, dtype=int); p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or len(y) == 0 or bins <= 0 or not ((0 <= p).all() and (p <= 1).all()):
        raise ValueError("invalid calibration inputs")
    error = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if mask.any():
            error += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def bootstrap_difference_ci(a, b, *, seed=0, samples=2000, confidence=0.95):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b) or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap inputs")
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(len(a), size=(samples, len(a)))].mean(1) - b[rng.integers(len(b), size=(samples, len(b)))].mean(1)
    alpha = (1 - confidence) / 2
    return float(a.mean() - b.mean()), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def sample_ratio_mismatch(control_count, treatment_count, expected_ratio=1.0, tolerance=0.05):
    if control_count <= 0 or treatment_count < 0 or expected_ratio <= 0 or tolerance < 0:
        raise ValueError("invalid experiment counts")
    observed = treatment_count / control_count
    return {"observed_ratio": observed, "mismatch": abs(observed - expected_ratio) > tolerance}


def aggregate_slices(rows, key="slice"):
    grouped = {}
    for row in rows:
        bucket = grouped.setdefault(row[key], {"count": 0, "correct": 0})
        bucket["count"] += 1; bucket["correct"] += int(bool(row["correct"]))
    return {name: {**values, "accuracy": values["correct"] / values["count"]} for name, values in grouped.items()}


def pass_at_k(total, successes, k):
    if not 0 <= successes <= total or not 0 < k <= total:
        raise ValueError("invalid pass@k counts")
    if total - successes < k:
        return 1.0
    return 1.0 - math.comb(total - successes, k) / math.comb(total, k)


def compare_regression_gate(candidate, baseline, *, higher_is_better=True, min_relative_change=0.0):
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    relative = (candidate - baseline) / abs(baseline)
    passed = relative >= min_relative_change if higher_is_better else relative <= -min_relative_change
    return {"relative_change": relative, "passed": bool(passed)}
