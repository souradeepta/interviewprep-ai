"""Data-safety, training-loop, retrieval, and evaluation references."""

from __future__ import annotations

import numpy as np


def temporal_split(timestamps: np.ndarray, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps)
    if times.ndim != 1 or len(times) < 2 or not 0 < train_fraction < 1:
        raise ValueError("timestamps must have at least two values and fraction in (0, 1)")
    order = np.argsort(times, kind="stable")
    cut = max(1, min(len(times) - 1, int(len(times) * train_fraction)))
    return order[:cut], order[cut:]


def point_in_time_features(events, *, entity: str = "entity", event_time: str = "event_time", feature_time: str = "feature_time"):
    """Select the latest feature strictly available at each event time.

    ``events`` is an iterable of mappings with entity, event_time, feature_time,
    and feature_value keys. The returned list is deterministic and preserves
    event order; future feature rows are rejected rather than silently joined.
    """
    rows = list(events)
    result = []
    for index, event in enumerate(rows):
        candidates = [row for row in rows if row.get(entity) == event.get(entity)
                      and row.get(feature_time) <= event.get(event_time)
                      and row.get(feature_time) != event.get(event_time)]
        if candidates:
            chosen = max(candidates, key=lambda row: row[feature_time])
            result.append({"event_index": index, "feature_value": chosen["feature_value"],
                           "feature_time": chosen[feature_time]})
        else:
            result.append({"event_index": index, "feature_value": None, "feature_time": None})
    return result


def choose_threshold(probabilities: np.ndarray, labels: np.ndarray, *, false_positive_cost: float = 1.0, false_negative_cost: float = 1.0) -> tuple[float, float]:
    probabilities = np.asarray(probabilities, dtype=float); labels = np.asarray(labels, dtype=int)
    if probabilities.shape != labels.shape or len(labels) == 0 or not (0 <= probabilities).all() or not (probabilities <= 1).all():
        raise ValueError("probabilities and labels must be aligned and probabilities in [0, 1]")
    if false_positive_cost < 0 or false_negative_cost < 0 or false_positive_cost + false_negative_cost == 0:
        raise ValueError("costs must be non-negative and not both zero")
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    costs = []
    for threshold in candidates:
        prediction = probabilities >= threshold
        fp = np.sum(prediction & (labels == 0)); fn = np.sum(~prediction & (labels == 1))
        costs.append(float(false_positive_cost * fp + false_negative_cost * fn))
    index = int(np.argmin(costs))
    return float(candidates[index]), float(costs[index])


def mini_batch_indices(n_samples: int, batch_size: int, *, seed: int = 0):
    if n_samples <= 0 or batch_size <= 0:
        raise ValueError("n_samples and batch_size must be positive")
    order = np.random.default_rng(seed).permutation(n_samples)
    return [order[start:start + batch_size] for start in range(0, n_samples, batch_size)]


def recall_at_k(relevant: set, ranked: list, k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def bootstrap_mean_ci(values: np.ndarray, *, seed: int = 0, samples: int = 2_000, confidence: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap inputs")
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(0, len(values), size=(samples, len(values)))], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(values.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def feature_freshness_status(feature_time: float, request_time: float, max_age: float) -> str:
    if max_age < 0 or feature_time > request_time:
        return "invalid"
    return "fresh" if request_time - feature_time <= max_age else "stale"
