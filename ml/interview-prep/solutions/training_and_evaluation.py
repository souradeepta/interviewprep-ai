"""Data-safety, training-loop, retrieval, and evaluation references."""

from __future__ import annotations

from datetime import datetime
from collections.abc import Callable, Mapping
import numpy as np


def temporal_split(timestamps: np.ndarray, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps)
    if times.ndim != 1 or len(times) < 2 or not 0 < train_fraction < 1:
        raise ValueError("timestamps must have at least two values and fraction in (0, 1)")
    order = np.argsort(times, kind="stable")
    cut = max(1, min(len(times) - 1, int(len(times) * train_fraction)))
    return order[:cut], order[cut:]


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamps must be ISO-8601 strings")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid timestamp: {value!r}") from error


def point_in_time_features(events, feature_rows, *, entity: str = "entity",
                           event_time: str = "event_time", feature_time: str = "feature_time"):
    """Select the latest feature strictly available at each event time.

    ``events`` and ``feature_rows`` are separate iterables of mappings. A feature
    is usable only when its availability timestamp is strictly before the event.
    The returned list is deterministic and preserves event order.
    """
    events = list(events); feature_rows = list(feature_rows)
    for event in events:
        if not isinstance(event, Mapping) or entity not in event or event_time not in event:
            raise ValueError("each event needs entity and event_time")
        _timestamp(event[event_time])
    for row in feature_rows:
        if (not isinstance(row, Mapping) or entity not in row or feature_time not in row
                or "feature_value" not in row):
            raise ValueError("each feature row needs entity, feature_time, and feature_value")
        _timestamp(row[feature_time])
    result = []
    for index, event in enumerate(events):
        event_dt = _timestamp(event[event_time])
        candidates = [row for row in feature_rows if row[entity] == event[entity]
                      and _timestamp(row[feature_time]) < event_dt]
        if candidates:
            chosen = max(candidates, key=lambda row: _timestamp(row[feature_time]))
            result.append({"event_index": index, "feature_value": chosen["feature_value"],
                           "feature_time": chosen[feature_time]})
        else:
            result.append({"event_index": index, "feature_value": None, "feature_time": None})
    return result


def choose_threshold(probabilities: np.ndarray, labels: np.ndarray, *, false_positive_cost: float = 1.0, false_negative_cost: float = 1.0) -> tuple[float, float]:
    probabilities = np.asarray(probabilities, dtype=float); labels = np.asarray(labels, dtype=int)
    if (probabilities.shape != labels.shape or probabilities.ndim != 1 or len(labels) == 0
            or not np.isfinite(probabilities).all() or not (0 <= probabilities).all()
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError("probabilities and labels must be aligned and probabilities in [0, 1]")
    if (isinstance(false_positive_cost, bool) or isinstance(false_negative_cost, bool)
            or not np.isfinite([false_positive_cost, false_negative_cost]).all()
            or false_positive_cost < 0 or false_negative_cost < 0
            or false_positive_cost + false_negative_cost == 0):
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


def train_with_minibatches(X, y, gradient_fn: Callable, *, batch_size: int,
                           epochs: int = 1, learning_rate: float = 0.1,
                           accumulation_steps: int = 1, clip_norm: float | None = None,
                           patience: int | None = None, seed: int = 0,
                           initial_weights=None, loss_fn: Callable | None = None):
    """Train a tiny model with deterministic batches and explicit safeguards.

    ``gradient_fn(X_batch, y_batch, weights)`` returns a gradient vector. Each
    accumulation group averages its gradients before one update. Clipping is
    global-norm clipping, and ``patience`` applies to the optional loss callback.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    if X.ndim != 2 or len(X) == 0 or y.shape[0] != len(X) or not np.isfinite(X).all():
        raise ValueError("X and y must be aligned finite arrays")
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0
            or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0
            or isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, int)
            or accumulation_steps <= 0 or learning_rate <= 0):
        raise ValueError("invalid training-loop parameters")
    if clip_norm is not None and (clip_norm <= 0 or not np.isfinite(clip_norm)):
        raise ValueError("clip_norm must be positive and finite")
    if patience is not None and (isinstance(patience, bool) or not isinstance(patience, int) or patience < 0):
        raise ValueError("patience must be a non-negative integer")
    weights = np.zeros(X.shape[1], dtype=float) if initial_weights is None else np.asarray(initial_weights, dtype=float).copy()
    if weights.shape != (X.shape[1],) or not np.isfinite(weights).all():
        raise ValueError("initial_weights must match feature width")
    history = []
    best_loss = float("inf"); stale = 0
    for epoch in range(epochs):
        gradients = []
        for batch in mini_batch_indices(len(X), batch_size, seed=seed + epoch):
            gradient = np.asarray(gradient_fn(X[batch], y[batch], weights.copy()), dtype=float)
            if gradient.shape != weights.shape or not np.isfinite(gradient).all():
                raise ValueError("gradient_fn returned an invalid gradient")
            gradients.append(gradient)
            if len(gradients) == accumulation_steps:
                weights = _apply_update(weights, np.mean(gradients, axis=0), learning_rate, clip_norm)
                gradients = []
        if gradients:
            weights = _apply_update(weights, np.mean(gradients, axis=0), learning_rate, clip_norm)
        loss = float(loss_fn(X, y, weights)) if loss_fn is not None else float("nan")
        if loss_fn is not None:
            if not np.isfinite(loss):
                raise ValueError("loss_fn returned a non-finite value")
            history.append(loss)
            if loss < best_loss - 1e-12:
                best_loss, stale = loss, 0
            else:
                stale += 1
                if patience is not None and stale > patience:
                    break
    return weights, history


def _apply_update(weights, gradient, learning_rate, clip_norm):
    if clip_norm is not None:
        norm = float(np.linalg.norm(gradient))
        if norm > clip_norm:
            gradient = gradient * (clip_norm / norm)
    return weights - learning_rate * gradient


def recall_at_k(relevant: set, ranked: list, k: int) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0 or not relevant:
        return 0.0
    return len(set(ranked[:k]).intersection(relevant)) / len(relevant)


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
