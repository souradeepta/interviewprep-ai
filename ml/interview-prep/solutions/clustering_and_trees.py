"""Deterministic clustering and tree-split references."""

from __future__ import annotations

import numpy as np


def kmeans(X: np.ndarray, k: int, *, max_iter: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Run seeded k-means with deterministic farthest-point empty-cluster repair."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a non-empty finite 2-D array")
    if not 1 <= k <= len(X) or max_iter <= 0:
        raise ValueError("k must be in [1, n] and max_iter positive")
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for cluster in range(k):
            members = X[new_labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
            else:
                # Re-seed with the point currently farthest from its center.
                new_centers[cluster] = X[np.argmax(distances.min(axis=1))]
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            centers = new_centers
            break
        labels, centers = new_labels, new_centers
    inertia = float(((X - centers[labels]) ** 2).sum())
    return centers, labels, inertia


def gini_impurity(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty vector")
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return float(1.0 - np.sum(probabilities**2))


def best_stump_split(X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
    """Return feature, threshold, and weighted Gini after the best split."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    if X.ndim != 2 or len(X) == 0 or y.shape != (len(X),):
        raise ValueError("X and y have incompatible shapes")
    best: tuple[int | None, float | None, float] = (None, None, gini_impurity(y))
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        thresholds = (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left = y[X[:, feature] <= threshold]; right = y[X[:, feature] > threshold]
            score = (len(left) * gini_impurity(left) + len(right) * gini_impurity(right)) / len(y)
            if score < best[2] - 1e-12:
                best = (feature, float(threshold), float(score))
    return best
