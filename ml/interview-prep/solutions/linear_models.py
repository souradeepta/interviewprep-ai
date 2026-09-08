"""Small, deterministic linear-model references for interview practice.

The functions intentionally expose the mechanics instead of wrapping sklearn.
They accept NumPy arrays, validate shapes, and never use global random state.
"""

from __future__ import annotations

import numpy as np


def _design_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a non-empty finite 2-D array")
    return X


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.05,
    steps: int = 1_000,
    l2: float = 0.0,
) -> tuple[np.ndarray, list[float]]:
    """Fit linear regression with full-batch gradient descent.

    The intercept is represented by the first coefficient and is not
    regularized. Returns ``(weights, loss_history)``.
    """
    X = _design_matrix(X)
    y = np.asarray(y, dtype=float)
    if y.shape != (X.shape[0],) or not np.isfinite(y).all():
        raise ValueError("y must be a finite vector aligned with X")
    if learning_rate <= 0 or steps <= 0 or l2 < 0:
        raise ValueError("learning_rate and steps must be positive; l2 non-negative")
    design = np.column_stack([np.ones(X.shape[0]), X])
    weights = np.zeros(design.shape[1])
    history: list[float] = []
    for _ in range(steps):
        residual = design @ weights - y
        history.append(float(np.mean(residual**2) + l2 * np.sum(weights[1:] ** 2)))
        grad = (2.0 / len(y)) * (design.T @ residual)
        grad[1:] += 2.0 * l2 * weights[1:]
        weights -= learning_rate * grad
    return weights, history


def logistic_loss(X: np.ndarray, y: np.ndarray, weights: np.ndarray, l2: float = 0.0) -> float:
    X = _design_matrix(X)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if y.shape != (X.shape[0],) or weights.shape != (X.shape[1] + 1,):
        raise ValueError("incompatible X, y, and weights")
    logits = np.column_stack([np.ones(len(X)), X]) @ weights
    # logaddexp(0, z) - y*z is stable binary cross entropy with logits.
    return float(np.mean(np.logaddexp(0.0, logits) - y * logits) + l2 * np.sum(weights[1:] ** 2))


def logistic_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.1,
    steps: int = 1_000,
    l2: float = 0.0,
) -> tuple[np.ndarray, list[float]]:
    """Fit binary logistic regression using stable logits and gradients."""
    X = _design_matrix(X)
    y = np.asarray(y, dtype=float)
    if y.shape != (len(X),) or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("y must contain only binary 0/1 labels")
    if learning_rate <= 0 or steps <= 0 or l2 < 0:
        raise ValueError("learning_rate and steps must be positive; l2 non-negative")
    design = np.column_stack([np.ones(len(X)), X])
    weights = np.zeros(design.shape[1])
    history: list[float] = []
    for _ in range(steps):
        probabilities = sigmoid(design @ weights)
        history.append(logistic_loss(X, y, weights, l2))
        grad = (design.T @ (probabilities - y)) / len(y)
        grad[1:] += 2.0 * l2 * weights[1:]
        weights -= learning_rate * grad
    return weights, history


def finite_difference_gradient(loss, weights: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Numerically estimate a scalar loss gradient for interview checks."""
    weights = np.asarray(weights, dtype=float)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    result = np.empty_like(weights)
    for index in np.ndindex(weights.shape):
        plus = weights.copy(); minus = weights.copy()
        plus[index] += epsilon; minus[index] -= epsilon
        result[index] = (loss(plus) - loss(minus)) / (2.0 * epsilon)
    return result
