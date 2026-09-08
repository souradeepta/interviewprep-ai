"""Behavioral contracts for repaired ML interview references."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ndcg_uses_untruncated_ideal_and_validates_inputs():
    evaluation = load("ml/interview-prep/solutions/evaluation_exercises.py")
    assert evaluation.ndcg_at_k([1, 0, 3], 2) == pytest.approx(0.131, abs=0.001)
    assert evaluation.ndcg_at_k([3, 1, 0], 2) == pytest.approx(1.0)
    assert evaluation.ndcg_at_k([0, 0], 2) == 0.0
    for bad_k in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            evaluation.ndcg_at_k([1], bad_k)
    with pytest.raises(ValueError):
        evaluation.ndcg_at_k([1, float("nan")], 2)
    for values in ([0, 1, 2], [3, 2, 1], [0, 0, 1]):
        result = evaluation.ndcg_at_k(values, 2)
        assert np.isfinite(result) and 0.0 <= result <= 1.0
    with pytest.raises(ValueError):
        evaluation.ndcg_at_k([-1, 0, 1], 2)


def test_point_in_time_and_threshold_contracts():
    module = load("ml/interview-prep/solutions/training_and_evaluation.py")
    events = [{"entity": "u1", "event_time": "2026-01-02T00:00:00"}]
    features = [
        {"entity": "u1", "feature_time": "2026-01-01T00:00:00", "feature_value": 1},
        {"entity": "u1", "feature_time": "2026-01-02T00:00:00", "feature_value": 2},
        {"entity": "u1", "feature_time": "2026-01-03T00:00:00", "feature_value": 3},
        {"entity": "u2", "feature_time": "2026-01-01T00:00:00", "feature_value": 4},
    ]
    assert module.point_in_time_features(events, features)[0]["feature_value"] == 1
    with pytest.raises(ValueError):
        module.choose_threshold([0.2, 0.8], [0, 2])
    assert module.choose_threshold([0.5], [0], false_positive_cost=0, false_negative_cost=1)[0] == 0.0


def test_training_loop_is_seeded_clipped_and_supports_early_stop():
    module = load("ml/interview-prep/solutions/training_and_evaluation.py")
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.zeros(4)

    def gradient(batch_x, batch_y, weights):
        return np.array([10.0, 0.0])

    kwargs = dict(batch_size=2, epochs=4, accumulation_steps=2, clip_norm=1.0,
                  seed=7, loss_fn=lambda x, y, w: float(np.sum(w * w)), patience=0)
    first = module.train_with_minibatches(X, y, gradient, **kwargs)
    second = module.train_with_minibatches(X, y, gradient, **kwargs)
    assert np.array_equal(first[0], second[0])
    assert np.linalg.norm(first[0]) <= 1.0
    assert len(first[1]) <= 2
