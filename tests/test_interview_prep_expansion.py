"""Regression coverage for the Terra/Luna ML and AI interview expansion."""

import importlib.util
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path):
    path = ROOT / relative_path
    name = "interview_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_interview_question_banks_have_expected_depth():
    llm = (ROOT / "llm/interview-prep/llm-interview-questions.md").read_text()
    agents = (ROOT / "agentic-ai/interview-prep/agent-interview-questions.md").read_text()
    assert len(re.findall(r"^### Q\d+ — ", llm, re.MULTILINE)) == 30
    assert len(re.findall(r"^### Q\d+ — ", agents, re.MULTILINE)) == 20
    assert len(re.findall(r"^### Q\d+ — ", (ROOT / "system-design/interview-prep/question-bank.md").read_text(), re.MULTILINE)) == 25


def test_ml_solution_modules_are_deterministic_and_numerically_stable():
    linear = load_module("ml/interview-prep/solutions/linear_models.py")
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    weights, history = linear.linear_regression_gd(X, y, steps=500)
    assert np.allclose(weights, [1.0, 2.0], atol=0.05)
    assert history[-1] < history[0]
    assert linear.sigmoid(np.array([-1000.0, 1000.0])).tolist() == [0.0, 1.0]

    evaluation = load_module("ml/interview-prep/solutions/evaluation_exercises.py")
    assert evaluation.binary_metrics([1, 0, 1], [0.9, 0.8, 0.2]) == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 0,
        "precision": 0.5, "recall": 0.5,
    }
    assert evaluation.pass_at_k(10, 2, 1) == pytest.approx(0.2)
    assert evaluation.bootstrap_difference_ci([1, 2, 3], [1, 1, 1], seed=3, samples=20)[0] == 1.0
    assert evaluation.ndcg_at_k([1, 0, 3], 2) == pytest.approx(1 / (7 + 1 / np.log2(3)))


def test_agent_exercises_cover_retry_checkpoint_and_budget():
    retry = load_module("agentic-ai/implementations/26-error-recovery.py")
    calls = []

    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("temporary")
        return "ok"

    result = retry.retry(operation, max_attempts=3, base_delay=0.5)
    assert (result.value, result.attempts, result.delays) == ("ok", 3, [0.5, 1.0])

    with pytest.raises(ValueError):
        retry.retry(operation, max_attempts=0)
    executor = retry.IdempotentExecutor()
    with pytest.raises(retry.ReconciliationRequired):
        executor.execute("payment-1", lambda: (_ for _ in ()).throw(RuntimeError("unknown")))
    assert executor.state("payment-1") == "unknown"
    assert executor.reconcile("payment-1", lambda key: {"payment-1": "settled"}[key]) == "settled"

    conversation = load_module("agentic-ai/implementations/61-multi-turn-conversation.py")
    state = conversation.ConversationState(max_messages=2)
    state.add("user", "one")
    state.add("assistant", "two")
    state.add("user", "three")
    assert [message["content"] for message in state.messages] == ["two", "three"]
    assert conversation.ConversationState.restore(state.checkpoint(), max_messages=2).messages == state.messages
    with pytest.raises(ValueError):
        conversation.ConversationState.restore(state.checkpoint(), max_messages=0)

    budget_module = load_module("agentic-ai/implementations/64-real-time-agent-systems.py")
    with pytest.raises(ValueError):
        budget_module.Budget(deadline_ms=10, token_limit=-1, cost_limit=1.0)
    budget = budget_module.Budget(deadline_ms=10, token_limit=10, cost_limit=1.0)
    assert budget_module.run_steps([
        {"name": "retrieve", "elapsed_ms": 4, "tokens": 3, "cost": 0.2},
        {"name": "answer", "elapsed_ms": 8, "tokens": 3, "cost": 0.2},
    ], budget)["completed"] == ["retrieve"]


def test_sql_exercises_execute_against_fresh_fixture():
    sql_dir = ROOT / "ml/interview-prep/sql"
    connection = sqlite3.connect(":memory:")
    connection.executescript((sql_dir / "fixtures.sql").read_text())
    for path in sorted(sql_dir.glob("[0-9][0-9]-*.sql")):
        connection.executescript(path.read_text())
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] > 0
    assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] > 0
    statuses = dict(connection.execute(
        "SELECT prediction_id, label_status FROM label_window_results"
    ).fetchall())
    assert statuses == {"p1": "mature_positive", "p2": "mature_negative", "p3": "unknown_not_mature"}
    assignments = connection.execute(
        "SELECT user_id, COUNT(DISTINCT variant) FROM experiment_assignments GROUP BY user_id"
    ).fetchall()
    assert all(variants == 1 for _, variants in assignments)
