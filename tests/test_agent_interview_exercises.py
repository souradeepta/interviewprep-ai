"""Behavioral tests for bounded agent reliability exercises."""

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uncertain_side_effect_requires_reconciliation_before_replay():
    module = load("agentic-ai/implementations/26-error-recovery.py")
    executor = module.IdempotentExecutor()
    calls = []

    def operation():
        calls.append("side-effect")
        raise RuntimeError("response lost after commit")

    with pytest.raises(module.ReconciliationRequired):
        executor.execute("payment-1", operation)
    assert executor.state("payment-1") == "unknown"
    with pytest.raises(module.ReconciliationRequired):
        executor.execute("payment-1", operation)
    assert calls == ["side-effect"]
    assert executor.reconcile("payment-1", lambda key: "committed") == "committed"
    assert executor.execute("payment-1", operation) == "committed"
    assert calls == ["side-effect"]


def test_checkpoint_validation_and_durable_progress():
    module = load("agentic-ai/implementations/61-multi-turn-conversation.py")
    clock = lambda: 100.0
    state = module.ConversationState(max_messages=2, clock=clock)
    state.mark_completed("write-1")
    state.add("user", "one")
    state.add("assistant", "two")
    state.add("user", "three")
    checkpoint = state.checkpoint()
    restored = module.ConversationState.restore(
        checkpoint, max_messages=2, clock=lambda: 101.0, max_checkpoint_age=2.0
    )
    assert restored.completed_action_ids == {"write-1"}
    assert [message["content"] for message in restored.messages] == ["two", "three"]
    with pytest.raises(ValueError, match="stale"):
        module.ConversationState.restore(checkpoint, clock=lambda: 104.0, max_checkpoint_age=2.0)
    with pytest.raises(ValueError):
        module.ConversationState(max_messages=0)
    malformed = dict(checkpoint, messages=[{"role": "system", "content": "bad"}])
    with pytest.raises(ValueError):
        module.ConversationState.restore(malformed)


def test_budget_rejects_invalid_and_cancel_is_structured_and_atomic():
    module = load("agentic-ai/implementations/64-real-time-agent-systems.py")
    budget = module.Budget(deadline_ms=10, token_limit=10, cost_limit=1.0)
    with pytest.raises(ValueError):
        budget.charge(tokens=-1)
    with pytest.raises(ValueError):
        budget.charge(cost=math.nan)
    assert (budget.elapsed_ms, budget.tokens, budget.cost) == (0, 0, 0.0)
    result = module.run_steps(
        [{"name": "retrieve", "elapsed_ms": 4, "tokens": 3, "cost": 0.2},
         {"name": "answer", "elapsed_ms": 4, "tokens": 3, "cost": 0.2}],
        budget, cancel_at_ms=4, cleanup_actions=["close-stream"],
    )
    assert result["status"] == "cancelled"
    assert result["completed"] == ["retrieve"]
    assert result["cleanup_actions"] == ["close-stream"]
    with pytest.raises(ValueError):
        module.run_steps([{"name": "bad", "elapsed_ms": 1, "tokens": 1}], budget)
