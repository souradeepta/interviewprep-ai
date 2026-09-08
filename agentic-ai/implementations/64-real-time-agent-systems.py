"""Deterministic deadline and budget accounting for a real-time agent."""

from dataclasses import dataclass
import math
from numbers import Real


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


@dataclass
class Budget:
    deadline_ms: int
    token_limit: int
    cost_limit: float
    elapsed_ms: int = 0
    tokens: int = 0
    cost: float = 0.0

    def __post_init__(self):
        for name in ("deadline_ms", "token_limit", "cost_limit", "elapsed_ms", "tokens", "cost"):
            setattr(self, name, _number(getattr(self, name), name))
        if self.elapsed_ms > self.deadline_ms or self.tokens > self.token_limit or self.cost > self.cost_limit:
            raise ValueError("initial usage exceeds budget")

    def can_continue(self, *, elapsed_ms=0, tokens=0, cost=0.0):
        elapsed_ms = _number(elapsed_ms, "elapsed_ms")
        tokens = _number(tokens, "tokens")
        cost = _number(cost, "cost")
        return (self.elapsed_ms + elapsed_ms <= self.deadline_ms
                and self.tokens + tokens <= self.token_limit
                and self.cost + cost <= self.cost_limit)

    def charge(self, *, elapsed_ms=0, tokens=0, cost=0.0):
        elapsed_ms = _number(elapsed_ms, "elapsed_ms")
        tokens = _number(tokens, "tokens")
        cost = _number(cost, "cost")
        if not self.can_continue(elapsed_ms=elapsed_ms, tokens=tokens, cost=cost):
            return False
        # Commit only after all three dimensions have passed: rejection is atomic.
        self.elapsed_ms += elapsed_ms
        self.tokens += tokens
        self.cost += cost
        return True


def run_steps(steps, budget: Budget, *, cancel_at_ms=None, cleanup_actions=()):
    """Run validated pre-measured steps with deterministic cancellation.

    ``cancel_at_ms`` is a logical clock boundary based on elapsed charged time;
    it avoids sleeping and returns a bounded partial result.
    """
    if cancel_at_ms is not None:
        cancel_at_ms = _number(cancel_at_ms, "cancel_at_ms")
    if not isinstance(cleanup_actions, (list, tuple)) or not all(
            isinstance(action, str) and action for action in cleanup_actions):
        raise ValueError("cleanup_actions must contain non-empty strings")
    completed = []
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("name"), str) or not step["name"]:
            raise ValueError("each step needs a non-empty name")
        for field in ("elapsed_ms", "tokens", "cost"):
            if field not in step:
                raise ValueError(f"step missing {field}")
            _number(step[field], f"step.{field}")
        if cancel_at_ms is not None and budget.elapsed_ms >= cancel_at_ms:
            return {"completed": completed, "status": "cancelled", "cancelled": step["name"],
                    "reason": "deadline", "cleanup_actions": list(cleanup_actions), "skipped": [step["name"]]}
        if not budget.charge(elapsed_ms=step["elapsed_ms"], tokens=step["tokens"], cost=step["cost"]):
            return {"completed": completed, "status": "budget_exhausted", "cancelled": None,
                    "reason": "budget_exhausted", "cleanup_actions": list(cleanup_actions),
                    "skipped": [step["name"]]}
        completed.append(step["name"])
        if cancel_at_ms is not None and budget.elapsed_ms >= cancel_at_ms:
            return {"completed": completed, "status": "cancelled", "cancelled": None,
                    "reason": "deadline", "cleanup_actions": list(cleanup_actions), "skipped": []}
    return {"completed": completed, "status": "complete", "cancelled": None,
            "reason": None, "cleanup_actions": list(cleanup_actions), "skipped": []}
