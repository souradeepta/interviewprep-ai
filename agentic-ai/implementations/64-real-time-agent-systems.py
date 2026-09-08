"""Deterministic deadline and budget accounting for a real-time agent."""

from dataclasses import dataclass


@dataclass
class Budget:
    deadline_ms: int
    token_limit: int
    cost_limit: float
    elapsed_ms: int = 0
    tokens: int = 0
    cost: float = 0.0

    def can_continue(self, *, elapsed_ms=0, tokens=0, cost=0.0):
        return (self.elapsed_ms + elapsed_ms <= self.deadline_ms
                and self.tokens + tokens <= self.token_limit
                and self.cost + cost <= self.cost_limit)

    def charge(self, *, elapsed_ms=0, tokens=0, cost=0.0):
        if not self.can_continue(elapsed_ms=elapsed_ms, tokens=tokens, cost=cost):
            return False
        self.elapsed_ms += elapsed_ms; self.tokens += tokens; self.cost += cost
        return True


def run_steps(steps, budget: Budget):
    """Run pre-measured steps until the first budget violation."""
    completed = []
    for step in steps:
        if not budget.charge(elapsed_ms=step["elapsed_ms"], tokens=step["tokens"], cost=step["cost"]):
            return {"completed": completed, "status": "budget_exhausted"}
        completed.append(step["name"])
    return {"completed": completed, "status": "complete"}
