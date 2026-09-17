"""Dependency-free Monte Carlo Tree Search for bounded agent planning."""

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Dict, Hashable, Iterable, Optional


State = Hashable


@dataclass
class Node:
    state: State
    parent: Optional["Node"] = None
    action: Optional[str] = None
    visits: int = 0
    value: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value / self.visits if self.visits else 0.0


class MonteCarloTreeSearch:
    """MCTS with bounded simulations and deterministic seeded rollouts."""

    def __init__(
        self,
        actions: Callable[[State], Iterable[str]],
        transition: Callable[[State, str], State],
        evaluate: Callable[[State], float],
        terminal: Callable[[State], bool],
        seed: int = 0,
        exploration: float = math.sqrt(2.0),
    ) -> None:
        if exploration < 0:
            raise ValueError("exploration must be non-negative")
        self.actions = actions
        self.transition = transition
        self.evaluate = evaluate
        self.terminal = terminal
        self.random = random.Random(seed)
        self.exploration = exploration

    def _expand(self, node: Node) -> Node:
        candidates = [a for a in self.actions(node.state) if a not in node.children]
        if not candidates:
            return node
        action = sorted(candidates)[0]
        child = Node(self.transition(node.state, action), parent=node, action=action)
        node.children[action] = child
        return child

    def _select(self, node: Node) -> Node:
        while not self.terminal(node.state):
            actions = list(self.actions(node.state))
            if not actions:
                return node
            if len(node.children) < len(actions):
                return self._expand(node)
            log_parent = math.log(max(node.visits, 1))
            node = max(
                node.children.values(),
                key=lambda child: (
                    child.mean_value
                    + self.exploration * math.sqrt(log_parent / child.visits),
                    child.action or "",
                ),
            )
        return node

    def _rollout(self, state: State, max_depth: int) -> float:
        for _ in range(max_depth):
            if self.terminal(state):
                break
            actions = sorted(self.actions(state))
            if not actions:
                break
            state = self.random.choice(actions)
        return float(self.evaluate(state))

    def search(self, root_state: State, simulations: int = 100, max_depth: int = 20) -> str:
        if simulations <= 0 or max_depth <= 0:
            raise ValueError("simulations and max_depth must be positive")
        root = Node(root_state)
        for _ in range(simulations):
            node = self._select(root)
            value = self._rollout(node.state, max_depth)
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent
        if not root.children:
            raise ValueError("root state has no available actions")
        return max(root.children.values(), key=lambda child: (child.visits, child.mean_value, child.action or "")).action  # type: ignore[return-value]

