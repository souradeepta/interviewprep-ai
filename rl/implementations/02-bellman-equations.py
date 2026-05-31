"""
Bellman Equations — Value Iteration and Q* Extraction
======================================================
Implements value_iteration to solve for V* and Q* exactly via dynamic programming.
Derives optimal policy from Q*. Visualises the value heatmap and policy arrows.

No external RL libraries required — only numpy.
"""

import numpy as np
import time
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# GridWorld MDP (self-contained, same physics as 01-markov-decision-processes.py)
# ---------------------------------------------------------------------------

class GridWorldMDP:
    """4x4 GridWorld for demonstrating Bellman equations."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, nrow: int = 4, ncol: int = 4, gamma: float = 0.99):
        self.nrow, self.ncol = nrow, ncol
        self.n_states = nrow * ncol
        self.n_actions = 4
        self.gamma = gamma
        self.terminal_states = {0, self.n_states - 1}
        self.P, self.R = self._build_model()

    def _rc_to_s(self, r: int, c: int) -> int:
        return r * self.ncol + c

    def _s_to_rc(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.ncol)

    def _next_state(self, s: int, a: int) -> int:
        if s in self.terminal_states:
            return s
        r, c = self._s_to_rc(s)
        if a == self.UP:    r = max(r - 1, 0)
        elif a == self.DOWN:  r = min(r + 1, self.nrow - 1)
        elif a == self.LEFT:  c = max(c - 1, 0)
        elif a == self.RIGHT: c = min(c + 1, self.ncol - 1)
        return self._rc_to_s(r, c)

    def _build_model(self) -> Tuple[np.ndarray, np.ndarray]:
        n_s, n_a = self.n_states, self.n_actions
        P = np.zeros((n_s, n_a, n_s))
        R = np.full((n_s, n_a), -1.0)
        for s in range(n_s):
            for a in range(n_a):
                s_next = self._next_state(s, a)
                P[s, a, s_next] = 1.0
                if s in self.terminal_states:
                    R[s, a] = 0.0
        return P, R


# ---------------------------------------------------------------------------
# Value Iteration: solves V* via Bellman optimality
# ---------------------------------------------------------------------------

def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    terminal_states: frozenset = frozenset(),
) -> Dict:
    """
    Solve for V*, Q*, and the optimal policy via value iteration.

    Update rule: V*(s) = max_a [ R(s,a) + gamma * sum_s' P(s,a,s') V*(s') ]

    Args:
        P: transition tensor (n_states, n_actions, n_states)
        R: reward matrix (n_states, n_actions)
        gamma: discount factor
        tol: convergence threshold (max |V_new - V_old|)
        terminal_states: set of terminal state indices (fixed V=0)

    Returns:
        dict with keys: V_star, Q_star, policy, n_iter, history
    """
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    history = []       # track max |delta V| per iteration

    for it in range(max_iter):
        V_old = V.copy()

        # Bellman optimality backup for all states
        # Q[s, a] = R[s, a] + gamma * sum_s' P[s, a, s'] * V[s']
        Q = R + gamma * (P @ V_old)    # shape (n_states, n_actions)

        # Terminal states have fixed V=0 and Q=0
        for ts in terminal_states:
            Q[ts, :] = 0.0

        V = Q.max(axis=1)
        delta = np.max(np.abs(V - V_old))
        history.append(delta)

        if delta < tol:
            break

    # Derive Q* from converged V*
    Q_star = R + gamma * (P @ V)
    for ts in terminal_states:
        Q_star[ts, :] = 0.0

    # Optimal policy: greedy w.r.t. Q*
    policy = Q_star.argmax(axis=1)

    return {
        "V_star": V,
        "Q_star": Q_star,
        "policy": policy,
        "n_iter": it + 1,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Policy from Q-function
# ---------------------------------------------------------------------------

def policy_from_Q(Q: np.ndarray) -> np.ndarray:
    """
    Extract a deterministic greedy policy from Q*.

    Args:
        Q: Q-value matrix (n_states, n_actions)
    Returns:
        policy: (n_states,) int array with the best action per state
    """
    return Q.argmax(axis=1)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_values(V: np.ndarray, nrow: int = 4, ncol: int = 4, label: str = "V*") -> None:
    """Print V as a grid with float precision."""
    print(f"\n{label}:")
    for r in range(nrow):
        row = "   ".join(f"{V[r * ncol + c]:+7.3f}" for c in range(ncol))
        print("  " + row)


def render_policy(policy: np.ndarray, nrow: int = 4, ncol: int = 4,
                  terminals: frozenset = frozenset()) -> None:
    """Print policy arrows on a grid."""
    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    print("\nOptimal Policy:")
    for r in range(nrow):
        row = "  ".join(
            " T " if r * ncol + c in terminals else f" {arrows[int(policy[r * ncol + c])]} "
            for c in range(ncol)
        )
        print("  " + row)


def render_q_slice(Q: np.ndarray, state: int) -> None:
    """Print Q-values for a single state."""
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    print(f"\nQ*(s={state}):")
    for a, name in enumerate(action_names):
        print(f"  {name:6s}: {Q[state, a]:+7.3f}")
    best = int(Q[state].argmax())
    print(f"  Best action: {action_names[best]}")


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def compare_gamma(
    P: np.ndarray,
    R: np.ndarray,
    gamma_values: list,
    terminal_states: frozenset,
) -> None:
    """Compare value iteration convergence rate across discount factors."""
    print("\n=== Gamma Sensitivity ===")
    print(f"{'gamma':>8}  {'iters':>6}  {'V*(state 5)':>12}  {'V*(state 10)':>12}")
    for gamma in gamma_values:
        result = value_iteration(P, R, gamma=gamma, tol=1e-8,
                                 terminal_states=terminal_states)
        print(f"  {gamma:>6.3f}  {result['n_iter']:>6d}  "
              f"{result['V_star'][5]:>12.4f}  {result['V_star'][10]:>12.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Bellman Equations — Value Iteration Demo")
    print("=" * 60)

    mdp = GridWorldMDP(nrow=4, ncol=4, gamma=0.99)
    terminals = frozenset(mdp.terminal_states)

    # --- Value Iteration ---
    print("\n--- Running Value Iteration (gamma=0.99) ---")
    t0 = time.time()
    result = value_iteration(
        mdp.P, mdp.R,
        gamma=0.99,
        tol=1e-8,
        terminal_states=terminals,
    )
    elapsed = time.time() - t0
    print(f"Converged in {result['n_iter']} iterations ({elapsed*1000:.1f} ms)")

    render_values(result["V_star"], nrow=4, ncol=4, label="V*")
    render_policy(result["policy"], nrow=4, ncol=4, terminals=terminals)

    # --- Q* slice ---
    render_q_slice(result["Q_star"], state=5)   # state 5 = row 1, col 1
    render_q_slice(result["Q_star"], state=14)  # state 14 = row 3, col 2

    # --- Policy extraction ---
    print("\n--- Policy from Q* ---")
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    extracted_policy = policy_from_Q(result["Q_star"])
    for s in [1, 5, 9, 13]:
        r, c = divmod(s, 4)
        print(f"  State {s:2d} (row={r}, col={c}): "
              f"{action_names[extracted_policy[s]]}")

    # --- Convergence history ---
    print("\n--- Convergence History (first 10 iters) ---")
    for i, delta in enumerate(result["history"][:10]):
        print(f"  Iter {i+1:3d}: max|delta V| = {delta:.8f}")

    # --- Gamma comparison ---
    compare_gamma(mdp.P, mdp.R,
                  gamma_values=[0.5, 0.8, 0.9, 0.95, 0.99, 0.999],
                  terminal_states=terminals)

    print("\nDone.")
