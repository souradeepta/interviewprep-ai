"""
Dynamic Programming for RL — Policy Evaluation, Improvement, Iteration
=======================================================================
Implements the three core DP algorithms for solving MDPs exactly:
  1. Policy Evaluation  — compute V^pi for a fixed policy
  2. Policy Improvement — derive a greedy policy from V^pi
  3. Policy Iteration   — alternate evaluation and improvement until convergence
  4. Value Iteration    — for comparison (one-step lookahead directly on V*)

Uses the same 4x4 GridWorld as notebooks 01-02.
No external RL libraries required — only numpy.
"""

import numpy as np
import time
from typing import Tuple, Dict, List


# ---------------------------------------------------------------------------
# GridWorld MDP (self-contained)
# ---------------------------------------------------------------------------

class GridWorldMDP:
    """Minimal 4x4 GridWorld for DP demonstrations."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, nrow: int = 4, ncol: int = 4, gamma: float = 0.99):
        self.nrow, self.ncol = nrow, ncol
        self.n_states = nrow * ncol
        self.n_actions = 4
        self.gamma = gamma
        self.terminal_states = {0, self.n_states - 1}
        self.P, self.R = self._build_model()

    def _next_state(self, s: int, a: int) -> int:
        if s in self.terminal_states:
            return s
        r, c = divmod(s, self.ncol)
        if a == self.UP:    r = max(r - 1, 0)
        elif a == self.DOWN:  r = min(r + 1, self.nrow - 1)
        elif a == self.LEFT:  c = max(c - 1, 0)
        elif a == self.RIGHT: c = min(c + 1, self.ncol - 1)
        return r * self.ncol + c

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
# 1. Policy Evaluation
# ---------------------------------------------------------------------------

def policy_evaluation(
    policy: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    tol: float = 1e-6,
    max_iter: int = 10_000,
    terminal_states: frozenset = frozenset(),
) -> Tuple[np.ndarray, int]:
    """
    Compute V^pi via iterative Bellman expectation backup.

    V^pi(s) = sum_a pi(a|s) * [R(s,a) + gamma * sum_s' P(s,a,s') * V^pi(s')]

    Args:
        policy: (n_states, n_actions) stochastic or (n_states,) deterministic int array.
    Returns:
        (V, n_iter): value function and number of iterations to converge.
    """
    n_states = R.shape[0]
    V = np.zeros(n_states)

    for it in range(max_iter):
        V_old = V.copy()
        for s in range(n_states):
            if s in terminal_states:
                V[s] = 0.0
                continue
            if policy.ndim == 1:
                # Deterministic
                a = int(policy[s])
                V[s] = R[s, a] + gamma * float(P[s, a] @ V_old)
            else:
                # Stochastic: E_pi[Q^pi(s,a)]
                V[s] = float(sum(
                    policy[s, a] * (R[s, a] + gamma * float(P[s, a] @ V_old))
                    for a in range(R.shape[1])
                ))
        if np.max(np.abs(V - V_old)) < tol:
            break
    return V, it + 1


# ---------------------------------------------------------------------------
# 2. Policy Improvement
# ---------------------------------------------------------------------------

def policy_improvement(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    terminal_states: frozenset = frozenset(),
) -> Tuple[np.ndarray, bool]:
    """
    Compute a greedy policy w.r.t. V, and check if policy changed (policy stable).

    pi'(s) = argmax_a [ R(s,a) + gamma * sum_s' P(s,a,s') V(s') ]

    Returns:
        (new_policy, policy_stable): deterministic policy and stability flag.
    """
    n_states, n_actions = R.shape
    Q = R + gamma * (P @ V)        # (n_states, n_actions)

    # Terminal states: no preferred action
    for ts in terminal_states:
        Q[ts, :] = 0.0

    new_policy = Q.argmax(axis=1)
    return new_policy, True  # stability tracked by policy_iteration


# ---------------------------------------------------------------------------
# 3. Policy Iteration
# ---------------------------------------------------------------------------

def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    eval_tol: float = 1e-6,
    max_pi_iter: int = 100,
    terminal_states: frozenset = frozenset(),
) -> Dict:
    """
    Solve MDP via policy iteration: alternate evaluation and greedy improvement.

    Guarantees convergence to the optimal policy in finite steps.

    Returns:
        dict with V_star, policy, n_eval_iters (list), n_pi_iters, history.
    """
    n_states = R.shape[0]
    # Start with uniform random deterministic policy
    policy = np.zeros(n_states, dtype=int)

    pi_history: List[np.ndarray] = [policy.copy()]
    eval_iters: List[int] = []

    for pi_it in range(max_pi_iter):
        # --- Policy Evaluation ---
        V, n_eval = policy_evaluation(
            policy, P, R, gamma, eval_tol,
            terminal_states=terminal_states,
        )
        eval_iters.append(n_eval)

        # --- Policy Improvement ---
        new_policy, _ = policy_improvement(V, P, R, gamma, terminal_states)

        # Check convergence: policy unchanged
        if np.all(new_policy == policy):
            # Policy is stable — converged
            return {
                "V_star": V,
                "policy": policy,
                "n_eval_iters": eval_iters,
                "n_pi_iters": pi_it + 1,
                "history": pi_history,
            }

        policy = new_policy
        pi_history.append(policy.copy())

    return {
        "V_star": V,
        "policy": policy,
        "n_eval_iters": eval_iters,
        "n_pi_iters": max_pi_iter,
        "history": pi_history,
    }


# ---------------------------------------------------------------------------
# 4. Value Iteration (for convergence comparison)
# ---------------------------------------------------------------------------

def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    terminal_states: frozenset = frozenset(),
) -> Dict:
    """Run value iteration and return V*, policy, n_iter."""
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    for it in range(max_iter):
        V_old = V.copy()
        Q = R + gamma * (P @ V_old)
        for ts in terminal_states:
            Q[ts, :] = 0.0
        V = Q.max(axis=1)
        if np.max(np.abs(V - V_old)) < tol:
            break
    Q_star = R + gamma * (P @ V)
    return {"V_star": V, "policy": Q_star.argmax(axis=1), "n_iter": it + 1}


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_convergence(mdp: GridWorldMDP) -> None:
    """Compare Policy Iteration vs Value Iteration on convergence speed."""
    terminals = frozenset(mdp.terminal_states)
    print("\n=== Policy Iteration vs Value Iteration ===")

    t0 = time.time()
    pi_result = policy_iteration(mdp.P, mdp.R, mdp.gamma, terminal_states=terminals)
    pi_time = time.time() - t0

    t0 = time.time()
    vi_result = value_iteration(mdp.P, mdp.R, mdp.gamma, terminal_states=terminals)
    vi_time = time.time() - t0

    print(f"{'Algorithm':25s}  {'Outer iters':>12}  {'Total eval iters':>16}  {'Time (ms)':>10}")
    print("-" * 70)
    print(f"{'Policy Iteration':25s}  {pi_result['n_pi_iters']:>12d}  "
          f"{sum(pi_result['n_eval_iters']):>16d}  {pi_time*1000:>10.2f}")
    print(f"{'Value Iteration':25s}  {vi_result['n_iter']:>12d}  "
          f"{'N/A':>16s}  {vi_time*1000:>10.2f}")

    # Verify both arrive at the same V*
    v_diff = np.max(np.abs(pi_result["V_star"] - vi_result["V_star"]))
    print(f"\nmax|V_PI - V_VI| = {v_diff:.2e}  ({'match' if v_diff < 1e-4 else 'MISMATCH'})")


def render_values(V: np.ndarray, nrow: int = 4, ncol: int = 4, label: str = "V") -> None:
    print(f"\n{label}:")
    for r in range(nrow):
        row = "   ".join(f"{V[r * ncol + c]:+7.3f}" for c in range(ncol))
        print("  " + row)


def render_policy(policy: np.ndarray, nrow: int = 4, ncol: int = 4,
                  terminals: frozenset = frozenset()) -> None:
    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    print("\nPolicy:")
    for r in range(nrow):
        row = "  ".join(
            " T " if r * ncol + c in terminals
            else f" {arrows[int(policy[r * ncol + c])]} "
            for c in range(ncol)
        )
        print("  " + row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Dynamic Programming for RL — Policy & Value Iteration")
    print("=" * 60)

    mdp = GridWorldMDP(nrow=4, ncol=4, gamma=0.99)
    terminals = frozenset(mdp.terminal_states)

    # --- Policy Iteration ---
    print("\n--- Policy Iteration ---")
    t0 = time.time()
    pi_res = policy_iteration(mdp.P, mdp.R, mdp.gamma, terminal_states=terminals)
    elapsed = time.time() - t0

    print(f"Converged in {pi_res['n_pi_iters']} policy improvement steps "
          f"({elapsed*1000:.1f} ms)")
    print(f"Eval iterations per PI step: {pi_res['n_eval_iters']}")
    render_values(pi_res["V_star"], label="V* (Policy Iteration)")
    render_policy(pi_res["policy"], terminals=terminals)

    # --- Policy history ---
    print("\n--- Policy Evolution Across PI Steps ---")
    action_names = ["^", "v", "<", ">"]
    for step, pol in enumerate(pi_res["history"]):
        actions = [action_names[a] for a in pol]
        print(f"  Step {step}: {' '.join(actions)}")

    # --- Manual policy evaluation demo ---
    print("\n--- Manual Policy Evaluation (uniform random) ---")
    uniform = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions
    V_rand, n_eval = policy_evaluation(uniform, mdp.P, mdp.R, mdp.gamma,
                                       terminal_states=terminals)
    print(f"Converged in {n_eval} iterations")
    render_values(V_rand, label="V(random policy)")

    # --- PI vs VI comparison ---
    compare_convergence(mdp)

    print("\nDone.")
