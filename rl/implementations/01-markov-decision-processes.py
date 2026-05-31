"""
Markov Decision Processes (MDP) - Standalone Implementation
============================================================
Implements a GridWorld MDP class with full transition matrix and reward model.
No external RL libraries required — only numpy.

Concepts: state space, action space, transition matrix P, reward matrix R,
          policy evaluation, sample trajectory collection.
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# GridWorld MDP
# ---------------------------------------------------------------------------

class GridWorldMDP:
    """
    A 4x4 GridWorld MDP.

    States: 0..15 (row-major). Terminal states: 0 (top-left) and 15 (bottom-right).
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.
    Transitions: deterministic; bumping a wall keeps the agent in place.
    Reward: -1 per step (standard cliff-walk formulation for policy comparison).
    """

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def __init__(self, nrow: int = 4, ncol: int = 4, gamma: float = 0.99):
        self.nrow = nrow
        self.ncol = ncol
        self.n_states = nrow * ncol
        self.n_actions = 4
        self.gamma = gamma
        self.terminal_states = {0, self.n_states - 1}

        # Build transition and reward matrices
        self.P, self.R = self._build_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _rc_to_s(self, r: int, c: int) -> int:
        return r * self.ncol + c

    def _s_to_rc(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.ncol)

    def _next_state(self, s: int, a: int) -> int:
        """Return next state after taking action a from state s (walls = stay)."""
        if s in self.terminal_states:
            return s
        r, c = self._s_to_rc(s)
        if a == self.UP:
            r = max(r - 1, 0)
        elif a == self.DOWN:
            r = min(r + 1, self.nrow - 1)
        elif a == self.LEFT:
            c = max(c - 1, 0)
        elif a == self.RIGHT:
            c = min(c + 1, self.ncol - 1)
        return self._rc_to_s(r, c)

    def _build_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build P[s, a, s'] = P(s' | s, a) and R[s, a] = E[r | s, a].
        Returns deterministic transition matrix and immediate reward matrix.
        """
        n_s, n_a = self.n_states, self.n_actions
        P = np.zeros((n_s, n_a, n_s))
        R = np.full((n_s, n_a), -1.0)   # -1 reward per step

        for s in range(n_s):
            for a in range(n_a):
                s_next = self._next_state(s, a)
                P[s, a, s_next] = 1.0
                if s in self.terminal_states:
                    R[s, a] = 0.0   # no cost in terminal state

        return P, R

    # ------------------------------------------------------------------
    # Core MDP interface
    # ------------------------------------------------------------------

    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """
        Take one step in the environment.

        Returns:
            (next_state, reward, done)
        """
        s_next = self._next_state(state, action)
        reward = self.R[state, action]
        done = s_next in self.terminal_states
        return s_next, reward, done

    def reset(self) -> int:
        """Reset to a non-terminal state (uniformly random)."""
        non_terminal = [s for s in range(self.n_states) if s not in self.terminal_states]
        return int(np.random.choice(non_terminal))

    # ------------------------------------------------------------------
    # Policy evaluation (exact, via Bellman equations)
    # ------------------------------------------------------------------

    def policy_eval(
        self,
        policy: np.ndarray,
        tol: float = 1e-6,
        max_iter: int = 10_000,
    ) -> np.ndarray:
        """
        Iterative policy evaluation for a deterministic or stochastic policy.

        Args:
            policy: shape (n_states,) int array (deterministic) or
                    (n_states, n_actions) float array (stochastic)
            tol: convergence threshold on max |V_new - V_old|
        Returns:
            V: value function array of shape (n_states,)
        """
        V = np.zeros(self.n_states)
        for _ in range(max_iter):
            V_old = V.copy()
            for s in range(self.n_states):
                if s in self.terminal_states:
                    V[s] = 0.0
                    continue
                if policy.ndim == 1:
                    # Deterministic policy
                    a = int(policy[s])
                    V[s] = self.R[s, a] + self.gamma * (self.P[s, a] @ V_old)
                else:
                    # Stochastic policy: sum over actions
                    V[s] = sum(
                        policy[s, a] * (self.R[s, a] + self.gamma * (self.P[s, a] @ V_old))
                        for a in range(self.n_actions)
                    )
            if np.max(np.abs(V - V_old)) < tol:
                break
        return V

    # ------------------------------------------------------------------
    # Trajectory sampling
    # ------------------------------------------------------------------

    def sample_trajectory(
        self,
        policy: np.ndarray,
        steps: int = 50,
        start_state: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict:
        """
        Sample a trajectory following policy for up to `steps` steps.

        Args:
            policy: (n_states,) deterministic or (n_states, n_actions) stochastic
            steps: maximum number of steps
            start_state: initial state (random non-terminal if None)
        Returns:
            dict with keys: states, actions, rewards, total_return
        """
        if rng is None:
            rng = np.random.default_rng()
        s = start_state if start_state is not None else self.reset()

        states, actions, rewards = [s], [], []
        total = 0.0
        discount = 1.0

        for _ in range(steps):
            if s in self.terminal_states:
                break
            if policy.ndim == 1:
                a = int(policy[s])
            else:
                a = int(rng.choice(self.n_actions, p=policy[s]))

            s_next, r, done = self.step(s, a)
            actions.append(a)
            rewards.append(r)
            total += discount * r
            discount *= self.gamma
            states.append(s_next)
            s = s_next
            if done:
                break

        return {"states": states, "actions": actions, "rewards": rewards, "total_return": total}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_values(self, V: np.ndarray, label: str = "V") -> None:
        """Print the value function as a grid."""
        print(f"\n{label} (4x4 grid):")
        for r in range(self.nrow):
            row_str = "  ".join(f"{V[self._rc_to_s(r, c)]:+6.2f}" for c in range(self.ncol))
            print("  " + row_str)

    def render_policy(self, policy: np.ndarray) -> None:
        """Print policy arrows as a grid."""
        arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
        print("\nPolicy:")
        for r in range(self.nrow):
            row_str = "  ".join(
                "  T  " if self._rc_to_s(r, c) in self.terminal_states
                else f"  {arrows[int(policy[self._rc_to_s(r, c)])]}  "
                for c in range(self.ncol)
            )
            print("  " + row_str)


# ---------------------------------------------------------------------------
# Policies for comparison
# ---------------------------------------------------------------------------

def random_policy(n_states: int, n_actions: int, seed: int = 0) -> np.ndarray:
    """Uniform random policy: each action equally likely."""
    policy = np.ones((n_states, n_actions)) / n_actions
    return policy


def uniform_deterministic_policy(n_states: int, seed: int = 42) -> np.ndarray:
    """Deterministic policy where each state maps to a fixed action (seeded)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=n_states)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Markov Decision Processes — GridWorld Demo")
    print("=" * 60)

    rng = np.random.default_rng(42)
    mdp = GridWorldMDP(nrow=4, ncol=4, gamma=0.99)

    print(f"\nGridWorld: {mdp.n_states} states, {mdp.n_actions} actions")
    print(f"Terminal states: {sorted(mdp.terminal_states)}")
    print(f"P shape: {mdp.P.shape}  (n_states x n_actions x n_states)")
    print(f"R shape: {mdp.R.shape}  (n_states x n_actions)")

    # --- Random policy evaluation ---
    print("\n--- Random Policy Evaluation ---")
    t0 = time.time()
    rand_pol = random_policy(mdp.n_states, mdp.n_actions)
    V_rand = mdp.policy_eval(rand_pol)
    print(f"Policy evaluation time: {1000*(time.time()-t0):.1f} ms")
    mdp.render_values(V_rand, label="V(random policy)")

    # --- Deterministic policy evaluation ---
    print("\n--- Deterministic Policy Evaluation ---")
    det_pol = uniform_deterministic_policy(mdp.n_states)
    V_det = mdp.policy_eval(det_pol)
    mdp.render_values(V_det, label="V(det policy)")
    mdp.render_policy(det_pol)

    # --- Sample trajectories ---
    print("\n--- Sample Trajectories (deterministic policy) ---")
    for episode in range(3):
        traj = mdp.sample_trajectory(det_pol, steps=30, rng=rng)
        print(f"  Episode {episode+1}: {len(traj['states'])} steps, "
              f"return={traj['total_return']:.3f}, "
              f"final state={traj['states'][-1]}")

    # --- Step interface ---
    print("\n--- Manual Step Demo ---")
    s = 5   # row 1, col 1
    for a_name, a in [("RIGHT", mdp.RIGHT), ("DOWN", mdp.DOWN)]:
        s_next, r, done = mdp.step(s, a)
        print(f"  State {s} --{a_name}--> State {s_next}, reward={r}, done={done}")
        s = s_next

    print("\nDone.")
