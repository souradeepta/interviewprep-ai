"""
Model-Based RL — Dyna-Q and Neural World Model
================================================
Implements:
  - Dyna-Q: combines real experience with simulated planning steps
  - NeuralWorldModel: a small MLP that learns environment dynamics
  - Comparison of K=0, K=5, K=20 planning steps on GridWorld

No external RL libraries required — only numpy.
"""

import numpy as np
import time
from typing import Tuple, Dict, List, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# GridWorld environment
# ---------------------------------------------------------------------------

class GridWorldEnv:
    """4x4 GridWorld. Terminal at state 0 and 15. Reward -1 per step."""

    N_STATES = 16
    N_ACTIONS = 4
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    TERMINAL = {0, 15}

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(42)
        self.state = self._random_start()

    def _random_start(self) -> int:
        non_terminal = [s for s in range(self.N_STATES) if s not in self.TERMINAL]
        return int(self.rng.choice(non_terminal))

    def _next_state(self, s: int, a: int) -> int:
        if s in self.TERMINAL:
            return s
        r, c = divmod(s, 4)
        if a == self.UP:    r = max(r - 1, 0)
        elif a == self.DOWN:  r = min(r + 1, 3)
        elif a == self.LEFT:  c = max(c - 1, 0)
        elif a == self.RIGHT: c = min(c + 1, 3)
        return r * 4 + c

    def reset(self) -> int:
        self.state = self._random_start()
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        s_next = self._next_state(self.state, action)
        reward = 0.0 if self.state in self.TERMINAL else -1.0
        done = s_next in self.TERMINAL
        self.state = s_next
        return s_next, reward, done


# ---------------------------------------------------------------------------
# Dyna-Q: Q-learning + tabular world model + planning
# ---------------------------------------------------------------------------

class DynaQ:
    """
    Dyna-Q (Sutton 1990): augments Q-learning with a simple model-based planning loop.

    After each real step:
      1. Update Q-table with TD(0) (real step)
      2. Update tabular world model: model[s, a] = (s', r)
      3. Run K simulated planning steps from random previously-seen (s,a) pairs
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        gamma: float = 0.99,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        k_planning: int = 5,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))
        self.model: Dict[Tuple, Tuple] = {}    # (s,a) -> (s', r)
        self.seen_sa: List[Tuple] = []         # all (s,a) pairs seen so far

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.k = k_planning
        self.n_actions = n_actions

    def act(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(self.Q[state].argmax())

    def real_step(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        """Q-learning update on real transition."""
        target = r if done else r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # Update deterministic model
        self.model[(s, a)] = (s_next, r)
        if (s, a) not in self.seen_sa:
            self.seen_sa.append((s, a))

    def model_update(self) -> None:
        """Update model (already done in real_step for tabular deterministic case)."""
        pass  # model is updated inline in real_step

    def plan(self, k_steps: Optional[int] = None) -> None:
        """K simulated planning steps from random past (s,a) pairs."""
        k = k_steps if k_steps is not None else self.k
        if len(self.seen_sa) == 0:
            return
        for _ in range(k):
            idx = int(self.rng.integers(len(self.seen_sa)))
            s, a = self.seen_sa[idx]
            s_next, r = self.model[(s, a)]
            done = s_next in GridWorldEnv.TERMINAL
            target = r if done else r + self.gamma * self.Q[s_next].max()
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def run_episode(self, env: GridWorldEnv, max_steps: int = 100) -> int:
        """Run one episode with Dyna-Q. Returns number of steps."""
        s = env.reset()
        for step in range(max_steps):
            a = self.act(s)
            s_next, r, done = env.step(a)
            self.real_step(s, a, r, s_next, done)
            self.plan()
            s = s_next
            if done:
                return step + 1
        return max_steps


# ---------------------------------------------------------------------------
# Neural World Model (MLP, numpy)
# ---------------------------------------------------------------------------

class NeuralWorldModel:
    """
    Two-layer MLP predicting next_state (16-dim one-hot) + reward from (state, action).
    Used for model-based rollouts in MBRL.
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        hidden: int = 32,
        lr: float = 1e-2,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        in_dim = n_states + n_actions   # one-hot encodings
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        # Output head 1: next state logits (n_states)
        self.W2_s = rng.normal(0, np.sqrt(2 / hidden), (hidden, n_states))
        self.b2_s = np.zeros(n_states)
        # Output head 2: reward (scalar)
        self.W2_r = rng.normal(0, np.sqrt(2 / hidden), (hidden, 1))
        self.b2_r = np.zeros(1)
        self.lr = lr
        self.n_states = n_states
        self.n_actions = n_actions

    def _encode(self, s: int, a: int) -> np.ndarray:
        """One-hot encode state and action, concatenate."""
        x = np.zeros(self.n_states + self.n_actions)
        x[s] = 1.0
        x[self.n_states + a] = 1.0
        return x

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return e / e.sum()

    def predict(self, s: int, a: int) -> Tuple[np.ndarray, float]:
        """Predict next state distribution (softmax) and reward."""
        x = self._encode(s, a)
        h = np.maximum(0, x @ self.W1 + self.b1)
        s_logits = h @ self.W2_s + self.b2_s
        r_pred = float(h @ self.W2_r + self.b2_r)
        return self._softmax(s_logits), r_pred

    def update(
        self,
        s: int, a: int,
        s_next: int, r: float,
    ) -> Tuple[float, float]:
        """One gradient step on (s, a, s_next, r). Returns (state_loss, reward_loss)."""
        x = self._encode(s, a)
        h = np.maximum(0, x @ self.W1 + self.b1)
        s_logits = h @ self.W2_s + self.b2_s
        s_probs = self._softmax(s_logits)
        r_pred = float(h @ self.W2_r + self.b2_r)

        # Cross-entropy loss for next state
        target_s = np.zeros(self.n_states)
        target_s[s_next] = 1.0
        d_s_logits = s_probs - target_s
        s_loss = float(-np.log(s_probs[s_next] + 1e-10))

        # MSE loss for reward
        r_err = r_pred - r
        r_loss = float(r_err**2)

        # Backprop W2_s, b2_s
        dW2_s = np.outer(h, d_s_logits)
        db2_s = d_s_logits

        # Backprop W2_r, b2_r
        dW2_r = h[:, None] * r_err * 2
        db2_r = np.array([r_err * 2])

        # Backprop shared trunk
        dh = d_s_logits @ self.W2_s.T + (self.W2_r.squeeze() * r_err * 2)
        dh_relu = dh * (h > 0).astype(float)
        dW1 = np.outer(x, dh_relu)
        db1 = dh_relu

        self.W2_s -= self.lr * dW2_s
        self.b2_s -= self.lr * db2_s
        self.W2_r -= self.lr * dW2_r
        self.b2_r -= self.lr * db2_r
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return s_loss, r_loss

    def sample_next(self, s: int, a: int) -> Tuple[int, float]:
        """Sample next state from predicted distribution and return predicted reward."""
        probs, r = self.predict(s, a)
        s_next = int(np.random.choice(len(probs), p=probs))
        return s_next, r


# ---------------------------------------------------------------------------
# Comparison experiment
# ---------------------------------------------------------------------------

def compare_planning_steps(
    k_values: List[int] = [0, 5, 20],
    n_episodes: int = 100,
    n_seeds: int = 5,
) -> Dict[int, List[float]]:
    """Compare Dyna-Q with different K planning steps. Returns steps-per-episode curves."""
    results = {}
    for k in k_values:
        all_runs = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            env = GridWorldEnv(rng=rng)
            agent = DynaQ(k_planning=k, seed=seed)
            ep_steps = [agent.run_episode(env) for _ in range(n_episodes)]
            all_runs.append(ep_steps)
        results[k] = list(np.mean(all_runs, axis=0))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Model-Based RL — Dyna-Q and Neural World Model")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Dyna-Q: K=0 vs K=5 vs K=20
    # ----------------------------------------------------------------
    print("\n--- Dyna-Q: Planning Steps Comparison ---")
    t0 = time.time()
    results = compare_planning_steps(k_values=[0, 5, 20], n_episodes=100, n_seeds=5)
    elapsed = time.time() - t0
    print(f"Comparison time: {elapsed:.1f}s")

    print(f"\n{'K':>4}  {'Ep 10 steps':>12}  {'Ep 50 steps':>12}  {'Ep 100 steps':>12}")
    for k, curve in results.items():
        print(f"  K={k:<2}  {curve[9]:>12.1f}  {curve[49]:>12.1f}  {curve[-1]:>12.1f}")

    # ----------------------------------------------------------------
    # 2. Neural World Model: train on GridWorld transitions
    # ----------------------------------------------------------------
    print("\n--- Neural World Model: Training on GridWorld Transitions ---")
    rng = np.random.default_rng(42)
    env = GridWorldEnv(rng=rng)
    model = NeuralWorldModel(n_states=16, n_actions=4, hidden=32, lr=5e-3)

    # Collect 500 random transitions for training
    transitions = []
    for _ in range(500):
        s = env.reset()
        a = int(rng.integers(4))
        s_next, r, _ = env.step(a)
        transitions.append((s, a, s_next, r))

    # Train the model
    s_losses, r_losses = [], []
    for epoch in range(20):
        for (s, a, s_next, r) in transitions:
            sl, rl = model.update(s, a, s_next, r)
        s_losses.append(sl)
        r_losses.append(rl)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: state_loss={sl:.4f}, reward_loss={rl:.6f}")

    # Test prediction accuracy
    correct_next = 0
    for (s, a, s_next, r) in transitions[:100]:
        pred_s, pred_r = model.sample_next(s, a)
        if pred_s == s_next:
            correct_next += 1
    print(f"\nNext-state prediction accuracy (100 test transitions): {correct_next}%")

    # ----------------------------------------------------------------
    # 3. Sample from neural model for planning
    # ----------------------------------------------------------------
    print("\n--- Neural Model Rollout (5 steps from state 5) ---")
    s = 5
    for step in range(5):
        a = int(rng.integers(4))
        probs, r_pred = model.predict(s, a)
        s_next = int(np.argmax(probs))   # greedy next state
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        print(f"  Step {step+1}: s={s:2d} --{action_names[a]:5s}--> "
              f"s_pred={s_next:2d}, r_pred={r_pred:.3f}")
        s = s_next

    print("\nDone.")
