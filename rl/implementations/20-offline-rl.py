"""
Offline RL — Standard Q-Learning vs Conservative Q-Learning (CQL)
==================================================================
Demonstrates the Q-value divergence problem in offline RL and how
Conservative Q-Learning (Kumar et al. 2020) addresses it.

Pipeline:
  1. Collect offline dataset from a behavioural policy (no further env interaction)
  2. Train standard OfflineQLearner — shows OOD Q-value divergence
  3. Train CQLAgent — penalises high Q-values on OOD actions, stays stable

GridWorld environment. No external RL libraries required — only numpy.
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from collections import deque


# ---------------------------------------------------------------------------
# GridWorld environment
# ---------------------------------------------------------------------------

class GridWorldEnv:
    """4x4 GridWorld for offline RL experiments."""

    N_STATES = 16
    N_ACTIONS = 4
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    TERMINAL = {0, 15}

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(0)
        self.state = self._rand_start()

    def _rand_start(self) -> int:
        non_t = [s for s in range(self.N_STATES) if s not in self.TERMINAL]
        return int(self.rng.choice(non_t))

    def _next(self, s: int, a: int) -> int:
        if s in self.TERMINAL:
            return s
        r, c = divmod(s, 4)
        if a == self.UP:    r = max(r - 1, 0)
        elif a == self.DOWN:  r = min(r + 1, 3)
        elif a == self.LEFT:  c = max(c - 1, 0)
        elif a == self.RIGHT: c = min(c + 1, 3)
        return r * 4 + c

    def reset(self) -> int:
        self.state = self._rand_start()
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        sn = self._next(self.state, action)
        r = 0.0 if self.state in self.TERMINAL else -1.0
        done = sn in self.TERMINAL
        self.state = sn
        return sn, r, done


# ---------------------------------------------------------------------------
# Offline dataset collection
# ---------------------------------------------------------------------------

def collect_offline_dataset(
    n_transitions: int = 2000,
    partial_coverage: bool = True,
    seed: int = 42,
) -> List[Tuple]:
    """
    Collect offline dataset using a sub-optimal behavioural policy.

    partial_coverage=True: only certain state-action pairs are visited
    (simulates the distribution shift problem in offline RL).
    """
    rng = np.random.default_rng(seed)
    env = GridWorldEnv(rng=rng)
    dataset = []
    s = env.reset()

    for _ in range(n_transitions):
        if partial_coverage:
            # Sub-optimal policy: heavily biased toward UP/LEFT (misses some states)
            weights = [0.5, 0.1, 0.3, 0.1]   # UP, DOWN, LEFT, RIGHT
            a = int(rng.choice(4, p=weights))
        else:
            a = int(rng.integers(4))

        sn, r, done = env.step(a)
        dataset.append((s, a, r, sn, done))
        s = env.reset() if done else sn

    return dataset


# ---------------------------------------------------------------------------
# Replay Buffer for offline training
# ---------------------------------------------------------------------------

class OfflineBuffer:
    """Pre-filled replay buffer from a fixed offline dataset."""

    def __init__(self, dataset: List[Tuple]):
        self.data = dataset

    def sample(self, batch_size: int, rng: np.random.Generator) -> List[Tuple]:
        idx = rng.choice(len(self.data), size=batch_size, replace=False)
        return [self.data[i] for i in idx]

    def __len__(self) -> int:
        return len(self.data)

    def covered_sa(self) -> set:
        """Return set of (s, a) pairs seen in the dataset."""
        return set((s, a) for (s, a, _, _, _) in self.data)


# ---------------------------------------------------------------------------
# Standard Offline Q-Learner (shows divergence on OOD actions)
# ---------------------------------------------------------------------------

class OfflineQLearner:
    """
    Standard Q-learning on offline data.
    Problem: OOD (out-of-distribution) actions have no data to constrain their
    Q-values. Bootstrapped updates propagate high Q(s', a_OOD) into Q(s, a),
    causing divergence (Q values grow unboundedly).
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        gamma: float = 0.99,
        alpha: float = 0.1,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))
        self.gamma = gamma
        self.alpha = alpha

    def update_batch(self, batch: List[Tuple]) -> float:
        """Standard Bellman update on a mini-batch."""
        total_loss = 0.0
        for (s, a, r, sn, done) in batch:
            target = r if done else r + self.gamma * self.Q[sn].max()
            td_err = target - self.Q[s, a]
            self.Q[s, a] += self.alpha * td_err
            total_loss += td_err ** 2
        return total_loss / len(batch)

    def train(self, buffer: OfflineBuffer, n_steps: int = 500, batch_size: int = 32) -> Dict:
        """Train on offline buffer. Returns Q-value and loss history."""
        max_q_history = []
        loss_history = []
        for step in range(n_steps):
            batch = buffer.sample(batch_size, self.rng)
            loss = self.update_batch(batch)
            loss_history.append(loss)
            max_q_history.append(float(self.Q.max()))
        return {"max_q": max_q_history, "loss": loss_history}


# ---------------------------------------------------------------------------
# Conservative Q-Learning (CQL)
# ---------------------------------------------------------------------------

class CQLAgent:
    """
    Conservative Q-Learning (Kumar et al. 2020).

    CQL loss = standard Bellman loss + alpha_cql * (E_{a~uniform}[Q(s,a)] - Q(s, a_data))

    The CQL penalty pushes Q-values for all actions down, but the Bellman update
    pushes Q(s, a_data) up (for high-reward data actions). The equilibrium is:
    Q(s, a_data) stays high, Q(s, a_OOD) is suppressed. This prevents bootstrapping
    from inflating OOD Q-values into the TD target.
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        gamma: float = 0.99,
        alpha: float = 0.1,
        alpha_cql: float = 1.0,    # CQL penalty strength
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_cql = alpha_cql
        self.n_actions = n_actions

    def update_batch(self, batch: List[Tuple]) -> Tuple[float, float]:
        """CQL update. Returns (td_loss, cql_penalty)."""
        td_loss = 0.0
        cql_penalty = 0.0

        for (s, a, r, sn, done) in batch:
            # Standard Bellman update
            target = r if done else r + self.gamma * self.Q[sn].max()
            td_err = target - self.Q[s, a]
            self.Q[s, a] += self.alpha * td_err
            td_loss += td_err ** 2

            # CQL conservative penalty:
            # Minimise E_{a_unif}[Q(s,a)] - Q(s, a_data)
            # Gradient: pushes down Q(s, a) uniformly, pushes up Q(s, a_data)
            q_all_mean = self.Q[s].mean()
            cql_penalty += q_all_mean - self.Q[s, a]

            # Apply CQL gradient: subtract from all actions, add to data action
            self.Q[s] -= self.alpha * self.alpha_cql / self.n_actions
            self.Q[s, a] += self.alpha * self.alpha_cql

        return td_loss / len(batch), cql_penalty / len(batch)

    def train(self, buffer: OfflineBuffer, n_steps: int = 500, batch_size: int = 32) -> Dict:
        """Train CQL on offline buffer. Returns diagnostic histories."""
        max_q_history = []
        td_loss_history = []
        cql_history = []

        for step in range(n_steps):
            batch = buffer.sample(batch_size, self.rng)
            td_loss, cql = self.update_batch(batch)
            max_q_history.append(float(self.Q.max()))
            td_loss_history.append(td_loss)
            cql_history.append(cql)

        return {"max_q": max_q_history, "td_loss": td_loss_history, "cql": cql_history}


# ---------------------------------------------------------------------------
# Evaluation: test policy in environment
# ---------------------------------------------------------------------------

def evaluate_offline_policy(
    Q: np.ndarray,
    n_episodes: int = 200,
    seed: int = 99,
) -> float:
    """Evaluate greedy policy in live environment. Returns mean return."""
    rng = np.random.default_rng(seed)
    env = GridWorldEnv(rng=rng)
    returns = []
    for _ in range(n_episodes):
        s = env.reset()
        G = 0.0
        for _ in range(50):
            a = int(Q[s].argmax())
            sn, r, done = env.step(a)
            G += r
            s = sn
            if done:
                break
        returns.append(G)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Offline RL — Standard Q-Learning vs CQL")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Collect offline dataset
    # ----------------------------------------------------------------
    print("\n--- Collecting Offline Dataset (partial coverage) ---")
    dataset = collect_offline_dataset(n_transitions=2000, partial_coverage=True, seed=42)
    buffer = OfflineBuffer(dataset)
    covered = buffer.covered_sa()
    total_sa = 16 * 4
    print(f"Dataset size: {len(dataset)} transitions")
    print(f"Coverage: {len(covered)}/{total_sa} state-action pairs ({len(covered)/total_sa:.0%})")

    # ----------------------------------------------------------------
    # 2. Standard offline Q-learning (shows divergence)
    # ----------------------------------------------------------------
    print("\n--- Standard Offline Q-Learning ---")
    t0 = time.time()
    oql = OfflineQLearner(alpha=0.1, gamma=0.99, seed=42)
    oql_hist = oql.train(buffer, n_steps=500, batch_size=32)
    oql_time = time.time() - t0

    oql_return = evaluate_offline_policy(oql.Q, seed=99)
    print(f"Training time: {oql_time:.2f}s")
    print(f"Max Q (start vs end): {oql_hist['max_q'][0]:.2f} -> {oql_hist['max_q'][-1]:.2f}")
    print(f"Evaluated policy return: {oql_return:.3f}")

    # Show Q-values on OOD state-action pairs
    ood_pairs = [(s, a) for s in range(16) for a in range(4) if (s, a) not in covered][:5]
    print(f"\nOOD Q-values (standard QL):")
    for (s, a) in ood_pairs:
        print(f"  Q[s={s}, a={a}] = {oql.Q[s, a]:.4f}")

    # ----------------------------------------------------------------
    # 3. CQL
    # ----------------------------------------------------------------
    print("\n--- Conservative Q-Learning (CQL) ---")
    t0 = time.time()
    cql = CQLAgent(alpha=0.1, gamma=0.99, alpha_cql=1.0, seed=42)
    cql_hist = cql.train(buffer, n_steps=500, batch_size=32)
    cql_time = time.time() - t0

    cql_return = evaluate_offline_policy(cql.Q, seed=99)
    print(f"Training time: {cql_time:.2f}s")
    print(f"Max Q (start vs end): {cql_hist['max_q'][0]:.2f} -> {cql_hist['max_q'][-1]:.2f}")
    print(f"Evaluated policy return: {cql_return:.3f}")

    print(f"\nOOD Q-values (CQL — should be lower/suppressed):")
    for (s, a) in ood_pairs:
        print(f"  Q[s={s}, a={a}] = {cql.Q[s, a]:.4f}")

    # ----------------------------------------------------------------
    # 4. Summary comparison
    # ----------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Metric':35s}  {'Standard QL':>12}  {'CQL':>8}")
    print("-" * 60)
    print(f"{'Max Q after 500 steps':35s}  {oql_hist['max_q'][-1]:>12.3f}  "
          f"{cql_hist['max_q'][-1]:>8.3f}")
    print(f"{'Mean OOD Q-value':35s}  "
          f"{float(np.mean([oql.Q[s,a] for (s,a) in ood_pairs])):>12.3f}  "
          f"{float(np.mean([cql.Q[s,a] for (s,a) in ood_pairs])):>8.3f}")
    print(f"{'Evaluated policy return':35s}  {oql_return:>12.3f}  {cql_return:>8.3f}")

    # ----------------------------------------------------------------
    # 5. CQL penalty sensitivity
    # ----------------------------------------------------------------
    print("\n--- CQL Alpha Sensitivity ---")
    print(f"  {'alpha_cql':>10}  {'Max Q':>8}  {'Policy return':>14}")
    for alpha_cql in [0.0, 0.1, 0.5, 1.0, 5.0]:
        agent = CQLAgent(alpha_cql=alpha_cql, seed=0)
        hist = agent.train(buffer, n_steps=300, batch_size=32)
        ret = evaluate_offline_policy(agent.Q, seed=99)
        print(f"  {alpha_cql:>10.1f}  {hist['max_q'][-1]:>8.3f}  {ret:>14.3f}")

    print("\nDone.")
