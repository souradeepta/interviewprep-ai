"""
Multi-Armed Bandit - Standalone Implementation
===============================================
K-armed bandit: epsilon-greedy, UCB1, Thompson Sampling, contextual LinUCB.
All environments are hand-coded numpy. No gym required.
"""

import numpy as np
import time
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Bandit Environment
# ---------------------------------------------------------------------------

class StationaryBandit:
    """K-armed bandit with fixed Gaussian reward distributions."""

    def __init__(self, k: int = 10, seed: int = 42):
        self.k = k
        rng = np.random.default_rng(seed)
        # True mean rewards sampled from N(0,1)
        self.true_means = rng.standard_normal(k)
        self.best_arm = int(np.argmax(self.true_means))

    def pull(self, arm: int) -> float:
        return float(np.random.normal(self.true_means[arm], 1.0))


class NonStationaryBandit:
    """Bandit where true means drift by a random walk each step."""

    def __init__(self, k: int = 10, drift: float = 0.01, seed: int = 42):
        self.k = k
        self.drift = drift
        rng = np.random.default_rng(seed)
        self.means = rng.standard_normal(k)

    def pull(self, arm: int) -> float:
        reward = float(np.random.normal(self.means[arm], 1.0))
        # Random walk on all means
        self.means += np.random.normal(0, self.drift, self.k)
        return reward

    def best_arm(self) -> int:
        return int(np.argmax(self.means))


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class EpsilonGreedy:
    """epsilon-greedy action selection."""

    def __init__(self, k: int, epsilon: float = 0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)       # value estimates
        self.N = np.zeros(k)       # pull counts

    def select(self) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.k))
        return int(np.argmax(self.Q))

    def update(self, arm: int, reward: float):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


class UCB1Agent:
    """UCB1: Q(a) + c * sqrt(ln t / N(a))."""

    def __init__(self, k: int, c: float = 2.0):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        # Force initial exploration: pull each arm once
        if self.t <= self.k:
            return self.t - 1
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + 1e-8))
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


class ThompsonSamplingAgent:
    """Thompson Sampling using Beta posterior for Bernoulli rewards."""

    def __init__(self, k: int):
        self.k = k
        # Beta(alpha, beta) posterior: alpha = successes+1, beta = failures+1
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def select(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """Binary reward assumed. Reward >= 0.5 counts as success."""
        if reward >= 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


class LinUCBAgent:
    """Contextual LinUCB: linear reward model with uncertainty exploration."""

    def __init__(self, k: int, context_dim: int, alpha: float = 1.0):
        self.k = k
        self.d = context_dim
        self.alpha = alpha
        # Per-arm ridge regression parameters
        self.A = [np.eye(context_dim) for _ in range(k)]   # A = X^T X + I
        self.b = [np.zeros(context_dim) for _ in range(k)] # b = X^T y

    def select(self, context: np.ndarray) -> int:
        scores = []
        for arm in range(self.k):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            # UCB: expected reward + alpha * uncertainty
            uncertainty = np.sqrt(context @ A_inv @ context)
            scores.append(theta @ context + self.alpha * uncertainty)
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


# ---------------------------------------------------------------------------
# Evaluation: cumulative regret
# ---------------------------------------------------------------------------

def run_bandit_experiment(
    bandit,
    agent,
    n_steps: int = 2000,
    optimal_reward: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a bandit experiment. Returns (rewards, cumulative_regret).
    """
    rewards = np.zeros(n_steps)
    regrets = np.zeros(n_steps)

    for t in range(n_steps):
        arm = agent.select()
        r = bandit.pull(arm)
        agent.update(arm, r)
        rewards[t] = r

        if optimal_reward is not None:
            regrets[t] = optimal_reward - r
        elif hasattr(bandit, 'true_means'):
            regrets[t] = bandit.true_means[bandit.best_arm] - r
        elif hasattr(bandit, 'best_arm'):
            regrets[t] = bandit.means[bandit.best_arm()] - r

    return rewards, np.cumsum(regrets)


# ---------------------------------------------------------------------------
# A/B test as bandit
# ---------------------------------------------------------------------------

def ab_test_simulation(
    n_visitors: int = 1000,
    true_cvr: List[float] = None,
    method: str = "thompson",
) -> Dict:
    """
    Simulate website A/B test with 3 variants.
    true_cvr: true conversion rates for each variant.
    """
    if true_cvr is None:
        true_cvr = [0.05, 0.08, 0.12]  # variant C is the winner
    k = len(true_cvr)

    if method == "thompson":
        agent = ThompsonSamplingAgent(k)
    else:
        agent = EpsilonGreedy(k, epsilon=0.1)

    arm_counts = np.zeros(k, dtype=int)
    conversions = np.zeros(k, dtype=int)

    for _ in range(n_visitors):
        arm = agent.select()
        reward = float(np.random.random() < true_cvr[arm])
        agent.update(arm, reward)
        arm_counts[arm] += 1
        conversions[arm] += int(reward)

    return {
        "traffic_allocation": arm_counts / n_visitors,
        "observed_cvr": conversions / (arm_counts + 1),
        "true_best_arm": int(np.argmax(true_cvr)),
        "selected_most": int(np.argmax(arm_counts)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Armed Bandit — Standalone Demo")
    print("=" * 60)

    np.random.seed(42)

    # 1. Stationary bandit comparison
    print("\n=== Stationary Bandit: epsilon-greedy vs UCB1 vs Thompson ===")
    bandit = StationaryBandit(k=10, seed=0)
    print(f"  True arm means: {bandit.true_means.round(2)}")
    print(f"  Best arm: {bandit.best_arm} (mean={bandit.true_means[bandit.best_arm]:.2f})")

    t0 = time.time()
    for name, agent in [
        ("eps=0.1", EpsilonGreedy(10, 0.1)),
        ("UCB1",    UCB1Agent(10, c=2.0)),
        ("Thompson", ThompsonSamplingAgent(10)),
    ]:
        _, regret = run_bandit_experiment(bandit, agent, n_steps=1000)
        print(f"  {name:12s}: cumulative regret after 1000 steps = {regret[-1]:.1f}")

    print(f"  Completed in {time.time()-t0:.2f}s")

    # 2. Non-stationary bandit: Thompson vs fixed-window
    print("\n=== Non-Stationary Bandit: UCB1 vs Thompson ===")
    ns_bandit = NonStationaryBandit(k=10, drift=0.05, seed=1)
    for name, agent in [
        ("UCB1",     UCB1Agent(10, c=2.0)),
        ("Thompson", ThompsonSamplingAgent(10)),
    ]:
        _, regret = run_bandit_experiment(ns_bandit, agent, n_steps=500)
        print(f"  {name:12s}: cumulative regret = {regret[-1]:.1f}")

    # 3. A/B test simulation
    print("\n=== A/B Test as Bandit ===")
    for method in ["thompson", "epsilon_greedy"]:
        result = ab_test_simulation(1000, method=method)
        print(f"  {method:15s}: traffic = {result['traffic_allocation'].round(2)}, "
              f"best_arm_selected = {result['selected_most'] == result['true_best_arm']}")

    # 4. Contextual LinUCB
    print("\n=== Contextual Bandit (LinUCB) ===")
    k, d = 4, 5
    lin_agent = LinUCBAgent(k=k, context_dim=d, alpha=1.0)
    # True reward = context @ theta_arm + noise
    true_thetas = np.random.randn(k, d)
    total_reward = 0.0
    for _ in range(500):
        ctx = np.random.randn(d)
        arm = lin_agent.select(ctx)
        r = float(true_thetas[arm] @ ctx + np.random.normal(0, 0.1))
        lin_agent.update(arm, ctx, r)
        total_reward += r
    print(f"  LinUCB 500 steps | avg reward = {total_reward/500:.3f}")

    print("\nDone.")
