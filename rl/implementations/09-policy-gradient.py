"""
Policy Gradient — REINFORCE with and without Baseline
======================================================
Implements the REINFORCE algorithm (Williams 1992) using only numpy.
CartPole physics are hand-coded — no gym required.

Variants:
  - Vanilla REINFORCE (no baseline)
  - REINFORCE with mean-return baseline (subtract mean episode return)
  - REINFORCE with learned value function baseline

Key concepts: log-derivative trick, policy gradient theorem, variance reduction.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# CartPole simulation (shared across RL notebooks)
# ---------------------------------------------------------------------------

def cartpole_step(state: np.ndarray, action: int, dt: float = 0.02) -> Tuple[np.ndarray, float, bool]:
    x, x_dot, theta, theta_dot = state
    force = 10.0 if action == 1 else -10.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    temp = (force + 0.05 * theta_dot**2 * sin_t) / 1.1
    theta_acc = (9.8 * sin_t - cos_t * temp) / (0.5 * (4/3 - 0.1 * cos_t**2 / 1.1))
    x_acc = temp - 0.05 * theta_acc * cos_t / 1.1
    x = x + dt * x_dot
    x_dot = x_dot + dt * x_acc
    theta = theta + dt * theta_dot
    theta_dot = theta_dot + dt * theta_acc
    done = bool(abs(x) > 2.4 or abs(theta) > 0.2095)
    reward = 1.0 if not done else 0.0
    return np.array([x, x_dot, theta, theta_dot]), reward, done


def cartpole_reset(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-0.05, 0.05, size=4)


# ---------------------------------------------------------------------------
# Policy Network: linear softmax (no autograd needed)
# ---------------------------------------------------------------------------

class PolicyNetwork:
    """
    Linear softmax policy: pi(a|s) = softmax(W^T s + b).
    Gradient w.r.t. W computed analytically via log-derivative trick.
    """

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, lr: float = 1e-2, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (obs_dim, n_actions))
        self.b = np.zeros(n_actions)
        self.lr = lr

    def logits(self, s: np.ndarray) -> np.ndarray:
        return s @ self.W + self.b

    def probs(self, s: np.ndarray) -> np.ndarray:
        lg = self.logits(s)
        lg -= lg.max()        # numerical stability
        p = np.exp(lg)
        return p / p.sum()

    def select_action(self, s: np.ndarray, rng: np.random.Generator) -> Tuple[int, float]:
        """Sample action and return (action, log_prob)."""
        p = self.probs(s)
        a = int(rng.choice(len(p), p=p))
        log_prob = np.log(p[a] + 1e-10)
        return a, log_prob

    def log_prob_and_grad(self, s: np.ndarray, a: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute log pi(a|s) and gradient grad_W log pi(a|s) analytically.
        d/dW log pi(a|s) = s * (e_a - pi(·|s))   [outer product]
        Returns: (log_prob, grad_W, grad_b)
        """
        p = self.probs(s)
        log_prob = float(np.log(p[a] + 1e-10))
        # Gradient of log pi(a|s) w.r.t. logits: e_a - p
        d_logits = -p.copy()
        d_logits[a] += 1.0
        grad_W = np.outer(s, d_logits)    # shape (obs_dim, n_actions)
        grad_b = d_logits
        return log_prob, grad_W, grad_b


# ---------------------------------------------------------------------------
# Value function baseline (linear)
# ---------------------------------------------------------------------------

class ValueBaseline:
    """Linear value function V(s) = w^T s used as REINFORCE baseline."""

    def __init__(self, obs_dim: int = 4, lr: float = 1e-2):
        self.w = np.zeros(obs_dim)
        self.b = 0.0
        self.lr = lr

    def predict(self, s: np.ndarray) -> float:
        return float(self.w @ s + self.b)

    def update(self, states: np.ndarray, returns: np.ndarray) -> float:
        """Gradient descent to minimise MSE(V(s) - G_t)."""
        preds = states @ self.w + self.b
        errs = preds - returns
        self.w -= self.lr * (states.T @ errs) / len(returns)
        self.b -= self.lr * errs.mean()
        return float((errs**2).mean())


# ---------------------------------------------------------------------------
# REINFORCE Agent
# ---------------------------------------------------------------------------

class REINFORCEAgent:
    """
    REINFORCE with optional baseline.
    baseline: None | 'mean' | 'value'
    """

    def __init__(
        self,
        obs_dim: int = 4,
        n_actions: int = 2,
        lr: float = 1e-2,
        gamma: float = 0.99,
        baseline: str = "none",
        seed: int = 42,
    ):
        self.policy = PolicyNetwork(obs_dim, n_actions, lr=lr, seed=seed)
        self.gamma = gamma
        self.baseline_type = baseline
        self.rng = np.random.default_rng(seed)

        if baseline == "value":
            self.value_fn = ValueBaseline(obs_dim, lr=lr * 5)
        else:
            self.value_fn = None

    def run_episode(self, max_steps: int = 200) -> Dict:
        """Run one episode, return trajectory."""
        state = cartpole_reset(self.rng)
        states, actions, log_probs, rewards = [], [], [], []

        for _ in range(max_steps):
            a, lp = self.policy.select_action(state, self.rng)
            next_state, r, done = cartpole_step(state, a)
            states.append(state.copy())
            actions.append(a)
            log_probs.append(lp)
            rewards.append(r)
            state = next_state
            if done:
                break

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "log_probs": log_probs,
            "rewards": rewards,
        }

    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """Compute discounted returns G_t = sum_{k=0}^{T-t} gamma^k r_{t+k}."""
        T = len(rewards)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def update_policy(
        self,
        episode_states: np.ndarray,
        episode_actions: np.ndarray,
        episode_returns: np.ndarray,
    ) -> float:
        """
        REINFORCE policy gradient update:
        grad_theta J = E[grad log pi(a|s) * (G_t - b_t)]
        Returns gradient norm as diagnostic.
        """
        # Compute baseline
        if self.baseline_type == "mean":
            baselines = np.full_like(episode_returns, episode_returns.mean())
        elif self.baseline_type == "value" and self.value_fn is not None:
            baselines = np.array([self.value_fn.predict(s) for s in episode_states])
            self.value_fn.update(episode_states, episode_returns)
        else:
            baselines = np.zeros_like(episode_returns)

        advantages = episode_returns - baselines

        # Accumulate policy gradient across timesteps
        total_grad_W = np.zeros_like(self.policy.W)
        total_grad_b = np.zeros_like(self.policy.b)

        for i in range(len(episode_states)):
            _, gW, gb = self.policy.log_prob_and_grad(episode_states[i], episode_actions[i])
            total_grad_W += advantages[i] * gW
            total_grad_b += advantages[i] * gb

        T = len(episode_states)
        # Gradient ascent (maximise)
        self.policy.W += self.policy.lr * total_grad_W / T
        self.policy.b += self.policy.lr * total_grad_b / T

        return float(np.linalg.norm(total_grad_W))


# ---------------------------------------------------------------------------
# Training with comparison
# ---------------------------------------------------------------------------

def train_reinforce(
    baseline: str = "none",
    n_episodes: int = 300,
    seed: int = 42,
    verbose: bool = False,
) -> List[float]:
    """Train REINFORCE and return episode returns."""
    agent = REINFORCEAgent(baseline=baseline, seed=seed)
    returns_log = []

    for ep in range(n_episodes):
        episode = agent.run_episode()
        ep_returns = agent.compute_returns(episode["rewards"])
        agent.update_policy(episode["states"], episode["actions"], ep_returns)
        ep_total = float(sum(episode["rewards"]))
        returns_log.append(ep_total)

        if verbose and (ep + 1) % 100 == 0:
            mean20 = np.mean(returns_log[-20:])
            print(f"  Ep {ep+1:4d} | total={ep_total:5.1f} | mean-20={mean20:.1f}")

    return returns_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Policy Gradient — REINFORCE with and without Baseline")
    print("=" * 60)

    N_SEEDS = 5
    N_EPISODES = 300

    # ----------------------------------------------------------------
    # 1. Single run demo
    # ----------------------------------------------------------------
    print("\n--- Single Run: REINFORCE with Mean Baseline ---")
    t0 = time.time()
    demo_returns = train_reinforce(baseline="mean", n_episodes=N_EPISODES,
                                   verbose=True, seed=42)
    print(f"Time: {time.time()-t0:.1f}s | Final mean-20: {np.mean(demo_returns[-20:]):.1f}")

    # ----------------------------------------------------------------
    # 2. Comparison across baselines (averaged over multiple seeds)
    # ----------------------------------------------------------------
    print("\n--- Baseline Comparison (avg over 5 seeds) ---")
    print(f"  {'Baseline':15s}  {'Mean return (last 50)':>22s}  {'Std (last 50)':>14s}")
    for bl in ["none", "mean", "value"]:
        all_returns = []
        for seed in range(N_SEEDS):
            r = train_reinforce(baseline=bl, n_episodes=N_EPISODES, seed=seed)
            all_returns.append(r)
        all_returns = np.array(all_returns)   # (N_SEEDS, N_EPISODES)
        mean_last = all_returns[:, -50:].mean()
        std_last = all_returns[:, -50:].std()
        print(f"  {bl:15s}  {mean_last:>22.2f}  {std_last:>14.2f}")

    # ----------------------------------------------------------------
    # 3. Learning rate sensitivity
    # ----------------------------------------------------------------
    print("\n--- LR Sensitivity (no baseline, mean of last 30 episodes) ---")
    for lr in [1e-3, 5e-3, 1e-2, 5e-2]:
        agent = REINFORCEAgent(lr=lr, baseline="mean", seed=0)
        rets = []
        for ep in range(200):
            ep_data = agent.run_episode()
            ep_rets = agent.compute_returns(ep_data["rewards"])
            agent.update_policy(ep_data["states"], ep_data["actions"], ep_rets)
            rets.append(sum(ep_data["rewards"]))
        print(f"  lr={lr:.4f} | mean(last 30)={np.mean(rets[-30:]):6.1f}")

    print("\nDone.")
