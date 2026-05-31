"""
Proximal Policy Optimization (PPO) - Standalone Implementation
=============================================================
Implements PPO-Clip without gym/gymnasium. All environments are hand-coded.
Uses only numpy (no torch/gym required).

Concepts: policy ratio clipping, GAE advantage estimation, multiple epochs per rollout.
"""

import numpy as np
import time
from typing import Tuple, List, Dict


# ---------------------------------------------------------------------------
# CartPole simulation (provided physics)
# ---------------------------------------------------------------------------

def cartpole_step(state: np.ndarray, action: int, dt: float = 0.02) -> Tuple[np.ndarray, float, bool]:
    """One step of CartPole dynamics."""
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
    done = bool(abs(x) > 2.4 or abs(theta) > 0.2)
    reward = float(not done)
    return np.array([x, x_dot, theta, theta_dot]), reward, done


def cartpole_reset(rng: np.random.Generator) -> np.ndarray:
    """Reset CartPole to a random starting state near the origin."""
    return rng.uniform(-0.05, 0.05, size=4)


# ---------------------------------------------------------------------------
# Linear policy (softmax) + linear value function
# ---------------------------------------------------------------------------

class LinearPolicy:
    """Softmax linear policy for discrete action spaces."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 3e-3):
        self.W = np.zeros((obs_dim, n_actions))
        self.b = np.zeros(n_actions)
        self.lr = lr

    def logits(self, s: np.ndarray) -> np.ndarray:
        return s @ self.W + self.b

    def probs(self, s: np.ndarray) -> np.ndarray:
        lg = self.logits(s)
        lg -= lg.max()          # numerical stability
        p = np.exp(lg)
        return p / p.sum()

    def log_prob(self, s: np.ndarray, a: int) -> float:
        return np.log(self.probs(s)[a] + 1e-8)

    def sample(self, s: np.ndarray, rng: np.random.Generator) -> int:
        p = self.probs(s)
        return int(rng.choice(len(p), p=p))


class LinearValue:
    """Linear value function baseline V(s) = w^T s + b."""

    def __init__(self, obs_dim: int, lr: float = 1e-2):
        self.w = np.zeros(obs_dim)
        self.b_val = 0.0
        self.lr = lr

    def predict(self, s: np.ndarray) -> float:
        return float(self.w @ s + self.b_val)

    def update(self, states: np.ndarray, targets: np.ndarray) -> float:
        """Gradient step to minimise MSE."""
        preds = states @ self.w + self.b_val
        errors = preds - targets
        self.w -= self.lr * (states.T @ errors) / len(targets)
        self.b_val -= self.lr * errors.mean()
        return float((errors**2).mean())


# ---------------------------------------------------------------------------
# GAE: Generalised Advantage Estimation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (advantages, returns) via GAE(gamma, lambda)."""
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_val * (1 - float(dones[t])) - values[t]
        gae = delta + gamma * lam * (1 - float(dones[t])) * gae
        advantages[t] = gae
    returns = advantages + np.array(values[:T])
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO Clip objective
# ---------------------------------------------------------------------------

def ppo_policy_loss(
    states: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    policy: LinearPolicy,
    clip_eps: float = 0.2,
) -> Tuple[float, np.ndarray]:
    """
    Compute PPO-Clip objective and gradient w.r.t. policy weights.
    Returns (loss, grad_W, grad_b).
    """
    T = len(states)
    total_loss = 0.0
    grad_W = np.zeros_like(policy.W)
    grad_b = np.zeros_like(policy.b)

    for i in range(T):
        s, a, old_lp, adv = states[i], int(actions[i]), old_log_probs[i], advantages[i]
        probs = policy.probs(s)
        new_lp = np.log(probs[a] + 1e-8)
        ratio = np.exp(new_lp - old_lp)
        clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)

        # PPO-Clip loss (negative because we maximise)
        obj = min(ratio * adv, clipped * adv)
        total_loss -= obj

        # Gradient through softmax log-prob
        d_new_lp = -adv * (ratio if ratio * adv <= clipped * adv else 0.0)
        # d log pi / d logit_a  = 1 - pi(a), d log pi / d logit_j (j!=a) = -pi(j)
        d_logits = -probs.copy()
        d_logits[a] += 1.0
        d_logits *= d_new_lp

        grad_W += np.outer(s, d_logits)
        grad_b += d_logits

    return total_loss / T, grad_W / T, grad_b / T


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    policy: LinearPolicy,
    value_fn: LinearValue,
    rng: np.random.Generator,
    n_steps: int = 256,
) -> Dict:
    """Collect a fixed-length rollout using the current policy."""
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    state = cartpole_reset(rng)

    for _ in range(n_steps):
        val = value_fn.predict(state)
        action = policy.sample(state, rng)
        lp = policy.log_prob(state, action)
        next_state, reward, done = cartpole_step(state, action)

        states.append(state.copy())
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(lp)
        values.append(val)

        state = cartpole_reset(rng) if done else next_state

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": rewards,
        "dones": dones,
        "log_probs": np.array(log_probs),
        "values": values,
    }


# ---------------------------------------------------------------------------
# PPO Training loop
# ---------------------------------------------------------------------------

def train_ppo(
    n_iterations: int = 80,
    n_steps: int = 256,
    k_epochs: int = 4,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    policy_lr: float = 3e-3,
    value_lr: float = 1e-2,
    seed: int = 42,
) -> List[float]:
    """Train PPO on CartPole. Returns list of mean episode returns per iteration."""
    rng = np.random.default_rng(seed)
    policy = LinearPolicy(obs_dim=4, n_actions=2, lr=policy_lr)
    value_fn = LinearValue(obs_dim=4, lr=value_lr)
    episode_returns = []

    for itr in range(n_iterations):
        # 1. Collect rollout with current policy
        rollout = collect_rollout(policy, value_fn, rng, n_steps)
        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]

        # 2. Compute GAE advantages and returns
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Multiple epochs of PPO update
        for _ in range(k_epochs):
            loss, gW, gb = ppo_policy_loss(states, actions, old_log_probs, advantages, policy, clip_eps)
            policy.W -= policy_lr * gW
            policy.b -= policy_lr * gb
            value_fn.update(states, returns)

        # 4. Track performance: mean return per episode
        ep_return = sum(rewards)
        episode_returns.append(ep_return / max(1, sum(dones)))

        if (itr + 1) % 20 == 0:
            print(f"  Iteration {itr+1:3d} | mean step return: {ep_return/n_steps:.3f}")

    return episode_returns


# ---------------------------------------------------------------------------
# Demo: policy ratio and clipping illustration
# ---------------------------------------------------------------------------

def demo_clipping():
    """Show how PPO clip prevents large policy updates."""
    ratios = np.linspace(0.5, 1.8, 200)
    adv = 1.0      # positive advantage
    eps = 0.2
    clipped = np.clip(ratios, 1 - eps, 1 + eps)
    ppo_obj = np.minimum(ratios * adv, clipped * adv)
    pg_obj = ratios * adv       # plain PG without clip

    print("=== PPO Clipping Demo ===")
    print(f"  Advantage = {adv}, epsilon = {eps}")
    print(f"  At ratio=1.5 (too large update):")
    print(f"    PG objective : {1.5 * adv:.3f}")
    print(f"    PPO objective: {min(1.5 * adv, np.clip(1.5, 1-eps, 1+eps) * adv):.3f}  (clipped!)")
    print(f"  At ratio=0.9 (conservative update):")
    print(f"    PG objective : {0.9 * adv:.3f}")
    print(f"    PPO objective: {min(0.9 * adv, np.clip(0.9, 1-eps, 1+eps) * adv):.3f}  (unclipped)")
    return ratios, pg_obj, ppo_obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Proximal Policy Optimization (PPO) — Standalone Demo")
    print("=" * 60)

    # 1. Clipping demo
    demo_clipping()

    # 2. Training
    print("\n=== Training PPO on CartPole (no gym) ===")
    t0 = time.time()
    returns = train_ppo(n_iterations=80, n_steps=256, k_epochs=4, clip_eps=0.2)
    elapsed = time.time() - t0

    print(f"\nTraining complete in {elapsed:.1f}s")
    final_returns = returns[-20:]
    print(f"Final 20-iter avg step-return: {np.mean(final_returns):.4f}")

    # 3. Hyperparameter comparison (clip_eps)
    print("\n=== Clip-epsilon sensitivity ===")
    for eps in [0.1, 0.2, 0.3]:
        r = train_ppo(n_iterations=40, n_steps=128, k_epochs=4, clip_eps=eps, seed=0)
        print(f"  clip_eps={eps:.1f} | avg final-10 step-return: {np.mean(r[-10:]):.4f}")

    print("\nDone.")
