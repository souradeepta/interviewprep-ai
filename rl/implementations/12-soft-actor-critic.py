"""
Soft Actor-Critic (SAC) - Standalone Implementation
====================================================
Entropy-regularised off-policy RL. All environments hand-coded in numpy.
No gym/gymnasium. Uses only numpy.

Key ideas: twin critics (min for target), soft value function, auto-tune temperature.
"""

import numpy as np
import time
from collections import deque
from typing import Tuple, List, Dict


# ---------------------------------------------------------------------------
# 1D Continuous Control Environment: Particle
# ---------------------------------------------------------------------------

def particle_step(state: float, action: float, dt: float = 0.1) -> Tuple[float, float, bool]:
    """
    1D particle: x_next = x + a + noise. Goal: reach x=0.
    state: position x in [-3, 3]
    action: force in [-1, 1]
    """
    action = np.clip(action, -1.0, 1.0)
    noise = np.random.normal(0, 0.01)
    state_new = state + action * dt + noise
    state_new = np.clip(state_new, -3.0, 3.0)
    reward = -abs(state_new)       # dense negative distance reward
    done = bool(abs(state_new) < 0.05)  # reached goal
    return state_new, reward, done


def particle_reset() -> float:
    """Random start position."""
    return float(np.random.uniform(-2.0, 2.0))


# ---------------------------------------------------------------------------
# Pendulum environment
# ---------------------------------------------------------------------------

def pendulum_step(state: np.ndarray, u: float, dt: float = 0.05) -> Tuple[np.ndarray, float, bool]:
    """Pendulum: theta'' = -(3g/2l)*sin(theta) + u/(ml^2)."""
    g, l, m = 9.8, 1.0, 1.0
    theta, theta_dot = state
    u = np.clip(u, -2.0, 2.0)
    theta_ddot = -(3 * g / (2 * l)) * np.sin(theta) + u / (m * l**2)
    theta_dot = theta_dot + dt * theta_ddot
    theta = theta + dt * theta_dot
    reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * u**2)
    done = bool(abs(theta) > np.pi)
    return np.array([theta, theta_dot]), reward, done


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Off-policy experience replay buffer."""

    def __init__(self, capacity: int = 50000, obs_dim: int = 1, act_dim: int = 1):
        self.cap = capacity
        self.ptr = 0
        self.size = 0
        self.S  = np.zeros((capacity, obs_dim))
        self.A  = np.zeros((capacity, act_dim))
        self.R  = np.zeros(capacity)
        self.NS = np.zeros((capacity, obs_dim))
        self.D  = np.zeros(capacity)

    def store(self, s, a, r, ns, d):
        i = self.ptr % self.cap
        self.S[i]  = np.atleast_1d(s)
        self.A[i]  = np.atleast_1d(a)
        self.R[i]  = r
        self.NS[i] = np.atleast_1d(ns)
        self.D[i]  = float(d)
        self.ptr += 1
        self.size = min(self.ptr, self.cap)

    def sample(self, batch_size: int = 256) -> Dict:
        idx = np.random.choice(self.size, batch_size, replace=False)
        return {"s": self.S[idx], "a": self.A[idx], "r": self.R[idx],
                "ns": self.NS[idx], "d": self.D[idx]}


# ---------------------------------------------------------------------------
# Linear Gaussian Actor (policy)
# ---------------------------------------------------------------------------

class LinearGaussianActor:
    """
    Gaussian policy pi(a|s) = N(Ws+b, exp(log_std)^2).
    """

    def __init__(self, obs_dim: int, act_dim: int = 1, lr: float = 3e-3):
        self.W = np.zeros((obs_dim, act_dim))
        self.b = np.zeros(act_dim)
        self.log_std = np.zeros(act_dim)
        self.lr = lr

    def mean(self, s: np.ndarray) -> np.ndarray:
        return s.reshape(1, -1) @ self.W + self.b

    def std(self) -> np.ndarray:
        return np.exp(np.clip(self.log_std, -4, 2))

    def sample(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (action, log_prob)."""
        mu = self.mean(s).flatten()
        sigma = self.std()
        noise = np.random.standard_normal(mu.shape)
        action = mu + sigma * noise
        log_prob = -0.5 * ((noise)**2) - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        return action, log_prob.sum()

    def log_prob(self, s: np.ndarray, a: np.ndarray) -> float:
        mu = self.mean(s).flatten()
        sigma = self.std()
        return float(np.sum(-0.5 * ((a - mu) / sigma)**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)))


# ---------------------------------------------------------------------------
# Linear Q-function
# ---------------------------------------------------------------------------

class LinearQFunction:
    """Q(s, a) = [s, a] @ W + b."""

    def __init__(self, obs_dim: int, act_dim: int = 1, lr: float = 1e-2):
        input_dim = obs_dim + act_dim
        self.W = np.zeros(input_dim)
        self.b = 0.0
        self.lr = lr

    def predict(self, s: np.ndarray, a: np.ndarray) -> float:
        sa = np.concatenate([np.atleast_1d(s), np.atleast_1d(a)])
        return float(sa @ self.W + self.b)

    def predict_batch(self, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        SA = np.concatenate([S, A], axis=1)
        return SA @ self.W + self.b

    def update(self, S: np.ndarray, A: np.ndarray, targets: np.ndarray) -> float:
        SA = np.concatenate([S, A], axis=1)
        preds = SA @ self.W + self.b
        errors = preds - targets
        self.W -= self.lr * SA.T @ errors / len(errors)
        self.b -= self.lr * errors.mean()
        return float((errors**2).mean())


# ---------------------------------------------------------------------------
# SAC Training
# ---------------------------------------------------------------------------

class SACAgent:
    """SAC with twin critics and auto-tuned temperature."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_entropy: float = -1.0,
        auto_alpha: bool = True,
        actor_lr: float = 3e-3,
        critic_lr: float = 1e-2,
        alpha_lr: float = 1e-3,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_entropy = target_entropy
        self.auto_alpha = auto_alpha
        self.log_alpha = np.log(alpha)
        self.alpha_lr = alpha_lr

        self.actor = LinearGaussianActor(obs_dim, act_dim, actor_lr)
        # Twin critics to reduce overestimation
        self.q1 = LinearQFunction(obs_dim, act_dim, critic_lr)
        self.q2 = LinearQFunction(obs_dim, act_dim, critic_lr)
        # Target critics (soft-updated copies)
        self.q1_target = LinearQFunction(obs_dim, act_dim, critic_lr)
        self.q2_target = LinearQFunction(obs_dim, act_dim, critic_lr)
        self.q1_target.W = self.q1.W.copy()
        self.q2_target.W = self.q2.W.copy()

    def select_action(self, s: np.ndarray) -> np.ndarray:
        a, _ = self.actor.sample(s)
        return np.clip(a, -1.0, 1.0)

    def _soft_update(self):
        """Polyak averaging: target_W = tau*W + (1-tau)*target_W."""
        self.q1_target.W = self.tau * self.q1.W + (1 - self.tau) * self.q1_target.W
        self.q2_target.W = self.tau * self.q2.W + (1 - self.tau) * self.q2_target.W

    def update(self, batch: Dict, actor_update: bool = True) -> Dict:
        S, A, R, NS, D = batch["s"], batch["a"], batch["r"], batch["ns"], batch["d"]

        # 1. Compute soft target Q-values for next states
        next_a_list, next_lp_list = [], []
        for i in range(len(NS)):
            na, nlp = self.actor.sample(NS[i])
            next_a_list.append(np.clip(na, -1.0, 1.0))
            next_lp_list.append(nlp)
        NA = np.array(next_a_list)
        nlps = np.array(next_lp_list)

        q1_next = self.q1_target.predict_batch(NS, NA)
        q2_next = self.q2_target.predict_batch(NS, NA)
        # Use min of twin critics to reduce overestimation
        q_next = np.minimum(q1_next, q2_next)
        # Soft Bellman target: adds entropy bonus
        targets = R + self.gamma * (1 - D) * (q_next - self.alpha * nlps)

        # 2. Update twin critics
        loss_q1 = self.q1.update(S, A, targets)
        loss_q2 = self.q2.update(S, A, targets)

        # 3. Update actor (minimise alpha*log_pi - Q)
        actor_loss = 0.0
        if actor_update:
            for i in range(len(S)):
                a_new, lp = self.actor.sample(S[i])
                a_new = np.clip(a_new, -1.0, 1.0)
                q1_val = self.q1.predict(S[i], a_new)
                q2_val = self.q2.predict(S[i], a_new)
                q_val = min(q1_val, q2_val)
                # Actor tries to maximise Q - alpha * log_pi
                obj = -(q_val - self.alpha * lp)
                actor_loss += obj
                # Gradient for actor mean (simplified)
                mu = self.actor.mean(S[i]).flatten()
                sigma = self.actor.std()
                d_mean = self.actor.lr * (-np.sign(q_val) * sigma)
                self.actor.W -= np.outer(S[i], d_mean)
                self.actor.b -= d_mean

            actor_loss /= len(S)

        # 4. Auto-tune temperature
        if self.auto_alpha:
            # Gradient: d/d(log_alpha) E[-(log_pi + H_target)]
            mean_lp = np.mean([self.actor.log_prob(S[i], A[i]) for i in range(min(16, len(S)))])
            alpha_grad = -(mean_lp + self.target_entropy)
            self.log_alpha += self.alpha_lr * alpha_grad
            self.alpha = float(np.exp(np.clip(self.log_alpha, -5, 2)))

        # 5. Soft update target networks
        self._soft_update()

        return {"q_loss": (loss_q1 + loss_q2) / 2, "actor_loss": actor_loss, "alpha": self.alpha}


def train_sac(
    n_steps: int = 5000,
    batch_size: int = 128,
    warmup_steps: int = 500,
    seed: int = 42,
    auto_alpha: bool = True,
) -> List[float]:
    """Train SAC on the 1D particle task. Returns per-step rewards."""
    np.random.seed(seed)
    buffer = ReplayBuffer(capacity=20000, obs_dim=1, act_dim=1)
    agent = SACAgent(obs_dim=1, act_dim=1, auto_alpha=auto_alpha)

    state = particle_reset()
    rewards = []

    for step in range(n_steps):
        if step < warmup_steps:
            action = np.array([np.random.uniform(-1, 1)])
        else:
            action = agent.select_action(np.array([state]))

        next_state, reward, done = particle_step(state, action[0])
        buffer.store(state, action, reward, next_state, done)
        state = particle_reset() if done else next_state
        rewards.append(reward)

        if step >= warmup_steps and buffer.size >= batch_size:
            batch = buffer.sample(batch_size)
            agent.update(batch)

    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Soft Actor-Critic (SAC) — Standalone Demo")
    print("=" * 60)

    # 1. Entropy effect demo
    print("\n=== Entropy Regularisation Demo ===")
    probs = np.array([0.7, 0.2, 0.05, 0.05])
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    print(f"  Deterministic-ish policy {probs}: H = {entropy:.3f}")
    uniform = np.ones(4) / 4
    entropy_u = -np.sum(uniform * np.log(uniform + 1e-8))
    print(f"  Uniform policy {uniform}: H = {entropy_u:.3f}")
    print(f"  Entropy bonus encourages more uniform (exploratory) policies")

    # 2. SAC training
    print("\n=== Training SAC on 1D Particle (auto-alpha) ===")
    t0 = time.time()
    rewards = train_sac(n_steps=3000, seed=42, auto_alpha=True)
    print(f"Done in {time.time()-t0:.1f}s")
    w = 200
    smoothed = np.convolve(rewards, np.ones(w)/w, 'valid')
    print(f"Final {w}-step avg reward: {smoothed[-1]:.4f} (closer to 0 is better)")

    # 3. alpha comparison
    print("\n=== Temperature Sensitivity ===")
    for alpha_str, auto in [("auto", True), ("fixed", False)]:
        r = train_sac(n_steps=2000, seed=0, auto_alpha=auto)
        print(f"  alpha={alpha_str}: final-200 avg = {np.mean(r[-200:]):.4f}")

    print("\nDone.")
