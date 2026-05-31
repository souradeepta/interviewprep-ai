"""
Actor-Critic — Shared Network with GAE Advantage Estimation (A2C)
=================================================================
Implements A2C (synchronous Advantage Actor-Critic) using a shared trunk
with separate policy and value heads. GAE advantage estimation reduces
variance compared to vanilla REINFORCE.

CartPole physics are hand-coded — no gym or torch required.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# CartPole simulation
# ---------------------------------------------------------------------------

def cartpole_step(state: np.ndarray, action: int, dt: float = 0.02) -> Tuple[np.ndarray, float, bool]:
    x, x_dot, theta, theta_dot = state
    force = 10.0 if action == 1 else -10.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    temp = (force + 0.05 * theta_dot**2 * sin_t) / 1.1
    theta_acc = (9.8 * sin_t - cos_t * temp) / (0.5 * (4/3 - 0.1 * cos_t**2 / 1.1))
    x_acc = temp - 0.05 * theta_acc * cos_t / 1.1
    x += dt * x_dot
    x_dot += dt * x_acc
    theta += dt * theta_dot
    theta_dot += dt * theta_acc
    done = bool(abs(x) > 2.4 or abs(theta) > 0.2095)
    reward = 1.0 if not done else 0.0
    return np.array([x, x_dot, theta, theta_dot]), reward, done


def cartpole_reset(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-0.05, 0.05, size=4)


# ---------------------------------------------------------------------------
# Shared Actor-Critic Network (two-layer MLP, numpy)
# ---------------------------------------------------------------------------

class ActorCritic:
    """
    Shared two-layer MLP trunk with separate policy and value heads.

    Architecture:
      Input (4) -> Hidden (64, ReLU) -> shared features
                 -> Policy head: softmax(W_pi * h + b_pi)   (n_actions,)
                 -> Value head:  linear(W_v * h + b_v)       (scalar)
    """

    def __init__(
        self,
        obs_dim: int = 4,
        n_actions: int = 2,
        hidden: int = 64,
        actor_lr: float = 3e-3,
        critic_lr: float = 1e-2,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        # Xavier init for shared trunk
        scale1 = np.sqrt(2.0 / obs_dim)
        self.W1 = rng.normal(0, scale1, (obs_dim, hidden))
        self.b1 = np.zeros(hidden)

        # Policy head (actor)
        scale_pi = np.sqrt(2.0 / hidden)
        self.W_pi = rng.normal(0, scale_pi, (hidden, n_actions))
        self.b_pi = np.zeros(n_actions)

        # Value head (critic)
        scale_v = np.sqrt(2.0 / hidden)
        self.W_v = rng.normal(0, scale_v, (hidden, 1))
        self.b_v = np.zeros(1)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.n_actions = n_actions

    def _shared_forward(self, s: np.ndarray) -> np.ndarray:
        """Compute shared hidden features (ReLU)."""
        return np.maximum(0, s @ self.W1 + self.b1)

    def policy(self, s: np.ndarray) -> np.ndarray:
        """Return action probabilities: pi(a|s)."""
        h = self._shared_forward(s)
        logits = h @ self.W_pi + self.b_pi
        logits -= logits.max()   # stability
        p = np.exp(logits)
        return p / p.sum()

    def value(self, s: np.ndarray) -> float:
        """Return state value estimate V(s)."""
        h = self._shared_forward(s)
        return float(h @ self.W_v + self.b_v)

    def act(self, s: np.ndarray, rng: np.random.Generator) -> Tuple[int, float]:
        """Sample action and return (action, log_prob)."""
        p = self.policy(s)
        a = int(rng.choice(self.n_actions, p=p))
        return a, float(np.log(p[a] + 1e-10))

    def update_critic(self, states: np.ndarray, returns: np.ndarray) -> float:
        """Update critic (value head + trunk) to minimise MSE(V(s) - G_t)."""
        batch_size = len(states)
        h = np.maximum(0, states @ self.W1 + self.b1)   # (B, hidden)
        preds = (h @ self.W_v + self.b_v).squeeze()     # (B,)
        errs = preds - returns                           # (B,)
        loss = float((errs**2).mean())

        # Gradient w.r.t. W_v, b_v
        dW_v = h.T @ errs[:, None] / batch_size * 2
        db_v = errs.mean() * 2

        # Gradient through shared trunk
        dh = errs[:, None] @ self.W_v.T * 2 / batch_size   # (B, hidden)
        dh_relu = dh * (h > 0).astype(float)
        dW1_v = states.T @ dh_relu                         # critic contribution

        self.W_v -= self.critic_lr * dW_v
        self.b_v -= self.critic_lr * db_v
        self.W1 -= self.critic_lr * dW1_v / batch_size

        return loss

    def update_actor(self, states: np.ndarray, actions: np.ndarray,
                     advantages: np.ndarray) -> float:
        """Update actor (policy head + trunk) via policy gradient."""
        batch_size = len(states)
        total_gW_pi = np.zeros_like(self.W_pi)
        total_gb_pi = np.zeros_like(self.b_pi)
        total_gW1 = np.zeros_like(self.W1)

        for i in range(batch_size):
            s, a, adv = states[i], int(actions[i]), advantages[i]
            h = self._shared_forward(s)
            p = self.policy(s)

            # Gradient of log pi(a|s) w.r.t. logits
            d_logits = -p.copy()
            d_logits[a] += 1.0

            # Gradient w.r.t. W_pi, b_pi (ascent)
            total_gW_pi += adv * np.outer(h, d_logits)
            total_gb_pi += adv * d_logits

            # Gradient through shared trunk
            dh = (d_logits @ self.W_pi.T) * adv
            dh_relu = dh * (h > 0).astype(float)
            total_gW1 += np.outer(s, dh_relu)

        # Gradient ascent on actor
        self.W_pi += self.actor_lr * total_gW_pi / batch_size
        self.b_pi += self.actor_lr * total_gb_pi / batch_size
        self.W1 += self.actor_lr * total_gW1 / batch_size

        return float(np.linalg.norm(total_gW_pi))


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
    """
    Compute GAE advantages and discounted returns.
    A_t = sum_{k=0}^{inf} (gamma*lambda)^k * delta_{t+k}
    where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if (t + 1 < len(values) and not dones[t]) else 0.0
        delta = rewards[t] + gamma * next_val * (1 - float(dones[t])) - values[t]
        gae = delta + gamma * lam * (1 - float(dones[t])) * gae
        advantages[t] = gae
    returns = advantages + np.array(values[:T])
    return advantages, returns


# ---------------------------------------------------------------------------
# A2C Agent
# ---------------------------------------------------------------------------

class A2CAgent:
    """A2C: synchronous rollout collection + GAE advantage estimation."""

    def __init__(
        self,
        n_steps: int = 128,
        gamma: float = 0.99,
        lam: float = 0.95,
        actor_lr: float = 3e-3,
        critic_lr: float = 1e-2,
        entropy_coef: float = 0.01,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.net = ActorCritic(actor_lr=actor_lr, critic_lr=critic_lr, seed=seed)
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef

    def collect_rollout(self) -> Dict:
        """Collect n_steps transitions."""
        state = cartpole_reset(self.rng)
        states, actions, rewards, dones, values = [], [], [], [], []

        for _ in range(self.n_steps):
            v = self.net.value(state)
            a, _ = self.net.act(state, self.rng)
            next_state, r, done = cartpole_step(state, a)

            states.append(state.copy())
            actions.append(a)
            rewards.append(r)
            dones.append(done)
            values.append(v)

            state = cartpole_reset(self.rng) if done else next_state

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": rewards,
            "dones": dones,
            "values": values,
        }

    def update(self, rollout: Dict) -> Dict:
        """Compute GAE, update actor and critic. Returns diagnostics."""
        advantages, returns = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"],
            gamma=self.gamma, lam=self.lam,
        )
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        critic_loss = self.net.update_critic(rollout["states"], returns)
        actor_grad = self.net.update_actor(rollout["states"], rollout["actions"], advantages)

        return {"critic_loss": critic_loss, "actor_grad": actor_grad,
                "mean_return": float(np.sum(rollout["rewards"]))}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_a2c(
    n_iterations: int = 100,
    n_steps: int = 128,
    seed: int = 42,
    verbose: bool = True,
) -> List[float]:
    """Train A2C for n_iterations rollouts. Returns mean step returns."""
    agent = A2CAgent(n_steps=n_steps, seed=seed)
    returns_log = []

    for itr in range(n_iterations):
        rollout = agent.collect_rollout()
        info = agent.update(rollout)
        returns_log.append(info["mean_return"] / n_steps)

        if verbose and (itr + 1) % 25 == 0:
            mean20 = np.mean(returns_log[-20:])
            print(f"  Iter {itr+1:4d} | step_return={info['mean_return']/n_steps:.3f} | "
                  f"mean20={mean20:.3f} | critic_loss={info['critic_loss']:.4f}")

    return returns_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Actor-Critic (A2C) — Shared Network + GAE")
    print("=" * 60)

    print("\n--- Single forward pass ---")
    net = ActorCritic()
    s = np.array([0.1, -0.05, 0.02, 0.1])
    rng = np.random.default_rng(0)
    p = net.policy(s)
    v = net.value(s)
    print(f"State: {s}")
    print(f"Policy probs: {p}")
    print(f"Value estimate: {v:.4f}")

    print("\n--- GAE demo (5-step trajectory) ---")
    rewards = [1.0, 1.0, 1.0, 1.0, 0.0]
    values  = [0.9, 0.85, 0.8, 0.7, 0.6]
    dones   = [False, False, False, False, True]
    advs, rets = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    for t in range(5):
        print(f"  t={t}: r={rewards[t]}, V={values[t]:.2f}, "
              f"Adv={advs[t]:.4f}, Return={rets[t]:.4f}")

    print("\n--- Training A2C on CartPole ---")
    t0 = time.time()
    returns = train_a2c(n_iterations=100, n_steps=128, verbose=True, seed=42)
    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")
    print(f"Final 20-iter avg step-return: {np.mean(returns[-20:]):.4f}")

    print("\n--- Lambda sensitivity (GAE) ---")
    for lam in [0.0, 0.5, 0.95, 1.0]:
        agent = A2CAgent(lam=lam, seed=7)
        rets = []
        for _ in range(50):
            ro = agent.collect_rollout()
            info = agent.update(ro)
            rets.append(info["mean_return"] / 128)
        print(f"  lambda={lam:.2f} | final mean-10 step-return: {np.mean(rets[-10:]):.4f}")

    print("\nDone.")
