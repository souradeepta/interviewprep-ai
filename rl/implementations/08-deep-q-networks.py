"""
Deep Q-Networks (DQN) — Replay Buffer, DQN, Target Network
===========================================================
Implements the core DQN algorithm from Mnih et al. (2015) without gym.
CartPole physics are hand-coded so the demo runs anywhere with only numpy.

Components:
  - ReplayBuffer with fixed maxlen (circular buffer)
  - DQN with two-layer MLP (numpy, no torch required for core logic)
  - DQNAgent with epsilon-greedy, learn(), update_target()

A PyTorch variant is shown in comments for reference.
"""

import numpy as np
import time
from typing import Tuple, List, Optional, Dict
from collections import deque


# ---------------------------------------------------------------------------
# CartPole simulation (no gym required)
# ---------------------------------------------------------------------------

def cartpole_step(state: np.ndarray, action: int, dt: float = 0.02) -> Tuple[np.ndarray, float, bool]:
    """One step of CartPole physics. action: 0=left, 1=right."""
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
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size circular replay buffer for experience replay.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, maxlen: int = 10_000):
        self.maxlen = maxlen
        self.buffer: deque = deque(maxlen=maxlen)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition. Oldest transitions are dropped when full."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Sample a random batch of transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} < {batch_size} samples")
        indices = rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN: two-layer MLP with ReLU (numpy only, forward and backprop)
# ---------------------------------------------------------------------------

class DQN:
    """
    Shallow two-layer MLP: input -> hidden -> Q-values.
    Parameters: W1, b1, W2, b2.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Xavier initialisation for stable gradients
        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, scale1, (state_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, scale2, (hidden, n_actions))
        self.b2 = np.zeros(n_actions)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (Q-values, hidden activations)."""
        h = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        q = h @ self.W2 + self.b2
        return q, h

    def predict(self, x: np.ndarray) -> np.ndarray:
        q, _ = self.forward(x)
        return q

    def update(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        actions: np.ndarray,
        lr: float = 1e-3,
    ) -> float:
        """
        One gradient step: minimise MSE(Q(s,a) - target).
        Returns mean batch loss.
        """
        batch_size = x.shape[0]
        q, h = self.forward(x)

        # Compute per-sample loss gradient w.r.t. Q(s, a)
        loss_grad = np.zeros_like(q)
        loss = 0.0
        for i in range(batch_size):
            a = actions[i]
            err = q[i, a] - targets[i]
            loss += err**2
            loss_grad[i, a] = 2 * err / batch_size

        # Backprop through W2, b2
        dW2 = h.T @ loss_grad / batch_size
        db2 = loss_grad.mean(axis=0)

        # Backprop through W1, b1 (ReLU gradient)
        dh = loss_grad @ self.W2.T
        dh_relu = dh * (h > 0).astype(float)
        dW1 = x.T @ dh_relu / batch_size
        db1 = dh_relu.mean(axis=0)

        # Gradient descent
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        return float(loss / batch_size)

    def copy_from(self, other: "DQN") -> None:
        """Hard copy weights from another DQN (target network update)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    DQN agent combining experience replay and a fixed target network.
    """

    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 2,
        hidden: int = 64,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_update_freq: int = 50,  # steps between hard target updates
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Online network + frozen target network
        self.online = DQN(state_dim, n_actions, hidden, seed=seed)
        self.target = DQN(state_dim, n_actions, hidden, seed=seed)
        self.target.copy_from(self.online)

        self.buffer = ReplayBuffer(maxlen=buffer_size)

        # Epsilon decay parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

    def epsilon(self) -> float:
        """Exponential epsilon decay."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.step_count / self.epsilon_decay)

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))
        q = self.online.predict(state[np.newaxis, :])[0]
        return int(q.argmax())

    def learn(self) -> Optional[float]:
        """
        Sample a minibatch, compute Bellman targets, do one gradient step.
        Returns loss or None if buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size, self.rng)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Target: r + gamma * max_a Q_target(s', a)   (0 if done)
        q_next = self.target.predict(next_states)
        targets = rewards + self.gamma * q_next.max(axis=1) * (1 - dones)

        loss = self.online.update(states, targets, actions, lr=self.lr)
        return loss

    def update_target(self) -> None:
        """Hard copy online weights to target network."""
        self.target.copy_from(self.online)

    def push(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_dqn(
    n_episodes: int = 200,
    max_steps: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> List[float]:
    """Train DQN on hand-coded CartPole. Returns episode returns."""
    rng = np.random.default_rng(seed)
    agent = DQNAgent(seed=seed)
    episode_returns: List[float] = []

    for ep in range(n_episodes):
        state = cartpole_reset(rng)
        ep_return = 0.0
        loss_sum = 0.0
        loss_count = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = cartpole_step(state, action)
            agent.push(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                loss_sum += loss
                loss_count += 1

            ep_return += reward
            state = next_state
            if done:
                break

        episode_returns.append(ep_return)
        mean_loss = loss_sum / max(loss_count, 1)

        if verbose and (ep + 1) % 50 == 0:
            last_returns = episode_returns[-20:]
            print(f"  Ep {ep+1:4d} | return={ep_return:6.1f} | "
                  f"mean-20={np.mean(last_returns):6.1f} | "
                  f"eps={agent.epsilon():.3f} | loss={mean_loss:.4f}")

    return episode_returns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Deep Q-Networks (DQN) — CartPole (no gym)")
    print("=" * 60)

    print("\n--- Replay Buffer Demo ---")
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(maxlen=100)
    for _ in range(50):
        s = rng.random(4).astype(np.float32)
        a = int(rng.integers(2))
        r = float(rng.random())
        s2 = rng.random(4).astype(np.float32)
        buf.push(s, a, r, s2, False)
    batch = buf.sample(16, rng)
    print(f"Buffer size: {len(buf)}, sampled batch states shape: {batch['states'].shape}")

    print("\n--- DQN Forward Pass ---")
    net = DQN(state_dim=4, n_actions=2, hidden=64, seed=0)
    dummy = np.random.randn(1, 4).astype(np.float32)
    q_out = net.predict(dummy)
    print(f"Q-values for dummy state: {q_out[0]}")

    print("\n--- Training DQN on CartPole ---")
    t0 = time.time()
    returns = train_dqn(n_episodes=200, verbose=True)
    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")
    print(f"Final 20-episode avg return: {np.mean(returns[-20:]):.1f}")
    print(f"Best episode return: {max(returns):.1f}")

    print("\n--- Hyperparameter Sensitivity (target_update_freq) ---")
    for freq in [10, 50, 200]:
        r_list = train_dqn(n_episodes=100, verbose=False, seed=7)
        print(f"  target_update_freq={freq:4d} | final-20 return: {np.mean(r_list[-20:]):.1f}")

    print("\nDone.")
