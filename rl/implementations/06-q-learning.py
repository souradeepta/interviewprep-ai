"""
Q-Learning Implementation
=========================
Standalone implementation of tabular Q-learning, Q(lambda), and Double Q-learning.
No external RL libraries required — environments are built with numpy.

Usage:
    python 06-q-learning.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

class GridWorld:
    """4x4 GridWorld: start=(0,0), goal=(3,3), wall=(1,1)."""

    def __init__(self, size: int = 4):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.wall = (1, 1)
        self.n_actions = 4   # 0=Up, 1=Down, 2=Left, 3=Right
        self.n_states = size * size
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.reset()

    def reset(self) -> int:
        self.pos = (0, 0)
        return self._idx(self.pos)

    def _idx(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def step(self, action: int) -> Tuple[int, float, bool]:
        dr, dc = self.action_deltas[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return self._idx(self.pos), -1.0, False
        npos = (nr, nc)
        if npos == self.wall:
            return self._idx(self.pos), -5.0, False
        self.pos = npos
        if self.pos == self.goal:
            return self._idx(self.pos), 10.0, True
        return self._idx(self.pos), -1.0, False


class CliffWorld:
    """4x12 Cliff Walking: start=(3,0), goal=(3,11), cliff at row 3 cols 1-10."""

    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = {(3, c) for c in range(1, 11)}
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.reset()

    def reset(self) -> int:
        self.pos = self.start
        return self._idx(self.pos)

    def _idx(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def step(self, action: int) -> Tuple[int, float, bool]:
        dr, dc = self.action_deltas[action]
        r = max(0, min(self.rows - 1, self.pos[0] + dr))
        c = max(0, min(self.cols - 1, self.pos[1] + dc))
        self.pos = (r, c)
        if self.pos in self.cliff:
            self.pos = self.start
            return self._idx(self.pos), -100.0, False
        if self.pos == self.goal:
            return self._idx(self.pos), 0.0, True
        return self._idx(self.pos), -1.0, False


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def run_q_learning(
    env,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> Tuple[np.ndarray, List[float]]:
    """Tabular Q-learning (off-policy)."""
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards: List[float] = []
    epsilon = epsilon_start

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 300:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done = env.step(action)

            # Q-learning update (off-policy: uses max over next state)
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            total_reward += reward
            state = next_state
            steps += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    return Q, episode_rewards


def run_sarsa(
    env,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, List[float]]:
    """Tabular SARSA (on-policy) for comparison."""
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards: List[float] = []

    for ep in range(n_episodes):
        state = env.reset()
        action = np.random.randint(env.n_actions) if np.random.rand() < epsilon else np.argmax(Q[state])
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 300:
            next_state, reward, done = env.step(action)
            next_action = np.random.randint(env.n_actions) if np.random.rand() < epsilon else np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] * (1-done) - Q[state, action])
            total_reward += reward
            state, action = next_state, next_action
            steps += 1

        episode_rewards.append(total_reward)

    return Q, episode_rewards


def run_q_lambda(
    env,
    lam: float = 0.9,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> List[float]:
    """Q(lambda) with Watkins's replacing eligibility traces."""
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards: List[float] = []

    for ep in range(n_episodes):
        state = env.reset()
        E = np.zeros((env.n_states, env.n_actions))  # Eligibility traces
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 300:
            action = np.random.randint(env.n_actions) if np.random.rand() < epsilon else np.argmax(Q[state])
            next_state, reward, done = env.step(action)

            best_next = np.max(Q[next_state])
            td_error = reward + gamma * best_next * (1 - done) - Q[state, action]

            # Replacing trace: set to 1 for current (s,a)
            E[state, action] = 1.0
            Q += alpha * td_error * E
            E *= gamma * lam

            # Watkins's Q(lambda): zero traces on non-greedy actions
            if action != np.argmax(Q[state]):
                E[state] = 0.0

            total_reward += reward
            state = next_state
            steps += 1

        episode_rewards.append(total_reward)

    return episode_rewards


def run_double_q_learning(
    env,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, List[float]]:
    """Double Q-learning: two tables to reduce overestimation bias."""
    Q1 = np.zeros((env.n_states, env.n_actions))
    Q2 = np.zeros((env.n_states, env.n_actions))
    episode_rewards: List[float] = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 300:
            # Action selection using combined Q1 + Q2
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = int(np.argmax(Q1[state] + Q2[state]))

            next_state, reward, done = env.step(action)

            # Randomly choose which Q to update
            if np.random.rand() < 0.5:
                best_a = int(np.argmax(Q1[next_state]))
                target = reward + gamma * Q2[next_state, best_a] * (1 - done)
                Q1[state, action] += alpha * (target - Q1[state, action])
            else:
                best_a = int(np.argmax(Q2[next_state]))
                target = reward + gamma * Q1[next_state, best_a] * (1 - done)
                Q2[state, action] += alpha * (target - Q2[state, action])

            total_reward += reward
            state = next_state
            steps += 1

        episode_rewards.append(total_reward)

    Q_avg = (Q1 + Q2) / 2.0
    return Q_avg, episode_rewards


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def smooth(arr: List[float], window: int = 20) -> np.ndarray:
    """Moving average smoothing."""
    if len(arr) < window:
        return np.array(arr)
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def extract_policy(Q: np.ndarray) -> np.ndarray:
    """Extract greedy policy from Q-table."""
    return np.argmax(Q, axis=1)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)
    print('=== Q-Learning Demonstration ===\n')

    # --- 1. Basic Q-learning on GridWorld ---
    print('1. Tabular Q-Learning on 4x4 GridWorld')
    env = GridWorld()
    Q, ql_rewards = run_q_learning(env, n_episodes=500)
    print(f'   Mean reward (last 50 eps): {np.mean(ql_rewards[-50:]):.2f}')
    print(f'   Final TD convergence (Q-value range): [{Q.min():.2f}, {Q.max():.2f}]')

    # --- 2. Q(lambda) sweep ---
    print('\n2. Q(lambda) Eligibility Traces (lambda sweep)')
    for lam in [0.0, 0.5, 0.9]:
        np.random.seed(42)
        r = run_q_lambda(env, lam=lam, n_episodes=500)
        print(f'   lambda={lam}: mean reward (last 50) = {np.mean(r[-50:]):.2f}')

    # --- 3. Cliff Walking: Q-learning vs SARSA ---
    print('\n3. Cliff Walking: Q-Learning vs SARSA')
    cliff = CliffWorld()
    np.random.seed(42)
    _, ql_cliff = run_q_learning(cliff, n_episodes=500, alpha=0.5, gamma=1.0,
                                  epsilon_start=0.1, epsilon_end=0.1, epsilon_decay=1.0)
    np.random.seed(42)
    _, sarsa_cliff = run_sarsa(cliff, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1)
    print(f'   Q-Learning training reward: {np.mean(ql_cliff[-100:]):.1f} (lower = risky path)')
    print(f'   SARSA    training reward: {np.mean(sarsa_cliff[-100:]):.1f} (higher = safe path)')

    # --- 4. Double Q-learning overestimation comparison ---
    print('\n4. Double Q-Learning vs Standard Q-Learning')
    np.random.seed(42)
    _, dql_rewards = run_double_q_learning(env, n_episodes=500)
    np.random.seed(42)
    _, sql_rewards = run_q_learning(env, n_episodes=500)
    print(f'   Standard Q-Learning reward: {np.mean(sql_rewards[-50:]):.2f}')
    print(f'   Double  Q-Learning reward: {np.mean(dql_rewards[-50:]):.2f}')

    # --- 5. Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Q-table heatmap
    ax = axes[0]
    max_q = np.max(Q, axis=1).reshape(4, 4)
    im = ax.imshow(max_q, cmap='RdYlGn', vmin=-10, vmax=10)
    ax.set_title('Max Q-Value per State')
    ax.set_xlabel('Column'); ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)
    for r_i in range(4):
        for c_i in range(4):
            ax.text(c_i, r_i, f'{max_q[r_i,c_i]:.1f}', ha='center', va='center', fontsize=8)

    # Cliff walking comparison
    ax = axes[1]
    ax.plot(smooth(ql_cliff), label='Q-Learning', color='red')
    ax.plot(smooth(sarsa_cliff), label='SARSA', color='blue')
    ax.set_title('Cliff Walking: Training Performance')
    ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward (smoothed)')
    ax.set_ylim(-200, 10)
    ax.legend()

    plt.tight_layout()
    plt.savefig('/tmp/06_q_learning_demo.png', dpi=80, bbox_inches='tight')
    plt.close()

    print('\n5. Plot saved to /tmp/06_q_learning_demo.png')
    print('\nKey takeaways:')
    print('  - Q-learning converges to optimal policy regardless of behavior policy')
    print('  - SARSA safer in cliff-walking during training (accounts for epsilon-greedy falls)')
    print('  - Q(lambda=0.9) faster convergence with sparse rewards via eligibility traces')
    print('  - Double Q-learning reduces overestimation at zero extra computation cost')
