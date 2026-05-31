"""
SARSA Implementation
====================
Standalone implementation of SARSA, Expected SARSA, and SARSA(lambda).
On-policy TD control: uses actual next action from behavior policy.

Usage:
    python 07-sarsa.py
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
        self.n_actions = 4
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
    """4x12 Cliff Walking: start=(3,0), goal=(3,11)."""

    def __init__(self):
        self.rows = 4; self.cols = 12
        self.start = (3, 0); self.goal = (3, 11)
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
        r = max(0, min(self.rows-1, self.pos[0]+dr))
        c = max(0, min(self.cols-1, self.pos[1]+dc))
        self.pos = (r, c)
        if self.pos in self.cliff:
            self.pos = self.start
            return self._idx(self.pos), -100.0, False
        if self.pos == self.goal:
            return self._idx(self.pos), 0.0, True
        return self._idx(self.pos), -1.0, False


class WindyGridWorld:
    """4x4 GridWorld with stochastic wind (probability p of being pushed up)."""

    def __init__(self, size: int = 4, wind_prob: float = 0.3):
        self.size = size; self.wind_prob = wind_prob
        self.goal = (size-1, size-1)
        self.n_actions = 4; self.n_states = size * size
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.reset()

    def reset(self) -> int:
        self.pos = (0, 0)
        return self._idx(self.pos)

    def _idx(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def step(self, action: int) -> Tuple[int, float, bool]:
        dr, dc = self.action_deltas[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if np.random.rand() < self.wind_prob:
            r -= 1  # Wind blows agent upward
        r = max(0, min(self.size-1, r))
        c = max(0, min(self.size-1, c))
        self.pos = (r, c)
        if self.pos == self.goal:
            return self._idx(self.pos), 10.0, True
        return self._idx(self.pos), -1.0, False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def epsilon_greedy(Q_row: np.ndarray, epsilon: float) -> int:
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        return int(np.random.randint(len(Q_row)))
    return int(np.argmax(Q_row))


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def run_sarsa(
    env,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, List[float]]:
    """Tabular SARSA (on-policy TD control)."""
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards: List[float] = []

    for ep in range(n_episodes):
        state = env.reset()
        # KEY: select first action BEFORE the loop (on-policy)
        action = epsilon_greedy(Q[state], epsilon)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 300:
            next_state, reward, done = env.step(action)
            # Sample next action from BEHAVIOR policy (on-policy — not greedy)
            next_action = epsilon_greedy(Q[next_state], epsilon)

            # SARSA update: uses Q(s', a') where a' is from behavior policy
            td_error = reward + gamma * Q[next_state, next_action] * (1-done) - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += reward
            state, action = next_state, next_action  # Advance both state AND action
            steps += 1

        episode_rewards.append(total_reward)

    return Q, episode_rewards


def run_expected_sarsa(
    env,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, List[float]]:
    """Expected SARSA: uses E_pi[Q(s',a')] instead of sampled Q(s',a')."""
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards: List[float] = []

    def expected_q(Q_row: np.ndarray) -> float:
        """E_{pi}[Q(s,a)] under epsilon-greedy policy."""
        n_a = len(Q_row)
        greedy = int(np.argmax(Q_row))
        probs = np.full(n_a, epsilon / n_a)
        probs[greedy] += (1.0 - epsilon)
        return float(np.dot(probs, Q_row))

    for ep in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(Q[state], epsilon)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 300:
            next_state, reward, done = env.step(action)

            # Expected SARSA: replace sampled Q(s',a') with expectation
            exp_q_next = expected_q(Q[next_state]) * (1 - done)
            td_error = reward + gamma * exp_q_next - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += reward
            action = epsilon_greedy(Q[next_state], epsilon)
            state = next_state
            steps += 1

        episode_rewards.append(total_reward)

    return Q, episode_rewards


def run_q_learning(
    env,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, List[float]]:
    """Q-learning baseline for comparison."""
    Q = np.zeros((env.n_states, env.n_actions))
    rewards: List[float] = []
    for ep in range(n_episodes):
        state = env.reset(); total = 0.0; done = False; steps = 0
        while not done and steps < 300:
            action = epsilon_greedy(Q[state], epsilon)
            ns, r, done = env.step(action)
            Q[state, action] += alpha * (r + gamma * np.max(Q[ns]) * (1-done) - Q[state, action])
            total += r; state = ns; steps += 1
        rewards.append(total)
    return Q, rewards


def run_sarsa_lambda(
    env,
    lam: float = 0.9,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> List[float]:
    """SARSA(lambda) with replacing eligibility traces."""
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards: List[float] = []

    for ep in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(Q[state], epsilon)
        E = np.zeros((env.n_states, env.n_actions))
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 300:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q[next_state], epsilon)

            # SARSA TD error (on-policy: uses actual next action)
            td_error = reward + gamma * Q[next_state, next_action] * (1-done) - Q[state, action]

            # Replacing trace: cap at 1.0 to prevent unbounded accumulation
            E[state, action] = 1.0

            # Update all Q values by eligibility-weighted TD error
            Q += alpha * td_error * E
            # Decay all traces
            E *= gamma * lam

            total_reward += reward
            state, action = next_state, next_action
            steps += 1

        episode_rewards.append(total_reward)

    return episode_rewards


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def smooth(arr: List[float], window: int = 20) -> np.ndarray:
    if len(arr) < window:
        return np.array(arr)
    return np.convolve(arr, np.ones(window) / window, mode='valid')


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)
    print('=== SARSA Demonstration ===\n')

    # --- 1. SARSA vs Q-learning on GridWorld ---
    print('1. SARSA vs Q-Learning on GridWorld')
    env = GridWorld()
    np.random.seed(42); Q_s, r_s = run_sarsa(env, n_episodes=500)
    np.random.seed(42); Q_q, r_q = run_q_learning(env, n_episodes=500)
    print(f'   SARSA:      mean reward (last 50) = {np.mean(r_s[-50:]):.2f}')
    print(f'   Q-learning: mean reward (last 50) = {np.mean(r_q[-50:]):.2f}')

    # --- 2. Expected SARSA variance comparison ---
    print('\n2. Expected SARSA vs SARSA (variance comparison, 5 seeds)')
    sarsa_finals, exp_finals = [], []
    for seed in range(5):
        np.random.seed(seed)
        _, rs = run_sarsa(env, n_episodes=300)
        np.random.seed(seed)
        _, re = run_expected_sarsa(env, n_episodes=300)
        sarsa_finals.append(np.mean(rs[-50:]))
        exp_finals.append(np.mean(re[-50:]))
    print(f'   SARSA:          mean={np.mean(sarsa_finals):.2f}, std={np.std(sarsa_finals):.2f}')
    print(f'   Expected SARSA: mean={np.mean(exp_finals):.2f}, std={np.std(exp_finals):.2f}')

    # --- 3. Cliff walking: SARSA safe path vs Q-learning risky path ---
    print('\n3. Cliff Walking: On-Policy Safety (SARSA vs Q-learning)')
    cliff = CliffWorld()
    np.random.seed(42); _, r_sarsa_cliff = run_sarsa(cliff, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1)
    np.random.seed(42); _, r_ql_cliff = run_q_learning(cliff, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1)
    print(f'   SARSA   training reward (last 100): {np.mean(r_sarsa_cliff[-100:]):.1f} (safe: avoids cliff)')
    print(f'   Q-learn training reward (last 100): {np.mean(r_ql_cliff[-100:]):.1f} (risky: walks cliff edge)')

    # --- 4. SARSA(lambda) sweep ---
    print('\n4. SARSA(lambda): Lambda Sweep on GridWorld')
    for lam in [0.0, 0.5, 0.9]:
        np.random.seed(42)
        r_lam = run_sarsa_lambda(env, lam=lam, n_episodes=500)
        print(f'   lambda={lam}: mean reward (last 50) = {np.mean(r_lam[-50:]):.2f}')

    # --- 5. Windy GridWorld: stochastic environment ---
    print('\n5. Windy GridWorld (wind_prob=0.3): SARSA vs Q-learning')
    windy = WindyGridWorld(size=4, wind_prob=0.3)
    windy_s, windy_q = [], []
    for seed in range(10):
        np.random.seed(seed)
        _, rs = run_sarsa(windy, n_episodes=400, alpha=0.1, gamma=0.95)
        np.random.seed(seed)
        _, rq = run_q_learning(windy, n_episodes=400, alpha=0.1, gamma=0.95)
        windy_s.append(np.mean(rs[-50:]))
        windy_q.append(np.mean(rq[-50:]))
    print(f'   SARSA:      mean={np.mean(windy_s):.2f}, std={np.std(windy_s):.2f}')
    print(f'   Q-learning: mean={np.mean(windy_q):.2f}, std={np.std(windy_q):.2f}')

    # --- 6. Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(smooth(r_sarsa_cliff), label='SARSA (safe)', color='blue')
    ax.plot(smooth(r_ql_cliff), label='Q-Learning (risky)', color='red')
    ax.set_title('Cliff Walking: SARSA vs Q-Learning')
    ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward (smoothed)')
    ax.set_ylim(-200, 10); ax.legend()

    ax = axes[1]
    lambdas = [0.0, 0.5, 0.9]
    colors = ['steelblue', 'darkorange', 'green']
    for lam, col in zip(lambdas, colors):
        np.random.seed(42)
        r = run_sarsa_lambda(env, lam=lam, n_episodes=500)
        ax.plot(smooth(r), label=f'lambda={lam}', color=col)
    ax.set_title('SARSA(lambda): Convergence Speed')
    ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward (smoothed)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/tmp/07_sarsa_demo.png', dpi=80, bbox_inches='tight')
    plt.close()
    print('\n6. Plot saved to /tmp/07_sarsa_demo.png')

    print('\nKey takeaways:')
    print('  - SARSA is on-policy: values reflect actual behavior (including exploration)')
    print('  - Safer in cliff environments during training; Q-learning optimal at test time')
    print('  - Expected SARSA reduces variance by replacing sample with expectation')
    print('  - SARSA(lambda=0.9) faster convergence in sparse-reward environments')
