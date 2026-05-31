"""
Reward Shaping - Standalone Implementation
==========================================
Potential-based shaping on GridWorld. Policy invariance theorem.
Non-potential shaping pitfalls. Curriculum via reward annealing.
No gym. Uses only numpy.
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Callable, Optional


# ---------------------------------------------------------------------------
# GridWorld Environment
# ---------------------------------------------------------------------------

class GridWorld:
    """N x N grid with configurable reward function."""

    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

    def __init__(self, size: int = 8, goal: Optional[Tuple] = None, seed: int = 42):
        self.size = size
        self.goal = goal or (size - 1, size - 1)
        self.rng = np.random.default_rng(seed)
        self.n_states = size * size
        self.n_actions = 4
        self.state = (0, 0)

    def reset(self) -> int:
        self.state = (0, 0)
        return self._encode(self.state)

    def step(self, action: int) -> Tuple[int, Tuple, bool]:
        dy, dx = self.ACTIONS[action]
        r, c = self.state
        nr = max(0, min(self.size - 1, r + dy))
        nc = max(0, min(self.size - 1, c + dx))
        prev_state = self.state
        self.state = (nr, nc)
        done = self.state == self.goal
        return self._encode(self.state), self._encode(prev_state), done

    def _encode(self, state: Tuple) -> int:
        return state[0] * self.size + state[1]

    def decode(self, s: int) -> Tuple:
        return (s // self.size, s % self.size)

    def manhattan_dist(self, s: int) -> int:
        r, c = self.decode(s)
        gr, gc = self.goal
        return abs(r - gr) + abs(c - gc)

    def euclidean_dist(self, s: int) -> float:
        r, c = self.decode(s)
        gr, gc = self.goal
        return float(np.sqrt((r - gr)**2 + (c - gc)**2))


# ---------------------------------------------------------------------------
# Reward shaping functions
# ---------------------------------------------------------------------------

def sparse_reward(env: GridWorld, s: int, ns: int, done: bool) -> float:
    """Only reward on reaching goal."""
    return 1.0 if done else 0.0


def manhattan_shaped(env: GridWorld, s: int, ns: int, done: bool,
                     gamma: float = 0.99) -> float:
    """Potential-based shaping: Phi(s) = -manhattan_dist(s) / max_dist."""
    max_dist = 2 * (env.size - 1)
    phi_s = -env.manhattan_dist(s) / max_dist
    phi_ns = -env.manhattan_dist(ns) / max_dist
    # F(s, s') = gamma * Phi(s') - Phi(s)
    shaping = gamma * phi_ns - phi_s
    return (1.0 if done else 0.0) + shaping


def euclidean_shaped(env: GridWorld, s: int, ns: int, done: bool,
                     gamma: float = 0.99) -> float:
    """Potential-based shaping with Euclidean distance."""
    max_dist = float(np.sqrt(2) * (env.size - 1))
    phi_s = -env.euclidean_dist(s) / max_dist
    phi_ns = -env.euclidean_dist(ns) / max_dist
    shaping = gamma * phi_ns - phi_s
    return (1.0 if done else 0.0) + shaping


def non_potential_shaping(env: GridWorld, s: int, ns: int, done: bool,
                          cycle_bonus: float = 0.5) -> float:
    """
    DANGEROUS: reward going right (positive) but penalise going left (negative).
    This is NOT potential-based, so it can change the optimal policy.
    A clever agent learns to exploit the right-left cycle for free reward.
    """
    sr, sc = env.decode(s)
    nsr, nsc = env.decode(ns)
    if nsc > sc:   # moved right
        return (1.0 if done else 0.0) + cycle_bonus
    elif nsc < sc:  # moved left
        return (1.0 if done else 0.0) - cycle_bonus * 0.5
    return 1.0 if done else 0.0


def curriculum_shaped(env: GridWorld, s: int, ns: int, done: bool,
                      progress: float = 0.0, gamma: float = 0.99) -> float:
    """
    Curriculum: start with dense Manhattan shaping, anneal toward sparse.
    progress: 0.0 (start, dense) -> 1.0 (end, sparse).
    """
    max_dist = 2 * (env.size - 1)
    phi_s = -env.manhattan_dist(s) / max_dist
    phi_ns = -env.manhattan_dist(ns) / max_dist
    shaping = gamma * phi_ns - phi_s
    # Linearly blend: shaping_weight decreases as training progresses
    shaping_weight = max(0.0, 1.0 - progress)
    return (1.0 if done else 0.0) + shaping_weight * shaping


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, ns: int, done: bool):
        td_target = r + self.gamma * np.max(self.Q[ns]) * (1 - float(done))
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_with_shaping(
    reward_fn: Callable,
    grid_size: int = 8,
    n_episodes: int = 400,
    max_steps: int = 200,
    seed: int = 42,
    **reward_kwargs,
) -> Tuple[List[int], List[float]]:
    """
    Train Q-learning agent on GridWorld with given reward function.
    Returns (steps_per_episode, cumulative_return_per_episode).
    """
    np.random.seed(seed)
    env = GridWorld(grid_size, seed=seed)
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.1, epsilon=0.15)
    steps_per_ep = []
    returns_per_ep = []

    for ep in range(n_episodes):
        s = env.reset()
        total_return = 0.0
        for step in range(max_steps):
            a = agent.select(s)
            ns, prev_s, done = env.step(a)

            # Compute shaped reward (pass progress for curriculum)
            if "progress" in reward_fn.__code__.co_varnames:
                r = reward_fn(env, prev_s, ns, done,
                              progress=ep / n_episodes, **reward_kwargs)
            else:
                r = reward_fn(env, prev_s, ns, done, **reward_kwargs)

            agent.update(s, a, r, ns, done)
            total_return += r
            s = ns
            if done:
                break

        steps_per_ep.append(step + 1)
        returns_per_ep.append(total_return)

    return steps_per_ep, returns_per_ep


# ---------------------------------------------------------------------------
# RLHF reward shaping: KL penalty demo
# ---------------------------------------------------------------------------

def rlhf_kl_shaping_demo(n_steps: int = 300, beta: float = 0.1) -> Dict:
    """
    Simulate RLHF reward shaping with KL penalty.
    Policy = softmax logits (10 tokens). Reference = initial policy.
    At each step: r_total = r_reward_model - beta * KL(pi || pi_ref)
    """
    vocab = 10
    logits = np.zeros(vocab)
    ref_logits = np.zeros(vocab)
    # Reward model: prefers tokens 6-9
    reward_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.7, 0.8, 1.0])
    lr = 0.05

    rewards_shaped, rewards_raw, kl_hist = [], [], []

    for _ in range(n_steps):
        # Compute probabilities
        p = np.exp(logits - logits.max()); p /= p.sum()
        p_ref = np.exp(ref_logits - ref_logits.max()); p_ref /= p_ref.sum()

        # Sample action (token)
        a = np.random.choice(vocab, p=p)

        # Rewards
        r_ext = float(reward_weights[a])
        kl = float(np.sum(p * np.log(p / (p_ref + 1e-8) + 1e-8)))
        r_shaped = r_ext - beta * kl

        # Policy gradient update
        d_logits = -p.copy(); d_logits[a] += 1.0
        logits += lr * r_shaped * d_logits

        rewards_shaped.append(r_shaped)
        rewards_raw.append(r_ext)
        kl_hist.append(kl)

    return {"shaped": rewards_shaped, "raw": rewards_raw, "kl": kl_hist}


# ---------------------------------------------------------------------------
# Circular reward loop: non-potential shaping pitfall
# ---------------------------------------------------------------------------

def circular_reward_pitfall_demo():
    """
    Show that non-potential shaping can make agent prefer a cycle over the goal.
    State: 0 -> 1 -> 2 -> 0 (cycle), goal is state 3.
    Non-potential bonus +0.3 for 0->1->2->0 cycle, sparse +1 for reaching 3.
    """
    # True optimal: go to state 3 for reward 1
    # Non-potential optimal: cycle 0->1->2->0 indefinitely for +0.3 each time
    cycle_reward = 0.3  # non-potential bonus per cycle step
    goal_reward = 1.0   # sparse final reward

    # Q-values for "go to goal" vs "cycle"
    gamma = 0.99
    # Value of cycling forever: 0.3 / (1 - 0.99) * per-step ~ very high
    V_cycle = cycle_reward / (1 - gamma)
    V_goal = goal_reward  # one-shot terminal reward

    print("  Non-potential shaping pitfall:")
    print(f"    V(cycle forever) = {V_cycle:.1f}  (exploits non-potential shaping!)")
    print(f"    V(reach goal)    = {V_goal:.1f}")
    print("    Agent learns to cycle instead of reaching goal.")
    print("    Fix: use potential-based shaping F = gamma*Phi(s') - Phi(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Reward Shaping — Standalone Demo")
    print("=" * 60)
    np.random.seed(42)

    # 1. Convergence speed comparison
    print("\n=== Convergence Speed: Sparse vs Shaped Reward ===")
    t0 = time.time()
    configs = [
        ("Sparse reward",      sparse_reward),
        ("Manhattan-shaped",   manhattan_shaped),
        ("Euclidean-shaped",   euclidean_shaped),
        ("Non-potential",      non_potential_shaping),
        ("Curriculum",         curriculum_shaped),
    ]
    for name, fn in configs:
        steps, returns = train_with_shaping(fn, n_episodes=300)
        # Convergence: first episode where steps < 50
        converged = next((i for i, s in enumerate(steps) if s < 50), None)
        print(f"  {name:20s}: final-20 avg steps={np.mean(steps[-20:]):.1f} "
              f"| converged ep: {converged or 'never'}")
    print(f"  Completed in {time.time()-t0:.1f}s")

    # 2. Non-potential pitfall
    print("\n=== Non-Potential Shaping Pitfall ===")
    circular_reward_pitfall_demo()

    # 3. RLHF KL penalty shaping
    print("\n=== RLHF Reward Shaping (KL penalty) ===")
    for beta in [0.0, 0.1, 0.5]:
        result = rlhf_kl_shaping_demo(n_steps=300, beta=beta)
        avg_kl = np.mean(result["kl"][-50:])
        avg_r = np.mean(result["raw"][-50:])
        print(f"  beta={beta:.1f}: final-50 avg reward={avg_r:.3f}, KL={avg_kl:.4f}")

    # 4. Potential function design
    print("\n=== Potential Function Design ===")
    env_demo = GridWorld(8)
    s = env_demo._encode((0, 0))   # start
    ns = env_demo._encode((0, 1))  # one step right
    prev_s = s
    # Temporarily move environment state
    env_demo.state = (0, 0)
    env_demo.step(0)               # move right
    ns_encoded = env_demo._encode(env_demo.state)
    for name, fn in [("Manhattan", manhattan_shaped), ("Euclidean", euclidean_shaped)]:
        r = fn(env_demo, prev_s, ns_encoded, False)
        print(f"  {name} shaped reward (step toward goal): {r:.4f}")

    print("\nDone.")
