"""
Multi-Agent RL — Cooperative Pursuit with IQL and CTDE
=======================================================
Implements:
  - MultiAgentEnv: 2-agent cooperative pursuit on an 8x8 grid (catch the prey)
  - IndependentQLearner (IQL): each agent learns a Q-table independently
  - CTDEAgent: Centralised Training Decentralised Execution (shared Q during training)

No external MARL libraries required — only numpy.
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# Multi-Agent Pursuit Environment
# ---------------------------------------------------------------------------

class MultiAgentEnv:
    """
    2-agent cooperative pursuit on an 8x8 grid.

    Agents: 2 pursuers.
    Prey: 1 target (moves randomly).
    Goal: both pursuers must be adjacent to the prey simultaneously to catch it.
    Reward: +10 for catch (shared), -0.1 per step (penalty for slow catching).
    State: (pursuer1_pos, pursuer2_pos, prey_pos) — each position is (row, col).
    """

    GRID = 8
    N_AGENTS = 2
    N_ACTIONS = 5   # 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT

    DELTAS = {
        0: (0, 0),    # STAY
        1: (-1, 0),   # UP
        2: (1, 0),    # DOWN
        3: (0, -1),   # LEFT
        4: (0, 1),    # RIGHT
    }

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(42)
        self.reset()

    def reset(self) -> Tuple:
        """Reset all positions. Returns (state_agent1, state_agent2, global_state)."""
        positions = set()
        agents = []
        for _ in range(3):   # 2 agents + 1 prey
            while True:
                pos = (int(self.rng.integers(self.GRID)), int(self.rng.integers(self.GRID)))
                if pos not in positions:
                    positions.add(pos)
                    agents.append(pos)
                    break
        self.p1, self.p2, self.prey = agents[0], agents[1], agents[2]
        self.step_count = 0
        return self._obs()

    def _clamp(self, r: int, c: int) -> Tuple[int, int]:
        return (max(0, min(self.GRID - 1, r)), max(0, min(self.GRID - 1, c)))

    def _move(self, pos: Tuple, action: int) -> Tuple[int, int]:
        dr, dc = self.DELTAS[action]
        return self._clamp(pos[0] + dr, pos[1] + dc)

    def _is_caught(self) -> bool:
        """Prey caught iff both agents are adjacent (Manhattan distance <= 1)."""
        d1 = abs(self.p1[0] - self.prey[0]) + abs(self.p1[1] - self.prey[1])
        d2 = abs(self.p2[0] - self.prey[0]) + abs(self.p2[1] - self.prey[1])
        return d1 <= 1 and d2 <= 1

    def _obs(self) -> Tuple:
        """Observations: (local_obs_agent1, local_obs_agent2, global_state)."""
        # Local obs: own position + relative prey position (4 ints each)
        obs1 = (self.p1[0], self.p1[1],
                self.prey[0] - self.p1[0], self.prey[1] - self.p1[1])
        obs2 = (self.p2[0], self.p2[1],
                self.prey[0] - self.p2[0], self.prey[1] - self.p2[1])
        global_state = (self.p1[0], self.p1[1], self.p2[0], self.p2[1],
                        self.prey[0], self.prey[1])
        return obs1, obs2, global_state

    def step(self, a1: int, a2: int) -> Tuple:
        """
        Step both agents simultaneously. Prey moves randomly.
        Returns: (obs1, obs2, global_state, reward, done)
        """
        self.p1 = self._move(self.p1, a1)
        self.p2 = self._move(self.p2, a2)

        # Prevent agents from occupying same cell
        if self.p1 == self.p2:
            self.p1 = self._move(self.p1, 0)   # agent 1 stays

        # Prey moves randomly
        prey_action = int(self.rng.integers(self.N_ACTIONS))
        self.prey = self._move(self.prey, prey_action)

        self.step_count += 1
        caught = self._is_caught()
        reward = 10.0 if caught else -0.1
        done = caught or self.step_count >= 100

        return *self._obs(), reward, done


# ---------------------------------------------------------------------------
# State encoding for tabular Q
# ---------------------------------------------------------------------------

def encode_obs(obs: Tuple) -> int:
    """Encode local 4-tuple observation to a single integer index."""
    # obs = (r, c, dr, dc) where r,c in [0,7] and dr,dc in [-7,7]
    r, c, dr, dc = obs
    dr += 7    # shift to [0, 14]
    dc += 7
    return r * (8 * 15 * 15) + c * (15 * 15) + dr * 15 + dc


def encode_global(global_state: Tuple) -> int:
    """Encode global 6-tuple state for CTDE centralised Q."""
    r1, c1, r2, c2, rp, cp = global_state
    return (r1 * 8**5 + c1 * 8**4 + r2 * 8**3 + c2 * 8**2 + rp * 8 + cp)


# ---------------------------------------------------------------------------
# Independent Q-Learner (IQL)
# ---------------------------------------------------------------------------

class IndependentQLearner:
    """
    Two independent tabular Q-learners. Each observes only its local observation.
    Q_i(obs_i, a_i) updated with TD(0) treating the other agent as part of the env.
    """

    def __init__(
        self,
        n_obs_states: int = 8 * 8 * 15 * 15,
        n_actions: int = 5,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.Q1 = np.zeros((n_obs_states, n_actions))
        self.Q2 = np.zeros((n_obs_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self, obs1: Tuple, obs2: Tuple) -> Tuple[int, int]:
        o1, o2 = encode_obs(obs1), encode_obs(obs2)
        a1 = (int(self.rng.integers(self.n_actions)) if self.rng.random() < self.epsilon
              else int(self.Q1[o1].argmax()))
        a2 = (int(self.rng.integers(self.n_actions)) if self.rng.random() < self.epsilon
              else int(self.Q2[o2].argmax()))
        return a1, a2

    def update(
        self,
        obs1: Tuple, a1: int,
        obs2: Tuple, a2: int,
        reward: float,
        next_obs1: Tuple, next_obs2: Tuple,
        done: bool,
    ) -> None:
        o1, o2 = encode_obs(obs1), encode_obs(obs2)
        no1, no2 = encode_obs(next_obs1), encode_obs(next_obs2)
        target1 = reward if done else reward + self.gamma * self.Q1[no1].max()
        target2 = reward if done else reward + self.gamma * self.Q2[no2].max()
        self.Q1[o1, a1] += self.alpha * (target1 - self.Q1[o1, a1])
        self.Q2[o2, a2] += self.alpha * (target2 - self.Q2[o2, a2])


# ---------------------------------------------------------------------------
# CTDE Agent (Centralised Training, Decentralised Execution)
# ---------------------------------------------------------------------------

class CTDEAgent:
    """
    Centralised Q-function Q(global_state, a1, a2) during training.
    Execution: each agent acts on own obs only (using marginalised Q).

    Q_central[global_state, a1, a2] = joint action-value.
    Execution: agent 1 uses argmax_{a1} E_{a2}[Q_central(s, a1, a2)].
    """

    def __init__(
        self,
        n_global_states: int = 8**6,
        n_actions: int = 5,
        n_obs_states: int = 8 * 8 * 15 * 15,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        # Joint Q-table (global state x a1 x a2) — stored flat
        # Using separate marginalised Q tables for memory efficiency
        # Q_marginal_i[obs_i, a_i] = mean_Q over other agent's actions
        self.Q_joint: Dict = {}     # {(global_state, a1, a2): float}
        self.Q1 = np.zeros((n_obs_states, n_actions))    # for execution
        self.Q2 = np.zeros((n_obs_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self, obs1: Tuple, obs2: Tuple) -> Tuple[int, int]:
        o1, o2 = encode_obs(obs1), encode_obs(obs2)
        a1 = (int(self.rng.integers(self.n_actions)) if self.rng.random() < self.epsilon
              else int(self.Q1[o1].argmax()))
        a2 = (int(self.rng.integers(self.n_actions)) if self.rng.random() < self.epsilon
              else int(self.Q2[o2].argmax()))
        return a1, a2

    def update(
        self,
        obs1: Tuple, a1: int,
        obs2: Tuple, a2: int,
        global_state: Tuple,
        reward: float,
        next_obs1: Tuple, next_obs2: Tuple,
        next_global: Tuple,
        done: bool,
    ) -> None:
        # Centralised update: Q(s, a1, a2) <- r + gamma * max_{a1',a2'} Q(s', a1', a2')
        gs_key = encode_global(global_state)
        ngs_key = encode_global(next_global)

        # Best next joint action
        if done:
            target = reward
        else:
            best_next = max(
                self.Q_joint.get((ngs_key, na1, na2), 0.0)
                for na1 in range(self.n_actions)
                for na2 in range(self.n_actions)
            )
            target = reward + self.gamma * best_next

        key = (gs_key, a1, a2)
        current = self.Q_joint.get(key, 0.0)
        self.Q_joint[key] = current + self.alpha * (target - current)

        # Update individual Q-tables for decentralised execution
        o1, o2 = encode_obs(obs1), encode_obs(obs2)
        # Q1(obs1, a1) = mean over a2 of Q_joint
        q_a1 = np.mean([self.Q_joint.get((gs_key, a1, a2_), 0.0)
                         for a2_ in range(self.n_actions)])
        q_a2 = np.mean([self.Q_joint.get((gs_key, a1_, a2), 0.0)
                         for a1_ in range(self.n_actions)])
        self.Q1[o1, a1] = q_a1
        self.Q2[o2, a2] = q_a2


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_agent(
    agent,
    n_episodes: int = 200,
    seed: int = 42,
    label: str = "Agent",
    verbose: bool = True,
) -> List[float]:
    """Train an agent for n_episodes. Returns episode rewards."""
    rng = np.random.default_rng(seed)
    env = MultiAgentEnv(rng=rng)
    rewards_log = []

    for ep in range(n_episodes):
        obs1, obs2, gs = env.reset()
        ep_reward = 0.0

        for _ in range(100):
            a1, a2 = agent.act(obs1, obs2)
            next_obs1, next_obs2, next_gs, reward, done = env.step(a1, a2)
            ep_reward += reward

            if isinstance(agent, CTDEAgent):
                agent.update(obs1, a1, obs2, a2, gs, reward, next_obs1, next_obs2, next_gs, done)
            else:
                agent.update(obs1, a1, obs2, a2, reward, next_obs1, next_obs2, done)

            obs1, obs2, gs = next_obs1, next_obs2, next_gs
            if done:
                break

        rewards_log.append(ep_reward)
        if verbose and (ep + 1) % 50 == 0:
            mean20 = np.mean(rewards_log[-20:])
            print(f"  [{label}] Ep {ep+1:4d} | reward={ep_reward:6.1f} | mean-20={mean20:.1f}")

    return rewards_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent RL — Cooperative Pursuit (IQL vs CTDE)")
    print("=" * 60)

    print("\n--- Environment Demo ---")
    rng = np.random.default_rng(0)
    env = MultiAgentEnv(rng=rng)
    obs1, obs2, gs = env.reset()
    print(f"Agent 1 pos: {gs[:2]}, Agent 2 pos: {gs[2:4]}, Prey pos: {gs[4:]}")
    for step in range(3):
        a1, a2 = int(rng.integers(5)), int(rng.integers(5))
        obs1, obs2, gs, r, done = env.step(a1, a2)
        print(f"  Step {step+1}: a1={a1}, a2={a2}, reward={r:.1f}, done={done}")
        print(f"    A1={gs[:2]}, A2={gs[2:4]}, Prey={gs[4:]}")

    N_EPISODES = 200

    print("\n--- Training IQL ---")
    t0 = time.time()
    iql = IndependentQLearner(epsilon=0.3, seed=42)
    iql_rewards = train_agent(iql, n_episodes=N_EPISODES, label="IQL", verbose=True)
    iql_time = time.time() - t0

    print("\n--- Training CTDE ---")
    t0 = time.time()
    ctde = CTDEAgent(epsilon=0.3, seed=42)
    ctde_rewards = train_agent(ctde, n_episodes=N_EPISODES, label="CTDE", verbose=True)
    ctde_time = time.time() - t0

    print(f"\n--- Results ---")
    print(f"  {'Algorithm':10s}  {'Mean reward (last 50)':>22s}  {'Training time':>14s}")
    print(f"  {'IQL':10s}  {np.mean(iql_rewards[-50:]):>22.2f}  {iql_time:>14.1f}s")
    print(f"  {'CTDE':10s}  {np.mean(ctde_rewards[-50:]):>22.2f}  {ctde_time:>14.1f}s")

    print("\nDone.")
