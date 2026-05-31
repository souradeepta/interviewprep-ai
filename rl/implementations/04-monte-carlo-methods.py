"""
Monte Carlo Methods for RL — First-Visit MC Prediction and Control
==================================================================
Implements:
  - Simple Blackjack-like environment (hand sum vs dealer card)
  - MCAgent with first-visit MC prediction (estimate V^pi)
  - MC control with epsilon-greedy (estimate Q*, derive pi)
  - Every-visit MC for comparison

No external RL libraries required — only numpy.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Blackjack-like environment (simplified)
# ---------------------------------------------------------------------------

class BlackjackEnv:
    """
    Simplified Blackjack (same structure as gym's Blackjack but self-contained).

    State: (player_sum, dealer_card, usable_ace)
      - player_sum: int in [4, 21]
      - dealer_card: int in [1, 10]
      - usable_ace: bool (True if ace counted as 11)

    Actions: 0=STICK, 1=HIT
    Reward: +1 win, -1 loss, 0 draw
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(42)

    def _draw_card(self) -> int:
        """Draw a card. Face cards count as 10; ace as 11 (handled separately)."""
        return min(self.rng.integers(1, 14), 10)

    def _hand_value(self, cards: List[int]) -> Tuple[int, bool]:
        """Compute hand value and whether an ace is used as 11."""
        total = sum(cards)
        usable_ace = 1 in cards and total + 10 <= 21
        if usable_ace:
            total += 10
        return total, usable_ace

    def reset(self) -> Tuple[int, int, bool]:
        """Start new episode. Returns initial state."""
        self.player = [self._draw_card(), self._draw_card()]
        self.dealer_showing = self._draw_card()
        self.dealer_hidden = self._draw_card()
        self.done = False
        p_sum, usable = self._hand_value(self.player)
        return p_sum, self.dealer_showing, usable

    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """
        Take a step.
        action: 0=STICK, 1=HIT
        Returns: (next_state, reward, done)
        """
        if self.done:
            raise RuntimeError("Episode already done. Call reset().")

        if action == 1:  # HIT
            self.player.append(self._draw_card())
            p_sum, usable = self._hand_value(self.player)
            if p_sum > 21:
                self.done = True
                return (p_sum, self.dealer_showing, usable), -1.0, True
            return (p_sum, self.dealer_showing, usable), 0.0, False

        else:  # STICK
            self.done = True
            # Dealer plays: must hit to 17
            dealer_hand = [self.dealer_showing, self.dealer_hidden]
            d_sum, _ = self._hand_value(dealer_hand)
            while d_sum < 17:
                dealer_hand.append(self._draw_card())
                d_sum, _ = self._hand_value(dealer_hand)

            p_sum, usable = self._hand_value(self.player)

            if d_sum > 21 or p_sum > d_sum:
                reward = 1.0
            elif p_sum == d_sum:
                reward = 0.0
            else:
                reward = -1.0

            return (p_sum, self.dealer_showing, usable), reward, True


# ---------------------------------------------------------------------------
# MC Agent
# ---------------------------------------------------------------------------

class MCAgent:
    """
    Monte Carlo agent for prediction (V^pi) and control (Q*/pi).

    Supports:
      - first-visit MC prediction
      - every-visit MC prediction
      - epsilon-greedy MC control (for Q*)
    """

    def __init__(
        self,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        first_visit: bool = True,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.first_visit = first_visit

        # For prediction: V(s) and returns
        self.V: Dict = defaultdict(float)
        self.V_returns: Dict = defaultdict(list)

        # For control: Q(s,a) and returns
        self.Q: Dict = defaultdict(lambda: np.zeros(2))
        self.Q_returns: Dict = defaultdict(lambda: [[], []])  # per action

    def run_episode(self, env: BlackjackEnv, policy: Optional[Dict] = None) -> List:
        """
        Run a single episode and return trajectory as list of (state, action, reward).
        If policy is None, uses epsilon-greedy w.r.t. self.Q.
        """
        state = env.reset()
        episode = []
        done = False

        while not done:
            if policy is not None:
                action = policy.get(state, 0)   # default: STICK
            else:
                # Epsilon-greedy w.r.t. Q
                if np.random.random() < self.epsilon:
                    action = np.random.randint(2)
                else:
                    action = int(np.argmax(self.Q[state]))

            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def update_V(self, episode: List) -> None:
        """
        First-visit (or every-visit) MC prediction: update V(s) from episode.
        Incremental average update avoids storing all returns.
        """
        G = 0.0
        visited = set()

        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G

            if self.first_visit and state in visited:
                continue
            visited.add(state)

            # Incremental mean: V(s) = V(s) + (G - V(s)) / N(s)
            self.V_returns[state].append(G)
            N = len(self.V_returns[state])
            self.V[state] += (G - self.V[state]) / N

    def update_Q_epsilon_greedy(self, episode: List) -> None:
        """
        MC control: update Q(s,a) using every-visit returns, epsilon-greedy policy.
        """
        G = 0.0
        visited_sa = set()

        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G

            if self.first_visit and (state, action) in visited_sa:
                continue
            visited_sa.add((state, action))

            self.Q_returns[state][action].append(G)
            N = len(self.Q_returns[state][action])
            # Incremental mean
            self.Q[state][action] += (G - self.Q[state][action]) / N

    def get_policy(self) -> Dict:
        """Extract greedy policy from Q."""
        return {s: int(np.argmax(q)) for s, q in self.Q.items()}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_policy(
    env: BlackjackEnv,
    policy: Dict,
    n_episodes: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate expected return under a deterministic policy."""
    total = 0.0
    env.rng = rng or np.random.default_rng(0)
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = policy.get(state, 0)   # default STICK
            state, reward, done = env.step(action)
            ep_return += reward
        total += ep_return
    return total / n_episodes


def summarise_V(agent: MCAgent) -> None:
    """Print value statistics for states with sum 18-21."""
    print("\nV(player_sum, dealer_card, usable_ace) for sum 18-21:")
    print(f"  {'State':35s}  {'V':>8s}  {'N':>6s}")
    for p_sum in [18, 19, 20, 21]:
        for dealer in [2, 6, 10]:
            for usable in [False, True]:
                s = (p_sum, dealer, usable)
                n = len(agent.V_returns[s])
                if n > 0:
                    print(f"  {str(s):35s}  {agent.V[s]:>8.4f}  {n:>6d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Monte Carlo Methods — Blackjack Prediction and Control")
    print("=" * 60)

    rng = np.random.default_rng(42)
    env = BlackjackEnv(rng=rng)

    # ----------------------------------------------------------------
    # 1. Always-STICK policy — a simple baseline to evaluate
    # ----------------------------------------------------------------
    print("\n--- MC Prediction: First-Visit on Always-STICK Policy ---")
    stick_policy = defaultdict(int)   # all states map to action 0 (STICK)

    agent_pred = MCAgent(gamma=1.0, first_visit=True)
    N_PRED = 10_000

    t0 = time.time()
    for ep in range(N_PRED):
        episode = agent_pred.run_episode(env, policy=stick_policy)
        agent_pred.update_V(episode)
    elapsed = time.time() - t0
    print(f"Ran {N_PRED} episodes in {elapsed:.2f}s")

    summarise_V(agent_pred)

    # ----------------------------------------------------------------
    # 2. MC Control with epsilon-greedy
    # ----------------------------------------------------------------
    print("\n--- MC Control: Epsilon-Greedy Q* Estimation ---")
    agent_ctrl = MCAgent(gamma=1.0, epsilon=0.1, first_visit=True)
    N_CTRL = 50_000

    t0 = time.time()
    for ep in range(N_CTRL):
        # Anneal epsilon over training
        agent_ctrl.epsilon = max(0.01, 0.1 - 0.09 * ep / N_CTRL)
        episode = agent_ctrl.run_episode(env, policy=None)
        agent_ctrl.update_Q_epsilon_greedy(episode)
    elapsed = time.time() - t0
    print(f"Ran {N_CTRL} MC control episodes in {elapsed:.2f}s")

    greedy_policy = agent_ctrl.get_policy()
    n_states_learned = len(greedy_policy)
    print(f"Learned Q-values for {n_states_learned} states")

    # Evaluate learned policy
    eval_rng = np.random.default_rng(99)
    eval_env = BlackjackEnv(rng=eval_rng)
    win_rate = evaluate_policy(eval_env, greedy_policy, n_episodes=10_000)
    stick_rate = evaluate_policy(BlackjackEnv(rng=np.random.default_rng(99)),
                                 stick_policy, n_episodes=10_000)
    print(f"\nAlways-Stick win rate: {stick_rate:+.4f}")
    print(f"MC-Control win rate:   {win_rate:+.4f}")
    print(f"Improvement:           {win_rate - stick_rate:+.4f}")

    # ----------------------------------------------------------------
    # 3. First-Visit vs Every-Visit comparison (on always-HIT policy)
    # ----------------------------------------------------------------
    print("\n--- First-Visit vs Every-Visit Comparison ---")
    hit_policy = defaultdict(lambda: 1)   # always HIT (high bust rate)

    for first_visit, label in [(True, "First-Visit"), (False, "Every-Visit")]:
        ag = MCAgent(gamma=1.0, first_visit=first_visit)
        for _ in range(5_000):
            ep = ag.run_episode(env, policy=hit_policy)
            ag.update_V(ep)
        n_visited = len(ag.V)
        mean_v = np.mean(list(ag.V.values()))
        print(f"  {label}: {n_visited} states visited, mean V = {mean_v:.4f}")

    print("\nDone.")
