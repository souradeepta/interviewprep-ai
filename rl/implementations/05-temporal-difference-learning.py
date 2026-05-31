"""
Temporal Difference Learning — TD(0), TD(lambda), N-Step TD
============================================================
Implements three TD methods and compares them on a 5-state random walk environment
where the true values are known analytically.

  - TD(0): one-step bootstrapped update
  - TD(lambda): eligibility traces for credit assignment across many steps
  - N-Step TD: explicit n-step return without eligibility traces

No external RL libraries required — only numpy.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Random Walk environment (5 non-terminal states, analytic true values known)
# ---------------------------------------------------------------------------

class RandomWalkEnv:
    """
    Classic 5-state random walk benchmark.

    States: 0 (left terminal), 1-5 (non-terminal), 6 (right terminal)
    From any non-terminal state: move left or right with probability 0.5 each.
    Reward: +1 on reaching state 6, 0 everywhere else.
    True values (gamma=1, uniform random policy):
      V*(1) = 1/6, V*(2) = 2/6, V*(3) = 3/6, V*(4) = 4/6, V*(5) = 5/6
    """

    N_STATES = 7          # 0=left-terminal, 1-5=non-terminal, 6=right-terminal
    START = 3             # always start at the centre

    TRUE_V = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 0])   # analytic solution

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(42)
        self.state: int = self.START

    def reset(self) -> int:
        self.state = self.START
        return self.state

    def step(self) -> Tuple[int, float, bool]:
        """Move left or right uniformly. Returns (next_state, reward, done)."""
        move = 1 if self.rng.random() < 0.5 else -1
        self.state += move
        done = (self.state == 0 or self.state == 6)
        reward = 1.0 if self.state == 6 else 0.0
        return self.state, reward, done


# ---------------------------------------------------------------------------
# TD(0)
# ---------------------------------------------------------------------------

class TD0Agent:
    """
    One-step TD prediction: V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]
    """

    def __init__(
        self,
        n_states: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        terminal_states: Tuple[int, ...] = (0, 6),
    ):
        self.V = np.zeros(n_states)
        self.alpha = alpha
        self.gamma = gamma
        self.terminal_states = set(terminal_states)

    def update(self, s: int, r: float, s_next: int, done: bool) -> float:
        """Apply one TD(0) update. Returns TD error."""
        v_next = 0.0 if done else self.V[s_next]
        td_error = r + self.gamma * v_next - self.V[s]
        self.V[s] += self.alpha * td_error
        return td_error

    def run_episode(self, env: RandomWalkEnv) -> List[float]:
        """Run one episode, collecting TD errors."""
        s = env.reset()
        errors = []
        done = False
        while not done:
            s_next, r, done = env.step()
            errors.append(self.update(s, r, s_next, done))
            s = s_next
        return errors

    def rmse(self, true_V: np.ndarray) -> float:
        """Root mean squared error against true values (non-terminal states only)."""
        non_terminal = [i for i in range(len(self.V)) if i not in self.terminal_states]
        return float(np.sqrt(np.mean((self.V[non_terminal] - true_V[non_terminal])**2)))


# ---------------------------------------------------------------------------
# TD(lambda) with eligibility traces
# ---------------------------------------------------------------------------

class TDLambdaAgent:
    """
    TD(lambda) via backward-view eligibility traces.
    e(s) <- gamma * lambda * e(s) for all s, then e(s_t) += 1
    V(s) <- V(s) + alpha * delta * e(s) for all s
    """

    def __init__(
        self,
        n_states: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        lam: float = 0.5,
        terminal_states: Tuple[int, ...] = (0, 6),
    ):
        self.V = np.zeros(n_states)
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.terminal_states = set(terminal_states)

    def run_episode(self, env: RandomWalkEnv) -> None:
        """Run one episode, updating V with eligibility traces."""
        e = np.zeros_like(self.V)      # eligibility trace vector
        s = env.reset()
        done = False

        while not done:
            s_next, r, done = env.step()
            v_next = 0.0 if done else self.V[s_next]
            td_error = r + self.gamma * v_next - self.V[s]

            # Decay all traces, then accumulate for current state
            e *= self.gamma * self.lam
            e[s] += 1.0

            # Update all states proportional to trace
            self.V += self.alpha * td_error * e
            s = s_next

    def rmse(self, true_V: np.ndarray) -> float:
        non_terminal = [i for i in range(len(self.V)) if i not in self.terminal_states]
        return float(np.sqrt(np.mean((self.V[non_terminal] - true_V[non_terminal])**2)))


# ---------------------------------------------------------------------------
# N-Step TD
# ---------------------------------------------------------------------------

class NStepTDAgent:
    """
    N-step TD prediction: accumulate n-step return G_t:t+n, then update V(s_t).
    G_t:t+n = r_{t+1} + gamma * r_{t+2} + ... + gamma^(n-1)*r_{t+n} + gamma^n * V(s_{t+n})
    """

    def __init__(
        self,
        n_states: int,
        n: int = 4,
        alpha: float = 0.1,
        gamma: float = 1.0,
        terminal_states: Tuple[int, ...] = (0, 6),
    ):
        self.V = np.zeros(n_states)
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.terminal_states = set(terminal_states)

    def run_episode(self, env: RandomWalkEnv) -> None:
        """Run one episode with n-step returns."""
        states = [env.reset()]
        rewards = [0.0]     # rewards[0] unused, index by time step
        done = False

        # Collect full episode
        while not done:
            s_next, r, done = env.step()
            states.append(s_next)
            rewards.append(r)

        T = len(rewards) - 1    # total timesteps

        for t in range(T):
            # Compute n-step return from t
            G = 0.0
            for i in range(1, self.n + 1):
                t_i = t + i
                if t_i > T:
                    break
                G += (self.gamma ** (i - 1)) * rewards[t_i]

            # Bootstrap at t+n if not terminal
            t_n = t + self.n
            if t_n <= T and states[t_n] not in self.terminal_states:
                G += (self.gamma ** self.n) * self.V[states[t_n]]

            # Update V(s_t)
            s_t = states[t]
            if s_t not in self.terminal_states:
                self.V[s_t] += self.alpha * (G - self.V[s_t])

    def rmse(self, true_V: np.ndarray) -> float:
        non_terminal = [i for i in range(len(self.V)) if i not in self.terminal_states]
        return float(np.sqrt(np.mean((self.V[non_terminal] - true_V[non_terminal])**2)))


# ---------------------------------------------------------------------------
# Training loop with RMSE tracking
# ---------------------------------------------------------------------------

def train_and_track(
    agent_cls,
    agent_kwargs: Dict,
    n_episodes: int = 100,
    rng_seed: int = 42,
) -> List[float]:
    """Train an agent for n_episodes and track RMSE after each episode."""
    rng = np.random.default_rng(rng_seed)
    env = RandomWalkEnv(rng=rng)
    agent = agent_cls(**agent_kwargs)
    rmse_curve = []

    for _ in range(n_episodes):
        agent.run_episode(env)
        rmse_curve.append(agent.rmse(RandomWalkEnv.TRUE_V))

    return rmse_curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Temporal Difference Learning — TD(0), TD(lambda), N-Step TD")
    print("=" * 60)

    N_EPISODES = 100
    N_RUNS = 30       # average over multiple seeds for smooth RMSE curves

    # ----------------------------------------------------------------
    # 1. Single-run demo: TD(0) value estimates
    # ----------------------------------------------------------------
    print("\n--- TD(0): Value Estimates After 100 Episodes ---")
    rng = np.random.default_rng(42)
    env = RandomWalkEnv(rng=rng)
    td0 = TD0Agent(n_states=7, alpha=0.1, gamma=1.0)
    for _ in range(N_EPISODES):
        td0.run_episode(env)

    print(f"  {'State':>6}  {'Learned V':>10}  {'True V':>10}  {'Error':>10}")
    for s in range(1, 6):
        print(f"  {s:>6}  {td0.V[s]:>10.4f}  "
              f"{RandomWalkEnv.TRUE_V[s]:>10.4f}  "
              f"{abs(td0.V[s] - RandomWalkEnv.TRUE_V[s]):>10.4f}")

    # ----------------------------------------------------------------
    # 2. RMSE curves: TD(lambda) at different lambda values
    # ----------------------------------------------------------------
    print("\n--- RMSE at 100 Episodes: TD(lambda) vs lambda ---")
    print(f"  {'lambda':>8}  {'Final RMSE':>12}  {'Best RMSE':>12}")

    lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = {}

    for lam in lambda_values:
        all_rmse = []
        for seed in range(N_RUNS):
            curve = train_and_track(
                TDLambdaAgent,
                {"n_states": 7, "alpha": 0.1, "gamma": 1.0, "lam": lam},
                n_episodes=N_EPISODES,
                rng_seed=seed,
            )
            all_rmse.append(curve)
        mean_curve = np.mean(all_rmse, axis=0)
        results[lam] = mean_curve
        print(f"  {lam:>8.1f}  {mean_curve[-1]:>12.5f}  {mean_curve.min():>12.5f}")

    # ----------------------------------------------------------------
    # 3. N-Step TD comparison
    # ----------------------------------------------------------------
    print("\n--- RMSE at 100 Episodes: N-Step TD vs n ---")
    print(f"  {'n':>6}  {'Final RMSE':>12}")

    for n in [1, 2, 4, 8, 16]:
        all_rmse = []
        for seed in range(N_RUNS):
            curve = train_and_track(
                NStepTDAgent,
                {"n_states": 7, "n": n, "alpha": 0.1, "gamma": 1.0},
                n_episodes=N_EPISODES,
                rng_seed=seed,
            )
            all_rmse.append(curve)
        mean_final = float(np.mean([c[-1] for c in all_rmse]))
        print(f"  {n:>6}  {mean_final:>12.5f}")

    # ----------------------------------------------------------------
    # 4. Alpha sensitivity for TD(0)
    # ----------------------------------------------------------------
    print("\n--- Alpha Sensitivity: TD(0) at 100 Episodes ---")
    print(f"  {'alpha':>8}  {'Final RMSE':>12}")
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.4]:
        t0 = time.time()
        all_rmse = []
        for seed in range(N_RUNS):
            curve = train_and_track(
                TD0Agent,
                {"n_states": 7, "alpha": alpha, "gamma": 1.0},
                n_episodes=N_EPISODES,
                rng_seed=seed,
            )
            all_rmse.append(curve[-1])
        elapsed = time.time() - t0
        print(f"  {alpha:>8.2f}  {np.mean(all_rmse):>12.5f}  ({elapsed:.2f}s)")

    print("\nDone.")
