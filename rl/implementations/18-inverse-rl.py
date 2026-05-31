"""
Inverse Reinforcement Learning — Behavioral Cloning and MaxEnt IRL
==================================================================
Implements two approaches to learning from expert demonstrations:
  1. BehavioralCloning: supervised policy learning from (s, a) pairs
  2. MaxEntIRL: iterative reward estimation (Ziebart et al. 2008)

Uses the 4x4 GridWorld. Expert demonstrations are generated from the known
optimal policy, then used to recover the reward function.

No external RL libraries required — only numpy.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# GridWorld MDP (shared physics)
# ---------------------------------------------------------------------------

class GridWorldMDP:
    """4x4 GridWorld for IRL experiments."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    N_STATES = 16
    N_ACTIONS = 4
    TERMINAL = {0, 15}

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.P, self.R_true = self._build_model()

    def _next(self, s: int, a: int) -> int:
        if s in self.TERMINAL:
            return s
        r, c = divmod(s, 4)
        if a == self.UP:    r = max(r - 1, 0)
        elif a == self.DOWN:  r = min(r + 1, 3)
        elif a == self.LEFT:  c = max(c - 1, 0)
        elif a == self.RIGHT: c = min(c + 1, 3)
        return r * 4 + c

    def _build_model(self) -> Tuple[np.ndarray, np.ndarray]:
        P = np.zeros((16, 4, 16))
        R = np.full((16, 4), -1.0)
        for s in range(16):
            for a in range(4):
                sn = self._next(s, a)
                P[s, a, sn] = 1.0
                if s in self.TERMINAL:
                    R[s, a] = 0.0
        return P, R

    def optimal_policy(self) -> np.ndarray:
        """Derive optimal deterministic policy via value iteration."""
        V = np.zeros(16)
        for _ in range(1000):
            V_old = V.copy()
            Q = self.R_true + self.gamma * (self.P @ V_old)
            for ts in self.TERMINAL:
                Q[ts, :] = 0.0
            V = Q.max(axis=1)
            if np.max(np.abs(V - V_old)) < 1e-8:
                break
        return Q.argmax(axis=1)

    def step(self, s: int, a: int) -> Tuple[int, float, bool]:
        sn = self._next(s, a)
        r = self.R_true[s, a]
        done = sn in self.TERMINAL
        return sn, r, done


# ---------------------------------------------------------------------------
# Expert demonstration generation
# ---------------------------------------------------------------------------

def generate_demonstrations(
    mdp: GridWorldMDP,
    policy: np.ndarray,
    n_demos: int = 50,
    max_steps: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> List[List[Tuple[int, int]]]:
    """
    Generate expert demonstrations as (state, action) sequences.
    Each demo starts from a random non-terminal state.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    demos = []
    non_terminal = [s for s in range(16) if s not in mdp.TERMINAL]
    for _ in range(n_demos):
        s = int(rng.choice(non_terminal))
        traj = []
        for _ in range(max_steps):
            a = int(policy[s])
            traj.append((s, a))
            sn, _, done = mdp.step(s, a)
            s = sn
            if done:
                break
        demos.append(traj)
    return demos


# ---------------------------------------------------------------------------
# 1. Behavioral Cloning
# ---------------------------------------------------------------------------

class BehavioralCloning:
    """
    Supervised cloning of expert policy from (state, action) pairs.
    Policy: softmax(W * one_hot(s) + b) — equivalent to a lookup table with softmax.
    """

    def __init__(self, n_states: int = 16, n_actions: int = 4, lr: float = 0.1):
        self.logits = np.zeros((n_states, n_actions))   # learnable logits
        self.lr = lr
        self.n_actions = n_actions

    def probs(self, s: int) -> np.ndarray:
        lg = self.logits[s] - self.logits[s].max()
        p = np.exp(lg)
        return p / p.sum()

    def predict(self, s: int) -> int:
        """Greedy action selection."""
        return int(self.probs(s).argmax())

    def update(self, s: int, a_expert: int) -> float:
        """
        Cross-entropy gradient step: log pi(a_expert | s).
        d/d_logits = e_a - pi(·|s)
        """
        p = self.probs(s)
        loss = float(-np.log(p[a_expert] + 1e-10))
        d_logits = p.copy()
        d_logits[a_expert] -= 1.0    # -(e_a - p) = gradient of CE loss
        self.logits[s] -= self.lr * d_logits
        return loss

    def train(self, demos: List[List[Tuple[int, int]]], n_epochs: int = 20) -> List[float]:
        """Train on all (s, a) pairs from demonstrations."""
        # Flatten demos into (s, a) pairs
        sa_pairs = [(s, a) for demo in demos for (s, a) in demo]
        losses = []
        for epoch in range(n_epochs):
            np.random.shuffle(sa_pairs)
            epoch_loss = sum(self.update(s, a) for s, a in sa_pairs)
            losses.append(epoch_loss / len(sa_pairs))
        return losses

    def evaluate(self, true_policy: np.ndarray) -> float:
        """State-level accuracy vs expert policy."""
        correct = sum(self.predict(s) == true_policy[s] for s in range(16)
                      if s not in {0, 15})
        return correct / 14.0


# ---------------------------------------------------------------------------
# 2. MaxEnt IRL
# ---------------------------------------------------------------------------

class MaxEntIRL:
    """
    Maximum Entropy IRL (Ziebart et al. 2008) — tabular version.

    Algorithm:
      1. Parameterise reward as R_theta(s) (feature weights theta * phi(s))
      2. Compute the policy induced by current R_theta via soft value iteration
      3. Compute expected state visitation frequencies mu_theta
      4. Gradient: mu_expert - mu_theta (match expert feature expectations)
      5. Update theta via gradient ascent
    """

    def __init__(
        self,
        n_states: int = 16,
        n_actions: int = 4,
        gamma: float = 0.99,
        lr: float = 0.1,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.temperature = temperature

        # Reward weights: R(s) = theta[s] (one weight per state = identity features)
        self.theta = rng.normal(0, 0.01, n_states)
        self.TERMINAL = {0, 15}

    def _soft_value_iteration(
        self,
        P: np.ndarray,
        R_est: np.ndarray,
        n_iter: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Soft value iteration for MaxEnt policy:
        Q(s,a) = R(s,a) + gamma * sum_s' P(s,a,s') V(s')
        V(s) = temperature * log sum_a exp(Q(s,a) / temperature)
        """
        V = np.zeros(self.n_states)
        for _ in range(n_iter):
            Q = R_est + self.gamma * (P @ V)
            # Soft max
            V_new = self.temperature * np.log(
                np.sum(np.exp(Q / self.temperature), axis=1) + 1e-10
            )
            for ts in self.TERMINAL:
                V_new[ts] = 0.0
            if np.max(np.abs(V_new - V)) < 1e-8:
                break
            V = V_new

        # Soft policy: pi(a|s) proportional to exp(Q(s,a) / T)
        Q = R_est + self.gamma * (P @ V)
        pi = np.exp(Q / self.temperature - Q.max(axis=1, keepdims=True))
        row_sums = pi.sum(axis=1, keepdims=True)
        pi /= (row_sums + 1e-10)
        for ts in self.TERMINAL:
            pi[ts, :] = 1.0 / self.n_actions    # uniform at terminal
        return V, pi

    def _state_visitation(
        self,
        pi: np.ndarray,
        P: np.ndarray,
        mu0: np.ndarray,
        horizon: int = 30,
    ) -> np.ndarray:
        """
        Compute expected state visitation frequencies under policy pi.
        mu[s] = sum_t gamma^t * P(s_t = s | pi)
        """
        mu = mu0.copy()
        total_mu = mu.copy()
        for _ in range(horizon):
            # Transition: mu_next[s'] = sum_s mu[s] * sum_a pi(a|s) * P(s,a,s')
            mu_next = np.zeros(self.n_states)
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    mu_next += mu[s] * pi[s, a] * P[s, a]
            mu = self.gamma * mu_next
            total_mu += mu
        return total_mu

    def _expert_visitation(
        self,
        demos: List[List[Tuple[int, int]]],
    ) -> np.ndarray:
        """Count normalised state visit frequencies in expert demonstrations."""
        mu = np.zeros(self.n_states)
        total = 0
        for demo in demos:
            for (s, _) in demo:
                mu[s] += 1
                total += 1
        return mu / max(total, 1)

    def train(
        self,
        demos: List[List[Tuple[int, int]]],
        P: np.ndarray,
        n_iter: int = 30,
    ) -> List[float]:
        """
        Train reward weights via gradient ascent on MaxEnt likelihood.
        Returns log-likelihood history.
        """
        mu_expert = self._expert_visitation(demos)
        non_terminal = [s for s in range(self.n_states) if s not in self.TERMINAL]
        # Uniform start state distribution
        mu0 = np.zeros(self.n_states)
        for s in non_terminal:
            mu0[s] = 1.0 / len(non_terminal)

        likelihoods = []

        for it in range(n_iter):
            # Current reward estimate: R(s,a) = theta[s] (broadcast over actions)
            R_est = np.tile(self.theta[:, None], (1, self.n_actions))
            for ts in self.TERMINAL:
                R_est[ts, :] = 0.0

            # Soft policy under current R
            V, pi = self._soft_value_iteration(P, R_est)

            # State visitation under soft policy
            mu_theta = self._state_visitation(pi, P, mu0)

            # Gradient: d_likelihood/d_theta = mu_expert - mu_theta
            gradient = mu_expert - mu_theta
            self.theta += self.lr * gradient

            # Log-likelihood (approximate)
            ll = float(np.sum(mu_expert * self.theta) - np.mean(V))
            likelihoods.append(ll)

        return likelihoods

    def recovered_reward(self) -> np.ndarray:
        """Return learned reward per state."""
        return self.theta.copy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compare_policies(
    bc_policy: BehavioralCloning,
    irl_theta: np.ndarray,
    mdp: GridWorldMDP,
    true_policy: np.ndarray,
) -> None:
    """Compare action agreement for BC and IRL-recovered policies vs true optimal."""
    # IRL policy: greedy w.r.t. recovered reward
    R_irl = np.tile(irl_theta[:, None], (1, 4))
    Q_irl = R_irl + mdp.gamma * (mdp.P @ irl_theta)
    irl_policy = Q_irl.argmax(axis=1)

    non_terminal = [s for s in range(16) if s not in {0, 15}]
    bc_acc = sum(bc_policy.predict(s) == true_policy[s] for s in non_terminal) / 14
    irl_acc = sum(irl_policy[s] == true_policy[s] for s in non_terminal) / 14

    print(f"\nPolicy Agreement vs Optimal (non-terminal states):")
    print(f"  Behavioral Cloning: {bc_acc:.1%}")
    print(f"  MaxEnt IRL:         {irl_acc:.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Inverse RL — Behavioral Cloning and MaxEnt IRL")
    print("=" * 60)

    rng = np.random.default_rng(42)
    mdp = GridWorldMDP(gamma=0.99)
    expert_policy = mdp.optimal_policy()

    print(f"\nExpert policy (deterministic greedy from V*):")
    action_names = ["^", "v", "<", ">"]
    for r in range(4):
        print("  " + "  ".join(
            " T " if r * 4 + c in mdp.TERMINAL else f" {action_names[expert_policy[r*4+c]]} "
            for c in range(4)
        ))

    # Generate expert demonstrations
    print("\n--- Generating Expert Demonstrations ---")
    demos = generate_demonstrations(mdp, expert_policy, n_demos=50, rng=rng)
    total_transitions = sum(len(d) for d in demos)
    print(f"Generated {len(demos)} demos, {total_transitions} total transitions")

    # ----------------------------------------------------------------
    # 1. Behavioral Cloning
    # ----------------------------------------------------------------
    print("\n--- Behavioral Cloning ---")
    t0 = time.time()
    bc = BehavioralCloning(lr=0.1)
    losses = bc.train(demos, n_epochs=20)
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.2f}s")
    print(f"Loss: initial={losses[0]:.4f}, final={losses[-1]:.4f}")
    print(f"Policy accuracy vs expert: {bc.evaluate(expert_policy):.1%}")

    # ----------------------------------------------------------------
    # 2. MaxEnt IRL
    # ----------------------------------------------------------------
    print("\n--- MaxEnt IRL ---")
    t0 = time.time()
    irl = MaxEntIRL(lr=0.1, temperature=0.5)
    ll_history = irl.train(demos, mdp.P, n_iter=30)
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.2f}s")
    print(f"Log-likelihood: initial={ll_history[0]:.4f}, final={ll_history[-1]:.4f}")

    # Recovered reward
    R_rec = irl.recovered_reward()
    print("\nRecovered reward per state (4x4 grid):")
    for r in range(4):
        row = "  ".join(
            f"{R_rec[r*4+c]:+.2f}" for c in range(4)
        )
        print("  " + row)

    # Compare
    compare_policies(bc, R_rec, mdp, expert_policy)

    print("\nDone.")
