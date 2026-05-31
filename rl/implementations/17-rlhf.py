"""
RLHF — Reward Model + Bradley-Terry + PPO with KL Penalty
==========================================================
Implements the full RLHF pipeline on a toy 8-token language model:
  1. SFT (supervised fine-tuning checkpoint — initialised, not trained here)
  2. RewardModel: learns human preferences from pairwise comparisons (Bradley-Terry)
  3. BradleyTerryTrainer: trains RM from preference pairs
  4. PPOWithKL: policy gradient update with KL penalty to prevent reward hacking

Everything is numpy. No torch, transformers, or gym required.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Toy Language Model (8 tokens, sequences of length 5)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 8
SEQ_LEN = 5


class ToyLM:
    """
    Softmax language model over 8 tokens.
    State: sequence of previously generated tokens (int list).
    Action: next token to generate (0-7).
    Policy: pi(a | state) = softmax(W_a * h(state) + b_a)
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        # Embedding: map token -> feature vector (dim=8)
        self.embeddings = rng.normal(0, 0.1, (vocab_size, vocab_size))
        # Policy head: 8 -> 8
        self.W = rng.normal(0, 0.1, (vocab_size, vocab_size))
        self.b = np.zeros(vocab_size)

    def _context_vec(self, tokens: List[int]) -> np.ndarray:
        """Mean of token embeddings as context representation."""
        if not tokens:
            return np.zeros(self.vocab_size)
        return np.mean([self.embeddings[t] for t in tokens], axis=0)

    def probs(self, context: List[int]) -> np.ndarray:
        """Return next-token probabilities."""
        h = self._context_vec(context)
        logits = h @ self.W + self.b
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def sample(self, context: List[int], rng: np.random.Generator) -> int:
        """Sample next token."""
        p = self.probs(context)
        return int(rng.choice(self.vocab_size, p=p))

    def log_prob_sequence(self, tokens: List[int]) -> float:
        """Log probability of a token sequence under this model."""
        lp = 0.0
        for t_idx in range(1, len(tokens)):
            context = tokens[:t_idx]
            p = self.probs(context)
            lp += np.log(p[tokens[t_idx]] + 1e-10)
        return lp

    def generate(self, rng: np.random.Generator, length: int = SEQ_LEN) -> List[int]:
        """Autoregressively generate a sequence."""
        tokens = [int(rng.integers(self.vocab_size))]   # start token
        for _ in range(length - 1):
            tokens.append(self.sample(tokens, rng))
        return tokens

    def update(self, tokens: List[int], advantages: np.ndarray, lr: float = 1e-2) -> None:
        """Simple policy gradient update for one sequence."""
        for t_idx in range(1, len(tokens)):
            context = tokens[:t_idx]
            a = tokens[t_idx]
            adv = advantages[t_idx - 1] if t_idx - 1 < len(advantages) else 0.0

            h = self._context_vec(context)
            p = self.probs(context)

            # Gradient of log pi(a|context) w.r.t. logits
            d_logits = -p.copy()
            d_logits[a] += 1.0

            # Gradient w.r.t. W and b
            dW = np.outer(h, d_logits) * adv
            db = d_logits * adv

            # Gradient ascent
            self.W += lr * dW
            self.b += lr * db


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------

class RewardModel:
    """
    Scalar reward model R_phi(sequence) -> float.
    Implemented as a linear model over mean token embeddings.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        # Token embeddings for reward model
        self.embeddings = rng.normal(0, 0.1, (vocab_size, vocab_size))
        # Linear head
        self.w = rng.normal(0, 0.1, vocab_size)
        self.b = 0.0

    def _seq_vec(self, tokens: List[int]) -> np.ndarray:
        return np.mean([self.embeddings[t] for t in tokens], axis=0)

    def score(self, tokens: List[int]) -> float:
        """Return a scalar reward score for a token sequence."""
        h = self._seq_vec(tokens)
        return float(h @ self.w + self.b)

    def update(self, winner: List[int], loser: List[int], lr: float = 1e-2) -> float:
        """
        Bradley-Terry update: R(winner) - R(loser) should be positive.
        Uses logistic loss: -log sigmoid(R(winner) - R(loser))
        Returns loss.
        """
        r_w = self.score(winner)
        r_l = self.score(loser)
        delta = r_w - r_l
        # Sigmoid and loss
        sig = 1.0 / (1.0 + np.exp(-delta))
        loss = -np.log(sig + 1e-10)

        # Gradient w.r.t. reward difference
        grad_delta = sig - 1.0    # d_loss / d_delta = sigmoid(delta) - 1

        # Update winner embedding sum (positive direction)
        h_w = self._seq_vec(winner)
        h_l = self._seq_vec(loser)

        self.w -= lr * grad_delta * (h_w - h_l)
        self.b -= lr * grad_delta

        return float(loss)


# ---------------------------------------------------------------------------
# Bradley-Terry Trainer
# ---------------------------------------------------------------------------

class BradleyTerryTrainer:
    """
    Train the reward model from pairwise preference data.
    Preferences: (sequence_a, sequence_b, preferred=a or b)
    """

    def __init__(self, reward_model: RewardModel, lr: float = 5e-3):
        self.rm = reward_model
        self.lr = lr

    def generate_preferences(
        self,
        n_pairs: int,
        rng: np.random.Generator,
        true_reward_fn,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Generate n_pairs preference comparisons.
        True reward function is used to determine human preference (oracle).
        Returns list of (winner, loser) pairs.
        """
        pairs = []
        for _ in range(n_pairs):
            seq_a = [int(rng.integers(VOCAB_SIZE)) for _ in range(SEQ_LEN)]
            seq_b = [int(rng.integers(VOCAB_SIZE)) for _ in range(SEQ_LEN)]
            ra = true_reward_fn(seq_a)
            rb = true_reward_fn(seq_b)
            if ra >= rb:
                pairs.append((seq_a, seq_b))
            else:
                pairs.append((seq_b, seq_a))
        return pairs

    def train(self, pairs: List[Tuple], n_epochs: int = 5) -> List[float]:
        """Train on preference pairs for n_epochs. Returns loss curve."""
        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for winner, loser in pairs:
                loss = self.rm.update(winner, loser, lr=self.lr)
                epoch_loss += loss
            losses.append(epoch_loss / len(pairs))
        return losses


# ---------------------------------------------------------------------------
# PPO with KL Penalty (RLHF-style)
# ---------------------------------------------------------------------------

class PPOWithKL:
    """
    RLHF-style PPO: maximise reward model score subject to KL penalty.
    r_total = r_RM(seq) - beta * KL(pi_RL || pi_SFT)
    """

    def __init__(
        self,
        policy: ToyLM,
        ref_policy: ToyLM,
        reward_model: RewardModel,
        beta: float = 0.1,
        lr: float = 5e-3,
    ):
        self.policy = policy
        self.ref = ref_policy
        self.rm = reward_model
        self.beta = beta
        self.lr = lr

    def _kl_from_ref(self, tokens: List[int]) -> float:
        """KL(pi_RL || pi_SFT) averaged over positions in sequence."""
        kl = 0.0
        for t_idx in range(1, len(tokens)):
            context = tokens[:t_idx]
            p_rl = self.policy.probs(context)
            p_sft = self.ref.probs(context)
            # KL(rl || sft) = sum pi_rl * log(pi_rl / pi_sft)
            kl += float(np.sum(p_rl * np.log(p_rl / (p_sft + 1e-10) + 1e-10)))
        return kl / max(len(tokens) - 1, 1)

    def shaped_reward(self, tokens: List[int]) -> float:
        """Total RLHF reward: r_RM - beta * KL(pi || pi_ref)."""
        r_rm = self.rm.score(tokens)
        kl = self._kl_from_ref(tokens)
        return r_rm - self.beta * kl

    def update(self, sequences: List[List[int]]) -> Dict:
        """
        Policy gradient update on a batch of sequences.
        Advantage = shaped_reward - mean(shaped_reward) [per-batch baseline].
        Returns diagnostics.
        """
        rewards = [self.shaped_reward(seq) for seq in sequences]
        rm_scores = [self.rm.score(seq) for seq in sequences]
        kl_vals = [self._kl_from_ref(seq) for seq in sequences]

        mean_reward = np.mean(rewards)
        advantages_scalar = [r - mean_reward for r in rewards]

        for seq, adv_scalar in zip(sequences, advantages_scalar):
            # Uniform advantage over all tokens in sequence
            advantages = np.full(len(seq) - 1, adv_scalar)
            self.policy.update(seq, advantages, lr=self.lr)

        return {
            "mean_reward": float(mean_reward),
            "mean_rm_score": float(np.mean(rm_scores)),
            "mean_kl": float(np.mean(kl_vals)),
        }


# ---------------------------------------------------------------------------
# True reward function (oracle — only used for preference generation)
# ---------------------------------------------------------------------------

def true_reward(tokens: List[int]) -> float:
    """
    Oracle reward: prefer sequences with more low-index tokens (0, 1, 2).
    Simulates "human preference" for concise, simple language.
    """
    return float(sum(1.0 / (t + 1) for t in tokens))


# ---------------------------------------------------------------------------
# Main: full RLHF pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("RLHF — Bradley-Terry Reward Model + PPO with KL Penalty")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # ----------------------------------------------------------------
    # Stage 1: Train Reward Model from preferences
    # ----------------------------------------------------------------
    print("\n--- Stage 1: Training Reward Model (Bradley-Terry) ---")
    rm = RewardModel(seed=1)
    sft_policy = ToyLM(seed=2)      # SFT checkpoint (reference policy)
    rl_policy = ToyLM(seed=2)       # RL policy (initialised from SFT)

    trainer = BradleyTerryTrainer(rm, lr=5e-3)
    pref_pairs = trainer.generate_preferences(500, rng, true_reward)
    losses = trainer.train(pref_pairs, n_epochs=10)
    print(f"RM training: initial loss={losses[0]:.4f}, final loss={losses[-1]:.4f}")

    # Validate RM alignment with true reward
    test_seqs = [[int(rng.integers(VOCAB_SIZE)) for _ in range(SEQ_LEN)] for _ in range(100)]
    true_scores = [true_reward(s) for s in test_seqs]
    rm_scores = [rm.score(s) for s in test_seqs]
    # Rank correlation
    rank_true = np.argsort(true_scores)
    rank_rm = np.argsort(rm_scores)
    overlap = len(set(rank_true[-20:]) & set(rank_rm[-20:]))
    print(f"Top-20 rank overlap (RM vs true): {overlap}/20")

    # ----------------------------------------------------------------
    # Stage 2: PPO with KL penalty
    # ----------------------------------------------------------------
    print("\n--- Stage 2: PPO with KL Penalty (RLHF Training) ---")
    ppo = PPOWithKL(
        policy=rl_policy,
        ref_policy=sft_policy,
        reward_model=rm,
        beta=0.1,
        lr=1e-2,
    )

    N_ITERATIONS = 50
    BATCH_SIZE = 32
    history: List[Dict] = []

    t0 = time.time()
    for itr in range(N_ITERATIONS):
        batch = [rl_policy.generate(rng, length=SEQ_LEN) for _ in range(BATCH_SIZE)]
        info = ppo.update(batch)
        history.append(info)
        if (itr + 1) % 10 == 0:
            print(f"  Iter {itr+1:3d} | mean_reward={info['mean_reward']:+.4f} | "
                  f"rm_score={info['mean_rm_score']:+.4f} | "
                  f"kl={info['mean_kl']:.4f}")
    elapsed = time.time() - t0
    print(f"PPO training time: {elapsed:.1f}s")

    # ----------------------------------------------------------------
    # Stage 3: Evaluate improvement
    # ----------------------------------------------------------------
    print("\n--- Stage 3: Policy Evaluation ---")
    eval_seqs = [rl_policy.generate(rng, length=SEQ_LEN) for _ in range(200)]
    sft_seqs  = [sft_policy.generate(rng, length=SEQ_LEN) for _ in range(200)]

    rl_true_rew = np.mean([true_reward(s) for s in eval_seqs])
    sft_true_rew = np.mean([true_reward(s) for s in sft_seqs])
    rl_kl = np.mean([ppo._kl_from_ref(s) for s in eval_seqs])

    print(f"SFT policy avg true reward:  {sft_true_rew:.4f}")
    print(f"RL  policy avg true reward:  {rl_true_rew:.4f}")
    print(f"Improvement:                 {rl_true_rew - sft_true_rew:+.4f}")
    print(f"RL policy avg KL from SFT:   {rl_kl:.4f}")

    # KL tracking across training
    print("\n--- KL Tracking Across Training ---")
    print(f"  {'Iter':>6}  {'KL':>8}  {'RM score':>10}  {'Total reward':>12}")
    for i in [0, 9, 24, 49]:
        h = history[i]
        print(f"  {i+1:>6}  {h['mean_kl']:>8.4f}  {h['mean_rm_score']:>10.4f}  "
              f"{h['mean_reward']:>12.4f}")

    print("\nDone.")
