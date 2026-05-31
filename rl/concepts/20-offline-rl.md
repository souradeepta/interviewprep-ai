# 20 — Offline Reinforcement Learning

## 1. Detailed Explanation

Offline (or batch) reinforcement learning learns a policy entirely from a fixed, pre-collected dataset D without any additional environment interaction. This matters enormously in domains where online exploration is dangerous (healthcare, autonomous driving), expensive (physical robotics), or simply impossible (historical records). The dataset was collected by some behavior policy π_β — a human, a heuristic, or a previous RL agent — and the offline RL agent must extract a better policy from this fixed data.

The central challenge is **distributional shift and bootstrapping error**: standard Q-learning applies the Bellman backup Q(s,a) = r + γ max_a' Q(s',a'). The max over a' may select actions not in the dataset — out-of-distribution (OOD) actions. When the Q-network has never seen these actions, it assigns arbitrarily high Q-values (it has no evidence they're bad). This **Q-value overestimation on OOD actions** causes the learned policy to select catastrophically bad actions, and because Q-values are used for subsequent bootstrapping, the error propagates and explodes.

**Conservative Q-Learning (CQL)** adds a penalty on Q-values under the current policy while maximizing Q-values under the dataset: L_CQL = α·(E_{a~π}[Q(s,a)] - E_{(s,a)~D}[Q(s,a)]) + Bellman error. This forces Q-values to be conservative (lower) on actions the behavior policy never took, preventing overestimation on OOD actions.

**BCQ (Batch-Constrained Q-learning)** takes a different approach: constrain the policy to only select actions similar to those in the dataset, using a generative model of the behavior policy.

Offline RL is used in recommendation systems (historical user logs), clinical decision support (patient records), and autonomous driving (simulation from logged data).

---

## 2. Core Intuition

Offline RL is like becoming a chess strategist by studying only recorded games — you can't play test games to see what works. The danger is convincing yourself that an untested move is brilliant because you've never seen evidence it fails. CQL is the discipline of being conservative about unplayed moves: assume they're bad unless the records say otherwise.

---

## 3. How It Works

**CQL (Conservative Q-Learning):**

1. **Initialize** Q-network Q_θ and policy π_φ.
2. **For each training batch** from fixed dataset D:
3. **Standard Bellman backup**: L_Bellman = E_{(s,a,r,s')~D}[(Q_θ(s,a) - (r + γ·max_a' Q_θ'(s',a')))²].
4. **CQL penalty**: L_CQL = E_{s~D}[E_{a~π_φ}[Q_θ(s,a)] - E_{a~D(·|s)}[Q_θ(s,a)]].
   - First term: penalize high Q-values for policy actions (may be OOD).
   - Second term: reward high Q-values for dataset actions (known to be reachable).
5. **Combined loss**: L_total = L_Bellman + α·L_CQL.
6. **Policy update**: π_φ = argmax_π E_{s~D}[E_{a~π}[Q_θ(s,a)]].
7. **Iterate** until Q-values stabilize (no divergence).

```mermaid
flowchart TD
    A[Fixed offline dataset D] --> B[Sample batch (s,a,r,s')]
    B --> C[Compute Bellman backup target]
    B --> D[Compute CQL penalty: penalize Q on policy actions]
    B --> E[Reward Q on dataset actions]
    C --> F[Combined CQL loss]
    D --> F
    E --> F
    F --> G[Update Q-network theta]
    G --> H[Update policy: max Q over dataset states]
    H --> I{Q-values stable?}
    I -- no --> B
    I -- yes --> J[Evaluate offline policy]
```

---

## 4. Architecture / Trade-offs

### Standard Q-learning vs CQL vs BCQ on Offline Data

| Method | OOD handling | Q-value behavior | Dataset coverage required | Typical offline perf |
|--------|-------------|-----------------|--------------------------|---------------------|
| Standard Q-learning | None | Diverges (OOD overestimation) | Full | Fails |
| Behavioral Cloning | N/A (no Q) | N/A | Any | Weak (copies behavior) |
| BCQ | Policy constraint | Stable | Moderate | Good |
| CQL | Q penalty | Conservative | Any | Strong |
| IQL (Implicit Q-learning) | Expectile regression | Stable | Any | Strong |

### CQL Penalty Strength α

| α value | Q conservatism | Policy performance | Risk |
|---------|---------------|-------------------|------|
| 0.0 (no CQL) | None | Diverges | OOD overestimation |
| 0.1 | Low | Near-optimal | Slight underperformance |
| 1.0 | Moderate | Good | Recommended default |
| 10.0 | High | Conservative | May be too restrictive |
| 100.0 | Very high | Matches BC | Over-conservative |

### Dataset Coverage Impact

| Behavior policy quality | Coverage | Offline RL outcome |
|------------------------|----------|--------------------|
| Near-optimal expert    | High     | Excellent (~expert perf) |
| Mixed (expert + random)| Moderate | Good              |
| Random only            | Low      | Poor (can't extract structure) |
| Adversarial/biased     | Skewed   | Poor or deceptive  |

---

## 5. Interview Q&A

**Q: Why does standard Q-learning diverge on offline data when it works fine online?**
A: Online, when Q selects an OOD action, the agent executes it and receives real reward feedback, correcting Q. Offline, OOD actions are never executed — Q-values on those actions are never corrected by real outcomes. Q keeps bootstrapping from these uncorrected (overestimated) values, leading to unbounded Q-values and policy collapse.

**Q: How does CQL prevent OOD overestimation without accessing the environment?**
A: CQL directly minimizes Q-values for actions the learned policy selects (which may be OOD) while maximizing Q-values for dataset actions. The dataset actions have real observed rewards — their Q-values are constrained by real Bellman targets. Policy actions that are OOD have their Q-values suppressed regardless, making the policy prefer dataset-supported actions.

**Q: When would offline RL outperform behavioral cloning even with the same dataset?**
A: When the dataset contains mixed-quality trajectories (some expert, some suboptimal). BC averages over all behaviors — it might clone random actions from suboptimal demonstrations. Offline RL with CQL can extract the high-value transitions and ignore low-value ones, potentially exceeding the average behavior quality. Stitching across trajectory segments is the key advantage.

**Q: What's the difference between offline RL and imitation learning?**
A: Imitation learning (behavioral cloning) only uses (s,a) pairs and learns π(a|s) — it doesn't use rewards. Offline RL uses (s,a,r,s') transitions with a reward signal, enabling the agent to reason about long-term value and potentially exceed demonstrator performance. IL matches behavior; offline RL optimizes reward.

**Q: How does dataset coverage affect CQL performance, and what can you do with limited coverage?**
A: Sparse coverage means many (s,a) pairs have never been seen — CQL's conservative Q-values may be overly pessimistic in unvisited regions, preventing generalization. Fix: use data augmentation (perturb states/actions within realistic bounds), or restrict policy to a support set estimated by a VAE/generative model of the behavior policy (BCQ approach).

**Q: What happens when you do online fine-tuning starting from a CQL-trained policy?**
A: The CQL policy is conservative but stable — it has a good initialization that doesn't take catastrophically bad actions. Online fine-tuning from this start is much safer than online RL from scratch, needing far fewer environment interactions to close the performance gap. This is the standard deployment pipeline: offline CQL → brief online RL fine-tuning.

---

## 6. Best Practices

- Always compare CQL against behavioral cloning as a baseline — if BC matches CQL, the dataset is sufficient and you don't need offline RL's complexity.
- Start with α=1.0 (CQL penalty); tune down if the policy is too conservative (underperforms BC significantly).
- Include both high-quality and diverse trajectories in the dataset — coverage matters more than size alone.
- Use offline policy evaluation metrics (OPE: doubly robust estimators, FQE) to select α before any real deployment.
- Monitor Q-value magnitudes during training; if max Q > 100x the maximum reward, OOD overestimation is occurring.
- Normalize rewards to [-1, 1] range; unnormalized rewards cause α to interact with reward scale unpredictably.
- After offline training, always fine-tune with a small number (100–1000) of online environment steps to close the distribution shift gap.

---

## 7. Common Pitfalls

- **OOD Q-value explosion**: Symptom: Q-values grow unboundedly after 10K training steps. Fix: add CQL penalty with α ≥ 0.5; verify α is non-zero in your loss.
- **Over-conservative policy (α too large)**: Symptom: CQL policy performs worse than BC. Fix: reduce α; monitor the gap between in-dataset and policy Q-values.
- **Dataset only contains one behavior**: Symptom: policy can't improve over the single demonstrator. Fix: collect diverse demonstrations; at minimum include some exploratory transitions.
- **Reward normalization ignored**: Symptom: CQL with α=1.0 behaves like α=0.001 or α=100 due to reward scale. Fix: normalize all rewards to [-1, 1] before training.
- **Assuming offline policy generalizes to all states**: Symptom: policy fails on states outside dataset distribution. Fix: use online fine-tuning; track distribution shift via feature drift metrics between dataset and deployment states.

---

## 8. Related Concepts

- [16-model-based-rl](./16-model-based-rl.md) — Model-based offline RL uses learned dynamics to avoid OOD actions
- [17-rlhf](./17-rlhf.md) — DPO is effectively offline RL on preference data
- [18-inverse-rl](./18-inverse-rl.md) — IRL also uses offline demonstrations; offline RL extends with reward
- [19-multi-agent-rl](./19-multi-agent-rl.md) — Offline MARL applies conservative Q-learning to logged multi-agent data
- [05-q-learning](./05-q-learning.md) — CQL extends Q-learning with conservative penalty
