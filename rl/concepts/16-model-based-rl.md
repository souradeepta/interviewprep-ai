# 16 — Model-Based Reinforcement Learning

## 1. Detailed Explanation

Model-based reinforcement learning (MBRL) fundamentally differs from model-free methods by learning an explicit internal model of the environment — a function that predicts how the world transitions given actions. Rather than requiring the agent to interact with the real environment millions of times, MBRL lets the agent *simulate* experience from its learned model, dramatically improving sample efficiency.

The central algorithm is Dyna-Q, introduced by Sutton: after each real environment step, the agent (1) updates its Q-table via standard Q-learning, (2) updates the model M(s,a) → (r, s') using the observed transition, and (3) performs K additional simulated Q-learning updates by sampling random previously-seen (s,a) pairs and querying the model for simulated outcomes. The planning ratio K controls the compute-vs-sample-efficiency tradeoff — K=20 achieves in 20× fewer real steps what K=0 (pure Q-learning) needs.

Modern variants scale this to neural networks. A neural world model f_θ(s,a) → (r, s') is trained on collected experience and used to generate synthetic rollouts. MBPO (Model-Based Policy Optimization) interleaves real rollout collection, world-model training, and policy optimization over blended real+simulated data. The key challenge is **model error compounding**: over long simulated horizons, small prediction errors accumulate, making long rollouts from imperfect models harmful rather than helpful. This is why MBPO uses short rollouts (1–5 steps) from the model.

MBRL is used in robotics (learning physical dynamics), game-playing (AlphaZero's planning), and any data-scarce domain where real interactions are expensive — drug discovery, clinical trials, industrial control.

---

## 2. Core Intuition

Think of MBRL as a student who, after each class, doesn't just memorize the teacher's answer — they also mentally rehearse similar problems using a mental model of how math works. The more accurate that internal model, the more useful the rehearsal. When the mental model has gaps, rehearsing too long on wrong assumptions makes things worse, not better.

---

## 3. How It Works

**Dyna-Q Algorithm:**

1. **Initialize** Q-table Q(s,a) to zeros; initialize empty model M.
2. **Real step**: observe state s, choose action a (ε-greedy), observe r, s'.
3. **Q-update**: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)].
4. **Model update**: store M(s,a) = (r, s') — the observed transition.
5. **Planning loop** (K iterations): sample random (s_sim, a_sim) from past experience; get (r_sim, s_sim') = M(s_sim, a_sim); update Q(s_sim, a_sim) ← Q(s_sim, a_sim) + α[r_sim + γ max Q(s_sim',·) - Q(s_sim, a_sim)].
6. **Repeat** from step 2 until convergence.

```mermaid
flowchart TD
    A[Agent in state s] --> B[Take action a]
    B --> C[Observe r, s' from environment]
    C --> D[Q-learning update on real transition]
    C --> E[Update model M(s,a) = (r, s')]
    D --> F{K planning steps}
    E --> F
    F --> G[Sample (s_i, a_i) from past experience]
    G --> H[Query model for simulated (r_i, s_i')]
    H --> I[Q-learning update on simulated transition]
    I --> J{More planning?}
    J -- yes --> G
    J -- no --> A
```

---

## 4. Architecture / Trade-offs

### Planning Ratio K vs Sample Efficiency

| K (planning steps) | Real steps to converge | Compute per real step | Risk of model error |
|--------------------|----------------------|-----------------------|---------------------|
| 0 (pure Q-learning)| High (baseline)      | Low                   | None                |
| 5                  | ~3x fewer            | Moderate              | Low                 |
| 20                 | ~8x fewer            | High                  | Moderate            |
| 50+                | Diminishing returns  | Very high             | High (if model wrong)|

### Model-Free vs Model-Based vs Hybrid

| Approach       | Sample Efficiency | Computation | Asymptotic Performance | When to Use |
|----------------|-------------------|-------------|----------------------|-------------|
| Q-learning (K=0)| Low              | Low         | Good                 | Cheap env, many interactions |
| Dyna-Q (K=20)  | High              | Medium      | Good                 | Moderate data scarcity |
| Neural MBPO    | Very high         | High        | Near-optimal         | Data-scarce, complex dynamics |
| Pure planning  | Highest           | Very high   | Depends on model     | Perfect model available |

### Rollout Length vs Model Error Compounding

| Rollout length | Bias (from model error) | Variance | Recommended for |
|----------------|------------------------|----------|-----------------|
| 1 step         | Minimal                | High     | Inaccurate models |
| 3–5 steps      | Low                    | Medium   | Typical MBPO    |
| 10+ steps      | High (compounding)     | Low      | Very accurate models only |

---

## 5. Interview Q&A

**Q: When would you increase planning ratio K, and when would it hurt?**
A: Increase K when the environment model is accurate (tabular/deterministic) and real interactions are expensive. K hurts when the model has high error — simulated Q-updates then reinforce wrong values. A practical heuristic: monitor real vs predicted reward; if mean absolute error exceeds 5–10% of reward range, cap K at 5.

**Q: What breaks first when a neural world model encounters out-of-distribution states?**
A: The model confidently predicts wrong next-states (hallucination). Since the policy was optimized against these wrong predictions, it catastrophically exploits model errors. This is model exploitation vs model uncertainty — add epistemic uncertainty (ensemble disagreement) and terminate rollouts when uncertainty is high.

**Q: How does MBPO differ from Dyna-Q in practice?**
A: Dyna-Q uses tabular model with deterministic lookups; MBPO uses an ensemble of neural networks for the dynamics model, samples short rollouts (1–5 steps) to limit error compounding, and blends real+simulated data in a replay buffer. MBPO's short rollouts are the key insight for neural settings.

**Q: Why does Dyna-Q need to sample from *past* (s,a) pairs rather than random ones?**
A: The model M(s,a) is only defined for observed transitions — it has no predictions for (s,a) pairs never seen. Planning on unvisited pairs would extrapolate through unknown model regions, producing garbage Q-updates.

**Q: What's the compute cost tradeoff between model training and planning?**
A: Model training is O(N·d) per batch where d is state dimension; planning is O(K·Q-update). For small K and large N, model training dominates. For K=50, planning dominates. Real-world practice (MBPO): retrain model every 250 steps, plan K=5 per real step — model training takes ~60% of total compute.

**Q: How would you detect that your learned world model is hurting rather than helping?**
A: Compare policy performance with K=0 (no planning) to K=5 and K=20. If K=5 outperforms K=0 but K=20 is worse than K=5, model error is compounding at longer horizons. Fix: reduce K, increase model training data, or add ensemble uncertainty gating.

---

## 6. Best Practices

- Keep K between 5–20 for tabular models; stay at 1–5 for neural models until model loss stabilizes below 0.01 MSE.
- Use an ensemble of 3–7 world models; terminate rollouts when ensemble disagreement (std across predictions) exceeds a threshold.
- Maintain separate replay buffers for real and simulated data; weight real transitions 3:1 over simulated.
- Retrain the world model every 250–1000 real environment steps to prevent distribution shift between policy and model.
- Monitor model prediction error on a held-out validation set; stop planning if error exceeds 10% of reward scale.
- For neural models, predict reward and next-state separately — joint prediction tends to let reward head dominate gradients.
- Use short rollout horizons (1–5 steps) in neural MBRL; only increase after confirming model accuracy empirically.

---

## 7. Common Pitfalls

- **Too many planning steps with a poor model**: Symptom: performance worse than pure Q-learning. Fix: reduce K to 1–5, evaluate model error first.
- **Using model for states never visited**: Symptom: Q-values diverge on early training steps. Fix: only plan from states that appear in the real-experience replay buffer.
- **Model doesn't track policy distribution shift**: Symptom: model has low error on old data but high error on current states. Fix: periodically purge stale model data and retrain on recent transitions.
- **Forgetting to update model after every real step**: Symptom: planning Q-updates use stale model, no benefit from K>0. Fix: always interleave real step → Q-update → model update → K planning steps.
- **Not separating model and Q-function learning rates**: Symptom: one dominates, the other oscillates. Fix: use lr=1e-3 for model, lr=0.1 for tabular Q-table.

---

## 8. Related Concepts

- [01-markov-decision-process](./01-markov-decision-process.md) — MDP framework that defines the dynamics model
- [05-q-learning](./05-q-learning.md) — Model-free baseline that Dyna-Q extends
- [15-temporal-difference](./15-temporal-difference.md) — TD learning underlying Q-updates in planning
- [19-multi-agent-rl](./19-multi-agent-rl.md) — Multi-agent planning requires multi-agent world models
- [20-offline-rl](./20-offline-rl.md) — Offline RL also uses learned models to avoid OOD transitions
