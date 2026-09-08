# 19 — Multi-Agent Reinforcement Learning

## 1. Detailed Explanation

Multi-Agent Reinforcement Learning (MARL) studies how multiple agents learn simultaneously in a shared environment, where each agent's actions affect others' observations, rewards, and optimal strategies. Unlike single-agent RL, MARL must address non-stationarity: as each agent improves, the environment appears to change from every other agent's perspective — making standard convergence guarantees break down.

MARL settings span a spectrum. **Cooperative** games have agents maximizing a shared team reward (warehouse robots, traffic control). **Competitive** (zero-sum) games have agents with opposing rewards (chess, poker). **Mixed** games combine both (autonomous driving has cooperative safety goals but competitive lane-merging).

The solution concept is **Nash equilibrium**: a strategy profile σ* where no agent can improve by unilaterally deviating. In zero-sum 2-player games, Nash equilibrium is unique (minimax theorem). In general-sum games, multiple equilibria exist and agents may oscillate between them.

**CTDE (Centralized Training, Decentralized Execution)** is the dominant MARL paradigm: during training, a centralized critic sees all agents' observations and actions, providing stable value estimates; at execution, each agent acts using only its local observation. This handles non-stationarity during training while keeping inference decentralized. MADDPG (Multi-Agent DDPG) implements CTDE: agent i has actor μ_i(o_i) and centralized critic Q_i(o_1,...,o_n, a_1,...,a_n).

MARL is deployed in game AI (OpenAI Five Dota), multi-robot coordination, adversarial cybersecurity, and financial market simulation.

---

## 2. Core Intuition

Multi-agent RL is like traffic engineering: each driver optimizes their own route, but their collective behavior determines whether the city gridlocks or flows. Training them independently ignores that every policy change shifts what "optimal" means for everyone else. CTDE is like giving each driver a GPS that trains using city-wide traffic data but routes using only local sensors at execution time.

---

## 3. How It Works

**MADDPG (Cooperative CTDE):**

1. **Each agent i** has actor μ_i(o_i) and centralized critic Q_i(o, a) where o = (o_1,...,o_n), a = (a_1,...,a_n).
2. **Collect transitions**: all agents execute actions simultaneously; store (o, a, r, o') in shared replay buffer.
3. **Train critics**: for each agent i, minimize Bellman error using full joint obs/actions.
4. **Train actors**: agent i's actor maximizes Q_i(o, a) where a_j = μ_j(o_j) for all j.
5. **Soft target updates**: slowly update target networks θ_i' ← τ·θ_i + (1−τ)·θ_i'.
6. **Execute**: each agent uses only μ_i(o_i) — local observation → action.

```mermaid
flowchart TD
    A[All agents observe environment] --> B[Each agent i acts: a_i = mu_i(o_i)]
    B --> C[Receive joint reward r, next obs o']
    C --> D[Store (o, a, r, o') in shared replay buffer]
    D --> E[Sample minibatch]
    E --> F[Update centralized critic Q_i using joint obs+actions]
    F --> G[Update actor mu_i using gradient from Q_i]
    G --> H[Soft update target networks]
    H --> I{Episode done?}
    I -- no --> A
    I -- yes --> J[Evaluation: each agent uses mu_i(o_i) only]
```

---

## 4. Architecture / Trade-offs

### Independent Q-learning vs CTDE vs Self-Play

| Method | Non-stationarity handling | Execution | Works for |
|--------|--------------------------|-----------|-----------|
| Independent Q-learning | None (agents ignore each other) | Decentralized | Small cooperative tasks |
| CTDE (MADDPG/QMIX) | Strong (joint critic) | Decentralized | Cooperative with many agents |
| Self-play | Strong (symmetric) | Decentralized | Competitive/symmetric |
| Centralized policy | Full | Centralized | Small teams, full observability |

### Cooperative vs Competitive vs Mixed

| Setting | Reward structure | Key challenge | Typical algorithm |
|---------|-----------------|---------------|-------------------|
| Cooperative | Shared team reward | Credit assignment | QMIX, MADDPG (coop) |
| Zero-sum | r_1 = -r_2 | Non-convergence | Nash-Q, self-play |
| Mixed | Partial overlap | Equilibrium selection | MADDPG (mixed) |
| Stackelberg | Leader-follower | Asymmetric optimization | Bi-level optimization |

### Scalability vs Centralization

| Approach | Max agents | Critic input size | Communication |
|----------|-----------|-------------------|---------------|
| MADDPG | ~5 agents | O(n·dim_o + n·dim_a) | None |
| QMIX | ~10 agents | Monotonic mixing | None |
| CommNet | ~20 agents | Variable | Continuous |
| MAAC (attention) | ~30 agents | Attention-weighted | Implicit |

---

## 5. Interview Q&A

**Q: Why does independent Q-learning fail in multi-agent settings even when it converges in single-agent?**
A: Each agent sees a non-stationary environment: as other agents' policies change, the transition dynamics and reward distributions change from any individual agent's perspective. Q-learning's convergence proof assumes a stationary MDP, which is violated here. In practice, agents' Q-values oscillate rather than converge, especially in competitive games.

**Q: What is the credit assignment problem in cooperative MARL, and how does QMIX address it?**
A: With a shared team reward, it's unclear which agent contributed to success. Independent agents can't distinguish "I did well while others failed" from "team succeeded because of me." QMIX addresses this by learning a monotonic mixing function Q_tot(o, a) = mix(Q_1,...,Q_n) where ∂Q_tot/∂Q_i ≥ 0 — ensuring agents can improve the team by improving individually.

**Q: When does CTDE break down in practice?**
A: CTDE assumes all agents' observations/actions are available during training — this fails when: (1) agents have heterogeneous action spaces making joint action space exponential, (2) number of agents is very large (>20), making centralized critic input huge, or (3) agents are trained asynchronously. For large n, attention-based critics (MAAC) help by weighting agent contributions.

**Q: What is reward shaping in cooperative MARL, and what can go wrong?**
A: Reward shaping adds potential-based terms to help agents learn: r_shaped = r + γΦ(s') − Φ(s). Done correctly (with proper potential function), it doesn't change the optimal joint policy. Done incorrectly, agents learn to exploit the shaping signal — e.g., an agent repeatedly triggers a "partial success" reward without completing the task.

**Q: How would you approach convergence in a competitive zero-sum game?**
A: Use self-play with a league: maintain a population of past policy checkpoints; train current agent against a mixture of historical opponents. This prevents cycling to exploiting specific weaknesses (rock-paper-scissors cycling) and converges toward Nash. Pure self-play against the latest opponent can cycle indefinitely.

**Q: What emergent behaviors arise in MARL that weren't explicitly programmed?**
A: Cooperative agents develop specialization (one agent scouts, another attacks) and communication protocols. Competitive agents discover deceptive strategies. Mixed settings produce social dilemma dynamics (tragedy of the commons). These emerge from reward structure and environment alone — no explicit programming needed.

---

## 6. Best Practices

- Use CTDE for cooperative tasks with up to 10 agents; switch to attention-based critics (MAAC) for larger teams.
- In self-play, maintain a league of at least 10–20 historical checkpoints to prevent policy cycling.
- Normalize rewards across agents to prevent one agent's scale from dominating joint critic gradients.
- For competitive games, start with self-play rather than hand-crafted opponents — human-designed heuristics create exploitable policies.
- Use communication bandwidth constraints in cooperative tasks; unlimited communication leads to lazy agents who defer to a "speaker."
- Monitor convergence via Nash-gap (measure of how much any agent gains by deviating) rather than just reward.
- In mixed games, separate reward components clearly — team reward and individual incentive should be logged separately.

---

## 7. Common Pitfalls

- **Non-stationarity ignored in independent learners**: Symptom: training reward oscillates without convergence. Fix: switch to CTDE or add opponent modeling (infer other agents' policies explicitly).
- **Centralized critic too large for many agents**: Symptom: critic loss doesn't decrease with n>8 agents. Fix: use attention-based mixing or QMIX's monotonic factorization.
- **Reward hacking in cooperative MARL**: Symptom: team metric doesn't improve despite individual rewards rising. Fix: use dense team reward with sparse individual bonuses; audit each agent's behavior separately.
- **Competitive agents cycling**: Symptom: agent A beats B, B beats C, C beats A — no improvement. Fix: league/historical self-play; Nash averaging over checkpoint population.
- **Exploration collapse in cooperative tasks**: Symptom: agents quickly converge to one coordination pattern even when better ones exist. Fix: use parameter noise, independent random seeds per agent, or diverse reward initialization.

---

## 8. Related Concepts

- [16-model-based-rl](./16-model-based-rl.md) — Multi-agent world models enable centralized planning
- [17-rlhf](./17-rlhf.md) — Constitutional AI uses multi-agent self-critique variants
- [18-inverse-rl](./18-inverse-rl.md) — Multi-agent IRL infers reward for interacting agents
- [20-offline-rl](./20-offline-rl.md) — Offline MARL trains on fixed multi-agent interaction logs
- [08-actor-critic](./10-actor-critic.md) — MADDPG extends Actor-Critic to multi-agent CTDE
