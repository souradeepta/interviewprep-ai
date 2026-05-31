# Monte Carlo Methods

## 1. Detailed Explanation

Monte Carlo (MC) methods in RL estimate value functions by averaging the actual returns observed from complete episodes—no model required, no bootstrapping. The core idea is simple: if you want to know how good state s is, run many episodes that visit s, record the total discounted reward from each visit to the end of the episode, and average those returns. With enough samples, the average converges to the true V_pi(s) by the law of large numbers.

Two variants differ in how they handle repeated state visits within an episode. First-visit MC uses only the return from the first time s is encountered in each episode. Every-visit MC uses all returns from every visit to s. Both are unbiased—first-visit has lower variance when visits are correlated, every-visit has lower variance when episodes are long and visits are sparse.

MC methods have two defining properties that distinguish them from DP and TD: (1) they require no model—all you need is the ability to generate episodes; (2) they do not bootstrap—each estimate is based on a complete return from the actual trajectory, not a guess about future states. The absence of bootstrapping means MC estimates are unbiased (they average actual returns, not approximations), but they have higher variance compared to TD, because the full return r_t + gamma*r_{t+1} + ... varies considerably across episodes.

In production, MC is used in Monte Carlo Tree Search (MCTS), the backbone of AlphaGo and AlphaZero: simulate random playouts from a state, average the outcomes, and use them to guide tree search. Off-policy MC with importance sampling is also the foundation of many evaluation methods in recommendation systems, where you want to evaluate a new policy using data collected under an old one.

## 2. Core Intuition

Monte Carlo RL works like a pollster: instead of analytically computing how popular a candidate will be, you just ask many people and average the answers. If you visit a state and always average the total reward you actually collected from that point to the end of the episode, the average eventually tells you exactly how good that state is—no model of how other people will vote is needed.

## 3. How It Works

```
flowchart TD
    A[Initialize V-s = 0 and Returns-s = empty list for all s] --> B[Generate episode: s0 a0 r1 s1 a1 r2 ... sT using policy pi]
    B --> C[Compute returns backward: G = 0, for t from T-1 to 0 set G = gamma times G plus r_{t+1}]
    C --> D{First-visit MC?}
    D -- Yes --> E[If s_t appeared for first time in episode, append G to Returns-s_t]
    D -- No --> F[Every-visit: append G to Returns-s_t regardless]
    E --> G[V-s_t = mean of Returns-s_t]
    F --> G
    G --> H{Enough episodes?}
    H -- No --> B
    H -- Yes --> I[V converged to V_pi-s]
```

Step-by-step:

1. **Generate episode**: Follow policy pi from start state s_0. At each step, take action a_t ~ pi(s_t), receive r_{t+1}, transition to s_{t+1}. Stop at terminal state T.
2. **Compute returns backward**: Start from T with G=0. For t from T-1 down to 0: G = r_{t+1} + gamma*G. This accumulates the discounted return efficiently without recomputation.
3. **First-visit or every-visit check**: For first-visit MC, only record G for states appearing for the first time in the episode.
4. **Update average**: Maintain a running average Returns[s] = [G_1, G_2, ...]. V(s) = mean(Returns[s]).
5. **Incremental update**: Use V(s) ← V(s) + alpha*(G - V(s)) with small alpha for non-stationary settings.
6. **MC control**: After each episode, improve the policy greedily: pi(s) = argmax_a Q(s,a). Use epsilon-greedy to maintain exploration.

## 4. Architecture and Trade-offs

### First-Visit vs Every-Visit MC

| Property | First-Visit MC | Every-Visit MC |
|----------|---------------|---------------|
| Bias | Unbiased | Unbiased |
| Variance | Lower (independent per episode) | Lower (uses more data) |
| Data efficiency | One sample per visit per episode | Multiple samples per episode |
| Convergence | Converges to V_pi | Converges to V_pi |
| Recommended for | Short episodes, revisited states | Long episodes, rare states |

### MC vs DP vs TD Comparison

| Property | Monte Carlo | DP | TD Learning |
|----------|-------------|----|----|
| Requires model | No | Yes | No |
| Bootstrapping | No | Yes | Yes |
| Bias | Zero (unbiased) | Zero (with exact model) | Nonzero (bootstrap bias) |
| Variance | High (full return) | Zero (deterministic) | Low (one-step) |
| Convergence condition | Many episodes | Exact model + finite MDP | Appropriate step size |
| Handles continuing tasks | No (needs episodes) | Yes | Yes |

### On-Policy vs Off-Policy MC

| Variant | Data source | Policy evaluated | Correction needed |
|---------|-------------|-----------------|-------------------|
| On-policy MC | Episodes from pi | pi | None |
| Off-policy MC (IS) | Episodes from behavior policy b | Target policy pi | Importance weights w = prod pi(a|s)/b(a|s) |
| Off-policy MC (WIS) | Same | Same | Weighted IS ratio |

## 5. Interview Q&A

**Q: When would you prefer MC over TD learning for policy evaluation?**
A: Prefer MC when episodes are short and complete, rewards are delayed to the end (e.g., win/lose games where reward comes only at termination), and you want unbiased estimates. TD is better for long or continuing tasks where waiting for episode completion is impractical, or when sample efficiency matters (TD uses partial trajectories and bootstraps, which reduces variance at the cost of some bias).

**Q: Why does MC have high variance, and what are your options to reduce it?**
A: High variance comes from averaging full returns G_t = r_t + gamma*r_{t+1} + ..., which compounds stochasticity across all future steps. Reduction options: (1) Use a baseline: replace G_t with G_t - b(s_t), where b is a state-dependent baseline (like V(s_t)), leaving the expectation unchanged but reducing variance. (2) Use n-step returns (TD(lambda)) to trade off variance and bias. (3) Use control variates. (4) Collect more episodes—variance scales as 1/N.

**Q: How does importance sampling work in off-policy MC, and when does it fail?**
A: Importance weights correct for the difference between the behavior policy b (used to generate data) and the target policy pi (being evaluated): w_t = prod_{k=t}^{T-1} pi(a_k|s_k)/b(a_k|s_k). The corrected return w_T * G_t is an unbiased estimator of V_pi(s_t). It fails when the policies diverge heavily: if b(a|s) ≈ 0 but pi(a|s) is large, importance weights explode, causing very high variance. In practice, clip weights at a maximum value (e.g., 100) and use weighted IS instead of ordinary IS to reduce but not eliminate the bias.

**Q: What happens if you use MC control without epsilon-greedy exploration?**
A: The policy collapses to always selecting the action that happened to give the highest return in the few episodes seen. States not visited don't get updated; the policy never explores alternatives. This is the exploration problem—without epsilon-greedy or some other exploration mechanism, MC control can cycle or converge to a suboptimal policy because it has never visited the states that would yield higher returns.

**Q: How would you apply MC estimation in a recommendation system?**
A: Off-policy MC evaluation: the production system collects logs under some logging policy b. To evaluate a new policy pi without deploying it, compute importance-weighted returns: for each user session, weight the observed cumulative reward by the IS ratio over all recommended actions. This gives an unbiased estimate of pi's expected return using only logged data. The practical challenge is IS ratio variance; doubly-robust estimators (combining MC and DM/direct methods) are the production standard.

**Q: Why doesn't MC apply to continuing tasks without modification?**
A: MC requires episodic structure because it needs a terminal state to compute the full return G_t = sum_{k=0}^{T-t-1} gamma^k r_{t+k+1}. In a continuing task, T = infinity. Solutions: (1) Use discounted returns and approximate with gamma^K ≈ 0 for large K (treat episodes as finite with a cutoff). (2) Use the average-reward formulation. (3) Switch to TD methods, which bootstrap and don't require episode completion.

**Q: How do you implement incremental MC updates without storing all returns?**
A: Use the running mean formula: V(s) ← V(s) + (1/N_s) * (G - V(s)), where N_s is the visit count. For non-stationary problems, use a fixed step size alpha instead: V(s) ← V(s) + alpha * (G - V(s)). This gives more weight to recent episodes, which is better when the environment is non-stationary. The cost is that old data never fully decays to zero influence.

## 6. Best Practices

- **Compute returns backward**: Always compute G = r_{T-1} + gamma*(r_{T-2} + gamma*(...)) by iterating from the end of the episode, not forward. Forward computation requires O(T^2) work; backward is O(T).
- **Use first-visit MC as default**: It has better theoretical properties (cleaner unbiasedness proof) and is the most widely validated variant. Switch to every-visit only when state visits are rare.
- **Track visit counts per state**: Store N[s] alongside V[s] to enable incremental averaging and to identify states that are rarely visited (high uncertainty in estimate).
- **Warm-start with optimistic initial values**: Initialize Q(s,a) to a high value (R_max/(1-gamma)) to encourage exploration without requiring epsilon-greedy. This is "optimism in the face of uncertainty."
- **Use epsilon-annealed exploration**: Start with epsilon=1.0 (pure random) and decay to epsilon=0.05. A simple schedule: epsilon = max(0.05, 1.0 - episode/total_episodes).
- **Batch episodes before updating**: In parallel environments, collect 16-64 episodes before each policy update to reduce update variance.
- **Clip importance weights to [1e-3, 100] in off-policy MC**: Prevents weight explosion that can cause learning instability, at the cost of a small bias.

## 7. Common Pitfalls

- **Computing returns forward instead of backward**: A loop from t=0 to T summing gamma^(k-t)*r_k is O(T^2) and causes incorrect discounting if indices are off.
  - Symptom: V estimates are consistently too high (under-discounting) or too low (over-discounting); slow runtime on long episodes.
  - Fix: Compute G = 0; iterate t from T-1 to 0: G = r[t+1] + gamma*G; store G[t] = G.

- **Forgetting to reset Returns lists per episode in first-visit MC**: If a state appears multiple times in one episode, all occurrences may be treated as first-visit, over-counting samples.
  - Symptom: V estimates have lower variance than expected; convergence appears faster than it should.
  - Fix: At the start of each episode, track visited_this_episode = set(); only append to Returns[s] if s not in visited_this_episode.

- **No exploration in MC control**: Pure greedy policy update causes policy collapse to whichever action happened to perform well in early episodes.
  - Symptom: Policy converges in a few episodes; Q(s,a) for non-chosen actions stagnates at initial values.
  - Fix: Use epsilon-greedy policy: with probability epsilon pick random action, else greedy.

- **High IS weights crashing learning in off-policy MC**: When behavior and target policies diverge, importance weights can be 10^6 or higher, causing V estimates to jump wildly.
  - Symptom: V(s) oscillates with very high amplitude across episodes; does not converge.
  - Fix: Use weighted importance sampling (sum of weights in denominator) instead of ordinary IS; alternatively, clip weights at a maximum value.

## 8. Related Concepts

- [Markov Decision Processes](./01-markov-decision-processes.md) — Defines the episodic structure MC methods require.
- [Bellman Equations](./02-bellman-equations.md) — The DP alternative to MC; Bellman equations bootstrap while MC does not.
- [Dynamic Programming for RL](./03-dynamic-programming-rl.md) — Model-based alternative; compare DP (exact) vs MC (sampled).
- [Temporal Difference Learning](./05-temporal-difference-learning.md) — Combines MC sampling with Bellman bootstrapping; the practical middle ground.
