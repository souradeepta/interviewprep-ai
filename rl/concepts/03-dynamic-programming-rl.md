# Dynamic Programming for Reinforcement Learning

## 1. Detailed Explanation

Dynamic programming (DP) in RL refers to a family of algorithms that compute exact optimal policies by exploiting the recursive Bellman equations. Unlike model-free RL, DP requires full knowledge of the MDP: the transition function P(s'|s,a) and reward function R(s,a,s'). Given this model, DP algorithms sweep through all states, computing and improving value estimates until convergence.

The two cornerstone DP algorithms are policy iteration and value iteration. Policy iteration alternates between policy evaluation (compute V_pi for a fixed policy until convergence) and policy improvement (greedily update the policy based on the current value function). It is guaranteed to converge in a finite number of policy improvement steps. Value iteration collapses this to a single Bellman optimality backup per state per sweep, trading the expensive full evaluation step for a simpler update—typically converging faster in practice despite requiring more sweeps.

Why does DP matter if it requires the model? Three reasons: (1) Many production systems have known dynamics—simulation-based planning, operations research, scheduling. (2) DP solutions serve as benchmarks to validate model-free algorithms on toy MDPs. (3) DP concepts (Bellman backup, policy evaluation, policy improvement, generalized policy iteration) underpin every modern RL algorithm, even deep RL. Understanding DP is understanding why actor-critic methods work.

A critical advanced variant is asynchronous DP, which updates states in any order, focusing computation on high-residual states (prioritized sweeping). This enables DP-style solutions on large state spaces that would be prohibitively expensive with synchronous full sweeps.

## 2. Core Intuition

Policy iteration is like repeatedly asking: "Given I think the world is worth this much, what is the best plan?" then asking "Given that plan, how much is the world actually worth?" Policy evaluation and policy improvement keep correcting each other until neither has anything more to say. Value iteration short-circuits this by always asking "what is the best possible value here?" without committing to a fixed plan.

## 3. How It Works

```
flowchart TD
    A[Initialize pi-0 randomly, V = 0] --> B[Policy Evaluation: iterate Bellman expectation for pi until V_pi converges]
    B --> C[Policy Improvement: for each state s set pi_new-s = argmax_a Q_pi-s-a]
    C --> D{pi_new == pi_old for all states?}
    D -- No --> E[Set pi = pi_new, go back to B]
    D -- Yes --> F[Policy is optimal: pi = pi-star, V = V-star]
```

Step-by-step:

1. **Initialize**: Set V(s) = 0 for all s; set pi(s) to any deterministic policy.
2. **Policy evaluation**: Apply Bellman expectation backup repeatedly: V(s) ← sum_a pi(a|s) sum_{s'} P(s'|s,a)[R + gamma*V(s')]. Repeat until max|V_new - V_old| < theta (typically 1e-6).
3. **Policy improvement**: For each state, compute Q_pi(s,a) = sum_{s'} P(s'|s,a)[R + gamma*V(s')] for all a. Set pi'(s) = argmax_a Q_pi(s,a).
4. **Convergence check**: If pi' == pi for all states, stop. Otherwise, set pi = pi' and return to step 2.
5. **Value iteration shortcut**: Skip full evaluation; just one step of Bellman optimality backup per state: V(s) ← max_a sum_{s'} P(s'|s,a)[R + gamma*V(s')]. Repeat until convergence.
6. **Extract final policy**: pi*(s) = argmax_a sum_{s'} P(s'|s,a)[R + gamma*V*(s')].

## 4. Architecture and Trade-offs

### Policy Iteration vs Value Iteration

| Property | Policy Iteration | Value Iteration |
|----------|-----------------|----------------|
| Per-step cost | O(|S|^3) for exact evaluation + O(|S|^2|A|) improvement | O(|S|^2|A|) per sweep |
| Sweeps to convergence | Few policy improvements (often < 20) | Many sweeps (hundreds) |
| Intermediate policies | Each intermediate pi is a valid policy | Intermediate V is not optimal policy |
| Best for | Small |S|, need policy at each step | Large |S|, care only about final V* |
| Modified variant | Truncated eval (K steps) | Same |

### Synchronous vs Asynchronous DP

| Variant | Update rule | Best for |
|---------|-------------|----------|
| Synchronous (full sweep) | All states updated once per sweep | Dense MDPs, exact convergence |
| Asynchronous | Any state, any order | Large sparse MDPs |
| Prioritized sweeping | Update states with largest Bellman residual first | Sparse rewards, large state spaces |
| Real-time DP | Update only visited states | When agent is interacting with environment |

### Evaluation Depth (Modified Policy Iteration)

| K (eval steps per improvement) | Behavior |
|---------------------------------|----------|
| K=1 | Value iteration |
| K=2 to 10 | Modified policy iteration (sweet spot) |
| K=infinity | Full policy iteration |

## 5. Interview Q&A

**Q: When would you choose policy iteration over value iteration in practice?**
A: Choose policy iteration when you have a small state space (< 10^4 states) and need an intermediate policy at each iteration for evaluation. The per-step cost is higher but convergence in policy space is much faster—often < 20 improvements vs hundreds of value iteration sweeps. Value iteration is better when the state space is large and you only care about the final V*.

**Q: What is modified policy iteration and why does it often outperform both extremes?**
A: Modified policy iteration performs K evaluation steps (instead of full convergence) before each policy improvement step. K=1 recovers value iteration; K=infinity recovers full policy iteration. In practice K=5-10 is often fastest overall: you get the fast policy improvement of policy iteration without the expensive full evaluation, and avoid the many sweeps needed by pure value iteration.

**Q: How does prioritized sweeping achieve faster convergence on sparse reward MDPs?**
A: It maintains a priority queue of states ordered by their Bellman residual |V(s) - backup(s)|. States with large residuals are updated first because they cause the most change to neighboring states when corrected. This focuses computation on the "frontier" of value propagation (near the reward), rather than wasting time on already-converged states. Speedups of 10-100x over synchronous DP on sparse MDPs are common.

**Q: What is the policy improvement theorem and why does it guarantee monotone improvement?**
A: The theorem states: if Q_pi(s, pi'(s)) >= V_pi(s) for all s, then V_{pi'}(s) >= V_pi(s) for all s. Intuitively, if a new policy is at least as good as the old one at the first step (and the old policy is followed afterwards), it is at least as good everywhere. This guarantees that greedy policy improvement never makes things worse, and the sequence of policies converges to pi*.

**Q: How would you handle a continuous state space with DP?**
A: Exact DP is intractable for continuous states. Options: (1) Discretize the state space (grid approximation)—works for low-dimensional problems up to ~3-4 dimensions. (2) Use linear function approximation for V_pi: V(s) ≈ theta^T phi(s), then solve the projected Bellman equation analytically. (3) Fitted value iteration: apply Bellman backup to a sampled set of states and fit a regression model. Limitations grow quickly with dimensionality (curse of dimensionality).

**Q: What causes oscillation in policy iteration and how do you detect it?**
A: Oscillation occurs when two or more policies have equal value at some state and the algorithm keeps switching between them. Detection: track the sequence of policies; if you see the same policy repeated after 2-4 improvement steps, you have a cycle. Fix: use a tie-breaking rule (e.g., prefer lower-index action on ties), or check for near-equality within a tolerance rather than exact equality.

**Q: How do you validate that your DP implementation is correct without a reference solution?**
A: Three checks: (1) Verify the Bellman residual is truly below epsilon after value iteration converges (recompute it from scratch). (2) Check that the extracted policy is greedy with respect to V*—compute Q(s,a) for 5-10 random states and verify pi(s) = argmax_a Q(s,a). (3) Roll out the extracted policy for 100+ episodes and compare average return to V*(s_0); they should match within 1-2%.

## 6. Best Practices

- **Use modified policy iteration (K=5-10 eval steps)**: This is almost always faster than both extremes. Start with K=5 and tune upward if policy quality is poor.
- **Warm-start policy evaluation**: Initialize V for the new policy using the V from the previous policy iteration; this dramatically reduces evaluation sweeps (the values change little between consecutive policies).
- **Use vectorized numpy operations**: Replace Python for-loops over states with matrix-vector products: V = (R_pi + gamma * P_pi @ V) where R_pi and P_pi are precomputed for the current policy.
- **Convergence tolerance strategy**: Use loose tolerance (1e-3) for early policy iterations; tighten to 1e-6 only in the final few iterations to save computation.
- **Always check sum(P[s,a,:]) == 1**: Transition matrix normalization bugs propagate silently and corrupt all value estimates.
- **Visualize value function heatmap**: For grid MDPs, plotting V as a 2D heatmap immediately reveals whether values propagate correctly from goal to start.
- **Benchmark against brute force on small MDPs**: For 3-5 state MDPs, enumerate all deterministic policies and compute V_pi by solving the linear system; use this to validate your DP code.

## 7. Common Pitfalls

- **Mixing up V_pi and V* updates**: Policy evaluation uses the current policy's action probabilities (Bellman expectation); value iteration takes the max over actions (Bellman optimality). Using max in policy evaluation inflates values and skips the policy improvement step.
  - Symptom: Policy iteration converges in 1-2 steps (suspiciously fast); values match value iteration not policy evaluation.
  - Fix: In the evaluation step, use V(s) = sum_a pi(a|s) * sum_{s'} P(s'|s,a)*(R + gamma*V(s')), not max.

- **Inefficient full matrix inversion for policy evaluation**: Solving V_pi = (I - gamma*P_pi)^{-1} R_pi via numpy.linalg.solve is O(|S|^3) and breaks for |S| > 10^4.
  - Symptom: Evaluation step hangs or runs out of memory for large state spaces.
  - Fix: Use iterative evaluation (Gauss-Seidel sweeps) instead of direct solve; stop at epsilon tolerance.

- **No convergence check inside policy evaluation**: Doing a fixed number of sweeps (e.g., always 100) instead of checking convergence causes over- or under-evaluation depending on the MDP.
  - Symptom: Policy iteration either wastes time evaluating a converged value function, or produces a poor policy from an unconverged one.
  - Fix: Track max|V_new - V_old| each sweep; stop when it falls below theta.

- **Ignoring numerical precision in policy comparison**: Comparing float V values with == causes the algorithm to never terminate (floats rarely match exactly).
  - Symptom: Policy iteration runs for hundreds of iterations without converging.
  - Fix: Compare policies by action index (argmax_a Q), not by V values.

## 8. Related Concepts

- [Markov Decision Processes](./01-markov-decision-processes.md) — DP requires a fully known MDP; understanding the MDP tuple is a prerequisite.
- [Bellman Equations](./02-bellman-equations.md) — DP algorithms are repeated applications of Bellman backup operators.
- [Monte Carlo Methods](./04-monte-carlo-methods.md) — Model-free alternative when P is unknown; compare convergence and bias properties.
- [Temporal Difference Learning](./05-temporal-difference-learning.md) — Bridges DP and MC; uses Bellman bootstrapping without full model.
