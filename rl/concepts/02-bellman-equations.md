# Bellman Equations

## 1. Detailed Explanation

Bellman equations are the recursive decomposition of the value function that makes dynamic programming tractable. Named after Richard Bellman (1957), they state that the value of being in a state equals the immediate reward plus the discounted value of the next state—a simple identity that, when applied iteratively, converges to the optimal value function.

There are two forms. The Bellman expectation equation computes the value under a fixed policy pi: V_pi(s) = sum_a pi(a|s) sum_{s'} P(s'|s,a)[R(s,a,s') + gamma*V_pi(s')]. The Bellman optimality equation finds the value under the best possible policy: V*(s) = max_a sum_{s'} P(s'|s,a)[R(s,a,s') + gamma*V*(s')]. The Q-function version is Q*(s,a) = sum_{s'} P(s'|s,a)[R + gamma*max_{a'} Q*(s',a')].

These equations matter because they turn the global optimization problem (maximize cumulative reward over an infinite horizon) into a local one (pick the best action at each state given future values). This local structure is what makes all DP algorithms, and ultimately deep Q-networks, tractable.

In model-free RL, agents do not have access to P—but the Bellman equations still provide targets. TD-learning uses r + gamma*V(s') as a bootstrap target; DQN minimizes the Bellman residual (y - Q(s,a))^2 where y = r + gamma*max_{a'} Q(s',a'). Understanding the Bellman equations is therefore foundational to understanding why these algorithms work and when they fail.

## 2. Core Intuition

The Bellman equation says: "the value of where you are equals what you gain right now, plus the discounted value of where you end up." It breaks a daunting infinite-horizon problem into a simple one-step look-ahead. Applied repeatedly like a recursive function call, this local rule propagates global optimality backward through the state space.

## 3. How It Works

```
flowchart TD
    A[Initialize V-s = 0 for all states s] --> B[For each state s: compute Bellman backup]
    B --> C[V_new-s = max over a of sum over s-prime of P-s-prime-given-s-a times R plus gamma times V-s-prime]
    C --> D[Compute delta = max over s of abs-V_new-s minus V-s]
    D --> E{delta < epsilon threshold?}
    E -- No --> F[Set V = V_new, go back to B]
    E -- Yes --> G[V has converged to V-star]
    G --> H[Extract policy: pi-star-s = argmax over a of Q-star-s-a]
```

Step-by-step:

1. **Initialize**: Set V(s) = 0 for all states (or use any finite initial values; convergence is guaranteed regardless).
2. **Bellman backup**: For each state s, compute the maximum expected value over all actions: V_new(s) = max_a sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')].
3. **Compute residual delta**: Track max|V_new(s) - V(s)| across all states as the convergence metric.
4. **Update and repeat**: Set V = V_new and repeat until delta < epsilon (typically 1e-6).
5. **Extract Q-values**: Q*(s,a) = sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V*(s')].
6. **Extract policy**: pi*(s) = argmax_a Q*(s,a). This greedy extraction is optimal by the Bellman optimality principle.

## 4. Architecture and Trade-offs

### Value Function vs Q-Function

| Representation | Formula | Memory | Policy extraction |
|----------------|---------|--------|-------------------|
| V(s) | V*(s) = max_a E[R + gamma*V*(s')] | O(|S|) | Requires knowing P: argmax_a sum_{s'} P*Q |
| Q(s,a) | Q*(s,a) = E[R + gamma*max_{a'} Q*(s',a')] | O(|S||A|) | argmax_a Q*(s,a) — no model needed |
| Advantage A(s,a) | A(s,a) = Q(s,a) - V(s) | O(|S||A|) | argmax_a A(s,a) |

### Convergence Comparison

| Algorithm | Updates per sweep | Memory | Convergence rate |
|-----------|------------------|--------|-----------------|
| Value iteration (Bellman optimality) | O(|S|^2|A|) | O(|S|) | Linear in 1/gamma |
| Policy iteration (Bellman expectation) | O(|S|^3) per eval | O(|S|) | Superlinear (fewer steps) |
| Q-value iteration | O(|S|^2|A|) | O(|S||A|) | Same as VI but stores Q-table |

### Error Propagation

| Source of error | Effect on V | Effect on policy |
|-----------------|------------|-----------------|
| Reward noise | O(sigma / (1-gamma)) amplification | Policy may change near borderline states |
| Transition noise | Smoothed value function | More conservative policies |
| Approximation error (VFA) | Bellman residual bound | Policy quality bounded by 2*epsilon/(1-gamma)^2 |

## 5. Interview Q&A

**Q: Why does the Bellman optimality equation guarantee we find the global optimum and not a local one?**
A: Because the value function space is a contraction mapping under the Bellman operator (with factor gamma). Banach's fixed-point theorem guarantees convergence to a unique fixed point—which is exactly V*. There are no local optima; any solution to the Bellman equation is the globally optimal value function.

**Q: What happens to Bellman residuals when you use function approximation (like a neural network)?**
A: The Bellman operator projected onto the function approximator's representable set is no longer a contraction in general. This can cause divergence—the "deadly triad" problem (function approximation + bootstrapping + off-policy data). DQN addresses this with a frozen target network: the target y = r + gamma*max Q_target(s',a') is held fixed for thousands of steps, breaking the feedback loop that causes divergence.

**Q: When would you use Q*(s,a) instead of V*(s)?**
A: Use Q* when you don't have access to the transition model P at decision time, because policy extraction from V* requires computing argmax_a sum_{s'} P(s'|s,a)*Q, which needs P. With Q*, argmax_a Q*(s,a) requires no model—just a table or network lookup. This is why Q-learning and DQN use Q-functions: they work model-free.

**Q: What is the Bellman residual and how do you use it as a diagnostic?**
A: The Bellman residual at state s is |V(s) - [max_a sum_{s'} P(s'|s,a)(R + gamma*V(s'))]|. In practice you track max_s(residual) across the state space; if it plateaus above your epsilon tolerance, either gamma is too high, the MDP has no fixed point (gamma=1 + continuing), or there's a bug in the transition matrix. Plot residual vs. iteration to diagnose convergence problems early.

**Q: How does the discount factor gamma interact with Bellman equation convergence speed?**
A: The contraction factor of the Bellman operator is exactly gamma. Iterations needed to reduce the error by factor epsilon scales as O(log(1/epsilon) / log(1/gamma)). At gamma=0.99, you need ~460 iterations to reach epsilon=0.01; at gamma=0.9, only ~44 iterations. High gamma gives better long-horizon policies but much slower convergence.

**Q: How would you extend Bellman equations to continuous action spaces?**
A: The max_a operator in Q*(s,a) becomes an optimization problem over a continuous domain, which is generally intractable for arbitrary Q. Solutions: (1) Restrict to simple function families where the max has a closed form (e.g., NAF: Normalized Advantage Functions where Q is quadratic in a). (2) Use a separate actor network trained to maximize Q (actor-critic methods like DDPG, SAC). (3) Use cross-entropy method or gradient ascent to approximately solve max_a Q(s,a) at each step.

**Q: What breaks if rewards are not bounded?**
A: The contraction property of the Bellman operator requires rewards to be bounded; otherwise V* may be infinite. In practice, reward clipping to [-1, 1] (as in original Atari DQN) bounds the value function and stabilizes learning. The cost is that you lose the magnitude information of the reward signal, which can hurt on tasks where reward scale matters.

## 6. Best Practices

- **Monitor Bellman residual not iteration count**: Convergence is detected by max|V_new - V_old| < epsilon, not a fixed number of sweeps. Use epsilon = 1e-6 for precise solutions; 1e-4 is usually fine for control.
- **Use in-place updates (Gauss-Seidel)**: Updating V(s) in-place (using already-updated values in the same sweep) converges faster in practice than synchronous updates. The order matters slightly—use a consistent order.
- **Prioritize high-residual states**: Prioritized sweeping updates states with the largest Bellman residuals first, achieving the same convergence with ~10x fewer state updates on sparse MDPs.
- **Verify Q extraction on toy example**: Before trusting extracted policy, manually check Q*(s,a) for 3-5 states by hand to confirm the backup is correct.
- **Scale rewards to [0,1] or [-1,1]**: Unnormalized rewards (e.g., dollar amounts) interact badly with a fixed epsilon convergence threshold; normalize rewards so epsilon=1e-4 is meaningful.
- **Separate terminal state handling**: Terminal states must have V(terminal)=0 permanently; exclude them from the Bellman backup loop or override after each update.
- **Test with known analytical solution**: 3-state chain MDPs have closed-form V*; use them as unit tests for your Bellman implementation.

## 7. Common Pitfalls

- **Not zeroing V(terminal) after each backup**: The Bellman backup for non-terminal states reads V(s') which may include terminal states. If terminal values drift from zero, reward-to-go estimates become inflated and the policy incorrectly avoids terminal states.
  - Symptom: V near the goal grows each iteration instead of being pulled toward the true reward.
  - Fix: After each sweep, force V[terminal_states] = 0.

- **Convergence criterion too loose**: Using epsilon=0.1 can stop iteration when V is far from V*, leading to suboptimal policies in states where V differences across actions are small.
  - Symptom: Policy evaluation shows lower-than-expected returns; switching to epsilon=1e-6 produces a different policy.
  - Fix: Use epsilon <= 1e-4 for grid worlds; tighter for tasks where small value differences lead to qualitatively different actions.

- **Using the wrong Q-extraction step after value iteration**: Extracting Q(s,a) requires one more Bellman step using V*; forgetting to do this and using V* directly as a proxy for Q leads to incorrect policies.
  - Symptom: Policy selects seemingly random actions at states near equal-value regions.
  - Fix: Compute Q*(s,a) = sum_{s'} P(s'|s,a)*(R + gamma*V*(s')) explicitly for policy extraction.

- **Bootstrapping from unreliable initial estimates**: When V(s) is initialized poorly (very large or negative), early Bellman targets are noisy and can slow convergence significantly.
  - Symptom: Convergence curves show very slow early descent before accelerating.
  - Fix: Initialize V to an optimistic upper bound (e.g., R_max/(1-gamma)) for faster convergence.

## 8. Related Concepts

- [Markov Decision Processes](./01-markov-decision-processes.md) — Defines the MDP tuple that Bellman equations operate on.
- [Dynamic Programming for RL](./03-dynamic-programming-rl.md) — Algorithms that apply Bellman equations iteratively to find V* or pi*.
- [Monte Carlo Methods](./04-monte-carlo-methods.md) — Model-free alternative that estimates V by sampling full trajectories instead of backing up Bellman.
- [Temporal Difference Learning](./05-temporal-difference-learning.md) — Combines Bellman bootstrapping with Monte Carlo sampling; bridges DP and MC.
