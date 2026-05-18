# Policy Gradients

## Detailed Explanation

Policy Gradient methods learn decision-making policies directly by adjusting the parameters of a neural network that outputs actions. Unlike value-based methods like Q-Learning that estimate future rewards and then act greedily, policy gradients use the gradient of expected reward with respect to policy parameters to update the policy towards better actions.

The fundamental idea is to increase the probability of actions that led to high rewards and decrease the probability of actions that led to low rewards. This is expressed as a gradient: ∇J(θ) = E[∇log π(a|s) R], which means we move the policy parameters in the direction that increases the log-probability of good actions scaled by their returns.

Policy gradients have several advantages: they handle continuous action spaces naturally (by outputting means and variances), they converge to local optima directly (not approximating values), and they support stochastic policies (useful for exploration). They power systems from robotic control to game-playing agents. The trade-off is higher variance in gradient estimates compared to value methods, requiring careful learning rate tuning and variance reduction techniques.

## Core Intuition

Instead of learning the value of each chess position, directly learn which moves are good in each position. The policy gradient approach adjusts move probabilities: if a move sequence led to victory, increase those move probabilities; if it led to defeat, decrease them. It's like a coach watching your game and saying 'do that move more often, do that move less often'.

## How It Works

1. Policy π(a|s): stochastic policy mapping states to action probabilities
2. Objective: maximize J(θ) = E[sum discounted rewards]
3. Policy gradient: ∇J(θ) = E[∇log π(a|s) × R(τ)]
4. REINFORCE: sample trajectory, compute gradients, update policy
5. Baseline: subtract baseline from reward to reduce variance (doesn't bias gradient)
6. Advantage: use advantage A(s,a) = Q(s,a) - V(s) instead of reward (lower variance)
7. Variants: PPO (clipped objective), TRPO (trust region), A3C (asynchronous)

```mermaid
graph TD
    A[Input] -->|Process| B[Model/Algorithm]
    B -->|Output| C[Result]
```

## Architecture / Trade-offs

Key trade-offs and design considerations for this concept.

## Interview Q&A


**Q: How do policy gradients differ from Q-learning?**
A: Q-learning: learn value function implicitly (derive policy by max). Policy gradient: directly optimize policy. Tradeoff: PG converges slower but to better optima, handles continuous actions naturally. Both have merits.

**Q: What is the baseline in policy gradients and why use it?**
A: Baseline: subtract moving average of returns from reward. Reduces variance (if return is 10 and baseline is 8, advantage is 2). Doesn't bias gradient (expected value still same). Critical for stable training.

**Q: What's the difference between REINFORCE and A3C?**
A: REINFORCE: accumulate trajectory, update once (on-policy). A3C: asynchronous (multiple workers), update frequently. A3C: faster (parallel) but more complex. REINFORCE: simpler but slower. Use REINFORCE for learning, A3C for scaling.

**Q: How do you handle high-variance policy gradients?**
A: Sources: rewards are noisy, variance grows with horizon. Solutions: (1) baseline (reduce magnitude), (2) advantage (relative comparison), (3) batch/normalization (average over samples), (4) trust regions (limit step size).

**Q: Can policy gradients handle discrete and continuous actions?**
A: Discrete: softmax over actions (same as classification). Continuous: output mean + variance of action distribution (Gaussian), sample from it. Much more natural for continuous than Q-learning (which requires discretization).


## Best Practices

- Apply best practices specific to this concept
- Consider edge cases and failure modes
- Test on representative data
- Evaluate comprehensively

## Common Pitfalls

- Avoid over-simplification
- Watch for incorrect assumptions
- Test edge cases thoroughly
- Monitor for degradation

## Code Examples

See the associated notebook for implementation and real-world examples.

## Related Concepts

- Understand prerequisites first
- Connect related topics
- Build integrated knowledge
