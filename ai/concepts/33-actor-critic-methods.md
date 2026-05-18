# Actor-Critic Methods

## Detailed Explanation

Actor-Critic methods combine policy gradients (actor: learns what action to take) with value functions (critic: evaluates quality of states/actions), balancing high variance of policy gradients with bias of value functions. The actor (policy network) generates actions; the critic (value network) estimates how good those actions were (temporal difference error). The actor uses critic's estimates to reduce variance in policy gradients. Advantage Actor-Critic (A2C) and Asynchronous A3C are practical algorithms combining both.

The actor-critic architecture mirrors human decision-making: the actor (conscious decision) is guided by the critic (evaluation/reward). Training uses two loss functions: actor loss (improve policy toward actions the critic values highly) and critic loss (make value estimates accurate). Entropy regularization encourages exploration by adding bonus for diverse policies. The advantage function (actual return - estimated value) reduces variance: tells actor which actions are better than average, focusing learning on important decisions.

Actor-Critic methods are widely used because they balance sample efficiency (value functions guide learning) with flexibility (policy gradients handle continuous actions). Understanding the actor-critic decomposition helps recognize similar architectures elsewhere: many modern systems combine prediction networks (critic) with generation networks (actor). Challenges include stable learning (both networks must improve together), function approximation errors in critic affecting actor, and tuning trade-offs between actor/critic learning rates.

## Core Intuition

Actor-Critic methods are like having a performer (actor) and a coach (critic): the actor tries different performances, the coach evaluates each attempt and gives feedback. The actor improves based on feedback, and the coach improves at evaluating performances. Together they become better than either alone.

## How It Works

1. Actor: policy network π(a|s), generates actions
2. Critic: value network V(s), estimates state value
3. Advantage: A(s,a) = r + γV(s') - V(s) (estimated using critic)
4. Actor loss: -log π(a|s) × A(s,a) (improve policy using critic's estimate)
5. Critic loss: MSE(V(s), target) where target = r + γV(s') (bootstrap from next state)
6. Update: compute both losses, backprop to both networks
7. Benefits: lower variance (critic baseline), more stable (two networks)

```mermaid
graph TD
    A[Input] -->|Process| B[Model/Algorithm]
    B -->|Output| C[Result]
```

## Architecture / Trade-offs

### Two-Network Architecture

```mermaid
graph TB
    subgraph "Actor-Critic System"
        A["Shared Features<br/>Feature Extractor"]

        A --> B["Actor Network<br/>π(a|s;θ_actor)"]
        A --> C["Critic Network<br/>V(s;θ_critic)"]

        D["Environment"] -->|State s, Reward r| E["TD Error<br/>δ = r + γV(s') - V(s)"]

        B -->|Action a| D
        C -->|Value estimate| E

        E -->|TD signal| B
        E -->|Value target| C

        B -->|Probability| D

        style B fill:#fff3e0
        style C fill:#e1f5ff
        style E fill:#f3e5f5
    end
```

### Actor vs Critic Responsibilities

| Component | Actor | Critic |
|-----------|-------|--------|
| **What it learns** | Policy π(a\|s) | Value function V(s) |
| **Output** | Action probabilities/means | Scalar value |
| **Loss function** | Policy gradient × TD error | TD loss (V(s) - target)² |
| **Training signal** | Critic's TD error | Ground truth return |
| **Role** | Decides what to do | Evaluates how good decision is |
| **Failure mode** | Policies underexplore | Overestimates/underestimates values |

### Architecture Variants

```mermaid
graph LR
    A["Actor-Critic Base"] --> B["Advantage Actor-Critic<br/>A3C"]
    A --> C["Asynchronous A3C<br/>Parallel workers"]
    A --> D["Deep Deterministic<br/>Policy Gradient<br/>DDPG"]
    A --> E["Soft Actor-Critic<br/>SAC"]

    B -->|Use| F["Discrete actions"]
    D -->|Use| G["Continuous actions"]
    E -->|Use| G

    style B fill:#e1f5ff
    style D fill:#fff3e0
    style E fill:#f3e5f5
```

### Bias-Variance Trade-off

| Aspect | Lower Bias (Critic) | Higher Bias (Advantage) |
|--------|-------------------|------------------------|
| **TD Target** | Full episode return (unbiased) | Critic prediction (biased) |
| **Variance** | High (entire episode affects gradient) | Low (critic reduces noise) |
| **Convergence speed** | Slow | Fast |
| **When to use** | Small variance in environment | Complex tasks with noise |
| **Stability** | More stable | Less stable |

### Multi-Step Learning

```mermaid
graph TD
    A["One-Step TD"] -->|δ = r + γV(s')| B["Low bias, High variance<br/>Fast learning, Unstable"]
    A -->|n-Step| C["n-Step TD<br/>δ = r + γr' + ... + γ^n V(s_n)"]
    A -->|∞-Step| D["Monte Carlo<br/>Use full return"]
    C -->|Balanced| E["Good trade-off<br/>n=3 or 4 typical"]

    style B fill:#fff3e0
    style E fill:#e1f5ff
    style D fill:#f3e5f5
```

### Trade-offs: Synchronous vs Asynchronous

| Property | Synchronous A3C | Asynchronous A3C |
|----------|-----------------|------------------|
| **Training** | Single worker, sequential | Multiple workers, parallel |
| **Convergence speed** | Slower | Faster (more samples) |
| **Complexity** | Simple | Complex (threading, locking) |
| **Hardware** | Single machine | Multi-core/GPU |
| **Stability** | Less stable | More stable (diverse experience) |
| **Memory** | Lower | Higher (multiple workers) |
## Interview Q&A


**Q: Why is actor-critic better than pure policy gradient?**
A: Pure PG: high variance (rewards are noisy). Actor-critic: critic provides baseline (reduces variance). Result: faster convergence, more stable training. Trade-off: slightly more complex (two networks).

**Q: What is TD error and how is it used?**
A: TD error: δ = r + γV(s') - V(s). Actor: use as advantage (update policy in gradient direction of advantage). Critic: update V(s) to minimize TD error. Both use same TD signal (efficient).

**Q: How do you avoid instability in actor-critic?**
A: Sources: two networks learning simultaneously (instability), high variance from policy gradients. Solutions: (1) target critic network (slowly updated copy), (2) experience replay (decorrelate samples), (3) entropy regularization (encourage exploration).

**Q: What is asynchronous advantage actor-critic (A3C)?**
A: A3C: multiple workers run episodes in parallel, asynchronously update shared networks. Benefits: more diverse experience, faster training. Implementation: careful synchronization (locks, atomic operations). Good for distributed systems.

**Q: Can you use actor-critic for continuous control?**
A: Yes, naturally: actor outputs mean+variance of action distribution. Critic estimates value. Works for both discrete and continuous. Popular in robotics (DDPG, TD3, SAC variants). Better than Q-learning for continuous actions.


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
