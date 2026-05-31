# Temporal Difference Learning

## 1. Detailed Explanation

Temporal Difference (TD) learning is the synthesis of dynamic programming and Monte Carlo methods. Like DP, it bootstraps—using current estimates of future values as targets. Like MC, it is model-free—learning directly from experience without knowing the transition probabilities. This combination makes TD learning the most practical foundation for real-world RL.

The simplest form, TD(0), updates the value of a state immediately after observing the next state and reward: V(s) ← V(s) + alpha * [r + gamma*V(s') - V(s)]. The term in brackets is the TD error delta = r + gamma*V(s') - V(s): how much better or worse the outcome was than the current estimate. This single-step update is efficient, online (no need to wait for episode end), and works in continuing tasks.

TD(lambda) generalizes TD(0) by blending returns of different depths using eligibility traces. At lambda=0 it reduces to TD(0); at lambda=1 it approaches MC (full returns). Eligibility traces e(s) track recency: recently visited states receive larger updates, propagating credit back through the episode. This makes TD(lambda) particularly effective when rewards are delayed—the trace keeps the relevant states "eligible" for credit long after they were visited.

SARSA and Q-learning are TD methods for control (finding the optimal policy): SARSA updates Q(s,a) using the action actually taken next (on-policy), while Q-learning uses the greedy next action regardless of what was taken (off-policy). Q-learning is the foundation of DQN; SARSA is safer in production environments where off-policy exploration could incur real costs.

## 2. Core Intuition

TD learning works like a GPS recalculating your ETA as you drive—you don't wait until you arrive to know if you're on schedule. At each step, you compare what you expected at the last decision point to what you now expect given where you are, and you immediately update your estimate. The difference between prediction and updated prediction is the TD error: a signal that tells you precisely which past predictions were too optimistic or too pessimistic.

## 3. How It Works

```
flowchart TD
    A[Initialize V-s = 0 for all s, choose step size alpha] --> B[Start episode: observe s_0]
    B --> C[Select action a_t from policy, observe r_{t+1} and s_{t+1}]
    C --> D[Compute TD error: delta = r_{t+1} + gamma times V-s_{t+1} minus V-s_t]
    D --> E[Update V-s_t = V-s_t + alpha times delta]
    E --> F[Update eligibility traces: e-s = gamma times lambda times e-s; e-s_t += 1]
    F --> G[For all s: V-s += alpha times delta times e-s]
    G --> H{Terminal state?}
    H -- No --> C
    H -- Yes --> I[Reset traces, start new episode from B]
```

Step-by-step:

1. **Initialize**: Set V(s) = 0 for all s. Choose alpha (learning rate, 0.01-0.5) and lambda (trace decay, 0-1).
2. **At each step**: Observe current state s_t, take action a_t, receive reward r_{t+1}, observe next state s_{t+1}.
3. **Compute TD error**: delta_t = r_{t+1} + gamma*V(s_{t+1}) - V(s_t). If s_{t+1} is terminal, use V(terminal)=0.
4. **Update TD(0)**: V(s_t) ← V(s_t) + alpha * delta_t. (For TD(lambda), update all states via traces.)
5. **Update eligibility traces (TD lambda)**: e(s) ← gamma*lambda*e(s) for all s; e(s_t) ← e(s_t) + 1 (accumulating traces). Then: V(s) ← V(s) + alpha*delta_t*e(s) for all s with nonzero trace.
6. **Reset at episode end**: Set all eligibility traces to 0. Start new episode.

## 4. Architecture and Trade-offs

### TD(0) vs TD(lambda) vs MC

| Property | TD(0) | TD(lambda), 0 < lambda < 1 | MC (lambda = 1) |
|----------|-------|---------------------------|-----------------|
| Bootstrapping | Full | Partial | None |
| Bias | High | Medium | Zero |
| Variance | Low | Medium | High |
| Online updates | Yes | Yes | No (waits for T) |
| Handles continuing tasks | Yes | Yes | No |
| Credit assignment | 1-step | Multi-step exponential | Full trajectory |

### n-Step TD Returns

| n | Target | Bias/Variance tradeoff |
|---|--------|----------------------|
| 1 (TD(0)) | r + gamma*V(s') | Lowest variance, highest bias |
| 2-4 | 2-4 step lookahead | Good balance |
| 16-32 | Long-range return | Lower bias, higher variance |
| infinity (MC) | Full return | Zero bias, highest variance |

### On-Policy vs Off-Policy TD Control

| Algorithm | Update target | Policy used | Safe for real environments? |
|-----------|--------------|------------|---------------------------|
| SARSA (on-policy) | r + gamma*Q(s',a') where a' ~ pi | Current (epsilon-greedy) policy | Yes |
| Q-learning (off-policy) | r + gamma*max_{a'} Q(s',a') | Behavior policy (any) | Risky (explores aggressively) |
| Expected SARSA | r + gamma*sum_a pi(a|s')*Q(s',a) | Current (expectation) | Yes, lower variance |

## 5. Interview Q&A

**Q: What is the bias-variance tradeoff in TD vs MC, and how does TD(lambda) address it?**
A: TD bootstraps from V(s'), which is an estimate (biased but low variance). MC uses the actual return (unbiased but high variance across trajectories). TD(lambda) blends both by computing a lambda-weighted combination of n-step returns: G_t^lambda = (1-lambda) * sum_{n=1}^{inf} lambda^{n-1} * G_t^{(n)}. Higher lambda gives more weight to longer-horizon returns (less bias, more variance); lower lambda gives more weight to short bootstrapped estimates. The optimal lambda on a specific task is empirically determined by sweeping lambda from 0 to 1.

**Q: Why is Q-learning off-policy, and when does that matter?**
A: Q-learning's update target uses max_{a'} Q(s',a') regardless of what action the agent actually took. The behavior policy (what actions you actually explore with) can be anything—random, epsilon-greedy, human demonstrations. The learned Q* converges to the optimal Q regardless of how data was collected. This matters when you want to reuse historical data (experience replay), learn from demonstrations, or train offline from logged trajectories. The risk: in systems where exploration itself has real costs (robotic control, medical dosing), the behavior policy must also be evaluated for safety.

**Q: How do eligibility traces in TD(lambda) help with delayed credit assignment?**
A: Without traces, TD(0) only updates the state visited at the last step. If a key decision was made 10 steps ago and reward is only observed now, TD(0) takes 10 separate episodes to propagate the credit back. Eligibility traces keep a decaying memory e(s) of which states were recently visited. When a reward arrives, all eligible states are updated proportionally to their trace value. This directly propagates reward to the states that caused it, dramatically speeding up learning when rewards are sparse and delayed.

**Q: What breaks in TD learning when the step size alpha is too large?**
A: With alpha too large, updates overshoot the correct V(s) estimate: a single high-reward sample can push V(s) far above its true value, and subsequent lower-reward samples then cause large negative updates. The value function oscillates rather than converging. Detection: plot V(s) for a few representative states over time; it should decrease monotonically in variance, not oscillate. Fix: use alpha = 0.01-0.1 for tabular TD; for neural networks, use Adam with lr = 1e-4 to 1e-3.

**Q: How does TD learning relate to the prediction error in neuroscience?**
A: The TD error delta_t = r + gamma*V(s') - V(s) matches the firing rate of dopamine neurons in mammalian brains. Dopamine neurons fire when reward is better than predicted (positive delta), are suppressed when worse (negative delta), and gradually stop firing when reward becomes fully predicted (delta → 0). This is not a coincidence: TD learning was partly inspired by animal learning theory, and neuroscience later confirmed the correspondence. This gives biological plausibility to TD as an account of how real brains learn value.

**Q: What is the deadly triad, and how does DQN solve it?**
A: The deadly triad is the combination of: function approximation (neural network for Q), bootstrapping (TD target), and off-policy data (experience replay). Together they can cause Q-value divergence—the target y = r + gamma*max Q(s',a') depends on Q itself, creating a moving target. DQN breaks the feedback loop with a frozen target network: for every C steps, copy the current Q-network to a target network Q_target; compute targets using Q_target (frozen) instead of the current Q. This stabilizes training significantly.

**Q: When would you use SARSA instead of Q-learning in production?**
A: Use SARSA when the behavior policy (exploration) is part of the real environment and unsafe exploration is costly—e.g., a factory robot or a healthcare dosing system. SARSA evaluates the policy you are actually executing (including its exploratory actions), so learned Q-values account for the cost of exploration. Q-learning ignores the behavior policy and may assign high Q-values to states that look good but are reached only by a risky exploration path in practice.

## 6. Best Practices

- **Set alpha by grid search, not intuition**: Try alpha in {0.001, 0.01, 0.1, 0.5} on a validation MDP; convergence speed is very sensitive to alpha, especially with function approximation.
- **Decay alpha over time for tabular TD**: Use alpha_t = alpha_0 / (1 + decay_rate * t). The Robbins-Monro conditions (sum alpha_t = inf, sum alpha_t^2 < inf) guarantee convergence.
- **Initialize V optimistically for exploration**: V(s) = R_max/(1-gamma) encourages the agent to visit all states before settling; without this, TD can converge to a suboptimal policy from greedy initialization.
- **Use TD(lambda) with lambda = 0.7-0.9 for delayed reward tasks**: The intermediate lambda balances bias and variance better than either extreme; 0.9 is a good default for sparse reward settings.
- **Clip TD errors to [-1, 1] for neural network stability**: Large TD errors cause gradient explosion; clipping is the standard fix in DQN and later architectures.
- **Track TD error magnitude as a training health metric**: Plot mean absolute TD error over time; it should decrease over training. If it plateaus high early or increases later, diagnose learning rate, target network freeze frequency, or data distribution.
- **For continuing tasks, use average-reward TD**: Differential TD subtracts an estimate of the average reward from each step; this keeps values bounded without requiring gamma < 1.

## 7. Common Pitfalls

- **Using V(s') = 0 at episode termination without checking terminal flag**: Non-terminal states with V(s) ≈ 0 near episode boundary get a wrong bootstrap target.
  - Symptom: V estimates near episode end are consistently too low; the policy avoids states close to the terminal state.
  - Fix: Check if s' is terminal before bootstrapping; if terminal, target = r (no V(s') term).

- **Not resetting eligibility traces between episodes**: Traces from episode k contaminate updates in episode k+1, propagating reward from one episode to the next.
  - Symptom: V estimates for early states in an episode are inflated (receiving reward from the previous episode's end).
  - Fix: Set e = np.zeros(n_states) at the start of every episode.

- **Step size too large with eligibility traces**: Eligibility traces amplify the TD error to all eligible states simultaneously; if alpha is large, this can cause large oscillating updates across many states at once.
  - Symptom: Many state values oscillate simultaneously; instability worse than TD(0) alone.
  - Fix: Reduce alpha by a factor of (1 / max_trace_sum) when using traces; for accumulating traces, max trace sum is 1/(1-gamma*lambda).

- **TD(lambda) with lambda=1 in stochastic environments**: At lambda=1, TD(lambda) becomes MC—it waits for episode end and uses the full return. But unlike MC, it still updates V during the episode via traces, which can cause double-counting in stochastic environments.
  - Symptom: V estimates at lambda=1 do not match first-visit MC; values are biased.
  - Fix: Use MC directly for lambda=1 scenarios; TD(lambda) is most useful for lambda in [0, 0.95].

## 8. Related Concepts

- [Markov Decision Processes](./01-markov-decision-processes.md) — Defines the environment structure TD learning operates in.
- [Bellman Equations](./02-bellman-equations.md) — TD targets are single-step Bellman backups; TD converges to the same V* as DP in tabular settings.
- [Dynamic Programming for RL](./03-dynamic-programming-rl.md) — Model-based counterpart; compare DP (requires P) vs TD (model-free bootstrapping).
- [Monte Carlo Methods](./04-monte-carlo-methods.md) — The other extreme: full returns, no bootstrapping. TD interpolates between MC (lambda=1) and pure bootstrap (lambda=0).
