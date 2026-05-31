# Markov Decision Processes (MDPs)

## 1. Detailed Explanation

A Markov Decision Process is the mathematical framework underlying nearly all of reinforcement learning. It formalizes sequential decision-making under uncertainty as a tuple (S, A, P, R, γ): a set of states S, a set of actions A, a transition probability function P(s'|s,a), a reward function R(s,a,s'), and a discount factor γ ∈ [0,1].

The "Markov" property is the load-bearing assumption: the probability of transitioning to the next state depends only on the current state and action, not on the history of how you arrived there. This is often called the "memoryless" property. In practice, many real problems are not perfectly Markovian—a game with hidden information, or a robot with noisy sensors—but framing them as MDPs still yields useful approximations.

MDPs matter because they give us a precise specification of what "solving" an RL problem means: find a policy π(a|s) that maximizes the expected discounted cumulative reward E[Σ_t γ^t R_t]. Every algorithm—Q-learning, policy gradients, actor-critic—is solving an MDP, whether explicitly or implicitly.

In production, MDPs appear in recommendation systems (user state, item actions, engagement reward), robotic control (joint angles as state, motor torques as actions), and operations research (inventory states, ordering actions, holding costs as negative rewards). Understanding MDP structure—sparse transitions, large state spaces, partial observability—is essential for choosing the right algorithm.

## 2. Core Intuition

Think of an MDP as a board game with stochastic dice. You are at some square (state), you pick a move (action), dice are rolled (stochastic transitions), and you land on a new square and collect a score (reward). The goal is a strategy (policy) that maximizes total score across the whole game, but future scores are worth slightly less (discount factor) than immediate ones.

## 3. How It Works

```
flowchart TD
    A[Agent observes state s_t] --> B[Agent selects action a_t via policy pi]
    B --> C[Environment transitions: sample s_{t+1} from P-s-prime-given-s-a]
    C --> D[Environment emits reward r_t from R-s-a-s-prime]
    D --> E[Agent receives r_t and observes s_{t+1}]
    E --> F{Terminal state?}
    F -- No --> A
    F -- Yes --> G[Episode ends, compute return G_t = sum gamma^k r_{t+k}]
```

Step-by-step:

1. **Define state space S**: Enumerate or parameterize all possible world configurations relevant to the decision problem.
2. **Define action space A**: Enumerate all actions available from each state (can be state-dependent).
3. **Specify transitions P(s'|s,a)**: For each (s,a) pair, a probability distribution over next states. Must sum to 1 over s'.
4. **Specify rewards R(s,a,s')**: Scalar signal emitted at each transition. Often simplified to R(s,a) or R(s).
5. **Choose discount factor gamma**: gamma=0 → myopic (only immediate reward). gamma→1 → long-horizon planning (future rewards count almost as much as present).
6. **Find optimal policy pi***: pi*(s) = argmax_a Q*(s,a), where Q* satisfies the Bellman optimality equation.

## 4. Architecture and Trade-offs

### Discount Factor Comparison

| gamma value | Behavior | Use case |
|-------------|----------|----------|
| 0.0 | Myopic — only immediate reward | Bandits, one-step tasks |
| 0.5 | Short-horizon planning | Tasks with clear near-term signals |
| 0.9 | Medium horizon | Most episodic RL tasks |
| 0.99 | Long horizon | Continuing tasks, sparse rewards |
| 1.0 | Undiscounted (only for episodic tasks) | Board games, finite episodes |

### Model Types

| Model type | P and R known? | Algorithm family | Data needed |
|------------|---------------|-----------------|-------------|
| Model-based (tabular) | Yes | DP (value/policy iteration) | None — plan directly |
| Model-based (learned) | Learned from data | Dyna, MBPO | Moderate |
| Model-free | No | Q-learning, SARSA, policy gradients | Many samples |

### State Space Complexity

| State space | Size | Approach |
|-------------|------|----------|
| Tabular | < 10^6 states | Exact DP, Q-table |
| Factored | States as feature vectors | Linear VFA |
| Continuous | Infinite | Neural network approximators |

## 5. Interview Q&A

**Q: When does the Markov property fail in practice, and how do you handle it?**
A: It fails when observations are partial—e.g., a robot can't see behind it, or a user's true intent is hidden. The standard fix is to use a Partially Observable MDP (POMDP) with belief states, or in deep RL, to stack the last k observations as the state so the agent has a short history window. This recovers approximate Markov structure at the cost of a larger effective state space.

**Q: How would you choose gamma for a production recommendation system?**
A: Start near 0.9-0.99 if long-term user retention matters more than immediate clicks. Use lower gamma (0.5-0.8) if the session is short and feedback loops are tight. In practice, gamma is a hyperparameter tuned on a holdout set; monitoring short-term vs long-term engagement metrics tells you whether gamma is too myopic.

**Q: What breaks when your reward function is misspecified?**
A: The agent finds a policy that maximizes the proxy reward, not the intended objective—known as reward hacking. Example: a cleaning robot given reward for not seeing mess learns to close its eyes. Debug by: logging what the agent actually does at high reward states, checking for degenerate behaviors, and involving domain experts in reward specification review.

**Q: How does MDP formulation differ for episodic vs continuing tasks?**
A: Episodic tasks have natural terminal states (a game ends, a robot finishes a task); returns are finite sums. Continuing tasks never end; you must discount (gamma < 1) or use average-reward formulations to keep returns finite. Most tabular RL theory assumes episodic; production systems often need continuing formulations.

**Q: When would value iteration fail to converge on a tabular MDP?**
A: It always converges given exact arithmetic, gamma < 1, and a finite MDP. Failures in practice come from: gamma = 1 in a continuing task (diverges), bugs in the transition matrix that don't sum to 1 (normalization errors), or floating-point accumulation over thousands of states. Always assert sum(P[s,a,:]) == 1 for all (s,a) before running DP.

**Q: What is the complexity of solving an MDP with value iteration?**
A: One sweep is O(|S|^2 |A|) for a dense transition matrix. Total iterations to epsilon-convergence is O(log(1/epsilon) / log(1/gamma)). For sparse transitions (each state reaches only k next states), one sweep is O(|S| |A| k). Large robotics MDPs with sparse transitions are tractable; grid worlds with 10^6 states and 4 actions are routine.

**Q: How do you debug a trained policy that performs well in simulation but poorly in deployment?**
A: This is the sim-to-real gap. Check: (1) whether transition dynamics P match real physics/behavior (simulation fidelity), (2) whether rewards in simulation match real-world proxies, (3) whether the state representation captures what matters in reality. Mitigation: domain randomization during training, adding real-world rollouts to the replay buffer.

## 6. Best Practices

- **Validate transition matrix normalization**: Assert that each row of P[s,a,:] sums to 1.0 (within 1e-9 tolerance) before any DP computation.
- **Start with small MDPs**: Debug your algorithm on 3-5 state examples where you can hand-verify V* before scaling up.
- **Sparse transition representation**: Store P as a dict {(s,a): [(prob, s_prime, reward), ...]} for large state spaces; dense numpy arrays are infeasible above ~10^4 states.
- **Discount factor search**: Treat gamma as a hyperparameter; log 5-10 values from 0.7-0.99 and compare long-term return on a held-out evaluation MDP.
- **Separate reward signal from transition dynamics**: Keep R and P as distinct data structures; this makes reward shaping experiments cleaner.
- **Check ergodicity**: Ensure every state is reachable under some policy, otherwise DP produces arbitrary values for unreachable states.
- **Use log-probabilities for numerical stability**: When composing many transition probabilities (planning over long horizons), work in log-space to avoid underflow.

## 7. Common Pitfalls

- **Forgetting the terminal state absorbs**: Terminal states should have V(terminal) = 0 and no outgoing transitions with non-zero reward. Failing to zero out the terminal value causes the Bellman backup to propagate phantom future rewards forever.
  - Symptom: value function keeps growing; policy near the terminal state looks wrong.
  - Fix: Set V[terminal] = 0 after each backup sweep; mark terminal states and skip their updates.

- **Reward at the wrong timestep**: Reward is often assigned to the transition (s,a,s') but mistakenly assigned only to (s,a) ignoring s'. This matters when reaching the goal from different directions gives different rewards.
  - Symptom: Policy avoids states with high reward because the reward is credited to the wrong state.
  - Fix: Implement R(s,a,s') and include s' in the reward lookup.

- **Using gamma=1 with continuing tasks**: Value iteration diverges; Q-values grow without bound.
  - Symptom: V(s) increases without converging over iterations.
  - Fix: Use gamma < 1, or switch to average-reward RL formulation with differential values.

- **Dense P matrix for large state spaces**: A 10^4-state MDP with 4 actions needs 4×10^8 floats for P—about 3 GB at float64.
  - Symptom: MemoryError on initialization.
  - Fix: Use sparse representation: dict of lists of (prob, s_prime) tuples.

- **Off-by-one in discount accumulation**: Computing G_t = r_t + gamma*r_{t+1} + ... is easy to shift by one step.
  - Symptom: Returns are systematically biased; MC estimates differ from DP values by a factor of gamma.
  - Fix: Start accumulation from the last timestep and work backwards: G = 0; for r in reversed(rewards): G = r + gamma * G.

## 8. Related Concepts

- [Bellman Equations](./02-bellman-equations.md) — The recursive equations used to compute optimal values in an MDP.
- [Dynamic Programming for RL](./03-dynamic-programming-rl.md) — Algorithms that exploit the Bellman equations to solve MDPs exactly.
- [Monte Carlo Methods](./04-monte-carlo-methods.md) — Model-free estimation of V by averaging sampled trajectories.
- [Temporal Difference Learning](./05-temporal-difference-learning.md) — Bootstrap updates combining DP and MC ideas.
