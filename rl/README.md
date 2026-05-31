# Reinforcement Learning

20 concepts from MDP foundations to RLHF. Each concept has a markdown explanation,
runnable notebook, and standalone implementation.

## Prerequisites

- Python, numpy, torch, matplotlib
- Basic probability (Markov chains, expectations)
- Basic linear algebra (matrix operations)

## Concepts

| # | Concept | Category | Algorithm |
|---|---------|----------|-----------|
| 01 | [Markov Decision Processes](concepts/01-markov-decision-processes.md) | Foundations | MDP tuple (S, A, P, R, γ) |
| 02 | [Bellman Equations](concepts/02-bellman-equations.md) | Foundations | V(s) = max_a [R + γ·V(s')] |
| 03 | [Dynamic Programming RL](concepts/03-dynamic-programming-rl.md) | Planning | Value iteration, policy iteration |
| 04 | [Monte Carlo Methods](concepts/04-monte-carlo-methods.md) | Model-Free | First-visit MC, every-visit MC |
| 05 | [Temporal Difference Learning](concepts/05-temporal-difference-learning.md) | Model-Free | TD(0), TD(λ), eligibility traces |
| 06 | [Q-Learning](concepts/06-q-learning.md) | Value-Based | Off-policy TD, Q(s,a) update |
| 07 | [SARSA](concepts/07-sarsa.md) | Value-Based | On-policy TD, ε-greedy |
| 08 | [Deep Q-Networks](concepts/08-deep-q-networks.md) | Deep RL | DQN, experience replay, target net |
| 09 | [Policy Gradient](concepts/09-policy-gradient.md) | Policy-Based | REINFORCE, log-gradient trick |
| 10 | [Actor-Critic](concepts/10-actor-critic.md) | Policy-Based | A2C, advantage function |
| 11 | [Proximal Policy Optimization](concepts/11-proximal-policy-optimization.md) | Advanced | PPO clip, KL penalty, RLHF use |
| 12 | [Soft Actor-Critic](concepts/12-soft-actor-critic.md) | Advanced | SAC, entropy regularization |
| 13 | [Multi-Armed Bandit](concepts/13-multi-armed-bandit.md) | Exploration | ε-greedy, UCB, Thompson sampling |
| 14 | [Exploration-Exploitation](concepts/14-exploration-exploitation.md) | Core Concept | ε-greedy, UCB, intrinsic motivation |
| 15 | [Reward Shaping](concepts/15-reward-shaping.md) | Engineering | Potential-based, sparse→dense reward |
| 16 | [Model-Based RL](concepts/16-model-based-rl.md) | Model-Based | Dyna-Q, world models, MBPO |
| 17 | [RLHF](concepts/17-rlhf.md) | LLM Alignment | Reward model, PPO, KL penalty |
| 18 | [Inverse RL](concepts/18-inverse-rl.md) | Advanced | Learning reward from demonstrations |
| 19 | [Multi-Agent RL](concepts/19-multi-agent-rl.md) | Advanced | Nash equilibrium, CTDE, MADDPG |
| 20 | [Offline RL](concepts/20-offline-rl.md) | Advanced | Conservative Q-learning, BCQ |

## Learning Paths

### RL Fundamentals (Concepts 01-07)
MDP → Bellman → DP → MC → TD → Q-Learning → SARSA

### Deep RL (Concepts 08-12)
DQN → Policy Gradient → Actor-Critic → PPO → SAC

### LLM Alignment (Concepts 11, 17)
PPO → RLHF (critical for understanding modern LLMs)

### Full Curriculum (4 weeks)
1. Week 1: Foundations (01-05)
2. Week 2: Value-Based Methods (06-10)
3. Week 3: Advanced Policy Methods (11-15)
4. Week 4: Modern Applications (16-20)

## Notebooks

Each notebook follows the 12-cell structure:
- Cells 1-2: Title + objectives, imports + device setup
- Cells 3-4: Level 1 basic numpy implementation
- Cells 5-6: Level 2 advanced torch implementation
- Cells 7-9: Three real-world examples
- Cells 10-11: Comparison visualization + key takeaways
- Cell 12: Exercises

**Note:** All environments are implemented from scratch using numpy/torch (no gym/gymnasium dependency).

## Standalone Implementations

Each `implementations/NN-topic.py` is a self-contained 150-250 line script that:
- Imports only numpy, torch, and matplotlib
- Defines 2-3 classes or functions
- Includes `if __name__ == "__main__"` demo with timing
- Shows expected output in comments
