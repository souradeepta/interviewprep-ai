import nbformat as nbf

nb = nbf.v4.new_notebook()

# Title
nb.cells.append(nbf.v4.new_markdown_cell("# Simulation for Agents\n\nObjectives: Environment simulation, performance evaluation, safety testing, metrics collection, curriculum learning"))

# Level 1: Basic Environment and Agent
code1 = """import numpy as np
from typing import Tuple, Dict, Any

# Level 1: Basic Environment and Agent Interaction

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size - 1, size - 1])
        self.steps = 0
        self.max_steps = 50

    def reset(self) -> Dict:
        self.agent_pos = np.array([0, 0])
        self.steps = 0
        return self._get_obs()

    def _get_obs(self) -> Dict:
        return {
            'agent': tuple(self.agent_pos),
            'goal': tuple(self.goal_pos),
            'steps': self.steps
        }

    def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool]:
        dx, dy = action
        self.agent_pos = np.clip(self.agent_pos + np.array([dx, dy]), 0, self.size - 1)
        self.steps += 1

        # Reward: penalty per step, bonus for goal
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -1.0 + (100.0 if distance == 0 else 0)

        done = (distance == 0) or (self.steps >= self.max_steps)
        return self._get_obs(), reward, done

class SimpleAgent:
    def act(self, obs: Dict) -> Tuple[int, int]:
        agent = np.array(obs['agent'])
        goal = np.array(obs['goal'])
        direction = np.sign(goal - agent)
        return tuple(direction.astype(int))

# Test Level 1
print('Level 1 - Basic Simulation:\\n')
env = GridWorld(size=5)
agent = SimpleAgent()

total_reward = 0
obs = env.reset()
done = False

while not done:
    action = agent.act(obs)
    obs, reward, done = env.step(action)
    total_reward += reward

print(f'✓ Episode completed: {env.steps} steps, reward={total_reward:.1f}\\n')"""

nb.cells.append(nbf.v4.new_code_cell(code1))
nb.cells.append(nbf.v4.new_markdown_cell("**Key Points:** Environment provides reset/step interface. Agent observes state, chooses action. Step returns observation, reward, done. Simple loop for single episode."))

# Level 2: Multi-Episode Simulation with Metrics
code2 = """# Level 2: Multi-Episode Simulation with Metrics

class SimulationRunner:
    def __init__(self, env, agent, num_episodes=100):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes

    def run_episode(self) -> Dict:
        obs = self.env.reset()
        done = False
        episode_reward = 0
        steps = 0
        success = False

        while not done:
            action = self.agent.act(obs)
            obs, reward, done = self.env.step(action)
            episode_reward += reward
            steps += 1

            if reward > 50:
                success = True

        return {
            'reward': episode_reward,
            'steps': steps,
            'success': success
        }

    def run_all(self) -> Dict:
        results = [self.run_episode() for _ in range(self.num_episodes)]

        rewards = [r['reward'] for r in results]
        steps_list = [r['steps'] for r in results]
        successes = [r['success'] for r in results]

        return {
            'success_rate': np.mean(successes),
            'success_std': np.std(successes),
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'avg_steps': np.mean(steps_list),
            'confidence_interval': (
                np.mean(successes) - 1.96 * np.std(successes) / np.sqrt(len(successes)),
                np.mean(successes) + 1.96 * np.std(successes) / np.sqrt(len(successes))
            )
        }

# Test Level 2
print('Level 2 - Metrics-Based Simulation:\\n')
runner = SimulationRunner(GridWorld(size=5), SimpleAgent(), num_episodes=100)
metrics = runner.run_all()

print(f'Success rate: {metrics["success_rate"]:.1%} ± {metrics["success_std"]:.1%}')
print(f'95% CI: [{metrics["confidence_interval"][0]:.2%}, {metrics["confidence_interval"][1]:.2%}]')
print(f'Avg steps: {metrics["avg_steps"]:.1f} ± {metrics["avg_steps"]:.1f}\\n')"""

nb.cells.append(nbf.v4.new_code_cell(code2))
nb.cells.append(nbf.v4.new_markdown_cell("**Key Takeaways:** Run multiple episodes to estimate performance. Collect reward, success, steps metrics. Report confidence intervals, not just means. Identifies robustness of agent."))

# Example 1: Stochastic Environment with Noise
code3 = """# Example 1: Stochastic Environment with Noise

class NoisyGridWorld(GridWorld):
    def __init__(self, size=5, noise_prob=0.2):
        super().__init__(size)
        self.noise_prob = noise_prob

    def step(self, action):
        # Sometimes action gets perturbed by noise
        if np.random.random() < self.noise_prob:
            action = (np.random.randint(-1, 2), np.random.randint(-1, 2))

        return super().step(action)

class AdaptiveAgent:
    def act(self, obs: Dict) -> Tuple[int, int]:
        agent = np.array(obs['agent'])
        goal = np.array(obs['goal'])

        # Prefer moving directly toward goal, but explore if stuck
        best_move = tuple(np.sign(goal - agent).astype(int))

        # Occasional exploration
        if np.random.random() < 0.1:
            return (np.random.randint(-1, 2), np.random.randint(-1, 2))

        return best_move

# Compare deterministic vs noisy
print('Example 1 - Stochastic Environment:\\n')

for noise_prob in [0.0, 0.2, 0.5]:
    env = NoisyGridWorld(size=5, noise_prob=noise_prob)
    agent = AdaptiveAgent()
    runner = SimulationRunner(env, agent, num_episodes=50)
    metrics = runner.run_all()

    print(f'Noise probability {noise_prob:.0%}:')
    print(f'  Success: {metrics["success_rate"]:.1%}, Steps: {metrics["avg_steps"]:.1f}')"""

nb.cells.append(nbf.v4.new_code_cell(code3))
nb.cells.append(nbf.v4.new_markdown_cell("**Example 1 Key Points:** Stochastic environments require more episodes to characterize. Adaptive agents handle noise better. Real-world sim needs noise to be realistic."))

# Example 2: Curriculum Learning
code4 = """# Example 2: Curriculum Learning

class CurriculumEnvironment:
    def __init__(self):
        self.difficulty = 1
        self.env = GridWorld(size=2)
        self.successes = 0
        self.episodes = 0

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done = self.env.step(action)

        if done:
            self.episodes += 1
            if reward > 50:
                self.successes += 1

            # Increase difficulty if succeeding
            if self.episodes % 10 == 0:
                success_rate = self.successes / self.episodes
                if success_rate > 0.8 and self.difficulty < 10:
                    self.difficulty += 1
                    self.env = GridWorld(size=self.difficulty + 1)
                    print(f'  Increased difficulty to {self.difficulty}')
                    self.successes = 0
                    self.episodes = 0

        return obs, reward, done

# Test curriculum learning
print('\\nExample 2 - Curriculum Learning:\\n')
env = CurriculumEnvironment()
agent = SimpleAgent()

for episode in range(100):
    obs = env.reset()
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, done = env.step(action)

print(f'Final difficulty reached: {env.difficulty}')"""

nb.cells.append(nbf.v4.new_code_cell(code4))
nb.cells.append(nbf.v4.new_markdown_cell("**Example 2 Key Points:** Start simple, increase difficulty as agent succeeds. Faster learning than starting hard. Prevents agent from getting stuck. Common in RL."))

# Example 3: Multi-Agent Competition
code5 = """# Example 3: Multi-Agent Competition

class CompetitiveEnvironment:
    def __init__(self, num_agents=3):
        self.num_agents = num_agents
        self.agents_pos = [np.array([i, 0]) for i in range(num_agents)]
        self.goal = np.array([2, 4])
        self.steps = 0
        self.max_steps = 20
        self.scores = [0] * num_agents

    def reset(self):
        self.agents_pos = [np.array([i, 0]) for i in range(self.num_agents)]
        self.steps = 0
        self.scores = [0] * self.num_agents
        return self._get_obs()

    def _get_obs(self):
        return {
            'agents': [tuple(p) for p in self.agents_pos],
            'goal': tuple(self.goal),
            'scores': self.scores.copy()
        }

    def step(self, actions):
        # Move agents
        for i, action in enumerate(actions):
            dx, dy = action
            self.agents_pos[i] = np.clip(self.agents_pos[i] + np.array([dx, dy]), 0, 4)

        self.steps += 1

        # Check if any agent reached goal
        rewards = [-1.0] * self.num_agents
        for i, pos in enumerate(self.agents_pos):
            if np.allclose(pos, self.goal):
                rewards[i] = 100.0
                self.scores[i] += 1

        done = (self.steps >= self.max_steps)
        return self._get_obs(), rewards, done

class RandomAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def act(self, obs):
        return (np.random.randint(-1, 2), np.random.randint(-1, 2))

# Run competitive simulation
print('\\nExample 3 - Multi-Agent Competition:\\n')

env = CompetitiveEnvironment(num_agents=3)
agents = [RandomAgent(i) for i in range(3)]

win_counts = [0] * 3

for episode in range(50):
    obs = env.reset()
    done = False

    while not done:
        actions = [agents[i].act(obs) for i in range(3)]
        obs, rewards, done = env.step(actions)

    winner = np.argmax(env.scores)
    win_counts[winner] += 1

print('Win rates:')
for i, wins in enumerate(win_counts):
    print(f'  Agent {i}: {wins / 50:.1%}')"""

nb.cells.append(nbf.v4.new_code_cell(code5))
nb.cells.append(nbf.v4.new_markdown_cell("**Example 3 Key Points:** Multi-agent simulation lets you study competition. Track individual scores. Enables relative performance comparison. Useful for testing agent robustness against adversaries."))

# Key Takeaways
nb.cells.append(nbf.v4.new_markdown_cell("""## Key Takeaways

**Simulation Pattern:**
1. Define environment (state, action, dynamics)
2. Implement agent decision-making
3. Run episodes, collect metrics
4. Analyze results, iterate on agent

**What to Measure:**
- Success rate (primary metric)
- Efficiency (steps, cost, time)
- Safety (rule violations)
- Variance (confidence intervals)
- Robustness (across noise, difficulty)

**When to Deploy After Simulation:**
- Success rate stable (multiple runs agree)
- Confidence intervals tight
- Edge cases tested explicitly
- Compared favorably to baselines
- Safety checks pass

**Related Concepts:** [[agent-loops]], [[error-recovery]], [[observability]], [[safety-alignment]], [[reinforcement-learning]]"""))

# Save notebook
nbf.write(nb, '/home/sbisw/github/interviewprep-ml/agentic-ai/notebooks/simulation-for-agents.ipynb')
print("✓ Notebook created successfully")
