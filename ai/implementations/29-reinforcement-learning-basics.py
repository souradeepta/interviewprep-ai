"""
Auto-generated from 29-reinforcement-learning-basics.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # 29 Reinforcement Learning Basics
# ## Learning Objectives
# 1. Understand core concepts of 29 reinforcement learning basics
# 2. Implement with production libraries
# ======================================================================

# ======================================================================
# ## Level 1: Basic Implementation
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

# Simple GridWorld environment
class GridWorld:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.agent_pos = (0, 0)
        self.goal_pos = (grid_size-1, grid_size-1)

    def reset(self):
        self.agent_pos = (0, 0)
        return self.agent_pos

    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        x, y = self.agent_pos

        if action == 0: x = max(0, x-1)
        elif action == 1: y = min(self.grid_size-1, y+1)
        elif action == 2: x = min(self.grid_size-1, x+1)
        elif action == 3: y = max(0, y-1)

        self.agent_pos = (x, y)

        # Reward: +1 at goal, -0.1 per step
        reward = 1.0 if self.agent_pos == self.goal_pos else -0.1
        done = self.agent_pos == self.goal_pos

        return self.agent_pos, reward, done

# Test environment
env = GridWorld()
state = env.reset()
print(f"Initial state: {state}")

# Take random action
action = np.random.randint(0, 4)
next_state, reward, done = env.step(action)
print(f"Action: {action}, Next state: {next_state}, Reward: {reward}")



# ======================================================================
# ## Level 2: Advanced Implementation
# ======================================================================

import numpy as np

class QLearning:
    def __init__(self, num_states, num_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((num_states, num_actions))
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def select_action(self, state):
        # Epsilon-greedy: explore with prob epsilon, exploit otherwise
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        # Q-learning update: Q(s,a) += lr * (r + gamma*max_a'Q(s',a') - Q(s,a))
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

    def train(self, env, num_episodes=100):
        episode_rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)

        return episode_rewards

# Train Q-learning agent
num_states = 25  # 5x5 gridworld
num_actions = 4
agent = QLearning(num_states, num_actions)

# Note: would train with actual environment
print("Q-learning agent initialized")
print(f"Q-table shape: {agent.Q.shape}")
print(f"Learning rate: {agent.lr}, Discount: {agent.gamma}, Epsilon: {agent.epsilon}")



# ======================================================================
# ## Real-World Example 1: Production Pattern
# ======================================================================

# Real-World: REINFORCE Policy Gradient
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        logits = self.fc2(x)
        return torch.softmax(logits, dim=-1)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        self.log_probs.append(torch.log(probs[action]))
        return action

    def update(self, gamma=0.99):
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = 0
        for log_prob, R in zip(self.log_probs, returns):
            loss += -log_prob * R

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

print("REINFORCE agent initialized")
print("Policy gradient learns by: log(pi(a|s)) * advantage")



# ======================================================================
# ## Real-World Example 2: Advanced Usage
# ======================================================================

# Real-World: Deep Q-Network (DQN)
import torch
import torch.nn as nn
from collections import deque
import numpy as np

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=10000)
        self.epsilon = 1.0

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self, batch_size=32, gamma=0.99):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch
        batch = np.random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q-learning loss
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

print("DQN agent initialized with experience replay")
print("Key: target network decouples action selection from evaluation")



# ======================================================================
# ## Real-World Example 3: Optimization
# ======================================================================

# Real-World: Actor-Critic Algorithm
import torch
import torch.nn as nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128, action_dim)  # Policy head
        self.critic = nn.Linear(128, 1)  # Value head

    def forward(self, state):
        x = torch.relu(self.fc(state))
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def train_step(self, state, action, reward, next_state, done, gamma=0.99):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Get predictions
        action_logits, value = self.network(state)
        _, next_value = self.network(next_state)

        # Compute advantage
        td_error = reward + gamma * next_value * (1-done) - value

        # Actor loss: -log(pi(a|s)) * advantage
        log_prob = torch.log_softmax(action_logits, dim=-1)[action]
        actor_loss = -log_prob * td_error.detach()

        # Critic loss: MSE of value prediction
        critic_loss = td_error**2

        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

print("Actor-Critic agent: combines policy gradient (actor) + value learning (critic)")
print("More stable than pure policy gradient, more flexible than pure Q-learning")




# ======================================================================
# ## Key Takeaways
# **When to use 29 reinforcement learning basics:**
# - Understand when this concept applies
# - Consider tradeoffs and constraints
# ======================================================================
