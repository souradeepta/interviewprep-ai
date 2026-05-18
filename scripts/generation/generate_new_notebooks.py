#!/usr/bin/env python3
"""Generate notebooks for newly created concepts."""

import json
import os

BASE = "/home/sbisw/github/interviewprep-ml"

NOTEBOOK_IMPLEMENTATIONS = {
    "33-vision-transformers": {
        "level1": """import torch
from torchvision.transforms import ToTensor
from PIL import Image

# Simple Vision Transformer from scratch
class SimpleViT:
    def __init__(self, image_size=224, patch_size=16, embedding_dim=768):
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_dim = embedding_dim

        # Initialize learnable parameters
        self.class_token = torch.randn(1, 1, embedding_dim)
        self.patch_embeddings = torch.randn(self.num_patches, embedding_dim)
        self.position_embeddings = torch.randn(self.num_patches + 1, embedding_dim)

    def extract_patches(self, image):
        # Image: (C, H, W), output patches
        patches = []
        for i in range(0, image.shape[1], self.patch_size):
            for j in range(0, image.shape[2], self.patch_size):
                patch = image[:, i:i+self.patch_size, j:j+self.patch_size]
                patch = patch.flatten()  # Flatten to 1D
                patches.append(patch)
        return torch.stack(patches)  # (num_patches, patch_size*patch_size*C)

# Load and process image
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Simulate with random image
image = torch.randn(3, 224, 224)
vit = SimpleViT()

patches = vit.extract_patches(image)
print(f"Image shape: {image.shape}")
print(f"Number of patches: {patches.shape[0]}")
print(f"Patch embedding dim: {vit.embedding_dim}")""",
        "level2": """import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

class ViTClassifier:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)

    def classify(self, image_url):
        # Download and load image
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Extract features
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get top-k predictions
        logits = outputs.logits
        top_k = torch.topk(logits, 5)

        return top_k

    def get_attention_maps(self, image_url):
        # Get attention from intermediate layers
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        attentions = outputs.attentions
        return attentions  # Tuple of attention maps per layer

# Initialize classifier
classifier = ViTClassifier()

# Example: classify ImageNet image
test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
print("ViT model initialized and ready for classification")
print(f"Model name: google/vit-base-patch16-224")
print(f"Number of patches: 196 (14x14 grid of 16x16 patches)")""",
        "example1": """# Real-World: Fine-tune ViT on Custom Dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import Dataset, DataLoader
import torch

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, feature_extractor):
        self.images = images  # List of PIL Images
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        inputs = self.feature_extractor(self.images[idx], return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'label': torch.tensor(self.labels[idx])
        }

# Load pretrained ViT
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10  # 10 custom classes
)

# Create dummy dataset
num_images = 100
dummy_images = [torch.randint(0, 256, (3, 224, 224)).permute(1,2,0).numpy().astype('uint8')
                 for _ in range(num_images)]
dummy_labels = [i % 10 for i in range(num_images)]

dataset = CustomImageDataset(dummy_images, dummy_labels, feature_extractor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

print(f"Dataset size: {len(dataset)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print("Ready for fine-tuning")""",
        "example2": """# Real-World: Visualize ViT Attention
import matplotlib.pyplot as plt
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

def visualize_attention(image_url, layer=0, head=0):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", output_attentions=True)

    # Load and process image
    from PIL import Image
    import requests
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Get attention maps
    with torch.no_grad():
        outputs = model(**inputs)
        attention = outputs.attentions[layer]  # (batch, heads, seq_len, seq_len)

    # Attention for specific head
    att_map = attention[0, head, 0, 1:].view(14, 14)  # CLS token attention to patches

    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(att_map.numpy(), cmap='hot')
    plt.title(f"Attention Map (Layer {layer}, Head {head})")

    plt.show()

print("ViT attention visualization code ready")
print("Attention reveals which patches the model focuses on for classification")""",
        "example3": """# Real-World: ViT for Zero-Shot Classification
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

class ViTZeroShot:
    def __init__(self):
        # Note: Zero-shot requires CLIP-like model, here using ViT for illustration
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    def classify_new_categories(self, image, categories):
        '''
        Zero-shot: classify into categories not in training data
        Approach: Use embedding similarity instead of softmax
        '''
        # Get image embedding
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embedding = outputs.logits  # Use last layer as embedding

        # In practice, would use text encoder for categories and compare embeddings
        # Here showing the pattern
        print(f"Image embedding: {image_embedding.shape}")
        print("Zero-shot requires dual encoders (image + text)")
        return image_embedding

# For true zero-shot, use CLIP
from transformers import CLIPProcessor, CLIPModel

class CLIPZeroShot:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def classify(self, image, categories):
        # Encode image and text
        inputs = self.processor(text=categories, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # (1, num_categories)

        scores = torch.softmax(logits_per_image, dim=1)[0]
        for category, score in zip(categories, scores):
            print(f"{category}: {score:.3f}")

print("CLIP zero-shot classification pattern:")
print("1. Encode image with vision model")
print("2. Encode categories with text model")
print("3. Compute similarity (logits)")
print("4. Apply softmax to get probabilities")
"""
    },
    "53-langchain-frameworks": {
        "level1": """from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = OpenAI(temperature=0.9)

# Create a simple prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a fun fact about {topic}",
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run(topic="machine learning")
print(f"Result: {result}")

# Verify components
print(f"\\nLLM: {llm.__class__.__name__}")
print(f"Prompt variables: {prompt.input_variables}")
print(f"Chain type: {chain.__class__.__name__}")""",
        "level2": """from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# Initialize components
llm = OpenAI(temperature=0.7)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
memory = ConversationBufferMemory(memory_key="chat_history")

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Run agent
response = agent.run(input="What is 2+2? Then search for recent AI news")

print(f"\\nAgent Response: {response}")
print(f"Memory: {memory.buffer}")""",
        "example1": """# Real-World: Build a Multi-Step Research Agent
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Tools for research
llm = OpenAI(temperature=0.7)
tools = load_tools(["serpapi", "summarization"], llm=llm)

# Agent for research
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.REACT,
    verbose=False
)

# Research prompt
research_prompt = PromptTemplate(
    template="Research and summarize: {query}",
    input_variables=["query"]
)

# Multi-step workflow
queries = [
    "Latest breakthroughs in transformer models",
    "Top papers on agents",
    "Comparison of frameworks"
]

results = []
for query in queries:
    result = agent.run(research_prompt.format(query=query))
    results.append(result)

print(f"Researched {len(results)} topics")
print(f"First result preview: {results[0][:100]}...")""",
        "example2": """# Real-World: Document Question-Answering
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create QA chain
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Ask questions
questions = [
    "What is the main topic?",
    "What are key points?",
]

for q in questions:
    answer = qa.run(q)
    print(f"Q: {q}\\nA: {answer}\\n")""",
        "example3": """# Real-World: Custom Tools and Tool Composition
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.llms import OpenAI

# Define custom tools
def calculate_product(a: str) -> str:
    '''Multiply two numbers'''
    nums = a.split('*')
    return str(float(nums[0]) * float(nums[1]))

def word_count(text: str) -> str:
    '''Count words in text'''
    return str(len(text.split()))

# Create Tool objects
tools = [
    Tool(name="Calculator", func=calculate_product, description="Multiply numbers"),
    Tool(name="WordCount", func=word_count, description="Count words"),
]

# Create agent with custom tools
llm = OpenAI(temperature=0.7)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Test agent
result = agent.run("What is 5*3? Count words: 'Hello world'")
print(f"Result: {result}")
"""
    },
    "29-reinforcement-learning-basics": {
        "level1": """import numpy as np
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
print(f"Action: {action}, Next state: {next_state}, Reward: {reward}")""",
        "level2": """import numpy as np

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
print(f"Learning rate: {agent.lr}, Discount: {agent.gamma}, Epsilon: {agent.epsilon}")""",
        "example1": """# Real-World: REINFORCE Policy Gradient
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
print("Policy gradient learns by: log(pi(a|s)) * advantage")""",
        "example2": """# Real-World: Deep Q-Network (DQN)
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
print("Key: target network decouples action selection from evaluation")""",
        "example3": """# Real-World: Actor-Critic Algorithm
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
"""
    }
}

TEMPLATE = """# {title}

## Learning Objectives
1. Understand core concepts of {lower_title}
2. Implement algorithms with PyTorch and HuggingFace
3. Apply to practical problems
4. Optimize for performance and stability

See also: `{concept_path}` for theory and interview Q&A

## Level 1: Basic Implementation

{level1_intro}

## Level 1: Code

```python
{level1_code}
```

## Level 2: Advanced Implementation

{level2_intro}

## Level 2: Code

```python
{level2_code}
```

## Real-World Example 1: {example1_title}

```python
{example1_code}
```

## Real-World Example 2: {example2_title}

```python
{example2_code}
```

## Real-World Example 3: {example3_title}

```python
{example3_code}
```

## Key Takeaways

- Understand the fundamental principles
- Practice implementation with real libraries
- Test on benchmarks and evaluate performance
- Consider production constraints and deployment

## Related Concepts

- Refer to the concept file for foundational knowledge
- Connect to related concepts for integrated understanding
"""

def create_notebook_cell(cell_type, source):
    """Create a notebook cell."""
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]

    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [] if cell_type == "code" else None,
        "source": source
    }

def generate_notebooks():
    """Generate notebooks for new concepts."""

    print("=== Generating Notebooks for New Concepts ===\n")

    sections = [
        ("llm", f"{BASE}/llm/notebooks", ["33-vision-transformers", "34-multimodal-fusion", "35-model-interpretability"]),
        ("agentic-ai", f"{BASE}/agentic-ai/notebooks", ["53-langchain-frameworks"]),
        ("ai", f"{BASE}/ai/notebooks", ["29-reinforcement-learning-basics"]),
    ]

    generated = 0

    for section_name, notebooks_dir, concept_slugs in sections:
        print(f"{section_name.upper()} Notebooks:")

        for concept_slug in concept_slugs:
            if concept_slug not in NOTEBOOK_IMPLEMENTATIONS:
                print(f"  - {concept_slug} (no implementation yet)")
                continue

            impl = NOTEBOOK_IMPLEMENTATIONS[concept_slug]
            title = concept_slug.replace("-", " ").title()
            lower_title = title.lower()

            # Get section-specific paths
            if "llm" in section_name:
                concept_path = f"../concepts/{concept_slug}.md"
            else:
                concept_path = f"../concepts/{concept_slug}.md"

            # Create notebook cells
            notebook = {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.10.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }

            # Cell 0: Title
            notebook["cells"].append(create_notebook_cell("markdown", f"""# {title}

## Learning Objectives
1. Understand core concepts of {lower_title}
2. Implement with production libraries
3. Apply to real-world problems
4. Optimize and evaluate

See also: `{concept_path}` for theory and interview Q&A"""))

            # Cell 1: Level 1 intro
            notebook["cells"].append(create_notebook_cell("markdown", f"## Level 1: Basic Implementation"))

            # Cell 2: Level 1 code
            notebook["cells"].append(create_notebook_cell("code", impl.get("level1", "# TODO")))

            # Cell 3: Level 2 intro
            notebook["cells"].append(create_notebook_cell("markdown", f"## Level 2: Advanced Implementation"))

            # Cell 4: Level 2 code
            notebook["cells"].append(create_notebook_cell("code", impl.get("level2", "# TODO")))

            # Cell 5: Example 1
            notebook["cells"].append(create_notebook_cell("markdown", f"## Real-World Example 1: Production Pattern"))

            # Cell 6: Example 1 code
            notebook["cells"].append(create_notebook_cell("code", impl.get("example1", "# TODO")))

            # Cell 7: Example 2
            notebook["cells"].append(create_notebook_cell("markdown", f"## Real-World Example 2: Advanced Usage"))

            # Cell 8: Example 2 code
            notebook["cells"].append(create_notebook_cell("code", impl.get("example2", "# TODO")))

            # Cell 9: Example 3
            notebook["cells"].append(create_notebook_cell("markdown", f"## Real-World Example 3: Optimization"))

            # Cell 10: Example 3 code
            notebook["cells"].append(create_notebook_cell("code", impl.get("example3", "# TODO")))

            # Cell 11: Takeaways
            notebook["cells"].append(create_notebook_cell("markdown", f"""## Key Takeaways

**When to use {lower_title}:**
- Understand when this concept applies
- Consider tradeoffs and constraints
- Evaluate against alternatives

**Related Concepts:**
- See concept file for theory
- Connect to related topics
- Build integrated knowledge"""))

            # Write notebook
            notebook_path = os.path.join(notebooks_dir, f"{concept_slug}.ipynb")
            with open(notebook_path, 'w') as f:
                json.dump(notebook, f, indent=1)

            print(f"  ✓ {concept_slug}.ipynb")
            generated += 1

    print(f"\n✅ Generated {generated} new notebooks (12 cells each)")

if __name__ == "__main__":
    generate_notebooks()
