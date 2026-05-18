#!/usr/bin/env python3
"""Create new concept markdown files for curriculum expansion."""

import os
from pathlib import Path

BASE = "/home/sbisw/github/interviewprep-ml"

NEW_LLM_CONCEPTS = [
    ("33-vision-transformers.md", {
        "title": "Vision Transformers (ViT)",
        "description": "Apply transformer architecture to computer vision by dividing images into patches and treating them as sequences",
        "how_it_works": """1. Divide image into fixed patches (16×16 pixels)
2. Flatten patches into vectors and linearly project to embedding dimension
3. Add positional embeddings (absolute position of each patch)
4. Pass through transformer encoder stack
5. Use [CLS] token embedding for image classification
6. Compare: ViT vs CNN receptive fields, ViT requires more data, ViT scales better""",
        "interview_qa": [
            ("Why divide images into patches instead of processing pixels directly?", "Patches reduce sequence length: 224×224 image = 196 patches vs 50K pixels. Transformers have O(n²) complexity, so patches are essential. Also enables positional embeddings."),
            ("How does ViT compare to CNNs?", "ViT: no inductive bias (worse small-data), requires more training data, scales better to large datasets, more interpretable attention. CNN: inductive bias (shift/scale invariance), data-efficient, good for small images."),
            ("What is the [CLS] token in ViT?", "Learned token prepended to patch sequence, similar to BERT. Its final representation is used as image embedding for classification. Learned end-to-end via supervised training."),
            ("How does positional encoding work in ViT?", "Learned absolute positional embeddings (no sinusoidal like NLP). Added to patch embeddings before transformer. Why learned? ViT authors found it matched sinusoidal, so learned is simpler."),
            ("Can ViT process images of different sizes?", "No, fixed input size by design. To handle different sizes: resize, crop, or retrain. Some methods use adaptive pooling or dynamic patch sizes, but vanilla ViT needs fixed size."),
        ]
    }),
    ("34-multimodal-fusion.md", {
        "title": "Multimodal Fusion",
        "description": "Combine image and text representations in a shared embedding space for tasks like image captioning, visual question answering, and image-text retrieval",
        "how_it_works": """1. Encode images with CNN/ViT → image embeddings (2048D)
2. Encode text with BERT/GPT → text embeddings (768D)
3. Project both to shared dimension (e.g., 256D)
4. Compute similarity: cosine distance in shared space
5. Contrastive learning: pull matching pairs close, push mismatched pairs apart
6. Applications: CLIP (image-text matching), BLIP (captioning), LLaVA (vision-language models)""",
        "interview_qa": [
            ("What is the key insight in CLIP?", "Train on 400M image-text pairs with contrastive loss: similar pairs have high cosine similarity, dissimilar pairs have low. Results in aligned embedding space without labeled data."),
            ("How do you handle modality gaps?", "Modalities have different dimensionality and structure (image 2048D, text 768D). Solution: project to shared space (256D). Train end-to-end so projections are mutually aligned."),
            ("What's the difference between early fusion and late fusion?", "Early: concatenate features before processing (sensitive to modality differences). Late: process separately, combine at end (more modality-agnostic). ViLBERT uses late fusion for vision-language."),
            ("How do you evaluate multimodal models?", "Image-text retrieval: rank images for query, measure recall@k. Image captioning: BLEU/CIDEr scores. VQA: accuracy on questions. Use cross-modal metrics (retrieval) and task-specific metrics."),
            ("What is the role of negative sampling in contrastive learning?", "Without negatives, model could collapse (all embeddings identical). Negatives force discrimination: positive pair distance < negative pair distance. Batch size matters: larger batches = more negatives = better learning."),
        ]
    }),
    ("35-model-interpretability.md", {
        "title": "Model Interpretability",
        "description": "Understand what features and patterns neural networks learn by visualizing attention, probing hidden representations, and analyzing decision boundaries",
        "how_it_works": """1. Attention visualization: show which tokens attend to which (heatmaps)
2. Feature attribution: which input features contributed most to prediction
   - Gradient-based: ∂output/∂input (saliency maps)
   - Perturbation: remove tokens, measure output change
3. Probing: train classifier on hidden representations to extract information
4. Representation analysis: PCA/t-SNE on embeddings to visualize clusters
5. Layer-wise relevance propagation: trace predictions back through layers""",
        "interview_qa": [
            ("What can attention heatmaps tell us?", "They show which tokens the model focuses on for a prediction. Not always interpretable (attention != explanation), but useful for debugging. Multi-head attention shows different patterns per head."),
            ("What is probing and how is it different from inspection?", "Probing: train a classifier on hidden representations to check if information exists. Inspection: look at activations directly. Probing is more rigorous: shows what a decoder can extract, not what the model actually uses."),
            ("What's the difference between saliency and attribution?", "Saliency: gradient ∂y/∂x (local linear approximation). Attribution: integrated gradients (path from baseline to input, sums gradients). Attribution more stable, saliency faster."),
            ("Can attention really explain model decisions?", "Not always. Attention is computed, but model may make decisions based on hidden computation before attention. Attention is correlated with important features but not causal. Use with caution."),
            ("How do you interpret embeddings from language models?", "PCA: reduce to 2D, look for clusters (synonyms close?). Nearest neighbors: find similar words. Probing classifiers: train on known properties (gender, tense). Combinations give best insight."),
        ]
    }),
]

NEW_AGENTIC_CONCEPTS = [
    ("53-langchain-frameworks.md", {
        "title": "LangChain Framework",
        "description": "Build complex language model applications using chains, agents, memory, and tool integrations",
        "how_it_works": """1. LLMs (language models): wrappers for OpenAI, HuggingFace, etc.
2. Chains: sequence of calls to LLMs and other tools
   - LLMChain: template → format input → LLM → parse output
   - SequentialChain: run chains in sequence, pass outputs
3. Agents: LLM decides which tools to use iteratively
   - Think: LLM decides next action
   - Act: execute tool
   - Observe: see result
   - Repeat
4. Memory: store conversation history, retrieved context
5. Tools: calculators, search, databases, APIs""",
        "interview_qa": [
            ("What's the difference between chains and agents?", "Chains: predetermined sequence of steps. Agents: LLM decides steps dynamically. Chains are deterministic, agents flexible. Use chains for fixed workflows, agents for reasoning-based decisions."),
            ("How does LangChain memory work?", "Stores messages (user, assistant) in conversation. Types: ConversationBufferMemory (all), ConversationSummaryMemory (summarize old), ConversationKGMemory (knowledge graph). Tradeoff: length vs richness."),
            ("What are prompts and prompt templates in LangChain?", "Templates: reusable prompt structures with variables. Example: 'Analyze {text} for {sentiment}'. At runtime, variables are filled. Enables prompt reuse, versioning, testing."),
            ("How do you handle errors in chains?", "Try-catch in code, but also: ValidationError for bad format, LLMException for API fails. Add fallbacks: if tool fails, try alternative. Retry with backoff for transient errors."),
            ("What is output parsing and why is it needed?", "LLMs output free text; need to extract structured data. Parsers: JSON (force format), PydanticOutputParser (validate schema), CommaSeparatedListOutputParser. Critical for using outputs downstream."),
        ]
    }),
    ("54-autogen-orchestration.md", {
        "title": "AutoGen Multi-Agent Orchestration",
        "description": "Coordinate multiple agents with different roles and capabilities to solve complex tasks through conversation",
        "how_it_works": """1. Define agents with specific roles (researcher, critic, executor)
2. Assign capabilities: tool access, knowledge base, reasoning
3. Agent interaction: send messages back-and-forth
   - Agent A → Agent B: question or result
   - Agent B → Agent A: response or next action
4. Termination: conversation ends when task solved or max rounds reached
5. Nested agents: agent calls another agent as a tool
6. Example: user → researcher (gathers info) → critic (evaluates) → executor (takes action)""",
        "interview_qa": [
            ("How do you define agent roles in AutoGen?", "System message defines role: 'You are a researcher. Your job is to...'. Role shapes how agent interprets requests and what tools it has. Clear roles reduce hallucination and wasted steps."),
            ("What's better: one smart agent or many specialized agents?", "Many agents: better division of labor, easier to debug (separate concerns), can verify work. One agent: simpler, fewer tokens (fewer messages), faster. Use many for complex tasks, one for simple."),
            ("How do you handle agent disagreement?", "Add a moderator agent that arbitrates. Or voting: each agent proposes solution, majority wins. Or escalation: complex cases go to human. Context matters: some tasks need consensus, others don't."),
            ("What is the cost of multi-agent systems vs single agent?", "Multi-agent: more API calls (message overhead), more tokens total. Single agent: fewer calls but may fail on complex tasks. Measure: token count × cost/token + latency tradeoff. Multi-agent often cheaper for complex tasks."),
            ("How do you prevent infinite loops in agent conversations?", "Set max_rounds (e.g., 20 turns max). Monitor: if same question repeated, stop. Termination condition: check if task complete (heuristic). Human-in-loop: user can stop anytime."),
        ]
    }),
]

NEW_AI_CONCEPTS = [
    ("29-reinforcement-learning-basics.md", {
        "title": "Reinforcement Learning Basics",
        "description": "Understand agents learning to maximize reward through interaction with an environment, without labeled data",
        "how_it_works": """1. Agent: makes decisions in environment
2. Environment: responds to actions, gives rewards and observations
3. Markov Decision Process (MDP): states, actions, transitions, rewards
4. Goal: maximize cumulative reward (discount future rewards: γ^t × reward_t)
5. Policy π: maps states to actions (deterministic or stochastic)
6. Value function V(s): expected cumulative reward from state s
7. Q-function Q(s,a): expected cumulative reward from state s, action a
8. Learning: improve policy by estimating V or Q, then act greedily""",
        "interview_qa": [
            ("What's the difference between on-policy and off-policy learning?", "On-policy: learn from current policy (REINFORCE). Off-policy: learn from different policy (Q-learning). Off-policy is sample-efficient but harder (importance weighting). On-policy is simpler but needs more samples."),
            ("Why is the discount factor γ important?", "γ < 1 makes future rewards worth less (rewards closer are worth more). γ close to 1: agent plans long-term. γ close to 0: agent myopic (immediate reward only). Choose based on task: long-horizon = larger γ."),
            ("What is the exploration-exploitation tradeoff?", "Exploitation: use best known action. Exploration: try new actions to find better ones. Epsilon-greedy: choose random action with probability ε, greedy otherwise. Critical for learning."),
            ("How do you handle continuous action spaces?", "Discrete: Q(s,a) table or network. Continuous: policy gradient (output action directly). Example: actor-critic where actor outputs mean/variance of action distribution, critic estimates value."),
            ("What makes RL training unstable?", "Non-stationary target (V(s) changes as policy improves). Correlation in samples (action depends on previous actions). Solution: target network (update slowly), replay buffer (decorrelate samples), gradient clipping."),
        ]
    }),
    ("30-markov-decision-processes.md", {
        "title": "Markov Decision Processes (MDPs)",
        "description": "Formalize sequential decision-making as MDPs with states, actions, transitions, and rewards",
        "how_it_works": """1. State space S: all possible states
2. Action space A: all possible actions
3. Transition function P(s'|s,a): probability of next state given state and action
4. Reward function R(s,a): immediate reward for action in state
5. Markov property: P(s'|s,a) depends only on s,a (not history)
6. Discount factor γ: weight of future rewards
7. Horizon T: episode length (finite or infinite)
8. Solution: policy π(a|s) that maximizes expected cumulative reward""",
        "interview_qa": [
            ("What does the Markov property mean?", "Future state depends only on current state and action, not the path taken to reach it. Implies no memory needed beyond current state. Simplifies computation but may not hold in partially observable environments."),
            ("What's the difference between episodic and continuous tasks?", "Episodic: finite horizon, clear endpoint (game ends, task completes). Continuous: infinite horizon, no natural end (robot control). Learning differs: episodic can use finite return, continuous needs discounting."),
            ("How do you define states for an MDP?", "Trade-off between: sufficient information to make decisions (Markov property) vs. tractability (small state space). May use hand-crafted features, learned representations (NN), or raw observations."),
            ("What is a stochastic vs deterministic policy?", "Deterministic: π(a|s) = 1 for one action. Stochastic: π(a|s) ∈ [0,1] for multiple actions. Stochastic better for exploration, deterministic after learning. Often start stochastic, anneal to deterministic."),
            ("How do you handle partial observability?", "MDP assumes full observability (see all relevant state info). If not, use POMDP (partially observable MDP). Solution: maintain belief state (distribution over possible states). Harder but more realistic."),
        ]
    }),
]

CONCEPT_TEMPLATE = """# {title}

## Detailed Explanation

{description}

## Core Intuition

{intuition}

## How It Works

{how_it_works}

```mermaid
graph TD
    A[Input] -->|Process| B[Model/Algorithm]
    B -->|Output| C[Result]
```

## Architecture / Trade-offs

Trade-off 1 vs trade-off 2 — consider context and requirements.

## Interview Q&A

{qa_section}

## Best Practices

- Research and implement best practices as you learn the concept
- Consider production implications and scalability
- Test on realistic data and benchmarks
- Monitor performance and iterate

## Common Pitfalls

- Oversimplifying the problem — understand nuances
- Ignoring computational costs and practicality
- Not validating assumptions with real data
- Premature optimization without profiling

## Code Examples

See concept implementation and real-world examples in the associated notebook.

## Related Concepts

- Review foundational concepts first
- Understand prerequisites before advanced topics
- Connect concepts to build integrated knowledge
"""

def create_concept_file(filepath, title, description, how_it_works, interview_qas):
    """Create a concept markdown file."""

    # Format interview Q&A
    qa_section = ""
    for q, a in interview_qas:
        qa_section += f"\n**Q: {q}**\nA: {a}\n"

    # Simple intuition
    intuition = description.split(".")[0] + " Core idea: understand the fundamental principle and how it applies."

    content = CONCEPT_TEMPLATE.format(
        title=title,
        description=description,
        intuition=intuition,
        how_it_works=how_it_works,
        qa_section=qa_section
    )

    with open(filepath, 'w') as f:
        f.write(content)

    return filepath

def main():
    """Create all new concept files."""

    print("=== Creating New Concept Files ===\n")

    # LLM concepts
    print("LLM Section (new concepts 33-44):")
    llm_dir = f"{BASE}/llm/concepts"
    for filename, metadata in NEW_LLM_CONCEPTS:
        filepath = os.path.join(llm_dir, filename)
        create_concept_file(
            filepath,
            metadata["title"],
            metadata["description"],
            metadata["how_it_works"],
            metadata["interview_qa"]
        )
        print(f"  ✓ {filename}")

    print("\nAgentic AI Section (new concepts 53-64):")
    agentic_dir = f"{BASE}/agentic-ai/concepts"
    for filename, metadata in NEW_AGENTIC_CONCEPTS:
        filepath = os.path.join(agentic_dir, filename)
        create_concept_file(
            filepath,
            metadata["title"],
            metadata["description"],
            metadata["how_it_works"],
            metadata["interview_qa"]
        )
        print(f"  ✓ {filename}")

    print("\nAI Fundamentals Section (new concepts 29-40):")
    ai_dir = f"{BASE}/ai/concepts"
    for filename, metadata in NEW_AI_CONCEPTS:
        filepath = os.path.join(ai_dir, filename)
        create_concept_file(
            filepath,
            metadata["title"],
            metadata["description"],
            metadata["how_it_works"],
            metadata["interview_qa"]
        )
        print(f"  ✓ {filename}")

    print("\n✅ Created 8 new concept files (more will be added in next phase)")

if __name__ == "__main__":
    main()
