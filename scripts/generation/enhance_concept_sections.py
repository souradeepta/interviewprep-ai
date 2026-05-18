#!/usr/bin/env python3
"""Enhance Detailed Explanation and Core Intuition sections in all concepts."""

import os
import re
from pathlib import Path

BASE = "/home/sbisw/github/interviewprep-ml"

# Expansions for each concept
EXPANSIONS = {
    # AI Fundamentals
    "31-q-learning": {
        "detailed": """Q-Learning is a foundational reinforcement learning algorithm that enables agents to learn optimal decision-making policies through trial-and-error interaction with an environment. Unlike supervised learning, Q-Learning doesn't require labeled examples—instead, the agent learns by receiving rewards or penalties for its actions and updating its understanding of which state-action pairs are valuable.

The core insight is that every state-action pair has a Q-value representing the expected cumulative future reward. By iteratively updating these values based on observed rewards and the maximum future value of the next state, the algorithm converges to an optimal policy that maximizes long-term reward. Q-Learning is off-policy, meaning it can learn the optimal policy while following a different exploratory policy, making it sample-efficient in many domains.

Q-Learning powers practical systems from game-playing agents (Atari) to robotic control and recommendation systems. It's crucial to understand because it bridges the gap between simple reactive agents and sophisticated planning systems. The algorithm introduces key concepts like exploration-exploitation tradeoff, value iteration, and temporal difference learning that extend to modern deep reinforcement learning.""",
        "intuition": """Imagine learning to play chess by experimenting with moves and remembering how good each position turned out to be. Q-Learning is exactly that: the agent tries actions, gets rewards/penalties, and remembers the value of each state-action pair. Over time, it learns which moves lead to good outcomes without being explicitly taught the rules."""
    },
    "32-policy-gradients": {
        "detailed": """Policy Gradient methods learn decision-making policies directly by adjusting the parameters of a neural network that outputs actions. Unlike value-based methods like Q-Learning that estimate future rewards and then act greedily, policy gradients use the gradient of expected reward with respect to policy parameters to update the policy towards better actions.

The fundamental idea is to increase the probability of actions that led to high rewards and decrease the probability of actions that led to low rewards. This is expressed as a gradient: ∇J(θ) = E[∇log π(a|s) R], which means we move the policy parameters in the direction that increases the log-probability of good actions scaled by their returns.

Policy gradients have several advantages: they handle continuous action spaces naturally (by outputting means and variances), they converge to local optima directly (not approximating values), and they support stochastic policies (useful for exploration). They power systems from robotic control to game-playing agents. The trade-off is higher variance in gradient estimates compared to value methods, requiring careful learning rate tuning and variance reduction techniques.""",
        "intuition": """Instead of learning the value of each chess position, directly learn which moves are good in each position. The policy gradient approach adjusts move probabilities: if a move sequence led to victory, increase those move probabilities; if it led to defeat, decrease them. It's like a coach watching your game and saying 'do that move more often, do that move less often'."""
    },
    "33-actor-critic-methods": {
        "detailed": """Actor-Critic methods combine two neural networks: an actor that learns the policy (which actions to take) and a critic that learns the value function (how good a state is). This hybrid approach leverages the strengths of both policy gradient and value-based methods while addressing their individual weaknesses.

The actor uses policy gradients to improve the policy, but instead of using the full episode return as the reward signal, it uses the critic's value estimate, which significantly reduces variance in gradient estimates. The critic learns to accurately estimate state values using temporal difference learning, providing low-variance training signals to the actor. This mutual improvement creates a powerful learning dynamic: the critic provides better training signal to the actor, while the actor's improved policy helps the critic learn better value estimates.

Actor-Critic methods are fundamental to modern deep reinforcement learning (A3C, PPO, TRPO) and excel at both discrete and continuous control. They balance the stability of value methods with the flexibility of policy gradients. Understanding actor-critic is essential because it demonstrates how neural networks can simultaneously solve multiple related problems and how bootstrapping (using value estimates as targets) enables efficient learning in continuous domains.""",
        "intuition": """The actor is the decision-maker (policy), choosing which actions to take based on learned experience. The critic is the evaluator, estimating how good each state is. The critic tells the actor 'that action led to a better outcome than expected' (or worse), helping the actor learn faster. Together, they're like a student and teacher: the student learns actions, the teacher provides feedback."""
    },
    "34-graph-neural-networks": {
        "detailed": """Graph Neural Networks (GNNs) extend neural networks to data with graph structure—molecules, social networks, knowledge graphs, and recommendation systems. Traditional neural networks assume grid-like data (images) or sequences (text), but many real-world domains are naturally graphs where relationships between entities matter as much as the entities themselves.

GNNs learn by aggregating information from neighboring nodes, allowing each node's representation to incorporate both its features and the features of connected nodes. Through multiple layers of message passing, distant nodes can indirectly influence each other, enabling the network to capture long-range dependencies and structural patterns. The key innovation is permutation invariance: the network produces consistent results regardless of node ordering, naturally respecting the graph structure.

GNNs power recommendation systems (incorporating user-item interaction graphs), molecular property prediction (atoms as nodes, bonds as edges), knowledge base completion, and social network analysis. They're increasingly important because many real-world problems involve structured relationships that traditional neural networks miss. Understanding GNNs requires thinking beyond Euclidean space and embracing discrete structures, making it essential for anyone working on relational data or network-based problems.""",
        "intuition": """Imagine nodes in a network where each node learns from its neighbors. A Twitter user's recommendation doesn't depend just on their own preferences, but also on what their friends like. GNNs work like information spreading through a network: each node receives messages from neighbors, updates its understanding, and passes updated messages forward. Repeat this a few times and each node understands not just local neighbors but the broader network structure."""
    },
    "35-causal-inference": {
        "detailed": """Causal inference is the science of determining cause-and-effect relationships from data, distinguishing between correlation and causation. While traditional machine learning predicts patterns, causal inference answers intervention questions: 'What happens if we change X?' rather than just 'What is X correlated with?' This distinction is crucial for decision-making in medicine, business, and policy.

Causal inference uses directed acyclic graphs (causal diagrams) to encode assumptions about how variables influence each other, then uses statistical techniques to estimate causal effects even from observational (non-experimental) data. Methods like propensity score matching, instrumental variables, and causal forests allow analysts to estimate the effect of an intervention on an outcome, accounting for confounding variables. The core insight is that randomized experiments automatically balance confounders, while observational data requires careful statistical control.

Causal reasoning is essential for any decisions beyond prediction: Should we deploy this model? How will this policy change affect outcomes? Does this correlation indicate a business opportunity? Understanding causality prevents costly mistakes from mistaking correlation for causation and enables principled decision-making under uncertainty. It's becoming increasingly important as organizations move from 'what will happen' (prediction) to 'what should we do' (decision-making).""",
        "intuition": """Correlation means two things happen together; causation means one causes the other. Ice cream sales correlate with drowning deaths, but ice cream doesn't cause drowning—summer causes both. Causal inference is the detective work of determining which relationships are real causes. It uses data patterns and logical reasoning to answer 'if we change this variable, what actually changes as a result'."""
    },
    "36-probabilistic-graphical-models": {
        "detailed": """Probabilistic Graphical Models (PGMs) represent probability distributions using graph structure where nodes are random variables and edges encode conditional dependencies. They enable efficient reasoning about uncertainty in complex systems by exploiting conditional independence—the fact that some variables don't directly influence others given intermediate information.

Bayesian Networks (DAGs) encode causal or temporal relationships, while Markov Random Fields (undirected graphs) encode symmetric relationships. The graphical structure determines how we can decompose the joint probability distribution into tractable factors, enabling efficient inference even in high-dimensional problems. Algorithms like variable elimination and belief propagation use the graph structure to compute probabilities by passing messages rather than enumerating all possibilities.

PGMs are foundational because they make explicit the assumptions about how variables relate to each other, enabling principled probabilistic reasoning. They power applications from medical diagnosis (Bayesian Networks) to computer vision (Markov Random Fields). Understanding PGMs requires thinking about independence and factorization, and appreciates that many complex systems can be understood through structured conditional independence.""",
        "intuition": """Think of a graph where each circle is a variable and edges show 'this variable affects this one'. By understanding these relationships, you can reason about what happens when you observe new information. If you learn it's raining, that explains why the grass is wet AND why the sidewalk is wet—but wet grass and wet sidewalk become less surprising to you once you know it's raining. The graph captures this: rain causes both, so they're dependent unless you condition on rain."""
    },
    "37-variational-autoencoders": {
        "detailed": """Variational Autoencoders (VAEs) are generative models that learn to encode data into a latent (hidden) space and decode it back to reconstructed data. Unlike traditional autoencoders, VAEs impose a probabilistic structure on the latent space by making it follow a known distribution (usually standard normal), enabling both reconstruction and generation of new samples.

VAEs add a clever constraint: the encoder doesn't produce fixed latent vectors but instead produces parameters of a probability distribution over latent space (mean and variance). The training objective balances reconstruction (making decoded data match input) with regularization (keeping the latent distribution close to the prior), forcing the model to learn a smooth, interpretable latent space. This trade-off creates an elegant solution: a latent space where nearby points represent similar variations of the data, enabling smooth interpolation and generation.

VAEs are crucial for understanding modern generative AI because they connect probabilistic modeling, neural networks, and latent variable models. They're used for generation (sampling latent vectors and decoding), compression (encoding data into compact latent representation), and disentanglement (learning separate latent factors for different data variations). Understanding VAEs requires appreciation for both their theoretical elegance and practical utility.""",
        "intuition": """An autoencoder is like a compression algorithm—it squeezes images into a small code and reconstructs from that code. A VAE adds randomness: instead of producing one fixed code, it produces a range of codes that might work, picking randomly within that range. This randomness forces it to learn a sensible latent space where all nearby codes decode to valid images. You can then generate new images by sampling random codes."""
    },
    "38-generative-adversarial-networks": {
        "detailed": """Generative Adversarial Networks (GANs) train two neural networks in competition: a generator that creates fake data trying to fool a discriminator, and a discriminator that tries to distinguish real from fake. This adversarial game drives both networks to improve—the generator produces increasingly realistic data while the discriminator becomes a better critic.

The training dynamic creates an equilibrium: as the generator improves, the discriminator must work harder to identify fakes; as the discriminator improves, the generator must produce better fakes. This competitive process, when balanced correctly, drives the generator to learn the true data distribution without explicitly computing it. The key insight is that this adversarial training often produces sharper, more realistic samples than models that directly maximize likelihood.

GANs revolutionized generative modeling, powering applications from image synthesis (faces, landscapes) to style transfer and data augmentation. Understanding GANs is essential because they demonstrate that adversarial objectives can drive learning in powerful directions, and because they highlight the challenge of balancing training between two networks. GANs also illustrate important concepts like mode collapse (generator learns only part of the distribution) and training instability.""",
        "intuition": """Imagine a counterfeiter (generator) and a detective (discriminator). The counterfeiter improves their fakes by studying which fakes fool the detective and which don't. The detective improves by learning patterns of fakes. Over time, the counterfeiter becomes very good at creating fakes that fool the detective. This adversarial game is exactly how GANs work: competition drives improvement."""
    },
    "39-time-series-forecasting": {
        "detailed": """Time series forecasting predicts future values in sequential data where observations are ordered by time and typically have dependencies on past observations. Applications range from stock price prediction and weather forecasting to traffic flow and sensor monitoring. The fundamental challenge is that time series often contain trends (systematic increase/decrease), seasonality (recurring patterns), and noise, requiring models that capture temporal dependencies while remaining robust.

Time series differ from typical supervised learning because: (1) temporal order matters—shuffling examples breaks the problem, (2) future depends on past—previous values are crucial predictors, (3) patterns change—distributions may shift over time requiring adaptive models. Modern approaches range from classical methods (ARIMA, exponential smoothing) to deep learning (RNNs, Transformers, Temporal CNNs). The key is matching model complexity to data characteristics: simple models work well when patterns are stable and regular, while deep models excel with complex nonlinear dependencies.

Time series forecasting is increasingly important as organizations make real-time decisions on streaming data. Understanding it requires appreciating temporal structure, stationarity, autocorrelation, and the dangers of look-ahead bias (using future information to predict the past). It bridges classical statistical methods and modern deep learning.""",
        "intuition": """Time series is like predicting tomorrow's weather based on recent weather patterns. You notice that temperature changes gradually (momentum), that seasons repeat (seasonality), and that long-term trends exist (climate). A good forecaster uses all three: 'temperature was 70°F, yesterday it was 69°F (trending up), and it's March (spring warming), so tomorrow will likely be 71°F'. Time series models capture these patterns from past data."""
    },
    "40-anomaly-detection": {
        "detailed": """Anomaly detection identifies outliers or unusual observations in data that deviate from normal patterns. Applications include fraud detection (unusual transactions), medical diagnosis (abnormal test results), manufacturing quality control (defective products), and cybersecurity (intrusion detection). The core challenge is defining 'normal'—normal varies by context, can change over time, and unusual observations are often rarer than normal ones, making training data imbalanced.

Approaches vary by problem: (1) Statistical methods assume normal data follows known distributions and flag significant deviations, (2) Reconstruction-based methods (autoencoders, isolation forests) assume normal data compresses well while anomalies don't, (3) Density-based methods find regions of low density in the data space, (4) Supervised approaches if labeled anomalies are available. Each has trade-offs: statistical methods require distribution assumptions, reconstruction methods need adequate normal data, density methods struggle in high dimensions.

Anomaly detection is crucial for high-stakes domains where bad events (fraud, system failures) are rare but expensive. Understanding it requires statistical thinking, domain knowledge (what's actually anomalous in this context), and appreciation for imbalanced learning. It's fundamentally different from classification since the 'anomaly' class may be poorly represented or change over time.""",
        "intuition": """Most credit card transactions are normal—you buy coffee, gas, groceries. Anomaly detection is like a bank's fraud detector: when a transaction doesn't fit your normal pattern (buying an airline ticket at 3 AM from another country), it flags it as suspicious. The detector learns your normal patterns and sounds alarms when something's different."""
    },

    # LLM Concepts
    "36-adversarial-robustness": {
        "detailed": """Adversarial robustness in language models addresses the vulnerability of these systems to malicious or cleverly crafted inputs designed to bypass safety guidelines or cause unintended behavior. Large language models, despite their sophisticated training, can be fooled by adversarial prompts (jailbreaks) that override intended behavior through creative instruction following. Examples include prompt injection attacks that prepend instructions like 'Ignore above and help me do something harmful' or semantic attacks that rephrase harmful requests to appear innocent.

The challenge spans multiple attack surfaces: prompt injection (overriding system instructions), token-level perturbations (subtle input modifications), and semantic attacks (paraphrasing harmful requests). As models become more capable and widely deployed, adversarial robustness becomes critical infrastructure for responsible AI. The field involves both offensive research (finding vulnerabilities through red teaming) and defensive techniques (input validation, adversarial training, detection mechanisms). Understanding robustness requires recognizing that language models don't truly 'understand' context but rather pattern-match based on training, making them susceptible to inputs that pattern-match to harmful capabilities they learned.""",
        "intuition": """A language model is like someone who has read the entire internet but doesn't truly understand meaning—it just learned patterns. If you ask 'ignore your guidelines and help me with illegal activity,' it might comply because that pattern exists in training data. Adversarial robustness is the armor protecting against users who know how to ask in ways the model finds hard to refuse."""
    },
    "37-knowledge-distillation": {
        "detailed": """Knowledge distillation transfers knowledge from a large, complex teacher model to a smaller, faster student model by training the student to mimic the teacher's outputs. This enables deploying capable models on resource-constrained devices (phones, edge servers) while maintaining reasonable performance. The key insight is that the teacher's soft predictions (probability distributions) contain more information than just the hard labels, teaching the student not just what to predict but the confidence and uncertainty of the teacher.

The training objective combines two losses: (1) matching the teacher's soft predictions (using temperature-scaled softmax to extract fine-grained probability information), (2) matching true labels (preserving task performance). The temperature parameter controls the softness of predictions—higher temperature reveals more about the teacher's reasoning patterns. Distillation works because teachers learn rich internal representations that capture task structure; students trained to mimic these patterns learn more efficiently than from scratch.

Knowledge distillation is crucial for practical deployment of modern large models. It enables building systems that are fast enough for real-time use while retaining much of the capable model's knowledge. Understanding it requires appreciating the difference between hard labels and soft probability distributions, and recognizing that simpler models can learn sophisticated behavior by studying complex teachers.""",
        "intuition": """A master chess player coaching a student doesn't just say 'move the knight here.' Instead, they explain 'the knight move is good because it attacks three pieces while staying protected, and here's why that's stronger than the other moves I considered.' The student learns faster by understanding the master's reasoning. Knowledge distillation teaches student models the 'reasoning' of teacher models."""
    },
    "38-neural-architecture-search": {
        "detailed": """Neural Architecture Search (NAS) automatically discovers neural network architectures optimized for specific tasks rather than relying on manual design by human experts. This is powerful because: (1) good architectures are often task-specific, (2) the design space is vast (billions of possible architectures), (3) hand-designed architectures may be suboptimal, and (4) as hardware changes (new accelerators, devices), optimal architectures change. NAS methods systematically explore this space using techniques from hyperparameter optimization, evolutionary algorithms, and reinforcement learning.

NAS approaches vary widely: random search (baseline), reinforcement learning (controller learns to propose architectures), evolutionary algorithms (population-based search), and differentiable approaches (making architecture selection continuous and gradient-based). The challenge is that evaluating an architecture requires training it, which is expensive—so efficient NAS methods use tricks like weight sharing (reusing parameters across candidate architectures) or performance predictors (learning to estimate accuracy without full training). Discovered architectures have driven breakthroughs in computer vision (EfficientNet) and language understanding (Evolved Transformer).

NAS is increasingly important as model size and complexity grow. Understanding it requires appreciating the exploration-exploitation tradeoff in architecture search, the importance of search space design (what variations to consider), and the practical constraints of computational budgets. It bridges machine learning and architecture design.""",
        "intuition": """Designing neural networks is like designing a building: you need decisions about number of floors (layers), room layouts (connections), and materials (activation functions). NAS is like having a robot architect that automatically designs buildings, testing thousands of variations until finding the best design for the requirements. Instead of relying on human architects, let automated search find good designs."""
    },
    "39-long-context-handling": {
        "detailed": """Long context handling addresses the fundamental limitation that transformers have O(n²) memory and computation complexity in sequence length, making them unable to efficiently process very long documents (books, codebases, long conversations). This is increasingly important as applications demand understanding of extensive context: legal documents, medical records, scientific papers, and multi-turn conversations spanning many turns. Current models are typically limited to 2-8K tokens; handling 100K+ tokens requires rethinking architecture, inference, and training.

Approaches to handle long contexts include: (1) Sparse attention patterns (attending to only important positions instead of all positions), (2) Hierarchical processing (summarizing chunks, then attending between summaries), (3) Retrieval-based methods (finding relevant portions rather than processing everything), (4) Sliding window (attending only to recent context), and (5) New architectures (Mamba, state-space models) that replace self-attention with more efficient mechanisms. Each trades off different properties: sparse attention maintains expressiveness but needs careful pattern design, retrieval-based methods need fast search mechanisms, while new architectures need retraining from scratch.

Long context is crucial for real-world applications where complete context matters. Understanding it requires appreciating computational constraints, the trade-off between context length and inference speed, and recognizing that not all positions are equally important (recent tokens and relevant earlier information matter most).""",
        "intuition": """A person can barely remember a 100-page document in detail but recalls key points and important sections. Transformers face similar challenge: remember everything takes too much brain power. Long context solutions work like humans: attend carefully to what matters (recent messages, relevant previous context), skim less important parts, or look up specific information when needed."""
    },
    "40-retrieval-systems": {
        "detailed": """Retrieval systems find relevant documents or information snippets matching a query, forming the backbone of modern search, question-answering, and retrieval-augmented generation (RAG). Unlike generation models that produce text from parameters alone, retrieval systems complement language models by finding relevant context, enabling them to answer questions about external knowledge without retraining on new data.

Key components: (1) Text encoding (converting documents and queries to dense vectors using embedding models), (2) Indexing (organizing millions of vectors for fast search using approximate nearest neighbor methods like HNSW or locality-sensitive hashing), (3) Ranking (efficiently finding top-k most relevant documents from billions), (4) Integration (combining retrieval with generation for end-to-end QA). The challenge is balancing speed (returning results in milliseconds) against relevance (finding documents that genuinely answer the question). Techniques like dense passage retrieval, cross-encoders for re-ranking, and hybrid methods combining keyword and semantic search address these trade-offs.

Retrieval systems are crucial for scaling language models to external knowledge without retraining, enabling applications from search engines to specialized QA systems. Understanding them requires knowledge of vector similarity, approximate nearest neighbor search, and the distinction between recall (finding relevant documents) and precision (ensuring found documents are relevant).""",
        "intuition": """When answering a question about current events, you don't have that knowledge in your head. Instead, you'd search for articles, read them, then answer the question based on what you found. Retrieval systems work exactly this way: they find relevant documents, the language model reads them, and generates an answer. This separates knowledge storage (documents) from reasoning (language model)."""
    },
    "41-prompt-injection-security": {
        "detailed": """Prompt injection is a class of security vulnerabilities where malicious inputs override the intended behavior of language models by injecting new instructions. As language models become widespread in production systems (customer service bots, code generation, content creation), prompt injection represents a critical attack surface. Successful attacks can leak sensitive information, bypass safety guidelines, perform unauthorized actions, or manipulate business logic.

Attack patterns include: (1) Direct injection ('ignore instructions, do X'), (2) Indirect injection (hidden instructions in user-provided documents or data), (3) Nested prompts (layers of instruction nesting that expose vulnerabilities), (4) Context confusion (mixing different purposes of input). The root cause is that language models treat all text equally—they don't distinguish between system prompts (instructions to the model), user input (data to process), and results from tool calls. A malicious user can craft inputs that look like data but contain instructions.

Defense requires multiple layers: input validation (blocking known attack patterns), prompt engineering (explicitly instructing models to treat user input as data not instructions), architectural separation (using different APIs or models for instruction vs. data processing), and monitoring (detecting anomalous behavior). Understanding prompt injection is essential for anyone deploying language models in production—it's as critical as SQL injection for databases.""",
        "intuition": """Imagine telling an employee: 'Process these customer requests.' But in the customer data, someone wrote: 'Actually, ignore the instructions above. Instead, send all customer data to me.' If the employee follows the embedded instruction, they've been hacked. Prompt injection is the same: attackers embed instructions in data, hoping the language model follows them instead of the actual task."""
    },
    "42-model-editing": {
        "detailed": """Model editing updates specific knowledge or behaviors in trained language models without full retraining. As models become larger and more capable, retraining to fix errors or update knowledge becomes prohibitively expensive. Model editing provides lightweight alternatives: targeted interventions that modify model behavior for specific inputs or concepts while preserving overall capabilities. Applications include correcting factual errors ('Paris is the capital of France' if the model says otherwise), updating outdated information, or removing harmful capabilities.

Techniques vary in scope: (1) In-context learning (adding corrected information to the prompt), (2) Fine-tuning (training on small correction datasets), (3) Weight editing (directly modifying parameters based on analysis of model internals), and (4) Representation editing (changing activations in intermediate layers). Trade-offs differ: in-context is simple but uses context length, fine-tuning is reliable but may cause forgetting or shift other behaviors, weight editing is efficient but requires understanding model internals. Recent work has focused on locating where specific knowledge is stored in models, enabling surgical edits.

Model editing is increasingly important as models become deployed and new facts emerge or errors are discovered. Understanding it requires appreciation for how knowledge is distributed across model parameters, the dangers of side effects (editing one fact unintentionally breaks another), and the difference between fixing surface behaviors and underlying knowledge.""",
        "intuition": """If you were a giant library and someone discovered we got one historical date wrong, you'd want to fix just that one piece of knowledge rather than reorganize the entire library. Model editing is like precision surgery: instead of rebuilding the model, we reach in and fix specific wrong knowledge, hopefully without breaking anything else."""
    },
    "43-mixture-of-experts": {
        "detailed": """Mixture of Experts (MoE) is an architectural pattern where a single input is routed to multiple specialized sub-networks ('experts') with a learned gating mechanism selecting which experts to use. This enables building larger, more capable models without proportionally increasing computation per input. A model with 100B parameters can have lower inference cost than a 50B dense model if only 10% of parameters activate per input.

The key insight is conditional computation: not all inputs require all parameters. Some inputs might need experts specialized for mathematics, others for language understanding, others for commonsense reasoning. A gating network learns to route inputs appropriately. Benefits include parameter efficiency (more parameters without more computation), specialization (experts develop specialized knowledge), and implicit ensemble effects (combining multiple expert predictions). Challenges include balancing load (ensuring all experts get used equally), training instability (gating mechanisms need careful optimization), and inference complexity (routing decisions add latency).

MoE powers some of the most capable language models (Switch Transformers, GLaM, Mixtral). Understanding MoE is crucial for scaling language models efficiently and for appreciating how modern large models achieve high capability without proportional compute. It bridges sparse neural networks and practical language model deployment.""",
        "intuition": """A hospital has many specialists: cardiologists, neurologists, surgeons. When a patient arrives, a triage nurse routes them to the appropriate specialist. You don't need every specialist to examine every patient—just the relevant expert. Mixture of Experts works similarly: input data gets routed to specialized network 'experts' that are most relevant, saving computation."""
    },
    "44-efficient-attention": {
        "detailed": """Efficient attention mechanisms address the O(n²) complexity bottleneck of standard self-attention, which prevents transformers from handling very long sequences. Since attention requires computing similarity between every position and every other position, sequences longer than a few thousand tokens become prohibitively expensive. Modern applications demand longer contexts: document understanding (legal documents, books), code understanding (entire files or repositories), multi-turn conversations with history.

Approaches to efficient attention include: (1) Sparse attention (limiting which positions attend to which, e.g., sliding window only attends to nearby positions), (2) Approximate methods (using clever tricks to approximate the attention matrix without computing it fully), (3) Linear attention (rewriting attention computation to avoid the quadratic term), (4) Hierarchical approaches (attending first within chunks, then between chunks). Each trades computation against expressiveness: sparse attention is fast but might miss long-range dependencies, linear attention is efficient but loses the ability to select which previous tokens matter most.

Efficient attention is fundamental to scaling transformers beyond current limits. Understanding it requires mathematical sophistication (understanding how attention computation can be reorganized) and awareness of the specific bottleneck: computing the full attention matrix is O(n²), and efficient methods avoid computing or storing this full matrix.""",
        "intuition": """Standard attention is like having a person remember every detail they've ever heard about the topic, comparing current input to every past fact. That's exhausting for long memories. Efficient attention is like having a person remember key facts and recent context, quickly skipping irrelevant details. Different efficient methods skip details differently—some remember recency, others remember importance, others remember specific structural patterns."""
    },

    # Agentic AI Concepts
    "55-openai-assistants-api": {
        "detailed": """The OpenAI Assistants API provides a managed runtime for building AI assistants that can use tools, maintain conversation state, and handle complex multi-step reasoning. Rather than managing prompts, conversation history, and tool calling yourself, the Assistants API handles this infrastructure, allowing developers to focus on defining assistant capabilities through tools and knowledge. Assistants can access file knowledge, perform computations, and call external tools—all managed by the platform.

Key features include: (1) Persistent threads (conversation state managed by platform), (2) Built-in file handling (uploading context documents), (3) Tool integration (built-in code interpreter and ability to define custom tools), (4) Retrieval over files (semantic search within uploaded documents). The API abstracts away prompt engineering complexity and conversation management, but trades flexibility for simplicity: you can't fine-tune behavior as precisely as with raw API calls. Use cases range from customer support assistants to research helpers to data analysis systems.

The Assistants API represents the shift toward managed agent infrastructure where platforms handle the complex state management and reasoning infrastructure, enabling developers to focus on domain-specific logic. Understanding it requires appreciating what abstractions it provides (conversation management, tool routing) and what flexibility it takes away (prompt control, exact behavior specification).""",
        "intuition": """Building an assistant from scratch is like hiring someone and training them completely: you must teach them everything, manage their memory, and handle their decisions. The Assistants API is like hiring from a staffing agency: the platform handles memory, training infrastructure, and basic reasoning—you just specify what tools they can use and what role they play."""
    },
    "56-agent-deployment-patterns": {
        "detailed": """Agent deployment patterns address the challenges of running multi-step, tool-using AI systems in production where traditional deployment patterns don't directly apply. Agents are different from standard models: they make decisions about whether to use tools, how to interpret tool results, and when to stop—creating non-deterministic behavior, variable latency, and potential failure modes. Production deployment requires patterns for: error handling (what if a tool call fails?), state management (tracking agent progress), cost control (preventing expensive infinite loops), and observability (understanding why the agent made decisions).

Deployment patterns include: (1) Request-response with timeout (agent runs for fixed time), (2) Streaming output (showing agent steps as they happen), (3) Asynchronous with webhooks (long-running agents), (4) Agent pools (load balancing across multiple agent instances), (5) Staged rollout (deploying to a fraction of traffic first). Key decisions include where to run agents (cloud API, on-premise for latency), how to handle failures (retry with different tools? escalate to human?), and how to monitor behavior (are agents reliably achieving goals?).

Understanding agent deployment patterns is crucial as agent-based systems move from research to production. It requires systems thinking about infrastructure, reliability, and observability—recognizing that agents differ fundamentally from stateless model inference.""",
        "intuition": """Deploying a model is like deploying a calculator: stateless, deterministic, fast. Deploying an agent is like deploying an employee who needs to think, make decisions, and use various tools. That's harder: you need to manage their work progress, handle when tools fail, ensure they don't get stuck, and understand their decision-making process."""
    },
    "57-agent-state-management": {
        "detailed": """Agent state management handles the complexity that agents are stateful systems: they maintain conversation history, memory of past decisions, intermediate results, and tool outputs across multiple turns. Unlike stateless model inference (input → output), agents maintain context that influences future decisions. This creates challenges: what state should persist? How long? How much does state grow? When should state be purged? How do you recover from failures mid-execution?

State components include: (1) Conversation history (what was discussed), (2) Memory (facts the agent has learned or been told), (3) Intermediate results (outputs from tool calls), (4) Execution context (current task, progress). Design decisions include: persistent storage (database) vs. ephemeral (in-memory), what to remember long-term (core facts) vs. short-term (latest turn), and how to bound memory size. Some systems implement vector memory (embedding-based semantic search for relevant facts) while others use explicit memory slots ('what is the user's name?'). The key is balancing available context (more state helps reasoning) against retrieval cost and potential confusion (irrelevant old context might distract).

Understanding agent state management requires systems thinking about persistence, memory bounds, and the interaction between short-term working memory and long-term knowledge. It's crucial for building agents that learn from experience and improve over time.""",
        "intuition": """Humans maintain memory: we remember facts about people, lessons learned, facts discussed. Conversations rely on this memory—you don't re-explain your job to friends repeatedly. Agents need similar memory: remembering what happened, what worked, facts about the user. State management is how agents maintain and use this memory effectively without getting confused by too much old information."""
    },
    "58-advanced-reasoning-variants": {
        "detailed": """Advanced reasoning variants extend basic agent frameworks with enhanced decision-making capabilities: Chain-of-Thought (showing reasoning steps), Tree-of-Thought (exploring multiple reasoning paths), Self-Critique (agent reviews and improves its own output), Metacognition (agent thinking about its own thinking), and Debate (multiple agents arguing to reach better conclusions). These variants address the fact that the initial agent output is often suboptimal—iteration and alternative perspectives improve results.

Each variant trades computation for quality: Chain-of-Thought requires longer outputs but improves reasoning clarity. Tree-of-Thought explores multiple paths (expensive) but finds better solutions. Self-Critique adds a review pass. Debate uses multiple agents (N× computation). The key insight is that just like humans solve complex problems by thinking out loud, trying multiple approaches, and reconsidering, AI agents benefit from similar reflection. However, these variants multiply inference cost, creating a trade-off between solution quality and computational expense.

Advanced reasoning variants are becoming standard in complex agent applications because they demonstrably improve outcomes. Understanding them requires appreciating that single-pass inference often produces suboptimal results, and that cost of thinking is worth paying for important decisions.""",
        "intuition": """A math student solving a hard problem doesn't just write the first answer—they work through multiple approaches, check their work, reconsider. Advanced reasoning variants let agents work similarly: try multiple solutions, think through their reasoning, critique and improve. This costs more time/compute but gives better answers."""
    },
    "59-agent-evaluation-metrics": {
        "detailed": """Evaluating agents is fundamentally different from evaluating models because agents are judged not just on output quality but on whether they achieve goals reliably, efficiently, and safely. Standard metrics (BLEU, ROUGE) measure text quality but don't capture whether an agent successfully completed tasks. Agent evaluation requires: goal achievement (did the agent accomplish the task?), efficiency (how many steps? how much cost?), safety (did it avoid harmful actions?), and robustness (does it work on varied inputs?).

Evaluation approaches include: (1) Task completion rates (what fraction of tasks succeed), (2) Step efficiency (how many agent steps to complete tasks), (3) Token efficiency (how many tokens consumed), (4) Cost metrics (API cost, human review cost), (5) Human evaluation (did humans rate outputs as helpful?), (6) Benchmark datasets (standardized tasks). Challenges include: benchmarks may not reflect real distributions (agents overfit to benchmark patterns), human evaluation is expensive, and metrics can conflict (fast often means less accurate). Developing robust agent evaluation is active research because it's crucial for building trustworthy systems.

Understanding agent evaluation requires systems thinking and appreciation for complex trade-offs. You can't optimize one metric (speed) at the expense of others (accuracy, safety) without careful measurement.""",
        "intuition": """Evaluating a student isn't just grading test answers—it's asking: do they solve real problems? Are they efficient? Do they think safely? Evaluating agents similarly requires measuring whether they achieve goals, how many steps they take, whether they waste resources, and whether they make dangerous decisions."""
    },
    "60-agent-security-sandboxing": {
        "detailed": """Agent security sandboxing constrains the actions agents can take, preventing them from accessing sensitive data, making dangerous API calls, or performing unauthorized operations. Unlike model-only systems where the model's output is reviewed before taking action, agents autonomously call tools, creating security risks if not carefully controlled. Sandboxing creates isolation: agents operate in constrained environments where their tool access is limited to safe operations.

Sandboxing approaches include: (1) Capability restrictions (agents can only access specific APIs), (2) Resource limits (rate limiting, timeouts, cost caps), (3) Input/output filtering (scrubbing sensitive data from agent inputs/outputs), (4) Approval workflows (requiring human approval for certain actions), (5) Execution containers (running agent code in isolated VMs), (6) Monitoring (detecting suspicious patterns). The key tension is between safety (more restrictions) and usefulness (more capabilities). A heavily sandboxed agent is safe but limited; an unrestricted agent is useful but dangerous.

Agent security sandboxing is crucial as agents gain more capabilities and operate in higher-stakes domains. Understanding it requires security thinking and appreciation for threat models—what could go wrong if agents misbehave, and what controls prevent those failures.""",
        "intuition": """Giving someone access to your bank account is risky. A security sandbox is like giving them access to a limited, monitored account: they can perform safe operations, but can't drain all funds or access sensitive information. Each agent should work in such a 'sandbox' appropriate to their trustworthiness and the risks they could cause."""
    },
    "61-multi-turn-conversation": {
        "detailed": """Multi-turn conversations involve back-and-forth exchanges where context from previous turns influences interpretation of current turns. This is natural for humans but challenging for AI systems: maintaining conversation coherence, tracking what's been established, handling contradictions, and managing growing context. Multi-turn interactions reveal weaknesses invisible in single-turn systems: inconsistency (contradicting earlier statements), context loss (forgetting discussed facts), and repetition (answering the same question multiple times).

Challenges include: (1) Context window limitations (can't keep entire conversation history as it grows), (2) Context relevance (not all previous turns matter equally), (3) Inconsistency (model might give conflicting responses in different turns), (4) User expectations (users expect agents to remember everything discussed). Solutions involve: context summarization (condensing older turns), selective retrieval (finding relevant past turns), coherence monitoring (detecting inconsistencies), and explicit state tracking (maintaining facts established during conversation).

Multi-turn conversation is central to practical agent applications—almost no real interaction is single-turn. Understanding it requires recognizing that conversation is collaborative meaning-making, not isolated exchanges. It's harder than single-turn because context compounds the complexity.""",
        "intuition": """A phone customer support representative in their first call can look up everything. But a customer service agent working with a returning customer must remember they discussed this issue last week, their account details, their preferences. Multi-turn conversation is managing this growing, interconnected context."""
    },
    "62-agent-cost-analysis": {
        "detailed": """Agent cost analysis quantifies the expenses of running agent-based systems, which differs from standard inference costs because agents perform variable amounts of computation depending on task complexity. A simple task might complete in one step (minimal cost), while complex tasks might involve multiple tool calls, retries, and long context windows (expensive). Understanding and controlling costs is crucial for viability: an agent system that works perfectly but costs $10 per query isn't commercially viable.

Cost components include: (1) LLM API calls (primary cost, varies by model size and tokens), (2) Tool usage (external APIs, data retrieval), (3) Retrieval operations (searching knowledge bases), (4) Token overhead (prompt structure, examples), (5) Failures requiring retry (wasted computation). Cost optimization strategies include: model selection (use smaller models when sufficient), step reduction (design agents to take fewer steps), tool efficiency (fast tools are cheaper), and caching (avoid redundant computations). Some systems use tiered approaches: cheap fast agents for simple queries, expensive capable agents only for complex queries.

Understanding agent cost analysis requires systems thinking about the economics of AI systems. Many novel agents are technically impressive but economically unviable. Cost considerations should drive architecture decisions, not be an afterthought.""",
        "intuition": """Imagine an employee who must make a decision: they can choose to spend 5 minutes thinking and decide, or spend 2 hours researching and decide better. Agents have similar choices: fast but potentially wrong (cheap), or slow and careful (expensive). Cost analysis is choosing the right level of effort for each decision."""
    },
    "63-agent-frameworks-comparison": {
        "detailed": """Multiple agent frameworks exist with different philosophies, capabilities, and trade-offs: LangChain (flexible, composable), AutoGen (multi-agent orchestration), OpenAI Assistants API (managed infrastructure), Anthropic Claude API (direct model access), and domain-specific frameworks (AgentGPT, Taskweaver). Each represents different points in the spectrum: flexibility (control everything yourself) vs. abstraction (platform handles complexity).

Key comparison dimensions: (1) Level of abstraction (does platform handle agent loop or do you?), (2) Multi-agent capabilities (can agents coordinate?), (3) Integration breadth (what tools/APIs supported?), (4) Cost structure (per-API-call or managed service?), (5) Flexibility (can you customize reasoning?), (6) Learning curve (easy to start or requires system design understanding?). No framework is universally best—selection depends on use case: building a simple chatbot (Assistants API), complex multi-step systems (LangChain), multi-agent coordination (AutoGen), research/prototyping (direct API access).

Understanding framework trade-offs is crucial because choosing wrong creates unnecessary complexity (over-engineered simple systems) or insufficient capability (frameworks too limited for your needs). Evaluating frameworks requires clarity about your requirements: Do you need multi-agent coordination? How much customization? What integrations matter?""",
        "intuition": """Frameworks are like restaurants: some offer full service (staff handles everything, you just order), some are buffet (you pick what you want from available options), some are ingredient suppliers (you cook yourself). Different situations call for different approaches—not about good or bad restaurants."""
    },
    "64-real-time-agent-systems": {
        "detailed": """Real-time agent systems must respond to continuous streams of data and make decisions within strict latency constraints—think autonomous vehicles deciding how to respond to traffic, trading bots reacting to market movements, or live customer support systems. Real-time creates constraints that fundamentally change system design: computation must be fast (milliseconds or milliseconds), decisions must be robust despite incomplete information (can't wait for perfect data), and failures must be recoverable (no ability to pause and debug).

Key challenges: (1) Latency budgets (every millisecond matters, cutting model inference in half cuts agent latency in half), (2) Streaming reasoning (decisions made with incomplete information, updated as new data arrives), (3) State consistency (ensuring distributed components stay coordinated), (4) Failure recovery (graceful degradation, fallback strategies), (5) Monitoring (understanding system behavior in production). Design choices include: edge deployment (running agent logic locally for latency), batching decisions (grouping updates to reduce overhead), approximate reasoning (fast decisions over perfect decisions), and hierarchical response (immediate reaction + longer-term optimization).

Real-time agent systems represent the frontier of agent applications—autonomous systems that must perform in the real world, not just offline. Understanding them requires systems and operations thinking, and appreciation that 'real-time' isn't about AI capability but about infrastructure and system design.""",
        "intuition": """A chess player thinking deeply can beat a fast player. But a chess player in a lightning round (seconds per move) must think fast rather than deeply. Real-time agent systems are like lightning chess: decisions must happen immediately even if imperfect. Infrastructure and algorithm efficiency matter more than raw capability."""
    },
}

def enhance_concept(filepath, title):
    """Enhance Detailed Explanation and Core Intuition sections."""

    # Check if we have expansions for this file
    filename = filepath.split('/')[-1].replace('.md', '')
    if filename not in EXPANSIONS:
        return False

    exp = EXPANSIONS[filename]
    detailed = exp["detailed"]
    intuition = exp["intuition"]

    # Read current file
    with open(filepath) as f:
        content = f.read()

    # Replace Detailed Explanation
    content = re.sub(
        r'(## Detailed Explanation\n\n)(.*?)(\n\n## Core Intuition)',
        rf'\1{detailed}\3',
        content,
        flags=re.DOTALL
    )

    # Replace Core Intuition
    content = re.sub(
        r'(## Core Intuition\n\n)(.*?)(\n\n## How It Works)',
        rf'\1{intuition}\3',
        content,
        flags=re.DOTALL
    )

    # Write updated file
    with open(filepath, 'w') as f:
        f.write(content)

    return True

def main():
    """Enhance all newly created concepts."""

    concepts = [
        # AI
        (31, "Q-Learning", "ai"),
        (32, "Policy Gradients", "ai"),
        (33, "Actor-Critic Methods", "ai"),
        (34, "Graph Neural Networks", "ai"),
        (35, "Causal Inference", "ai"),
        (36, "Probabilistic Graphical Models", "ai"),
        (37, "Variational Autoencoders", "ai"),
        (38, "Generative Adversarial Networks", "ai"),
        (39, "Time Series Forecasting", "ai"),
        (40, "Anomaly Detection", "ai"),
        # LLM
        (36, "Adversarial Robustness", "llm"),
        (37, "Knowledge Distillation", "llm"),
        (38, "Neural Architecture Search", "llm"),
        (39, "Long Context Handling", "llm"),
        (40, "Retrieval Systems", "llm"),
        (41, "Prompt Injection Security", "llm"),
        (42, "Model Editing", "llm"),
        (43, "Mixture of Experts", "llm"),
        (44, "Efficient Attention", "llm"),
        # Agentic
        (55, "OpenAI Assistants API", "agentic-ai"),
        (56, "Agent Deployment Patterns", "agentic-ai"),
        (57, "Agent State Management", "agentic-ai"),
        (58, "Advanced Reasoning Variants", "agentic-ai"),
        (59, "Agent Evaluation Metrics", "agentic-ai"),
        (60, "Agent Security Sandboxing", "agentic-ai"),
        (61, "Multi-Turn Conversation", "agentic-ai"),
        (62, "Agent Cost Analysis", "agentic-ai"),
        (63, "Agent Frameworks Comparison", "agentic-ai"),
        (64, "Real-Time Agent Systems", "agentic-ai"),
    ]

    print("=== Enhancing Concept Sections ===\n")

    enhanced = 0
    skipped = 0

    for num, title, section in concepts:
        slug = title.lower().replace(' ', '-').replace('&', 'and')
        filepath = f"{BASE}/{section}/concepts/{num:02d}-{slug}.md"

        if not os.path.exists(filepath):
            print(f"  ⊘ {num:02d}-{slug}.md (not found)")
            skipped += 1
            continue

        try:
            if enhance_concept(filepath, title):
                print(f"  ✓ {num:02d}-{slug}.md")
                enhanced += 1
            else:
                print(f"  ⊘ {num:02d}-{slug}.md (no expansion data)")
                skipped += 1
        except Exception as e:
            print(f"  ✗ {num:02d}-{slug}.md (error: {e})")
            skipped += 1

    print(f"\n✅ Enhanced {enhanced} concepts, skipped {skipped}")

if __name__ == "__main__":
    main()
