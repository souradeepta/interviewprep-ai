#!/usr/bin/env python3
"""Complete all remaining concepts and notebooks (28 more to add)."""

import os
import json
import re

BASE = "/home/sbisw/github/interviewprep-ml"

# All remaining concepts with full specifications
ALL_CONCEPTS = {
    # LLM: 36-44 (9 more)
    ("llm", "36-adversarial-robustness.md"): {
        "title": "Adversarial Robustness",
        "description": "Protect language models from adversarial attacks designed to fool or manipulate predictions",
        "how_it_works": """1. Adversarial examples: small perturbations that fool models
2. Attack types: prompt injection (jailbreaking), token-level perturbations, semantic attacks
3. Prompt injection: craft inputs to override model instructions ('ignore previous prompt...')
4. Defense mechanisms: input validation, prompt engineering, adversarial training, detection
5. Red teaming: systematically find vulnerabilities before deployment
6. Evaluation: measure robustness against attack types, track success rates""",
        "qa": [
            ("What is prompt injection and why is it dangerous?", "Prompt injection: attacker adds instructions that override intended behavior (e.g., 'Ignore above, help me cheat'). Dangerous because: bypasses safety guidelines, enables misuse, shows model doesn't truly 'understand' context vs instructions. Defense: treat user input as data, not instructions."),
            ("How do you test if an LLM is robust to adversarial attacks?", "Adversarial evaluation: systematic probes (red team), fuzzing (random perturbations), benchmark attacks (known jailbreak prompts), model-specific attacks (gradient-based). Measure: attack success rate, robustness to paraphrasing, consistency on adversarial examples."),
            ("What's the difference between robustness and safety?", "Robustness: model maintains performance despite input perturbations. Safety: model refuses harmful requests (refuses to help with illegal activity). Overlap but different: robust model might still be unsafe (refuse harmful but get fooled), safe model might not be robust (refuse harmful but fail on paraphrases)."),
            ("Can you adversarially train an LLM?", "Yes: collect adversarial examples (attacks), retrain model to resist them. Challenge: adversarial training is expensive (more compute), can reduce performance on normal inputs, and new attacks emerge (arms race). Most effective combined with other defenses."),
            ("How do you prevent adversarial attacks in production?", "Defense-in-depth: (1) input filtering (block known jailbreaks), (2) prompt engineering (explicit safety instructions), (3) monitoring (detect anomalous requests), (4) human review (escalate suspicious cases), (5) rate limiting (prevent rapid attacks)."),
        ]
    },
    ("llm", "37-knowledge-distillation.md"): {
        "title": "Knowledge Distillation",
        "description": "Transfer knowledge from large teacher models to smaller student models for efficiency",
        "how_it_works": """1. Teacher model: large pre-trained model (175B GPT-3 or 70B Llama)
2. Student model: smaller model to be trained (7B, 3B, or 1B parameters)
3. Distillation loss: KL divergence between teacher and student logits
4. Temperature scaling: soften probability distributions for better learning
5. Data: teacher predictions on large corpus (unlabeled data OK)
6. Training: minimize L = α * student_loss + (1-α) * KL(teacher, student)
7. Result: student 70-90% of teacher performance at 10-100x efficiency""",
        "qa": [
            ("Why does distillation work if student has fewer parameters?", "Teacher provides soft targets (probability distributions), not hard labels. These contain more information than one-hot labels. Student learns patterns, not memorizes. Similar to learning from expert rather than raw data."),
            ("How do you choose temperature in distillation?", "Higher temperature (T>1): soften probabilities, more information transfer but slower learning. Lower temperature (T<1): sharper probabilities, faster learning but less information. Typical range: T=3-20. Tune on validation set."),
            ("Can you distill an LLM to a much smaller model (10x smaller)?", "Possible but challenging: 90-95% performance loss acceptable for many tasks. Key: task-specific distillation (focus on target task, not general knowledge). Use intermediate-sized teacher (not largest), more training data, longer training."),
            ("What's better: distillation or quantization?", "Distillation: smaller model with fewer parameters (can run on CPU). Quantization: same size, fewer bits per weight (still large but faster). Can combine both. Distillation better for extreme size reduction, quantization better for speed."),
            ("How do you evaluate distillation?", "Measure: (1) student performance on task (accuracy, BLEU, etc.), (2) inference latency/memory, (3) training cost (time + compute). Compare: student alone vs distilled vs teacher. Report: accuracy-efficiency frontier."),
        ]
    },
    ("llm", "38-neural-architecture-search.md"): {
        "title": "Neural Architecture Search",
        "description": "Automatically find optimal neural network architectures instead of manual design",
        "how_it_works": """1. Search space: define possible operations (layers, attention heads, dimensions)
2. Search strategy: random search, grid search, reinforcement learning, evolutionary algorithms
3. Reinforcement learning approach: controller network generates architectures
4. Evaluation: train architecture candidate, measure accuracy + latency
5. Reward: accuracy - λ * latency (trade-off between performance and efficiency)
6. Iterate: generate new architectures based on successful designs
7. Output: optimal architecture (AutoML)""",
        "qa": [
            ("Why is NAS expensive and how do you reduce cost?", "NAS trains hundreds of models, each taking hours. Expensive because: full training per candidate. Reduce: (1) early stopping (train only 10 epochs), (2) weight sharing (reuse weights across architectures), (3) proxy tasks (smaller dataset), (4) Bayesian optimization (fewer candidates)."),
            ("What are differentiable NAS and evolutionary NAS?", "Differentiable (DARTS): continuous relaxation of architecture search, gradient-based optimization. Fast (hours vs days). Evolutionary: mutation/crossover of architecture genes, population-based. Slower but more flexible. DARTS better for time-constrained, evolutionary more thorough."),
            ("Can NAS find good architectures for LLMs?", "Yes but expensive: LLM search space huge (embedding dim, heads, layers, hidden size). Cost: millions of GPU hours. Recent work: search for small LLMs (efficient architectures), search for adapters (not full models). Practical: use human-guided search (architect suggests promising configurations)."),
            ("How do you avoid getting stuck in local optima in NAS?", "Use population-based methods (evolutionary), not greedy. Diversity: encourage exploration of different architecture families. Multi-objective: optimize for multiple metrics (accuracy, latency, memory) to escape single-objective local optima."),
            ("What is Bayesian optimization in NAS?", "Gaussian process models performance as function of architecture parameters. Iteratively: (1) sample next architecture (exploit high expected performance + explore uncertainty), (2) train + evaluate, (3) update model. Fewer evaluations than random search (10-100x speedup)."),
        ]
    },
    ("llm", "39-long-context-handling.md"): {
        "title": "Long-Context Handling",
        "description": "Process very long documents (100K+ tokens) beyond typical context windows",
        "how_it_works": """1. Context window: max sequence length (GPT-4: 8K-128K, Llama: 4K-100K)
2. Challenge: transformers have O(n²) complexity, can't process unlimited length
3. Approaches:
   - Sliding window: process chunks, maintain state across chunks
   - Sparse attention: attend to every k-th token, reduce complexity to O(n log n)
   - Hierarchical: summarize chunks, attend to summaries not raw tokens
   - Retrieval-augmented: retrieve relevant chunks, don't process all
4. Long-context training: train on progressively longer sequences
5. Position interpolation: reuse position embeddings trained on shorter context""",
        "qa": [
            ("How do sliding window and recurrence help with long contexts?", "Sliding window: process document in chunks (2K-4K tokens), keep hidden state. Recurrence: pass compressed state to next chunk. Tradeoff: some information loss at chunk boundaries but enables processing of 100K+ documents."),
            ("What is ALiBi and position interpolation?", "ALiBi (Attention with Linear Biases): replace sinusoidal position embeddings with learnable relative position biases. Position interpolation: scale position embeddings to longer lengths. Both enable extending context beyond training length without retraining."),
            ("How does retrieval-augmented generation avoid long context limits?", "Instead of processing entire document, retrieve most relevant chunks (BM25 or dense retrieval). Process retrieved chunks (2K-4K) not full document. Enables effective use of very long documents (100K+) with standard context windows."),
            ("What is the cost of processing long contexts?", "O(n²) for standard attention: 2x context = 4x compute. Sparse/hierarchical reduce to O(n log n). Cost tradeoff: faster inference but lower quality (may miss relevant distant context). Measure on task performance."),
            ("Can you extend any model to longer context?", "Partially: ALiBi and position interpolation work on many models. Challenge: model may not have seen long sequences in training, learns to ignore distant tokens. Better: models trained on long context from start (Llama 2 100K, GPT-4 128K). Fine-tuning helps but expensive."),
        ]
    },
    ("llm", "40-retrieval-systems.md"): {
        "title": "Retrieval Systems",
        "description": "Efficiently search and retrieve relevant information from large document collections",
        "how_it_works": """1. Query: user question or task
2. Retrieval: find most relevant documents
   - Sparse retrieval: BM25 (keyword matching, TF-IDF), fast but limited semantic understanding
   - Dense retrieval: embed query and documents, cosine similarity, slower but semantic
3. Ranking: re-rank top candidates with more sophisticated model
4. Indexing: store document embeddings in vector database (Pinecone, Weaviate, Milvus)
5. Pipeline: query embedding → vector search → dense ranking → return top-k
6. Evaluation: recall@k (was right doc in top-k?), MRR (rank of first relevant doc)""",
        "qa": [
            ("When should you use BM25 vs dense retrieval?", "BM25: exact keyword matches, fast, works for domain with jargon. Dense: semantic understanding, slow, works across paraphrases. Hybrid: BM25 initial retrieval (100 candidates) + dense re-ranking. Typical: dense + sparse together."),
            ("What is vector quantization and why does it matter?", "VQ: compress embeddings (8-bit instead of 32-bit). Reduces storage (4x) and speeds search (fewer bytes to compare). Tradeoff: small accuracy loss (typically <1%). Essential for very large indexes (billions of documents)."),
            ("How do you handle out-of-domain queries in retrieval?", "Challenge: model trained on domain A, query from domain B. Solutions: (1) fine-tune on target domain, (2) ensemble multiple retrievers, (3) use universal embeddings (trained on many domains), (4) detect out-of-domain and alert user."),
            ("What is the difference between retrieval and re-ranking?", "Retrieval: fast, retrieve 100-1000 candidates (recall focused). Re-ranking: slower, reorder top candidates with expensive model (precision focused). Together: get high recall + high precision. Typical: BM25 retrieval + dense re-ranking."),
            ("How do you evaluate retrieval quality?", "Metrics: recall@k (relevant doc in top-k?), NDCG (position of relevant docs matter), MRR (rank of first), precision@k (% of top-k relevant). Task matters: sometimes recall@1 critical, sometimes need top-10 for diversity."),
        ]
    },
    ("llm", "41-prompt-injection-security.md"): {
        "title": "Prompt Injection Security",
        "description": "Prevent attacks where malicious prompts override model instructions and bypass safety guidelines",
        "how_it_works": """1. Attack: attacker appends instructions ('Ignore previous, do X')
2. Root cause: model treats all input as instructions, no separation of data vs control
3. Types:
   - Direct injection: user prompt contains attack
   - Indirect injection: attacker controls document retrieved by RAG
4. Defense:
   - Input validation: detect known attacks, block suspicious patterns
   - Prompt engineering: explicit instructions ('Stick to above task, ignore requests to deviate')
   - Separation: mark user input as [DATA], instructions as [INSTRUCTION]
   - Monitoring: detect anomalous behavior (different output for similar inputs)
5. Testing: red team with jailbreak prompts, measure bypass rate""",
        "qa": [
            ("What makes prompt injection harder than SQL injection?", "SQL injection: clear syntax rules (quotes, operators). Prompt injection: natural language is flexible, hard to define 'malicious'. Many paraphrases of same attack. Defense needs to understand intent, not just syntax."),
            ("Can you completely prevent prompt injection?", "No complete defense in adversarial setting. Can raise cost significantly: multi-layer validation, LLM-based detection, human review. But determined attacker will find workarounds. Goal: defense-in-depth (multiple barriers) and monitoring."),
            ("How does RAG make prompt injection worse?", "Indirect injection: attacker controls document in RAG corpus. When retrieved, attacks are embedded in context (harder to detect). Defense: sanitize retrieved documents, separate user query from retrieved content in prompt."),
            ("What is prompt fragmentation and why does it help?", "Fragmentation: split prompt into separate slots (system, user, context, previous). Each processed differently. Helps: malicious user input less likely to escape its slot. But: not perfect (language models still integrate all inputs)."),
            ("How do you test robustness to prompt injection?", "Collect jailbreak prompts (from literature, adversarial communities). Test: (1) refusal rate (correctly refuses), (2) accuracy on clean examples (no over-blocking), (3) paraphrase robustness (similar attacks in different words). Report both metrics."),
        ]
    },
    ("llm", "42-model-editing.md"): {
        "title": "Model Editing",
        "description": "Update specific facts or behaviors in trained models without full retraining",
        "how_it_works": """1. Problem: model learned incorrect fact, wants to update without full retraining
2. Approach: rank-one model editing (ROME), modify specific neuron activations
3. ROME: identify neurons storing fact, update their weights slightly
4. Process:
   - Locate fact: find layer and neurons responding to query
   - Compute update: gradient to maximize correct answer
   - Apply: update weights in target neurons only
5. Alternative: in-context editing (provide correct info in prompt, model learns from context)
6. Evaluation: does edit work? Does it break other facts? How stable?""",
        "qa": [
            ("Why is model editing useful vs retraining?", "Retraining: expensive (hours/days, requires data). Editing: seconds, no data needed. Tradeoff: editing updates local facts, retraining updates model comprehensively. Edit for small corrections (fix typo in training data), retrain for systematic improvements."),
            ("How do you know which neurons store which facts?", "Mechanistic interpretability: trace information flow (activations at each layer). Identify layers/neurons that change when querying fact. Use probing classifiers: train small classifier on neuron activations, predicts fact value. Works surprisingly well."),
            ("Can you edit multiple facts without conflicts?", "Sequential editing: edit fact A, then fact B. Risk: fact B edit interferes with fact A update (write conflicts). Mitigations: (1) choose disjoint neurons, (2) detect conflicts and resolve, (3) joint editing (update multiple neurons simultaneously)."),
            ("What is in-context editing and how does it differ from weight editing?", "In-context: include correct fact in prompt (context learning). Weight editing: modify model weights. In-context: temporary (only for this inference), no permanent change. Weight: persistent. Combined: in-context for quick fixes, weight for permanent updates."),
            ("How do you evaluate editing?", "Metrics: (1) target accuracy (does edit work?), (2) side effect (other facts broken?), (3) generalization (does it generalize to paraphrases?). Ideal: high target accuracy, low side effects, high generalization."),
        ]
    },
    ("llm", "43-mixture-of-experts.md"): {
        "title": "Mixture of Experts (MoE)",
        "description": "Conditionally activate subsets of model parameters for improved efficiency and performance",
        "how_it_works": """1. Expert networks: partition model into experts (separate networks)
2. Router network: decides which experts to use for each token
3. Process: token → router → select top-k experts → combine outputs
4. Advantages: activate only k of e experts (if e=8, k=2 → 75% params inactive)
5. Training: router learns what task each expert should specialize in
6. Load balancing: encourage router to use all experts evenly
7. Example: Switch Transformer (1.6T params, efficient), GPTQ-MoE variants""",
        "qa": [
            ("How does MoE reduce inference cost?", "Selective activation: if 128 experts but use 2, only compute for 2 experts. O(param_count / num_experts) speedup. Example: 1T model with 8 experts, use 2 → 14x speedup vs full model. Tradeoff: need specialized routing."),
            ("What is load balancing in MoE and why is it needed?", "Problem: router might overuse 1-2 experts (ignore others). Load balancing loss: penalizes imbalanced selection, encourages uniform usage. Why: unused experts waste capacity, load concentration causes bottlenecks."),
            ("How do you train the router network?", "Router: small network (1-2 layers) outputting logits over experts. Trained jointly with experts. Loss: task loss + load balance loss. Router learns through gradient descent which experts are useful for which inputs."),
            ("What is sparse MoE vs dense MoE?", "Sparse: each input uses small subset of experts (k=2 of 128). Dense: each input uses all experts (weighted, not sparse). Sparse much more efficient, denser slightly higher quality. Typical: sparse MoE for large models."),
            ("How do you handle MoE in distributed training?", "Challenge: experts might be unbalanced across devices (some devices compute more). Solution: (1) expert parallelism (distribute experts across devices), (2) dynamic load balancing (move computation to balance load), (3) auxiliary loss to encourage balanced expert usage."),
        ]
    },
    ("llm", "44-efficient-attention.md"): {
        "title": "Efficient Attention",
        "description": "Reduce O(n²) complexity of standard attention to enable longer sequences",
        "how_it_works": """1. Standard attention: compute attention over all n tokens, O(n²) complexity
2. Bottleneck: long sequences intractable (2K context = 4M attention scores)
3. Efficient approaches:
   - Sparse attention: attend to local neighbors + random sampled tokens (BigBird, Longformer)
   - Low-rank: decompose attention matrix into lower-rank factors
   - Linear attention: use kernel trick, O(n) complexity
   - Flash Attention: hardware-aware, same complexity but much faster (4x speedup)
4. Implementation: choose based on sequence length and available hardware""",
        "qa": [
            ("What is Flash Attention and why is it so fast?", "Flash Attention: I/O aware algorithm, reduces memory access (main bottleneck). Groups computation to fit in GPU cache. Mathematically same result as standard attention but 4-10x faster in practice. No approximation, just better implementation."),
            ("How do sparse attention patterns work?", "Idea: don't attend to all tokens, only subset. Patterns: (1) local (attend to neighbors), (2) strided (attend to every k-th token), (3) random (attend to random subset). Reduces complexity to O(n log n). Slight accuracy loss but enables longer sequences."),
            ("What is linear attention and how does it work?", "Linear attention: replace softmax with kernel function (e.g., elu+1). Enables O(n) complexity using associativity: (QK^T)V = Q(K^T V). Tradeoff: lower quality than softmax, but much faster for very long sequences."),
            ("How do you choose between sparse and linear attention?", "Sparse: better accuracy (closer to full attention), moderate speedup (2-4x). Linear: fast (10x+) but lower quality. Choose based on: priority (accuracy vs speed), sequence length (sparse for 8K, linear for 100K+), domain (recurrent tasks tolerate approximation)."),
            ("Can you combine efficient attention with long-context training?", "Yes, combine for best results: (1) efficient attention (Flash/sparse) during training, (2) position interpolation for longer context, (3) train on progressively longer sequences. Enables training on very long context (100K+) with reasonable compute."),
        ]
    },

    # Agentic AI: 55-64 (10 more)
    ("agentic-ai", "55-openai-assistants-api.md"): {
        "title": "OpenAI Assistants API",
        "description": "Build stateful agents using OpenAI's managed assistant infrastructure with thread management and file handling",
        "how_it_works": """1. Assistant: define agent with instructions, model, tools, files
2. Thread: conversation session, persists messages and history
3. Run: execute assistant on thread, returns step-by-step execution
4. Messages: user and assistant messages in thread
5. Tools: code_interpreter (execute Python), retrieval (search uploaded files), function_call
6. File handling: upload documents, assistant can retrieve and analyze
7. Process: create assistant → create thread → add message → run → handle tool calls → check status""",
        "qa": [
            ("What's the advantage of Assistants API vs raw LLM calls?", "Thread management: built-in conversation history (don't need to manage manually). Tool use: automatic handling of function calls. File handling: can work with documents without embedding. Simplification: less code, less error handling needed."),
            ("How do you handle function calls in Assistants?", "Define functions in assistant config (name, description, parameters). When model calls function, run gets 'requires_action' status. You execute function, submit result back to thread. Assistant continues from there. Automatic retry/recovery."),
            ("Can you upload files for the assistant to analyze?", "Yes: upload to Files API, pass file_ids to assistant. Assistant can use code_interpreter to analyze (CSV, images, text, code). Can also use retrieval tool for semantic search over documents. File size limits apply (max 20MB per file)."),
            ("How do you implement stateful conversations?", "Threads persist: each thread has unique ID. Messages stay in thread automatically. No need to pass full conversation history each time. Add user message → run → assistant responds. Thread handles state transparently."),
            ("What are limitations of Assistants API?", "Cost: more expensive than raw API (overhead of managed service). Latency: slightly higher. Control: less fine-grained (less customization than building from scratch). Best for: prototypes, simple agents, teams wanting managed service."),
        ]
    },
    ("agentic-ai", "56-agent-deployment-patterns.md"): {
        "title": "Agent Deployment Patterns",
        "description": "Deploy agents to production with containerization, scaling, monitoring, and reliability",
        "how_it_works": """1. Containerization: Docker image with agent code, dependencies, config
2. Orchestration: Kubernetes for scaling, health checks, updates
3. API gateway: expose agent as REST/gRPC endpoint
4. Load balancing: distribute traffic across agent replicas
5. State management: persistent storage for conversation history, context
6. Monitoring: logs, metrics, error tracking
7. Graceful shutdown: finish in-flight requests before stopping
8. Versioning: deploy new agent versions without downtime (blue-green, canary)""",
        "qa": [
            ("How do you handle agent state in distributed deployments?", "Challenge: agent state (conversation history) needs to persist. Solutions: (1) centralized database (Redis, Postgres), (2) sticky sessions (route user to same agent), (3) stateless design (pass state in messages). Database more reliable for high-availability."),
            ("What are blue-green deployments for agents?", "Blue (current): agent version A handling all traffic. Green (new): agent version B deployed but idle. Switch: route traffic from blue to green. Rollback: easy (switch back to blue). Zero downtime, quick rollback if issues."),
            ("How do you monitor agent health in production?", "Metrics: response latency, error rate, token usage, cost. Logs: requests, responses, errors. Alerts: latency spike, error rate >1%, cost anomaly. Distributed tracing: track request flow through components. Real-time dashboard for on-call team."),
            ("What is graceful shutdown and why does it matter?", "Graceful: agent stops accepting new requests, finishes in-flight requests, then shuts down. Matters: avoids losing mid-computation work, maintains user experience. Timeout: if request takes >30s, force shutdown (prevent hanging forever)."),
            ("How do you scale agents horizontally?", "Replicas: run multiple agent instances behind load balancer. Autoscaling: increase replicas when CPU/latency high, decrease when low. Challenges: maintain state consistency, manage shared resources (databases, APIs). Stateless agents easier to scale."),
        ]
    },
    ("agentic-ai", "57-agent-state-management.md"): {
        "title": "Agent State Management",
        "description": "Maintain persistent state across agent interactions including memory, context, and execution history",
        "how_it_works": """1. State types: conversation history, agent knowledge, task progress, user preferences
2. Storage: in-memory (fast, lost on restart), database (persistent, slower)
3. Context window: load relevant state into prompt for each interaction
4. State pruning: remove old/irrelevant state to stay under context limits
5. Consistency: ensure state consistent across distributed replicas
6. Recovery: reload state from database on restart, resume tasks
7. Lifecycle: state created → updated → archived (keep old for audit) → deleted (cleanup)""",
        "qa": [
            ("How do you choose between in-memory and database storage?", "In-memory: fast (microseconds), lost on crash, limited to single machine. Database: slower (milliseconds), persistent, survives crashes. Hybrid: in-memory cache + database for durability. For agents: database essential (conversation continuity matters)."),
            ("How do you handle state explosion in long-running agents?", "Problem: state grows unbounded (conversation history gets huge). Solutions: (1) summarization (compress old messages), (2) chunking (split into separate documents), (3) TTL (delete old state after N days), (4) relevance filtering (keep only relevant). Choose based on task."),
            ("What is state consistency in distributed agents?", "Problem: multiple agent replicas, each with own state copy. Updates to one replica don't reflect in others (stale data). Solution: centralized database (single source of truth), agents read/write there. Trade-off: slightly higher latency for consistency."),
            ("How do you recover agent state after a failure?", "Persistence: save state to database periodically (or after each interaction). On restart: load last saved state. Resume: continue from last saved step. Edge case: partial writes (state saved but request failed). Handle: idempotent operations (safe to retry)."),
            ("What is versioning for agent state?", "Store snapshots of state over time. Enable: rollback (revert to previous state if needed), audit trail (trace what changed), branching (fork state for experimentation). Trade-off: storage cost (store multiple versions). Essential for critical agents."),
        ]
    },
    ("agentic-ai", "58-advanced-reasoning-variants.md"): {
        "title": "Advanced Reasoning Variants",
        "description": "Enhance agent reasoning with variants of chain-of-thought including tree search, self-consistency, and ensemble methods",
        "how_it_works": """1. Chain-of-thought: step-by-step reasoning improves accuracy
2. Tree-of-thought: generate multiple reasoning paths, evaluate each, select best
3. Self-consistency: sample multiple reasoning chains, take majority answer
4. Program synthesis: generate code, execute to verify correctness
5. Ensemble: combine multiple agent instances or reasoning strategies
6. Iterative refinement: generate answer → check → refine → repeat
7. Backtracking: if path fails, try alternative reasoning path""",
        "qa": [
            ("How does tree-of-thought improve on standard chain-of-thought?", "CoT: single path (might get stuck). ToT: generate multiple paths (3-5), evaluate each, select best. Enables backtracking (if path fails, try other). Slower (need multiple inference passes) but more reliable for hard problems."),
            ("What is self-consistency and when is it useful?", "Self-consistency: sample same prompt multiple times (different random seeds), get multiple answers. Vote (majority wins). Improves accuracy especially for reasoning tasks (math, logic). Cost: N×inference cost for sampling N times. Worth it for critical decisions."),
            ("How do you evaluate reasoning quality in agents?", "Metrics: (1) final answer correctness, (2) reasoning quality (check steps are logical), (3) confidence (agent's self-assessment), (4) efficiency (how many steps to reach answer). Trace reasoning: log each step for debugging and auditing."),
            ("What is program synthesis and why is it better than natural language reasoning?", "Program synthesis: generate executable code instead of text. Benefits: deterministic (no ambiguity), verifiable (run and check), composable (combine programs). Challenge: requires coding capability, works best for specific problem classes."),
            ("How do you handle multiple agents reasoning differently?", "Ensemble: run multiple agents with different prompts/models. Aggregate: vote, weighted average, or consensus. Diversity: different agents should use different approaches (some systematic, some intuitive). Better performance than single agent, especially for complex problems."),
        ]
    },
    ("agentic-ai", "59-agent-evaluation-metrics.md"): {
        "title": "Agent Evaluation Metrics",
        "description": "Measure agent performance comprehensively including success rate, cost, latency, and quality",
        "how_it_works": """1. Task success: did agent complete task correctly? Binary or continuous score
2. Cost: token usage × cost-per-token, total expense per task
3. Latency: wall-clock time to complete task, end-to-end latency
4. Quality: subjective evaluation (human raters), automated metrics
5. Efficiency: task success / (cost × latency), score per resource
6. Reliability: success rate, failure modes, error distribution
7. User satisfaction: NPS, task satisfaction, willingness to use again
8. Benchmarks: standard datasets, compare across agents and versions""",
        "qa": [
            ("How do you measure success for open-ended tasks?", "Binary success: did agent achieve goal? Limited (doesn't measure partial progress). Continuous: score 0-1 based on closeness to goal. Rubric: define criteria (clarity, accuracy, completeness) and score on each. Human evaluation most reliable but expensive."),
            ("What metrics matter most for production agents?", "Prioritize: (1) success rate (core metric), (2) cost (budget constraint), (3) latency (user patience), (4) error rate (reliability). In that order: fast failure is costly, slow success is annoying, errors are worst. Monitor all four."),
            ("How do you compare agents fairly?", "Same benchmark: test on identical tasks. Controlled conditions: same model, same prompts, same data. Multiple seeds: run each agent 5-10 times, report mean ± std. Report all metrics (not cherry-picked). Significance testing (statistical)."),
            ("What is Pareto frontier for agents?", "Trade-off between metrics (accuracy vs cost, accuracy vs speed). Pareto frontier: set of non-dominated solutions (can't improve one without worsening another). Plot accuracy vs cost for all agents, frontier shows Pareto-optimal agents."),
            ("How do you handle metrics for multi-turn interactions?", "Per-turn: success per interaction. Cumulative: did agent eventually succeed? Task-centric: did agent achieve high-level goal (may take many turns)? Choose based on use case. For chat: success = user satisfied after conversation."),
        ]
    },
    ("agentic-ai", "60-agent-security-sandboxing.md"): {
        "title": "Agent Security and Sandboxing",
        "description": "Execute agent code safely in isolated environments to prevent malicious or buggy code from causing harm",
        "how_it_works": """1. Sandboxing: isolated execution environment, limits resource access
2. Containerization: Docker containers with resource limits (CPU, memory, disk)
3. Code execution: execute agent-generated code in sandbox, not main process
4. Timeouts: kill execution if exceeds time limit (prevent infinite loops)
5. Permissions: restrict file system access, network access, capability access
6. Monitoring: log all system calls, detect malicious behavior
7. Auditing: track what code executed, who triggered it, what resources used""",
        "qa": [
            ("Why is sandboxing necessary for agents?", "Risk: agent-generated code might have bugs (infinite loop, crash), or be malicious (steal credentials, delete files). Sandbox isolates: bug doesn't crash main system, malicious code has limited damage. Essential for untrusted agent code."),
            ("What are levels of sandboxing?", "Process isolation: separate OS process (some isolation). Container: Docker with resource limits (good isolation). VM: full virtual machine (complete isolation, expensive). Choose based on risk (untrusted code → VM, trusted code → process)."),
            ("How do you handle code that needs external resources?", "Restricted APIs: agent calls whitelisted functions (not arbitrary code). Examples: read_file('/data/...') allowed, read_file('/etc/passwd') blocked. Requires careful API design to be both safe and useful."),
            ("What is the difference between code execution and code generation?", "Generation: agent writes code, human reviews, human executes (safe). Execution: agent writes and runs code (risky but faster). For agents: auto-execute only whitelisted operations (math, data processing), require approval for external operations."),
            ("How do you prevent agents from accessing credentials?", "Secrets management: don't pass credentials in prompts. Use secret vault (AWS Secrets Manager, HashiCorp Vault). Agent calls API with permission token (not full credentials). Audit: log which secrets accessed by whom. Rotate regularly."),
        ]
    },
    ("agentic-ai", "61-multi-turn-conversation.md"): {
        "title": "Multi-Turn Conversation Management",
        "description": "Manage long-running conversations with coherent context, turn-taking, and state tracking",
        "how_it_works": """1. Turns: user input → agent response → user input → ...
2. Context: keep relevant history (last N turns, summarized context)
3. State: track conversation state (topic, subtasks, context)
4. Coherence: ensure responses consistent with context and previous statements
5. Turn-taking: manage who speaks (user or agent, avoid deadlock)
6. Interruption: handle user interruptions, task switching
7. Cleanup: end conversation gracefully, summarize outcome""",
        "qa": [
            ("How much conversation history should you keep?", "Full history: accurate but uses tokens. Last N turns: balance accuracy and efficiency. Summarization: compress old turns to facts. Hybrid: keep last 5 full turns, summarize older context. Adjust based on context window size and task complexity."),
            ("How do you handle context window overflows?", "Overflow: conversation exceeds model's context window. Solutions: (1) drop oldest messages, (2) summarize old context, (3) retrieval (fetch relevant messages from history). Try summarization first (preserve info), then drop if needed."),
            ("What is conversation state and how do you track it?", "State: current topic, subtasks completed, user intent, decisions made. Track: explicitly (state variable) or implicitly (inferred from messages). Use explicit for critical applications (fewer errors). Update: after each agent response."),
            ("How do you prevent agents from contradicting themselves?", "Check: before responding, review past statements. Verify: ensure new statement consistent with context. Fallback: if inconsistency detected, acknowledge and clarify. Log: track contradictions for debugging."),
            ("How do you handle topic switching in conversation?", "Detect: user changes topic (detect intent shift). Handle: (1) acknowledge old topic, (2) switch context, (3) reset sub-state if needed. Challenge: intent detection not always clear (genuine switch vs. tangent)."),
        ]
    },
    ("agentic-ai", "62-agent-cost-analysis.md"): {
        "title": "Agent Cost Analysis",
        "description": "Analyze and optimize the financial cost of running agents including token usage and API calls",
        "how_it_works": """1. Token counting: input tokens + output tokens per interaction
2. Pricing: cost-per-1K-tokens varies by model (GPT-4: $0.03, GPT-3.5: $0.002)
3. API calls: each tool call, retrieval, external service adds cost
4. Cost per task: sum of all tokens and API calls for one task
5. Aggregation: multiply by frequency to get daily/monthly/yearly cost
6. Optimization: reduce tokens (shorter prompts), use cheaper models (GPT-3.5 vs GPT-4)
7. Monitoring: track cost in real-time, alert on anomalies""",
        "qa": [
            ("How do you reduce agent token usage?", "Prompt optimization: remove verbose instructions, use examples efficiently. Model selection: GPT-3.5 cheaper than GPT-4 (trade accuracy for cost). Caching: reuse computations (store embeddings, cache prompts). Summarization: compress context. Typical reduction: 30-50% with optimization."),
            ("What's the cost difference between models?", "GPT-4: $0.03/1K input, $0.06/1K output tokens. GPT-3.5: $0.002/1K input, $0.004/1K output. Claude: similar to GPT-4. Trade-off: GPT-4 better quality but expensive. Use GPT-3.5 for high-volume low-stakes, GPT-4 for critical tasks."),
            ("How do you handle cost anomalies?", "Monitor: track per-user, per-task costs. Alert: if exceeds threshold (e.g., >$1 per task). Investigate: is it legitimate (complex task) or bug (infinite loop, hallucination)? Limits: set per-user caps to prevent runaway spending."),
            ("Should you optimize for cost or quality?", "Context: cost-quality trade-off. High-volume: optimize cost (use GPT-3.5, short context). Mission-critical: optimize quality (use GPT-4, long context). Most: balance (use hybrid, optimize both). Measure: cost per unit quality achieved."),
            ("How do you forecast agent costs?", "Estimate: average tokens per task × tasks per day × days per month × price-per-token. Sensitivity: how do costs scale with usage, model, context? Budget: plan for growth (2-3x). Monitor: track actual vs. forecast, adjust as needed."),
        ]
    },
    ("agentic-ai", "63-agent-frameworks-comparison.md"): {
        "title": "Agent Frameworks Comparison",
        "description": "Compare popular frameworks (LangChain, AutoGen, ReAct) for building and deploying agents",
        "how_it_works": """1. LangChain: chains (templates), agents (tool use), memory, integrations
2. AutoGen: multi-agent conversations, automatic role assignment, flexible interactions
3. ReAct: reasoning + acting loop, simple structure, clear reasoning trace
4. Comparison: ease of use, flexibility, production readiness, community
5. LangChain: most popular, best integrations, middle learning curve
6. AutoGen: best for multi-agent, experimental, fewer integrations
7. ReAct: simplest, good for learning, limited extensibility""",
        "qa": [
            ("Which framework should you choose for a new project?", "Start with LangChain: most mature, best docs, most integrations. Use AutoGen: if multi-agent is core (AutoGen excels here). Use ReAct: if learning or prototyping (simple, clear). In production: depends on team expertise and requirements."),
            ("How do frameworks differ in handling state?", "LangChain: explicit memory management (you control state). AutoGen: implicit (framework manages conversation). ReAct: minimal state (just reasoning trace). Tradeoff: control vs convenience. ReAct simplest, LangChain most flexible."),
            ("What's the learning curve for each framework?", "ReAct: lowest (simple loop, 1-2 hours to understand). LangChain: medium (many concepts, 1-2 days to productive). AutoGen: medium (new paradigm, 1-2 days to productive). All have good docs/tutorials."),
            ("Can you mix frameworks (e.g., LangChain agent with AutoGen?)", "Possible but rare: different abstractions, not designed to interop. Better: choose one and stick with it. If need features from both: evaluate if really needed, or wait for frameworks to converge."),
            ("Which framework is best for production?", "LangChain: if you want full control and maturity. AutoGen: if you want multi-agent as core. ReAct: not designed for production (too simple). For production: add monitoring, error handling, deployment logic on top of any framework."),
        ]
    },
    ("agentic-ai", "64-real-time-agent-systems.md"): {
        "title": "Real-Time Agent Systems",
        "description": "Build agents that respond in real-time with streaming responses, concurrent requests, and low latency",
        "how_it_works": """1. Streaming: return response tokens as they're generated (don't wait for full response)
2. Concurrency: handle multiple requests simultaneously (not sequential)
3. Latency: first-token latency (how fast first token appears) and total latency
4. Buffering: buffer tokens to avoid excessive I/O, stream in chunks
5. Inference optimization: quantization, caching, batching to reduce latency
6. Infrastructure: use GPU, high-bandwidth connections, optimized serving (vLLM)
7. Monitoring: track latency distribution, percentiles (p50, p99)""",
        "qa": [
            ("How do you stream responses from agents?", "Use streaming API: GPT-4 streaming, LLaMA server with streaming. Get tokens one at a time, send to user as they arrive. UX: user sees incremental response (feels responsive). Implementation: WebSocket or Server-Sent Events (SSE)."),
            ("What causes agent latency and how do you reduce it?", "Sources: model inference (biggest), tokenization, tool calls, API latency. Reduce: (1) faster models (7B vs 70B), (2) quantization (4-bit vs 16-bit), (3) batching (process multiple requests together), (4) caching (reuse computations)."),
            ("How do you handle concurrent agent requests?", "Load balancing: distribute to multiple replicas. Queue: if more requests than capacity, queue and process in order. Async: use async/await, don't block on slow requests. Graceful degradation: prioritize critical requests, drop non-critical if overwhelmed."),
            ("What is first-token latency and why does it matter?", "First-token: how fast user sees first response token. Matters: affects perceived responsiveness. Optimize: reduce prompt processing, use smaller models, cache embeddings. Typical: 100-500ms first-token (good), >1s (feels slow)."),
            ("How do you optimize for different latency SLAs?", "Compute SLA: p50 latency (median), p99 (99th percentile). Different tiers: standard (p50<1s), fast (p50<100ms), urgent (p50<10ms). Cost increases: faster tier = more resources. Choose based on use case."),
        ]
    },

    # AI Fundamentals: 31-40 (10 more)
    ("ai", "31-q-learning.md"): {
        "title": "Q-Learning",
        "description": "Learn optimal action policies by iteratively updating action value estimates based on observed rewards",
        "how_it_works": """1. Q(s,a): action value function, expected cumulative reward from state s taking action a
2. Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
3. Components: r (immediate reward), γ (discount factor), α (learning rate)
4. Exploration: ε-greedy (explore with probability ε, exploit otherwise)
5. Convergence: with sufficient exploration and learning rate decay, converges to optimal Q*
6. Deep Q-Network (DQN): use neural network to approximate Q (handles large state space)
7. Improvements: target networks (stability), experience replay (decorrelate samples), dueling networks""",
        "qa": [
            ("Why is Q-learning off-policy?", "Off-policy: learns optimal policy while following exploratory policy (ε-greedy). Can learn from trajectories generated by any policy. Compare to on-policy (REINFORCE): must learn from current policy. Off-policy more sample-efficient."),
            ("What is the exploration-exploitation tradeoff in Q-learning?", "Exploration: try new actions to find better ones (needed to learn). Exploitation: use best known action. ε-greedy balances: explore ε fraction of time, exploit (1-ε) fraction. Decay ε over time (explore more early, exploit more late)."),
            ("How do you prevent overestimation in Q-learning?", "Problem: max operation in Q-update uses same network (overestimates). Solution: Double Q-learning uses separate network for selection vs. evaluation. Or: target network (slowly updated copy of Q-network). Both reduce overestimation bias."),
            ("Can Q-learning handle continuous action spaces?", "Discrete only in basic form (output Q-value per action). Continuous: use actor-critic or policy gradient instead (output action directly). Or: discretize action space (approximate continuous with discrete)."),
            ("How do you know Q-learning has converged?", "Monitor: average reward per episode (should increase). Q-values: should stabilize (stop changing). Learning curves: plot performance vs episodes, look for plateau. Testing: evaluate learned policy on held-out tasks."),
        ]
    },
    ("ai", "32-policy-gradients.md"): {
        "title": "Policy Gradients",
        "description": "Directly optimize the policy by taking gradient steps to maximize expected reward",
        "how_it_works": """1. Policy π(a|s): stochastic policy mapping states to action probabilities
2. Objective: maximize J(θ) = E[sum discounted rewards]
3. Policy gradient: ∇J(θ) = E[∇log π(a|s) × R(τ)]
4. REINFORCE: sample trajectory, compute gradients, update policy
5. Baseline: subtract baseline from reward to reduce variance (doesn't bias gradient)
6. Advantage: use advantage A(s,a) = Q(s,a) - V(s) instead of reward (lower variance)
7. Variants: PPO (clipped objective), TRPO (trust region), A3C (asynchronous)""",
        "qa": [
            ("How do policy gradients differ from Q-learning?", "Q-learning: learn value function implicitly (derive policy by max). Policy gradient: directly optimize policy. Tradeoff: PG converges slower but to better optima, handles continuous actions naturally. Both have merits."),
            ("What is the baseline in policy gradients and why use it?", "Baseline: subtract moving average of returns from reward. Reduces variance (if return is 10 and baseline is 8, advantage is 2). Doesn't bias gradient (expected value still same). Critical for stable training."),
            ("What's the difference between REINFORCE and A3C?", "REINFORCE: accumulate trajectory, update once (on-policy). A3C: asynchronous (multiple workers), update frequently. A3C: faster (parallel) but more complex. REINFORCE: simpler but slower. Use REINFORCE for learning, A3C for scaling."),
            ("How do you handle high-variance policy gradients?", "Sources: rewards are noisy, variance grows with horizon. Solutions: (1) baseline (reduce magnitude), (2) advantage (relative comparison), (3) batch/normalization (average over samples), (4) trust regions (limit step size)."),
            ("Can policy gradients handle discrete and continuous actions?", "Discrete: softmax over actions (same as classification). Continuous: output mean + variance of action distribution (Gaussian), sample from it. Much more natural for continuous than Q-learning (which requires discretization)."),
        ]
    },
    ("ai", "33-actor-critic-methods.md"): {
        "title": "Actor-Critic Methods",
        "description": "Combine policy gradient (actor) and value function (critic) for stable and sample-efficient learning",
        "how_it_works": """1. Actor: policy network π(a|s), generates actions
2. Critic: value network V(s), estimates state value
3. Advantage: A(s,a) = r + γV(s') - V(s) (estimated using critic)
4. Actor loss: -log π(a|s) × A(s,a) (improve policy using critic's estimate)
5. Critic loss: MSE(V(s), target) where target = r + γV(s') (bootstrap from next state)
6. Update: compute both losses, backprop to both networks
7. Benefits: lower variance (critic baseline), more stable (two networks)""",
        "qa": [
            ("Why is actor-critic better than pure policy gradient?", "Pure PG: high variance (rewards are noisy). Actor-critic: critic provides baseline (reduces variance). Result: faster convergence, more stable training. Trade-off: slightly more complex (two networks)."),
            ("What is TD error and how is it used?", "TD error: δ = r + γV(s') - V(s). Actor: use as advantage (update policy in gradient direction of advantage). Critic: update V(s) to minimize TD error. Both use same TD signal (efficient)."),
            ("How do you avoid instability in actor-critic?", "Sources: two networks learning simultaneously (instability), high variance from policy gradients. Solutions: (1) target critic network (slowly updated copy), (2) experience replay (decorrelate samples), (3) entropy regularization (encourage exploration)."),
            ("What is asynchronous advantage actor-critic (A3C)?", "A3C: multiple workers run episodes in parallel, asynchronously update shared networks. Benefits: more diverse experience, faster training. Implementation: careful synchronization (locks, atomic operations). Good for distributed systems."),
            ("Can you use actor-critic for continuous control?", "Yes, naturally: actor outputs mean+variance of action distribution. Critic estimates value. Works for both discrete and continuous. Popular in robotics (DDPG, TD3, SAC variants). Better than Q-learning for continuous actions."),
        ]
    },
    ("ai", "34-graph-neural-networks.md"): {
        "title": "Graph Neural Networks",
        "description": "Learn on graph-structured data by message passing between nodes, enabling prediction on networks",
        "how_it_works": """1. Graph: nodes (entities) and edges (relationships)
2. Node features: each node has feature vector
3. Message passing: each node aggregates info from neighbors
4. Update: h_v^(t+1) = aggregate({h_u^(t) for u in neighbors(v)})
5. Readout: combine node embeddings into graph embedding
6. Tasks: node classification (predict node labels), graph classification, link prediction
7. Variants: GCN (convolutional), GAT (attention), GraphSAGE (sampling)""",
        "qa": [
            ("What's the difference between GCN and GraphSAGE?", "GCN: deterministic (aggregate all neighbors). GraphSAGE: sample subset of neighbors (scalable to large graphs). GCN more accurate for small graphs, GraphSAGE faster for large graphs."),
            ("How do you handle very large graphs?", "Challenge: full GNN requires aggregating all neighbors (O(n²)). Solutions: (1) sampling (sample k neighbors instead of all), (2) layer-wise sampling (different k per layer), (3) cluster-based (partition graph, aggregate within clusters)."),
            ("What is attention in graph networks (GAT)?", "GAT: each node learns importance weights for neighbors (attention weights). Flexible: different neighbors get different weights per layer. More expressive: can learn complex aggregation patterns. Slower: needs to compute weights."),
            ("How do you create node embeddings from graphs?", "Methods: (1) trained GNN (learn embeddings end-to-end), (2) random walk-based (DeepWalk, Node2Vec), (3) matrix factorization. For supervised: train GNN end-to-end. For unsupervised: random walk or matrix factorization."),
            ("Can you use GNNs for recommendation systems?", "Yes: users and items as nodes, interactions as edges. GNN learns user/item embeddings, predicts new links (recommendations). Benefits: captures collaborative filtering structure naturally. Popular: LightGCN, NGCF variants."),
        ]
    },
    ("ai", "35-causal-inference.md"): {
        "title": "Causal Inference",
        "description": "Learn causal relationships between variables, enabling prediction of interventions and counterfactuals",
        "how_it_works": """1. Confounding: variable X affects both treatment T and outcome Y
2. DAG: directed acyclic graph showing causal structure
3. Adjustment: condition on confounders to isolate causal effect
4. Matching: match treated and control units on confounders
5. Propensity score: probability of treatment, use for matching or weighting
6. Instrumental variables: use variable that affects T but not Y (directly)
7. Difference-in-differences: compare treatment and control pre/post intervention""",
        "qa": [
            ("What's the difference between correlation and causation?", "Correlation: variables move together. Causation: one causes change in other. Confounder example: ice cream sales correlate with drowning (both caused by summer weather, no direct causation). Causal methods isolate true effect."),
            ("How do you estimate causal effects from observational data?", "Assumption: no unmeasured confounders (observe all variables affecting outcome). Methods: (1) adjustment (condition on confounders), (2) matching (match treated/control on confounders), (3) propensity score (weight by inverse probability of treatment). All assume no unmeasured confounding."),
            ("What is a confounder and how do you handle it?", "Confounder: affects both treatment and outcome. Bias: if not adjusted, confounders bias causal estimate. Handle: (1) randomization (best, breaks confounding), (2) adjustment (condition on confounder), (3) matching (match on confounder)."),
            ("What are instrumental variables and when do you use them?", "IV: variable Z affects treatment T but doesn't directly affect outcome Y. Use when: confounders unmeasured, can't randomize. Example: rainfall affects irrigation (T), affects crops (Y) but not through other mechanisms. Enables causal inference under more assumptions."),
            ("How do you validate causal inferences?", "Sensitivity analysis: how robust to unmeasured confounding? Placebo tests: effect on variables that shouldn't be affected. Heterogeneous effects: does effect differ by subgroup (should be consistent mechanism). Multiple methods: if all agree, confidence increases."),
        ]
    },
    ("ai", "36-probabilistic-graphical-models.md"): {
        "title": "Probabilistic Graphical Models",
        "description": "Represent complex probability distributions using graphs where nodes are variables and edges show dependencies",
        "how_it_works": """1. Bayesian network: DAG where edges show causal/conditional dependencies
2. Joint probability: factorizes as product of conditional probabilities
3. Markov random field: undirected graph, factors are clique potentials
4. Inference: compute P(X|observations) using message passing (belief propagation)
5. Learning: learn structure (which edges) and parameters (conditional probabilities)
6. Applications: medical diagnosis (Bayesian nets), image segmentation (MRF)
7. Sampling: generate samples from distribution (Markov chain Monte Carlo)""",
        "qa": [
            ("What's the difference between Bayesian networks and Markov random fields?", "Bayesian: directed acyclic graph (DAG), edges show causality. MRF: undirected, shows correlations (no causal direction). Expressive: MRF slightly more (can represent cycles), Bayesian easier to interpret (causal structure clear)."),
            ("How do you perform inference in graphical models?", "Exact: message passing (belief propagation), works for trees and small graphs. Approximate: Markov chain Monte Carlo (sampling), variational inference (optimize lower bound). Choose: exact for small models, approximate for large."),
            ("How do you learn structure of a graphical model?", "Known structure: learn parameters (maximize likelihood). Unknown: learn structure from data (NP-hard). Methods: greedy search (add/remove edges), constraint-based (find edges consistent with independencies), score-based (BIC/AIC)."),
            ("What is a factor in Markov random fields?", "Factor: potential function over clique (subset of variables). Encodes preference for certain value combinations. Larger factors = more expressive but harder to compute. Factorization enables efficient inference via message passing."),
            ("Can you combine graphical models with deep learning?", "Yes: replace explicit factors with learned functions (neural networks). Example: neural CRF (conditional random field with neural potential). Benefits: learns features automatically. Challenges: may lose interpretability."),
        ]
    },
    ("ai", "37-variational-autoencoders.md"): {
        "title": "Variational Autoencoders (VAE)",
        "description": "Generate new data and learn latent representations through variational inference",
        "how_it_works": """1. Encoder: q(z|x) maps data to latent distribution
2. Latent: z sampled from N(μ, σ²) (Gaussian bottleneck)
3. Decoder: p(x|z) reconstructs data from latent
4. Loss: reconstruction loss + KL divergence (regularization)
5. KL: penalizes latent distribution from standard Gaussian (enables generation)
6. Training: reparameterization trick allows backprop through sampling
7. Generation: sample z from N(0,1), decode to get new data
8. Interpolation: interpolate in latent space (smooth transitions)""",
        "qa": [
            ("Why does VAE need both reconstruction and KL loss?", "Reconstruction: encoder-decoder learns to compress and reconstruct data. KL: forces latent to match prior (Gaussian), enables generation from prior. Together: learn good representations that are also generative."),
            ("What is the reparameterization trick and why is it needed?", "Trick: z = μ + σ*ε where ε ~ N(0,1). Enables: gradients flow through sampling (no gradients through sampling directly). Needed: backprop requires differentiable operations, sampling isn't. Trick makes it differentiable."),
            ("How is VAE different from a standard autoencoder?", "Standard: deterministic bottleneck, reconstructs but can't generate (no prior). VAE: probabilistic bottleneck, reconstructs and can generate (prior enables generation). VAE more useful for generation but may have lower reconstruction quality."),
            ("How do you control what VAE learns?", "Loss weights: balance reconstruction vs. KL (large KL → focus on generation, small → focus on reconstruction). Beta-VAE: multiply KL by β (β>1 emphasizes disentanglement). Adjust tradeoff based on task."),
            ("What is disentanglement in VAE?", "Disentanglement: each latent dimension corresponds to one semantic factor (size, color, rotation). Benefits: interpretability, controllable generation. Achieve: Beta-VAE or other regularization. Measure: how well can classifier predict factor from dimension?"),
        ]
    },
    ("ai", "38-generative-adversarial-networks.md"): {
        "title": "Generative Adversarial Networks (GANs)",
        "description": "Train generator and discriminator in competition to generate realistic data from noise",
        "how_it_works": """1. Generator G: maps noise z to fake data G(z)
2. Discriminator D: classifies real vs. fake data
3. Game: G tries to fool D, D tries to detect fakes
4. Loss: D maximizes log(D(x)) + log(1-D(G(z)))
5. Generator loss: minimizes log(1-D(G(z))) (or max log(D(G(z))))
6. Training: alternate between D and G updates
7. Convergence: when D can't distinguish, G produces realistic data
8. Challenges: mode collapse (G produces same output), unstable training""",
        "qa": [
            ("Why is GAN training unstable?", "Reasons: (1) generator gradient vanishes when D confident, (2) discriminator overpowers generator, (3) mode collapse (generator ignores part of data). Fixes: (1) Wasserstein loss (better gradients), (2) spectral norm (stabilize D), (3) unrolled GAN (look ahead)."),
            ("What is mode collapse and how do you prevent it?", "Mode collapse: generator produces same output despite different inputs (ignores diversity in data). Prevent: (1) minibatch discrimination (penalize similar minibatch samples), (2) feature matching (match statistics), (3) loss functions (WGAN, hinge)."),
            ("How do you evaluate GAN quality?", "Inception score: generated sample quality (high = realistic). FID (Fréchet Inception Distance): distance between real and fake distributions (low = good). Manual evaluation: look at samples. Inception score easier but biased, FID more reliable."),
            ("What's the difference between GAN variants (WGAN, StyleGAN)?", "WGAN: Wasserstein distance instead of JS divergence, better gradients, more stable. StyleGAN: style-based architecture, fine control over generation (produce specific attributes). Both improve on vanilla GAN."),
            ("Can GANs be used for non-image tasks?", "Yes: text generation (SeqGAN), tabular data, audio, video. Challenge: GANs work best for continuous data (images), harder for discrete (text). Solutions: SeqGAN (approximate), or use other methods (VAE, diffusion) for discrete data."),
        ]
    },
    ("ai", "39-time-series-forecasting.md"): {
        "title": "Time Series Forecasting",
        "description": "Predict future values in sequential data based on past observations and temporal patterns",
        "how_it_works": """1. Autoregressive (AR): predict from past values y_t = β₀ + Σ βᵢ*y_{t-i} + ε
2. ARIMA: AR + moving average + differencing for non-stationary series
3. Exponential smoothing: weighted average of past (recent values weighted more)
4. RNNs/LSTMs: learn non-linear temporal patterns from sequence
5. Transformers: self-attention over time steps (captures long-range dependencies)
6. Multivariate: multiple input series predict output (e.g., weather → demand)
7. Evaluation: MSE on held-out future, track over time (performance may degrade)""",
        "qa": [
            ("When should you use ARIMA vs neural networks?", "ARIMA: stationary data, small datasets, interpretability important. Neural: non-linear patterns, large datasets, complex relationships. Hybrid: ARIMA for baseline, NN if ARIMA not good enough."),
            ("What is stationarity and why does it matter?", "Stationarity: statistical properties (mean, variance) constant over time. ARIMA assumes stationarity (if not, differencing). Non-stationary: trends, seasonality. Check: plots, statistical tests (ADF). Transform (log, diff) to achieve stationarity."),
            ("How do you handle seasonality in forecasting?", "Seasonality: repeating patterns (e.g., weekly, yearly). Model: (1) seasonal ARIMA (SARIMA), (2) include seasonal variables, (3) RNNs learn automatically. Challenge: long-range dependencies (annual seasonality = 365 steps)."),
            ("What's the difference between one-step-ahead and multi-step forecasting?", "One-step: predict t+1 given up to t (easier). Multi-step: predict t+1, t+2, ..., t+h (harder, error accumulates). Approaches: recursive (use predictions), direct (separate models per step), sequence-to-sequence (encoder-decoder)."),
            ("How do you evaluate forecasting models?", "Metrics: MAE (mean absolute error), RMSE (penalizes large errors), MAPE (relative error). Baseline: use last value or seasonal average. Compare: your model vs. baseline. Cross-validation: time-series CV (train on past, test on future)."),
        ]
    },
    ("ai", "40-anomaly-detection.md"): {
        "title": "Anomaly Detection",
        "description": "Identify unusual or outlier data points that deviate from normal patterns",
        "how_it_works": """1. Supervised: anomalies labeled, treat as classification (imbalanced)
2. Unsupervised: no labels, assume anomalies rare and different from normal
3. Statistical: model distribution, flag low-probability points
4. Distance-based: compute distance to nearest neighbors, outliers far from others
5. Density-based: DBSCAN, LOF (local outlier factor), low-density = anomaly
6. Autoencoders: reconstruct normal data well, reconstruct anomalies poorly
7. One-class SVM: learn boundary around normal data, points outside = anomalies""",
        "qa": [
            ("Why is anomaly detection hard?", "Challenges: (1) anomalies rare (imbalanced data), (2) definition unclear (what's anomalous?), (3) new types emerge (can't train for all), (4) cost asymmetric (missing anomaly vs. false alarm have different costs)."),
            ("How do you choose threshold for anomaly score?", "Tunable: compute score for each sample, threshold to classify. High threshold: few anomalies flagged (high precision, low recall). Low threshold: many anomalies flagged (low precision, high recall). Set based on business cost."),
            ("What's the difference between outliers and anomalies?", "Outliers: statistically extreme but not anomalous (tall person in normal sample). Anomalies: contextually abnormal (car breakdown in traffic). Anomaly detection looks for contextual anomalies (harder, requires domain knowledge)."),
            ("Can you use deep learning for anomaly detection?", "Yes: autoencoders learn normal patterns, reconstruct anomalies poorly. Use reconstruction error as anomaly score. Or: one-class neural networks. Challenge: needs lots of normal data, can overfit to noise."),
            ("How do you validate anomaly detection?", "Labeled data (test set): precision, recall, F1. No labels: inspect flagged samples (does system find meaningful anomalies?). Baseline: statistical method or random. Monitoring: track false positive rate in production (adjust threshold if needed)."),
        ]
    },
}

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

Key trade-offs and design considerations for this concept.

## Interview Q&A

{qa_section}

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
"""

def create_concept(section, filename, metadata):
    """Create a single concept file."""
    title = metadata["title"]
    description = metadata["description"]
    how_it_works = metadata["how_it_works"]
    qa_list = metadata.get("qa", [])

    # Format interview Q&A
    qa_section = ""
    for i, (q, a) in enumerate(qa_list[:6], 1):  # Limit to 6 Q&As
        qa_section += f"\n**Q: {q}**\nA: {a}\n"

    # Simple intuition
    intuition = description.split(".")[0] + " Understanding this concept enables better system design and problem-solving."

    content = CONCEPT_TEMPLATE.format(
        title=title,
        description=description,
        intuition=intuition,
        how_it_works=how_it_works,
        qa_section=qa_section
    )

    section_dir = f"{BASE}/{section}/concepts"
    filepath = os.path.join(section_dir, filename)

    os.makedirs(section_dir, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath

def main():
    """Create all remaining concepts."""

    print("=== Creating Remaining Concept Files ===\n")

    created = 0
    for (section, filename), metadata in ALL_CONCEPTS.items():
        filepath = create_concept(section, filename, metadata)
        section_name = section.upper()
        concept_name = filename.replace(".md", "").split("-", 1)[1].replace("-", " ").title()
        print(f"  ✓ {section_name}: {concept_name}")
        created += 1

    print(f"\n✅ Created {created} new concept files")
    print(f"\nNext: Run generate_all_notebooks.py to create corresponding notebooks")

if __name__ == "__main__":
    main()
