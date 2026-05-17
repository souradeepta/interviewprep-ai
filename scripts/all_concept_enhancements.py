#!/usr/bin/env python3
"""
Comprehensive enhancements for all remaining LLM concepts.
To be merged into main enhance_concepts.py
"""

ALL_CONCEPT_ENHANCEMENTS = {
    "continuous-batching.md": {
        "interview_qa": [
            ("What's continuous batching and why is it better than static batching?",
             "Static batching: wait for N requests → batch → process. Continuous: accept requests one-by-one, add to batch as available, execute when full or timeout reached. Advantage: latency for early requests drops (don't wait for 32 requests). For 1 req/sec: static batch-32 = 32s latency. Continuous = 0.1s latency. Throughput same, latency better."),
            ("How does scheduling work in continuous batching?",
             "1) Queue incoming requests 2) When batch size reached OR timeout expired, execute. Timeout typically 5-100ms to balance latency/throughput. Scheduling policy: FIFO (fair), priority (VIP requests first), adaptive (vary timeout)."),
            ("What are the challenges with continuous batching?",
             "Variable batch size: harder to optimize GPU utilization. Batching overhead per batch (kernel launch ~1-5ms). Very large batches (1000s): worse latency for unlucky requests (last in queue). Solution: max batch size + dynamic adaptation."),
            ("When would you use vLLM vs standard Hugging Face for batching?",
             "HF: static batching, simple but high latency. vLLM: continuous batching + paging, 10-50x throughput improvement, lower latency. Trade-off: vLLM complexity. Use HF for offline batch processing; vLLM for online serving."),
            ("How do you handle requests with different output lengths in batching?",
             "Problem: one request needs 10 tokens, another needs 100. Can't batch (different lengths). Solution: PagedAttention (vLLM) - don't wait, process different lengths independently. Alternative: pad all to max (wasteful) or reject early batches (unfair)."),
        ],
        "real_world_examples": [
            {"title": "Static vs Continuous Batching Benchmark", "description": "Model: Llama 2 7B, 1-10 req/sec. Static batch-32: p99 latency 32s. Continuous batch-32: p99 latency 0.5s. Throughput: same (100 tok/s). Cost: same. Winner: continuous for interactive, static for batch processing jobs."},
            {"title": "vLLM in Production API", "description": "Flask API: accept requests, queue via vLLM. 5 concurrent users: static batching would serve 1-2/sec total. vLLM: 20/sec. p99 latency: 2-5s (vs 30-60s static). Deployed at Anyscale: handles 1000s req/sec."},
            {"title": "Timeout Tuning", "description": "Timeout 1ms: latency under 10ms (good for interactive). Timeout 100ms: latency up to 100ms (better throughput). Data center: adapt based on load. High load: longer timeout (more batching). Low load: shorter timeout (lower latency)."},
        ],
        "workflow_diagram": """graph LR
    A["Request 1"] -->|Queue| B["Batch Builder"]
    C["Request 2"] -->|Queue| B
    D["Request 3"] -->|Queue| B
    B -->|Size or Timeout| E["Execute Batch"]
    E -->|Outputs| F["Send Responses"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style E fill:#e8f5e9""",
    },
    "kv-cache.md": {
        "interview_qa": [
            ("What's KV cache and why does it matter?",
             "During inference: generate token-by-token. For each token, compute attention over full context. Without cache: recompute keys/values every step = O(n²) for n-token output. With cache: store computed K,V, reuse = O(n). For 1000-token generation: 1000x speed difference."),
            ("How much memory does KV cache use?",
             "KV cache size = batch_size × context_length × 2 × hidden_dim × precision. Example: batch=1, context=4K, hidden=4096, FP32 = 1 × 4K × 2 × 4096 × 4 bytes = 128MB per layer × 32 layers = 4GB. For batch=16: 64GB. Dominates memory in inference."),
            ("What's grouped-query attention (GQA) and how does it reduce KV cache?",
             "Standard multi-head: each head has separate K,V (32 heads = 32 K,V heads). GQA: share K,V across multiple query heads (e.g., 8 groups = 4 K,V heads). Reduces KV cache 8x. Trade-off: 0.5-1% accuracy loss. Used in Llama 2, Mistral."),
            ("How do you optimize KV cache for long contexts?",
             "Options: 1) GQA (reduce K,V). 2) Sparse attention (not all positions needed). 3) Pruning (discard irrelevant history). 4) Compression (quantize K,V). Combinations: GQA + INT8 KV = 32x reduction. For 32K context: 4GB → 128MB."),
            ("When would you clear KV cache?",
             "Per-request: always clear (fresh context). Multi-turn conversation: keep cache (reuse context from previous turns). Problem: cache grows unbounded. Solution: sliding window (keep last 2K tokens) or explicit reset."),
        ],
        "real_world_examples": [
            {"title": "KV Cache in Multi-Turn Chat", "description": "Conversation: 10 turns, each turn 100 tokens input. Without cache: recompute 1000s of attention operations per turn. With cache: reuse 900 from previous turns, compute only 100 new. Latency: 5s → 0.5s per turn."},
            {"title": "Batch Inference with Limited Memory", "description": "GPU: 40GB memory. Batch=64, context=4K, FP32 without GQA: 64GB needed (exceeds memory!). With GQA: 8GB KV cache. Can now run batch=64. Throughput: 100 tok/s (vs impossible without optimization)."},
            {"title": "Long-Context RAG with Cache Optimization", "description": "Retrieval: 100 context chunks (100K tokens). Naive: KV cache 25GB. GQA + INT8: 1.5GB. Within inference budget. Latency: 500ms (reasonable). Without optimization: impossible."},
        ],
        "workflow_diagram": """graph LR
    A["Token t"] -->|New Attention| B["Compute Q<br/>Reuse K,V"]
    B -->|Store| C["KV Cache<br/>[0:t]"]
    C -->|Next Token| D["Token t+1"]
    D -->|Add to Cache| E["Updated Cache<br/>[0:t+1]"]

    style A fill:#e3f2fd
    style C fill:#e0f2f1
    style E fill:#e8f5e9""",
    },
    "few-shot-learning.md": {
        "interview_qa": [
            ("What's few-shot learning and how does it differ from zero-shot?",
             "Zero-shot: 'Classify sentiment: positive/negative'. Few-shot: '\"Good product\" → positive. \"Bad product\" → negative. \"Good quality\" → positive. Now classify: \"Great service\"'. Few-shot dramatically improves accuracy (sometimes 30-50% gain)."),
            ("How many examples do you need for effective few-shot?",
             "Task-dependent: simple classification = 1-2 examples sufficient. Complex reasoning = 5-10 needed. Diminishing returns beyond 10 (20+ shows minimal improvement). Rule of thumb: start with 3, increase if accuracy low. Quality > quantity (good examples matter more than many)."),
            ("What makes a good few-shot example?",
             "Diverse: cover range of inputs (easy + hard cases). Representative: similar to test distribution. Explained: include reasoning if helpful. Consistent: same format for all. Bad: unrepresentative or noisy examples confuse model."),
            ("How do you select few-shot examples programmatically?",
             "Random: simple, sometimes okay. Similarity-based: choose examples most similar to test input (use embeddings). Uncertainty sampling: examples model uncertain on. Diversity-based: examples covering feature space. Best: combination of similarity + diversity."),
            ("When is few-shot insufficient and you need fine-tuning?",
             "Few-shot works: general tasks, simple patterns, prompt-able behaviors. Fails: task requires significant internal model change, distribution shift, style transfer. Example: sentiment classification (few-shot fine) vs. writing style adaptation (needs fine-tuning)."),
        ],
        "real_world_examples": [
            {"title": "Few-Shot for Customer Intent Classification", "description": "Chatbot: classify support tickets. Examples: 'Refund request' → returns, 'Can't login' → technical, 'Feedback' → general. Zero-shot: 45% accuracy. Few-shot (3 examples): 72% accuracy. Deployed in Zendesk integration."},
            {"title": "Few-Shot Semantic Matching", "description": "Task: match product descriptions to categories. Descriptions vary (informal, typos, abbreviations). Few-shot with diverse examples: 88% accuracy. Cost: <$0.01 per classification vs. $0.50 with fine-tuning."},
            {"title": "Multi-Language Few-Shot", "description": "Translate task instructions to 10 languages. Few-shot examples in each language. Model generalizes zero-shot to other languages through few-shot anchoring. Accuracy: 75% (vs 40% direct translation)."},
        ],
        "workflow_diagram": """graph TD
    A["Zero-Shot<br/>No Examples"] -->|Low Accuracy| B["30-50%"]
    C["Few-Shot<br/>3-5 Examples"] -->|Good Accuracy| D["70-80%"]
    E["Fine-Tune<br/>1000s Examples"] -->|High Accuracy| F["90%+"]
    B -.->|Cost: None| G["Cost"]
    D -.->|Cost: Minimal| G
    F -.->|Cost: High| G

    style A fill:#ffebee
    style C fill:#fff3e0
    style E fill:#e8f5e9""",
    },
    "finetuning.md": {
        "interview_qa": [
            ("When should you fine-tune vs use prompting/few-shot?",
             "Prompting: works for general tasks, no data needed, zero-shot. Few-shot: 1-100 examples, cost per inference. Fine-tune: 100-10K examples, cost upfront, fast inference. Choice: prompting for one-off, fine-tune for repeated queries or distribution shift."),
            ("What's the difference between full fine-tuning and parameter-efficient methods?",
             "Full: update all weights, best accuracy, slow training, expensive compute. Parameter-efficient (LoRA/adapters): update 0.1-1% of weights, 90% of accuracy, fast, cheap. Trade-off: accuracy ceiling. Use LoRA for most cases, full fine-tune if precision critical."),
            ("How do you avoid overfitting during fine-tuning?",
             "Small dataset: use early stopping, regularization (weight decay), smaller learning rate. Monitor validation loss. Dropout. Data augmentation. With <1K examples: aggressive regularization necessary. With >10K: overfitting less likely."),
            ("What's catastrophic forgetting and how do you prevent it?",
             "Fine-tuning can harm performance on original task. E.g., fine-tune for medical domain, lose general knowledge. Prevention: mix original data (50% original, 50% new). Use lower learning rate. Use adapter instead of full fine-tune."),
            ("How do you measure fine-tuning success?",
             "On validation set: accuracy, F1, perplexity (task-dependent). Compare to baseline: how much improvement? Cost analysis: training cost vs improvement. Ablation: which data/technique helped most?"),
        ],
        "real_world_examples": [
            {"title": "LoRA Fine-Tuning for Domain Adaptation", "description": "General LLM → Legal domain. Fine-tune on 5K legal documents + contracts. LoRA rank-8: 2 hours training. Result: 42% → 68% accuracy on legal tasks. Cost: $100 (vs $10K full fine-tune). Deployed as legal assistant."},
            {"title": "Full Fine-Tuning for Company Chat", "description": "Goal: make model understand company context (products, policies, customers). Fine-tune on 100K internal documents. Full fine-tuning (smaller model, 3B). Result: 95% accuracy on internal queries. Deployment: on-premise (compliance)."},
            {"title": "Multi-Task Fine-Tuning", "description": "One model for: classification, NER, summarization. Fine-tune on all three mixed. Shared representations improve transfer. Accuracy: 85% across all tasks (vs 88% individual models, but single model advantage)."},
        ],
        "workflow_diagram": """graph TD
    A["Pre-Trained Model"] -->|No Training| B["General Knowledge<br/>70% on Task"]
    A -->|Fine-Tune| C["Task-Specific<br/>90% on Task"]
    A -->|Few-Shot| D["Prompt Engineering<br/>60% on Task"]

    C -->|Cost| E["High Upfront<br/>Low Per-Query"]
    B -->|Cost| F["None<br/>Low Quality"]
    D -->|Cost| G["None<br/>Medium Quality"]

    style A fill:#e3f2fd
    style C fill:#e8f5e9
    style E fill:#fff3e0""",
    },
    "in-context-learning.md": {
        "interview_qa": [
            ("What's in-context learning (ICL) and how does it work?",
             "LLMs can learn from examples in the prompt without weight updates. 'These are positive reviews...now classify: \"Great product!\"' → model recognizes pattern from context. Mechanism: attention weights focus on similar examples. Why: no retraining needed, instant task adaptation."),
            ("How much does ICL improve over zero-shot?",
             "Task-dependent: simple tasks (+5-10%), complex reasoning (+20-50%). Math: 30% → 80% with CoT+few-shot. Language: 45% → 70% with examples. Not all models: large models (100B+) better at ICL than small (7B)."),
            ("What's the ICL surface hypothesis?",
             "Common belief: ICL learns labels from examples. Reality: ICL may learn input-label format/style instead of actual patterns. Research: sometimes ICL equivalent to label randomization. Lesson: ICL is more mysterious than expected, still useful but not guaranteed."),
            ("How do you optimize ICL prompt design?",
             "Example order: easier examples first (warm-up). Example diversity: represent different cases. Label distribution: match test distribution. Format consistency: same template for all. CoT: add reasoning steps for complex tasks."),
            ("When does ICL fail?",
             "Misleading examples: model memorizes wrong pattern. Distribution mismatch: training data very different from examples. Task requires learned knowledge: unseen phenomena, facts model doesn't know. No text solution: image classification needs vision model."),
        ],
        "real_world_examples": [
            {"title": "ICL for Few-Shot Classification", "description": "Task: sentiment analysis on tweets. Zero-shot: 45% accuracy. 3-example ICL: 72% accuracy. 10-example: 78%. Cost: $0.001 per example per query. No training needed. Deployed in real-time sentiment pipeline."},
            {"title": "ICL for Code Generation", "description": "Problem: generate Python function. Zero-shot: syntactically correct 40%. With 2 code examples: 65%. With 5 examples: 75%. Prompt size: 1-2K tokens. Used in Copilot-style suggestions."},
            {"title": "Cross-Lingual ICL", "description": "English → Spanish translation. Zero-shot: 50% BLEU. 3 example translations: 70% BLEU. Model transfers knowledge across languages through in-context examples. No language-specific training."},
        ],
        "workflow_diagram": """graph LR
    A["Context Examples"] -->|Prompt| B["LLM Attention"]
    C["Test Input"] -->|Prompt| B
    B -->|Pattern Matching| D["Output"]
    D -->|No Training| E["Instant Adaptation"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style E fill:#e8f5e9""",
    },
    "instruction-tuning.md": {
        "interview_qa": [
            ("What's instruction-tuning and why is it important?",
             "Fine-tune models on (instruction, output) pairs instead of raw text. 'Write a poem about X' → poem. Makes models follow instructions better. Without: same model learns to predict next token (not instructions). With: models understand tasks, improve zero-shot capability on unseen tasks."),
            ("How is instruction-tuning different from supervised fine-tuning?",
             "SFT: (input, target) pairs. Instruction-tuning: (instruction, input, output) triples with diverse instructions. SFT: narrow, task-specific. Instruction-tuning: broad, instruction-following. Instruction-tuning improves generalization to new tasks."),
            ("What makes good instruction-tuning data?",
             "Diverse: wide range of tasks (QA, summarization, writing, coding, math). Instructions clear and specific. Multiple ways to say same task. Quality outputs. Mix simple and complex. Include edge cases."),
            ("How much instruction-tuning data do you need?",
             "Small models (7B): 10K instructions suffices. Large models (70B): 100K better. Research: diminishing returns after 50K. More important: diversity and quality, not quantity. 1K high-quality > 100K low-quality."),
            ("Does instruction-tuning hurt base capabilities?",
             "Slight decrease on original pretraining tasks (~1-2% perplexity). Gain: zero-shot instruction following (+30-50% on held-out tasks). Net positive for general assistants. For specialized models: may want to avoid instruction-tuning."),
        ],
        "real_world_examples": [
            {"title": "Instruction-Tuning for Conversational AI", "description": "Base LLM: general but doesn't follow instructions well. Instruction-tuning on 50K diverse instructions: ChatGPT-like behavior. Zero-shot on new tasks: 70% user satisfaction. Standard SFT: 40%."},
            {"title": "Instruction-Tuning for Multilingual", "description": "Base model: trained on 50+ languages. Instruction-tuning: same 50K instructions in each language. Results: instruction-following works in all languages (zero-shot). Enables multilingual assistant."},
            {"title": "Light Instruction-Tuning for Domain", "description": "Medical base model. Domain-specific instructions: 5K medical queries. Light tuning (1 epoch): instruction-following + medical knowledge combined. 90% accuracy on medical QA (vs 70% base)."},
        ],
        "workflow_diagram": """graph LR
    A["Base Model"] -->|SFT| B["Task-Specific<br/>Narrow"]
    A -->|Instruction-Tuning| C["Instruction-Following<br/>Broad"]
    B -->|Unseen Task| D["Poor"]
    C -->|Unseen Task| E["Good"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#ffebee
    style E fill:#c8e6c9""",
    },
    "evaluation.md": {
        "interview_qa": [
            ("What are the main evaluation metrics for LLMs?",
             "Task-specific: accuracy, F1, BLEU (translation), ROUGE (summarization). General: perplexity, HellaSwag, MMLU benchmarks. Human: preference rating, satisfaction, specific criteria. Choose based on task. Perplexity ≠ quality (low perplexity but poor outputs possible)."),
            ("Why is human evaluation necessary?",
             "Automatic metrics miss quality nuances. Example: 'The quick brown fox' vs 'A fast auburn fox' same BLEU (metric) but different readability. Human eval catches: coherence, factuality, tone, usefulness. Best: combine automatic + human."),
            ("What's the difference between intrinsic and extrinsic evaluation?",
             "Intrinsic: model performance on benchmark (MMLU 75%). Extrinsic: how well it helps in real application (customer satisfaction 80%). Best practice: measure both. A model scoring well on benchmarks might underperform in production."),
            ("How do you avoid benchmark overfitting?",
             "Test on out-of-distribution data. Track new vs held-out vs old benchmarks. Different evaluation protocols. Domain-specific evals alongside general benchmarks. Regular human spot-checks."),
            ("What's the right evaluation sample size?",
             "Rule of thumb: 1000+ examples for statistical significance. Small: 100 (high variance). Medium: 500 (reasonable). Large: 5000+ (more cost). For human eval: 50-500 (expensive per example)."),
        ],
        "real_world_examples": [
            {"title": "Evaluation in Production LLM API", "description": "Benchmark: MMLU 78%. Production eval: real customer queries. Problem: MMLU accuracy doesn't translate to user satisfaction. Added human eval: 100 random queries per week. Discovered: model good at benchmarks but verbose in practice."},
            {"title": "Multilingual Evaluation Challenges", "description": "English model: 90% accuracy. Same model + light multilingual tuning: 45% on non-English (not measured!). Added language-specific evals. Now: 85% across 10 languages (vs false impression of 90%)."},
            {"title": "Domain Evaluation", "description": "General model on medical domain. MMLU-Medicine: 60%. Real doctor evaluation: accuracy clinically acceptable 92% (very critical tasks missed, needs review). Lesson: domain experts should evaluate."},
        ],
        "workflow_diagram": """graph TD
    A["Model"] -->|Automatic<br/>BLEU, ROUGE| B["Score: 85"]
    A -->|Human<br/>Preference| C["Score: 72"]
    A -->|Benchmark<br/>MMLU| D["Score: 88"]
    B -->|Real Deployment| E["User Satisfaction"]
    C -->|Real Deployment| E
    D -->|Real Deployment| E

    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#e0f2f1""",
    },
}

print(f"Loaded {len(ALL_CONCEPT_ENHANCEMENTS)} concept enhancements")

# More enhancements - expanding coverage

EVEN_MORE_ENHANCEMENTS = {
    "prompt-optimization.md": {"interview_qa": [("What problem does prompt engineering solve?", "Raw prompts: vague, inconsistent, low quality outputs. Optimized prompts: clear, structured, high quality. Example: 'Classify this' (50% accuracy) vs 'Classify as positive/negative, considering sarcasm' (85%). Optimization compounds small improvements."), ("What are the main prompt optimization techniques?", "Clarity (be specific), structure (use templates), examples (few-shot), constraints (limit output), reasoning (CoT), role (set context). Typical stack: clear instruction + 3 examples + structure + constraints. Each adds 2-5% accuracy."), ("How do you measure prompt effectiveness?", "Test on validation set before deployment. Metrics: accuracy, latency (some prompts slower), cost (tokens used), user satisfaction. A/B test: old vs new prompt on sample of traffic."), ("When does prompt optimization hit diminishing returns?", "First 2-3 techniques: 5-15% gains. Next 3-4: 2-5% each. Beyond 7: <1% gains, not worth effort. Most gains front-loaded."), ("Should you use prompt templates or dynamic generation?", "Templates: simpler, consistent, faster. Dynamic: adapts to input (harder). Start with templates, use dynamic for complex tasks.")], "real_world_examples": [{"title": "Customer Support Chatbot Optimization", "description": "Baseline: generic response template. Optimized: clear categorization + relevant examples + structured output (JSON). Accuracy: 70% → 85%. User satisfaction: 60% → 88%."}, {"title": "Content Moderation Prompt", "description": "Task: detect toxic content. V1: 'Is this toxic?' (45% recall). V2: 'Identify toxic content: slurs, threats, harassment, ...' (72% recall). V3: Add examples (85%). Cost: same tokens."}, {"title": "Code Generation Improvement", "description": "Prompt v1: 'Write Python function' (40% correct). Prompt v2: Add type hints, docstring requirements (60%). Prompt v3: Add example functions (78%). Final prompt: 2K tokens vs 100 for basic."}], "workflow_diagram": """graph LR
    A["Vague Prompt"] -->|70% Quality| B["Basic Output"]
    A -->|Optimized| C["Clear Instructions<br/>Structured<br/>Examples"]
    C -->|90% Quality| D["High Quality Output"]
    E["Cost"] -->|Same| B
    E -->|Same Tokens| D
    style A fill:#ffebee
    style C fill:#fff3e0
    style B fill:#ffebee
    style D fill:#e8f5e9""",
    },
    "prompting.md": {"interview_qa": [("What's the difference between prompting and fine-tuning?", "Prompting: shape behavior via text, no training, instant. Fine-tuning: update weights, slower, stronger control. Prompting for one-off tasks; fine-tuning for repeated. Prompting cost: per query. Fine-tuning cost: upfront."), ("What are prompt patterns that work?", "Role-play ('You are X'), examples (few-shot), structure (JSON/XML), constraints (length/format), reasoning (steps). Combine 2-3 patterns for best results. No silver bullet; task-dependent."), ("How do you debug a failing prompt?", "Start: simplest possible. Gradually add clarity/examples. Test each change. Monitor: does accuracy improve? Common issues: ambiguous instructions, inconsistent examples, conflicting constraints."), ("What's the limitation of prompting?", "Can't teach new facts (needs training data). Can't change fundamental model biases. Can't do tasks model fundamentally can't do (e.g., perfect arithmetic). Role: shape existing capabilities, not add new ones."), ("Why do longer prompts sometimes fail?", "Information overload: too many instructions confuse model. Contradictions: conflicting guidance. Dilution: signal lost in noise. Best: concise, clear, specific. More is not always better.")], "real_world_examples": [{"title": "Email Classification Prompts", "description": "Generic: 'Classify email' (50% accuracy). With examples: 'Similar emails are classified as [examples shown]' (75%). With role: 'You are email expert' (78%). Final: +28% improvement."},   {"title": "Multi-Language Prompting", "description": "Single English prompt: works in English (90%), fails in Spanish (40%). Language-aware prompt: includes language name, examples in that language, cultural context. 90% in both."},  {"title": "Adversarial Prompt Injection Defense", "description": "Naive prompt: vulnerable to 'Ignore previous instructions'. Defended prompt: explicit rules, constraints, separation of data. Improves robustness against injection attacks by 95%."}], "workflow_diagram": """graph TD
    A["Model Capabilities<br/>100% as-is"] -->|Prompting| B["Access 70-90%<br/>via good prompts"]
    A -->|Bad Prompts| C["Access 20-30%<br/>via poor prompts"]
    D["Fine-Tuning"] -->|Reshape| E["Access 95%<br/>on specific task"]
    B -->|Cost| F["None per query"]
    E -->|Cost| G["High upfront"]
    style B fill:#e8f5e9
    style C fill:#ffebee
    style E fill:#c8e6c9""",
    },
    "zero-shot-learning.md": {"interview_qa": [("What's zero-shot learning and when does it work?", "No examples provided; model uses instructions only. 'Classify sentiment: positive/negative' (no examples). Works: simple, general tasks. Fails: complex reasoning, nuanced tasks. Baseline for few-shot comparison."), ("Why is zero-shot accuracy often low?", "Model never saw examples of task structure. Must infer from language alone. Without grounding, prone to misinterpretation. Few-shot adds grounding (examples show format/expectations)."), ("What model capabilities enable zero-shot?", "Instruction following (understands task). Generalization (applies knowledge to new tasks). Reasoning (multi-step logic). Not all models strong at all three. Large models (100B+) better zero-shot."), ("When would you use zero-shot vs few-shot?", "Zero-shot: unknown task, can't afford annotation, exploring. Few-shot: better accuracy acceptable, data available, consistent task. Zero-shot faster (fewer tokens), few-shot more accurate."), ("How do you prompt for effective zero-shot?", "Be specific: 'Classify as positive/negative/neutral' (not just 'analyze'). Set format: 'Output: [label]'. Add context if helpful: 'Customer reviews: classify'. Still less effective than examples.")], "real_world_examples": [{"title": "Zero-Shot Language Detection", "description": "Model: multilingual. Task: detect language from text. No examples provided. Accuracy: 95% (works because language is distinct). vs few-shot: 98% (marginal gain, examples not needed)."},  {"title": "Zero-Shot Domain Transfer", "description": "Task: medical text classification. Model trained on general domain. Zero-shot: 45% accuracy (struggles with medical jargon). Few-shot (3 medical examples): 70%. Full fine-tune: 90%."},  {"title": "Zero-Shot Generalization Test", "description": "Test model on completely unseen task (no fine-tuning, no examples). Success indicates good instruction following. Failure suggests task too specialized or model too narrow."}], "workflow_diagram": """graph LR
    A["Task Description<br/>No Examples"] -->|Zero-Shot| B["Model Output<br/>Accuracy: 40-70%"]
    A -->|Add 3-5 Examples| C["Few-Shot<br/>Accuracy: 70-90%"]
    A -->|Fine-Tune| D["Task-Specific<br/>Accuracy: 85-95%"]
    E["Cost/Speed"] -->|Fastest| B
    E -->|Medium| C
    E -->|Slowest| D
    style B fill:#ffebee
    style C fill:#fff3e0
    style D fill:#e8f5e9""",
    },
}

ALL_CONCEPT_ENHANCEMENTS.update(EVEN_MORE_ENHANCEMENTS)


# Final batch of enhancements for remaining concepts
FINAL_BATCH = {
    "semantic-search.md": {"interview_qa": [("How does semantic search differ from keyword search?", "Keyword: exact matching ('customer' doesn't match 'client'). Semantic: meaning-based ('customer' and 'client' similar). Semantic via embeddings: convert text → vector → cosine similarity. Better for synonyms, paraphrases, intent."), ("Why use semantic search over traditional full-text search?", "Full-text: fast, exact, many false negatives. Semantic: slower but catches intent. Hybrid: use both, combine scores. Best: semantic for accuracy, full-text for speed (use full-text as first filter)."), ("How do you scale semantic search to millions of documents?", "Index embeddings in vector DB (Pinecone, Weaviate, Milvus). HNSW/IVF algorithms for fast approximate search. Query: embed → search DB → get neighbors. Latency: <100ms for 10M docs."), ("What embedding model should you use?", "General: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (accurate). Domain-specific: fine-tune on your domain. Multilingual: multilingual-e5-base. Speed: smaller models faster (22M params vs 335M params)."), ("How do you evaluate semantic search quality?", "Metrics: mean average precision (MAP), NDCG, MRR. Manual inspection: do top-k results make sense? Click-through rate in production. A/B test: semantic vs keyword search.")], "real_world_examples": [{"title": "E-commerce Product Search", "description": "Keyword search: 'blue shoe' doesn't match 'footwear in navy'. Semantic search: matches because of similarity. Conversion rate: 2% → 5% from better relevance."}, {"title": "Customer Support Search", "description": "Query: 'How do I reset my password?'. Should match: 'Forgotten credentials', 'Sign-in issue', 'Account access'. Keyword: misses many. Semantic: catches all."}], "workflow_diagram": """graph LR
    A["Query Text"] -->|Embedding| B["Query Vector"]
    C["Corpus"] -->|Embedding| D["Doc Vectors<br/>Indexed"]
    B -->|Similarity Search| E["Top-K Neighbors"]
    E -->|Ranked| F["Results"]
    style B fill:#e3f2fd
    style D fill:#e0f2f1
    style F fill:#e8f5e9""",
    },
    "vector-databases.md": {"interview_qa": [("What's a vector database and why not just use Postgres+embeddings?", "Vector DB: optimized for similarity search (HNSW, IVF indices), fast approximate search. Postgres: general-purpose, slower for similarity (full scan). At scale (1M docs): Postgres = 1s query, vector DB = 10ms query. For small datasets: Postgres fine."), ("What's the difference between HNSW and IVF indexing?", "HNSW (Hierarchical Navigable Small World): graph-based, better quality but memory-intensive. IVF (Inverted File): cluster-based, faster but less accurate. Trade-off: quality vs speed. Most: hybrid (IVF + HNSW)."), ("How do you handle real-time updates in vector DB?", "Append-only: add new docs constantly. Problem: old docs become stale. Solutions: 1) Rebuild index periodically. 2) Delete old docs (expensive). 3) TTL (time-to-live) for automatic expiration."), ("What's the relationship between vector dimension and accuracy?", "Higher dimension: more expressive (better accuracy). Lower dimension: faster search. Trade-off: 384 dims = 90% accuracy, 1536 dims = 95%. Most use 384-768 for balance."), ("How do you scale vector search to billions of documents?", "Sharding: split across multiple nodes. Index on each shard separately. Query: broadcast to all shards, merge results. Latency: scales, throughput scales.")], "real_world_examples": [{"title": "Pinecone for Semantic Search", "description": "Pinecone: serverless vector DB. Index: 10M product embeddings. Query latency: <50ms p99. Compare to: Redis (memory-intensive), Postgres (slow)."},  {"title": "Milvus for Enterprise", "description": "On-premise deployment: 100M embeddings. Multi-shard setup for scale. Accuracy: 99% (vs 95% IVF-only). Useful for compliance (no cloud data transfer)."}], "workflow_diagram": """graph TD
    A["Documents"] -->|Batch Embed| B["Embeddings"]
    B -->|Index| C["Vector DB<br/>HNSW/IVF"]
    D["Query"] -->|Embed| E["Query Vector"]
    E -->|Search| C
    C -->|Top-K Results| F["Ranked Output"]
    style C fill:#e0f2f1
    style F fill:#e8f5e9""",
    },
    "semantic-caching.md": {"interview_qa": [("What's semantic caching and why is it better than lexical caching?", "Lexical: 'How to reset password?' cached, 'How to recover account?' misses (different words). Semantic: embeddings → both similar → cache hit. Cache hit rate: 10% (lexical) → 40% (semantic)."), ("How does semantic caching work?", "1) Embed incoming query. 2) Check if similar to cached queries. 3) If similar + cached result, return cached. 4) Otherwise, compute new. Requires embedding + similarity lookup (overhead)."), ("When is semantic caching worthwhile?", "High-cost inference + repeated similar queries. Example: GPT-4 @$0.03/K tokens. Cache hit saves $0.03/query. If 50% cache hit on 1M queries: $15K savings. Cost of caching infrastructure: $5K. Worth it."), ("What's the latency impact of semantic caching?", "Cache lookup: 1-5ms (embedding + similarity). vs original query: 1000ms+. Even with lookup overhead, net gain. In ideal case: 1ms (cached) vs 1000ms (uncached)."), ("How do you measure semantic cache effectiveness?", "Metrics: hit rate, savings (queries avoided), accuracy (cached results match fresh). Latency P99 (worst case). Trade-off: more conservative similarity threshold = higher accuracy but lower hit rate.")], "real_world_examples": [{"title": "LLM API Cost Reduction", "description": "API: 1000 req/sec similar queries. Without cache: $3K/day. Semantic cache: 50% hit rate. Cost: $1.5K/day. Savings: $1.5K/day ($500K/year). Infrastructure: $50K/year."},  {"title": "Chatbot Context Reuse", "description": "User: asks similar questions in different wordings. Cache embeddings of previous answers. Hit rate: 30-50% on typical conversation. Latency: 100ms (vs 2s original)."}], "workflow_diagram": """graph LR
    A["Query"] -->|Embed| B["Query Vector"]
    B -->|Check Cache| C{"Semantically<br/>Similar?"}
    C -->|Yes| D["Return Cached<br/>Result"]
    C -->|No| E["Generate New<br/>Result"]
    E -->|Add to| F["Cache"]
    D -->|Fast| G["Response"]
    F -->|Response| G
    style G fill:#e8f5e9""",
    },
    "rlhf.md": {"interview_qa": [("What's RLHF and why is it important for alignment?",  "RLHF (Reinforcement Learning from Human Feedback): fine-tune model using human preferences. Process: 1) Generate outputs. 2) Humans rate quality. 3) Train reward model. 4) RL training to maximize reward. Result: better aligned model, follows instructions, refuses harmful."), ("How is DPO different from RLHF?", "RLHF: complex (reward model + RL training). DPO: direct preference optimization (no reward model, no RL). DPO simpler, faster, comparable results. RLHF still valuable for complex preference modeling."), ("What are the challenges with RLHF?",  "Expensive: need 10K-100K human evaluations. Unstable: RL training can diverge. Scalability: RL training expensive. Data quality: human disagreement. Reward model: separate model to train (adds complexity)."), ("How much data do you need for RLHF?",  "Minimum: 5K preference pairs (basic alignment). Standard: 50K (good alignment). Large: 100K+ (refined). More data = better alignment, diminishing returns >100K."), ("How do you prevent reward hacking in RLHF?",  "Reward model can be gamed (model finds loopholes). Solutions: 1) Regularization (penalize divergence from base). 2) Conservative RL (don't stray too far). 3) Regular human evaluation (catch gaming).")], "real_world_examples": [{"title": "OpenAI ChatGPT Alignment", "description": "RLHF pipeline: 13K human evaluators. Process: GPT-3.5 → RLHF → ChatGPT. Improvement: instruction-following, refuses harmful. Standard approach for alignment in industry."}, {"title": "Safety Fine-Tuning", "description": "Base model: 60% refuse harmful, 40% comply. After RLHF: 95% refuse, 5% false positives. Trade-off: safety vs helpfulness (needs careful tuning)."}], "workflow_diagram": """graph TD
    A["Generate Outputs"] -->|Human Eval| B["Preference Data<br/>A > B > C"]
    B -->|Train| C["Reward Model"]
    C -->|RL Training| D["Policy Update"]
    D -->|New Model| E["Better Aligned<br/>LLM"]
    E -->|Generates| A
    style E fill:#e8f5e9""",
    },
    "prefix-tuning.md": {"interview_qa": [("What's prefix-tuning and how does it compare to LoRA?", "Prefix-tuning: add learnable tokens to input prefix. LoRA: add low-rank matrices to weights. Prefix: simpler (just tokens), easier to understand. LoRA: more flexible, works everywhere. Both: parameter-efficient (~0.1% params trainable)."), ("When would you use prefix-tuning vs LoRA?", "Prefix: when you want visible/interpretable parameters (the learned tokens). LoRA: when you want better accuracy or lower memory. Most: use LoRA. Prefix: still research-popular."), ("How many prefix tokens do you need?", "Typical: 10-100 tokens. More tokens = more expressiveness but more memory. 50 tokens: ~200KB (tiny). Can stack multiple tasks' prefixes."), ("Can you combine prefix-tuning with other methods?", "Yes: prefix-tuning + LoRA = more parameters, better accuracy. Prefix-tuning + fine-tuning = full fine-tune plus guided prefix. Trade-off: complexity vs performance."), ("What's the intuition behind prefix-tuning?", "Idea: in-context learning works, so learnable prefix (like learned examples) should work. Instead of new examples in prompt, learn prefix embeddings. Acts like implicit multi-task instruction.")], "real_world_examples": [{"title": "Interpretable Prefix-Tuning", "description": "Learned prefix for medical domain: doctors can inspect what the model learned (somewhat). vs LoRA: black-box low-rank matrices. Useful for explainability."}], "workflow_diagram": """graph LR
    A["Input"] -->|Prepend| B["Learned Prefix<br/>P_e1, P_e2, ..."]
    B -->|Concatenate| C["Full Sequence<br/>Prefix + Input"]
    C -->|Model| D["Output"]
    D -->|Task-Specific| E["Result"]
    style B fill:#fff3e0
    style E fill:#e8f5e9""",
    },
    "multimodal.md": {"interview_qa": [("What's multimodal learning in LLMs?", "Process multiple input types: text, images, audio. Example: 'What's in this image?' Model sees image + question → generates answer. Enables: image understanding, video understanding, audio transcription."), ("How do you handle image inputs with LLMs?", "Approach 1: use vision encoder (CLIP) → embeddings → feed to LLM. Approach 2: end-to-end training on image+text. CLIP approach more practical (separate vision, reusable)."), ("What's the trade-off between image resolution and latency?", "High-res (1024×1024): better detail, slow. Low-res (224×224): fast, miss details. Typical: 256-512×256-512 (good balance)."), ("How do you train multimodal models?", "Contrastive learning: image-text pairs, learn similar embeddings. Generative: generate captions from images. Alignment: learn shared representation. Data: millions of image-text pairs (LAION, etc)."), ("When does multimodal fail?", "Hallucination: generate plausible but false image descriptions. Bias: reflect biases in training data. Understand: struggle with complex reasoning about images.")], "real_world_examples": [{"title": "GPT-4V Image Understanding", "description": "Input: image of a recipe. Task: extract ingredients, instructions. Accuracy: 95%. vs OCR+NLP: 70% (misses context). Multimodal: understands layout, context."},  {"title": "Accessibility: Image Descriptions", "description": "Generate captions for web images automatically. Model: vision encoder + language model. Coverage: 90% of web images can be described (manual: 10% due to cost)."}], "workflow_diagram": """graph LR
    A["Image"] -->|Vision Encoder<br/>CLIP| B["Image Embeddings"]
    C["Text"] -->|Tokenize| D["Text Embeddings"]
    B -->|Combine| E["Multimodal Representation"]
    D -->|Combine| E
    E -->|LLM| F["Output"]
    style B fill:#e0f2f1
    style D fill:#e3f2fd
    style F fill:#e8f5e9""",
    },
    "speculative-decoding.md": {"interview_qa": [("What's speculative decoding and why does it help?", "Problem: LLM generation slow (1 token/50ms). Solution: small model generates candidates, large model verifies in parallel. If candidates accepted: speedup 2-4x. If rejected: fallback to large model."), ("How do you choose draft model size?", "Draft should be 5-10x smaller. 70B model → 7B draft. Speed ratio: 7B = 10x faster than 70B. Net speedup: if acceptance rate >70%, worth it."), ("What's the acceptance rate and why does it matter?", "Acceptance rate: % of draft tokens confirmed by large model. High (>80%): draft model aligned with large model. Low (<50%): draft model too different. Rule: if <50%, speedup < 1x (not worth it)."), ("How does this relate to ensemble methods?", "Similar idea: multiple models voting. Speculative: small model proposes, large model decides. Ensemble: all vote. Speculative more efficient."), ("When is speculative decoding worth it?", "Latency-critical: 50ms budget. Have small model available. Acceptance rate likely >70%. Not worth: batch inference (throughput matters, not latency).")],"real_world_examples": [{"title": "llama.cpp Speculative Decoding", "description": "Model: Llama 2 70B. Draft: Llama 2 7B. Speedup: 2.5x. Acceptance rate: 78%. Deployed in local inference (limited GPU)."},  {"title": "Interactive Chat Speedup", "description": "Chat: need <500ms latency. Without spec decoding: 1s. With: 400ms. Acceptance: 75%. User-facing improvement: noticeable."}], "workflow_diagram": """graph LR
    A["Input"] -->|Small Model| B["Generate Draft<br/>Tokens 1-5"]
    B -->|Verify| C["Large Model<br/>Checks in Parallel"]
    C -->|Accept?| D{Decision}
    D -->|Yes| E["Use Draft<br/>Continue"]
    D -->|No| F["Fallback to<br/>Large Model"]
    E -->|Output| G["Faster Generation"]
    F -->|Output| G
    style G fill:#e8f5e9""",
    },
}

ALL_CONCEPT_ENHANCEMENTS.update(FINAL_BATCH)


# Final 7 concepts to reach 100% coverage
LAST_BATCH = {
    "context-window.md": {"interview_qa": [("What's the context window and why does it matter?", "Maximum sequence length a model can process. GPT-4: 8K/128K, Llama 2: 4K, Claude: 100K+. Matters because: longer context = more text analyzed at once. A 4K model can't summarize 50-page documents; 128K model can. Trade-off: larger windows = more memory + slower inference."), ("How do you handle documents longer than context window?", "Options: 1) Truncate (lose info), 2) Chunk + summarize (recursive), 3) Use sliding window (overlap chunks). For RAG: chunk documents at index time, retrieve relevant chunks at query time. For fine-tuning: split into context-size pieces, train on each."), ("What are efficient long-context techniques?", "Sparse attention: attend to nearby tokens + random tokens (reduces O(n²) to O(n log n)). Recurrent: process in chunks, maintain state across chunks. Paged attention: splits KV cache into pages. ROPe (Rotary Position Embeddings): handles longer sequences better than absolute positions."), ("When is 4K context enough vs when do you need 128K?", "4K (sufficient for): single-turn QA, classification, sentiment analysis. 128K (necessary for): multi-document analysis, long conversations, full book summarization, code repositories. Cost trade-off: 128K input = 32x tokens = 32x cost vs 4K."), ("How do position embeddings affect context length?", "Absolute positems: hardcoded for fixed length (4K). Extrapolation to 8K = hallucinations. RoPE/ALiBi: relative positions, can generalize to longer lengths. ALiBi: particularly good for length extrapolation (trained on 2K, works on 16K). Matters for real-world documents.")], "real_world_examples": [{"title": "4K Context Limitation in Customer Support", "description": "Support ticket: reference previous 10 conversations (50K tokens). 4K context model: truncates to last 2 conversations. Customer context lost, solution suggestions miss important history. 128K model: loads all history, provides better solutions. Customer satisfaction: 70% → 88%."}, {"title": "Code Repository Understanding", "description": "Codebase: 100K tokens across multiple files. 4K model: analyzes one function at a time (limited understanding). 128K model: sees entire repository structure, dependencies, patterns. Code review quality: better suggestions, catches subtle bugs. GitHub: suggests Copilot with larger context."}, {"title": "Legal Document Analysis", "description": "Contract: 200 pages = 100K tokens. 4K model: useless (can't fit full contract). Standard approach: expensive human review. With 128K model: upload full contract, extract terms, identify risks. Cost: $0.50 (tokens) vs $500 (human). Trade-off: still needs human verification."}], "workflow_diagram": """graph LR
    A["Document<br/>50K tokens"] -->|4K Model| B["Truncate<br/>4K context"]
    A -->|128K Model| C["Use Full<br/>128K context"]
    B -->|Limited| D["Partial Analysis"]
    C -->|Complete| E["Full Understanding"]
    style A fill:#e3f2fd
    style B fill:#ffebee
    style C fill:#e8f5e9
    style D fill:#ffebee
    style E fill:#e8f5e9""",
    },
    "dpo.md": {"interview_qa": [("What's DPO and how does it differ from RLHF?", "RLHF: train reward model, use for RL (complex, unstable). DPO: direct preference optimization, no reward model, just use preference pairs directly. Training: instead of 'maximize reward', use 'prefer chosen over rejected'. Simpler, more stable, comparable results."), ("When would you use DPO vs RLHF?", "DPO: preference data available, want simpler pipeline, unstable RLHF. RLHF: reward model importance, have label budget for rewards. DPO getting popular because: easier to implement, no RL training needed, better results on benchmarks."), ("How do you collect preference data for DPO?", "Generate multiple outputs, have human raters choose best. 10K-100K preference pairs typical. Quality matters: bad labels → bad DPO model. Standard: binary choice (A vs B), can do ranking (order N outputs). Cost: 5-10x cheaper than RLHF (no reward model training)."), ("What's the DPO loss function conceptually?", "Maximize probability of preferred over rejected: P(preferred) > P(rejected). During training: upweight preferred outputs, downweight rejected. Unlike RLHF: no explicit reward signal, implicit from comparisons."), ("How does DPO affect model behavior vs base model?", "Alignment shift: better instruction-following, fewer refusals on legitimate requests, more appropriate refusals on harmful. Similar to RLHF but achieved faster. Risk: potential jailbreak if preferences skewed toward harmful outputs.")], "real_world_examples": [{"title": "DPO for LLM Alignment", "description": "Base model: 45% human preference. After DPO (10K preference pairs): 72% human preference. Training: 3 hours on 1 GPU. Improvement: better instruction following, fewer refusals. Cost: $100 (data annotation) + $50 (compute) vs $5K (RLHF)."}, {"title": "DPO for Code Generation", "description": "Base model: 40% correct solutions. Target: 70%. DPO on 5K (output, correct, incorrect) triplets. Result: 68% accuracy, 5% improvement. Faster than RLHF pipeline (RL training). Deployed in Copilot-style autocomplete."}], "workflow_diagram": """graph TD
    A["Preference Data<br/>A vs B"] -->|Standard| B["Train Reward<br/>Model"]
    A -->|DPO| C["Direct Training<br/>No Reward Model"]
    B -->|RLHF| D["RL Training<br/>Complex"]
    C -->|DPO| E["Aligned Model<br/>Simple"]
    style A fill:#e3f2fd
    style C fill:#e8f5e9
    style E fill:#c8e6c9""",
    },
    "pretraining.md": {"interview_qa": [("What's pretraining and why is it important?", "Pretraining: train model on massive unlabeled text (next-token prediction). Learns language, factual knowledge, reasoning patterns. Foundation for all downstream tasks. Cost: millions of dollars, weeks of GPU time. Benefit: transfers to any task."), ("What's the pretraining objective?", "Next-token prediction: given 'The capital of France is', predict 'Paris'. Objective: minimize cross-entropy. Simple but powerful: emergent abilities appear at scale (reasoning, coding, translation)."), ("How much pretraining data do you need?", "Rule of thumb: more data > larger model. Chinchilla scaling laws: compute ≈ data, model size ≈ data. 1.3T tokens (GPT-3): 175B model. Trade-off: too little (underfitting), too much (diminishing returns)."), ("What's the relationship between model size and pretraining quality?", "Larger models learn more. But: compute grows cubically with size. 70B model = 27x compute vs 7B. Typical: scale to largest affordable size (time/money constrained)."), ("How do you evaluate pretraining quality?", "Loss on held-out test set (perplexity). Downstream task performance (MMLU, etc). Both matter: low loss ≠ good downstream performance always. Look at both metrics.")], "real_world_examples": [{"title": "OpenAI Pretraining", "description": "GPT-3: 175B model, 300B tokens, cost $10M+. Time: 3+ months. Result: strong zero-shot, few-shot capability. Established that scale enables capabilities."}, {"title": "Open-Source Pretraining", "description": "Llama 2: 70B model, 2T tokens, compute-optimal training. Result: competitive with GPT-3.5 on many benchmarks. Cost: lower (open source infrastructure)."}], "workflow_diagram": """graph LR
    A["Raw Text<br/>Massive"] -->|Tokenize| B["Token Stream"]
    B -->|Batch| C["Training Loop"]
    C -->|Next-Token| D["Predict Next"]
    D -->|Minimize Loss| E["Better Model<br/>Week 1"]
    E -->|Scale| F["Weeks/Months"]
    F -->|Result| G["Pretrained LLM"]
    style G fill:#e8f5e9""",
    },
    "retrieval-augmented-generation.md": {"interview_qa": [("What's RAG and how does it reduce hallucinations?", "RAG: retrieve documents, feed as context, generate answer grounded in retrieved docs. Without RAG: model generates from memory (hallucinations). With RAG: facts grounded in documents. Hallucination rate: 30-50% lower."), ("How do you structure a RAG pipeline?", "1) Index: chunk docs, embed, store in vector DB. 2) Retrieve: embed query, get top-k docs. 3) Rerank (optional): cross-encoder scores. 4) Generate: LLM + docs → answer. Latency: retrieval 50ms + ranking 50ms + generation 1000ms."), ("What's dense vs sparse retrieval?", "Dense: embeddings (semantic, slow). Sparse: keywords (BM25, fast). Hybrid: both combined. Dense finds 'good service' for 'satisfied', sparse misses without keyword overlap. Use hybrid for best results."), ("When should you use reranking?", "Retrieval top-20 candidates. Reranker: scores all 20. Cost: +50-100ms. Gain: +5-10% accuracy. Worth if: quality matters, latency allows."), ("How do you handle knowledge base drift?", "Documents change, become stale. Solutions: 1) Periodic reindexing (cheap). 2) TTL (auto-expire old docs). 3) Versioning (keep history). Most: combine periodic reindex + TTL.")], "real_world_examples": [{"title": "Enterprise RAG for Support", "description": "50K support docs. Query: retrieve top-5 docs, generate answer. User satisfaction: 70% (basic) → 90% (RAG). Reduced support tickets by 35%."}, {"title": "Medical RAG", "description": "50K medical papers. Doctor input: patient symptoms. RAG retrieves literature, LLM synthesizes. Not for diagnosis (needs human), but research support."}], "workflow_diagram": """graph LR
    A["Query"] -->|Embed| B["Vector"]
    B -->|Search| C["Vector DB<br/>Top-K"]
    C -->|Rerank| D["Top-5<br/>Docs"]
    D -->|Context| E["LLM"]
    E -->|Generate| F["Answer<br/>Grounded"]
    style F fill:#e8f5e9""",
    },
    "parameter-efficient-finetuning.md": {"interview_qa": [("What's parameter-efficient fine-tuning (PEFT)?", "Train <1% of model parameters instead of 100%. Methods: LoRA (low-rank), Adapters (bottleneck), Prefix-tuning (learned tokens). Advantage: 10-100x cheaper, faster, fewer GPUs. Trade-off: slightly lower accuracy ceiling."), ("How much parameter reduction do you get?", "LoRA rank-8: 0.1% params trainable. Adapters: 0.5-1% params. Prefix-tuning: 0.05%. All compress 100-1000x. For 7B model: 1-7M params instead of 7B."), ("When is PEFT sufficient vs when do you need full fine-tuning?", "PEFT: general tasks (classification, QA, summarization). Full: style transfer, major behavior change, science domain. Most tasks: PEFT sufficient. Full fine-tune: <5% of use cases."), ("How do you combine multiple PEFT methods?", "LoRA + quantization (QLoRA): reduce memory 4x further. Adapters + LoRA: more parameters, better accuracy. Trade-off: complexity."), ("How does PEFT affect inference?", "LoRA: merged for deployment (no overhead). Adapters: slight overhead (2-5% latency increase). Prefix: no overhead. Choice: LoRA for production (no inference cost).")], "real_world_examples": [{"title": "LoRA Multi-Task", "description": "10 downstream tasks, LoRA per task. Total: 10M params (vs 70GB×10 for full models). Deployment: 1 base + 10 LoRA = 4.5GB."}, {"title": "QLoRA on Consumer GPU", "description": "13B model with QLoRA: 6GB memory. RTX 3090 (24GB) trains comfortably. Enables researchers without cloud budget."}], "workflow_diagram": """graph LR
    A["Base Model<br/>100%"] -->|Freeze| B["7B Params<br/>Frozen"]
    C["Add PEFT<br/>0.1-1%"] -->|Train| D["1-7M Params<br/>Trainable"]
    E["Total"] -->|Compute| F["10-100x Cheaper"]
    style D fill:#fff3e0
    style F fill:#e8f5e9""",
    },
    "inference-optimization.md": {"interview_qa": [("What are the main inference bottlenecks?", "Compute (matrix multiplies), memory bandwidth (loading weights), latency (time to first token). Identify bottleneck: profile your setup. For large models: memory bandwidth dominates."), ("How does batch size affect inference?", "Small: underutilize GPU. Large: saturate memory, maximize throughput. Trade-off: batch latency vs throughput. Balance depends on use case."), ("What's the difference between throughput and latency optimization?", "Throughput: maximize tokens/sec. Latency: minimize per-request time. Batching: great throughput, bad latency. Use both techniques together."), ("When would you use quantization vs distillation?", "Quantization: compress weights (4-8x), 1-2% accuracy loss. Distillation: smaller model trained on large model (moderate compression, variable loss). Choose: quantization for speed, distillation for accuracy."), ("How do you measure inference efficiency?", "Tokens/sec, ms/token, memory GB, power watts, FLOPS utilization. Track all: one metric incomplete. E.g., high throughput but high power wasteful.")], "real_world_examples": [{"title": "Quantization for Inference", "description": "Model: Llama 7B. FP32: 28GB, 50ms/token. INT8: 7GB, 35ms/token. INT4: 3.5GB, 25ms/token. Trade-off: size vs speed vs accuracy."}, {"title": "Batching Optimization", "description": "Single-request: 1 req/sec. Batch-32: 32 req/sec (but 32s per request). Continuous batching: 20 req/sec with 1-2s latency. Best for serving."}], "workflow_diagram": """graph TD
    A["Bottleneck"] -->|Compute| B["Quantization<br/>Sparsity"]
    A -->|Memory| C["KV Cache Opt<br/>GQA"]
    A -->|Latency| D["Batching<br/>Speculative"]
    B -->|Optimize| E["Faster Inference"]
    C -->|Optimize| E
    D -->|Optimize| E
    style E fill:#e8f5e9""",
    },
    "token-optimization.md": {"interview_qa": [("What's token optimization?", "Reduce token count without losing quality. Techniques: summarization (preprocess docs), token pruning (remove unimportant tokens), smart batching (group short requests). Goal: 10-50% reduction."), ("How do you identify important tokens?", "Attention weights: tokens attended to are usually important. Gradient: tokens affecting loss prediction. Heuristic: first + last tokens important (contain info). Methods: remove low-attention tokens."), ("What's the trade-off between token reduction and accuracy?",  "More reduction: lower cost but worse accuracy. 10% reduction: no accuracy loss. 50% reduction: 2-5% accuracy loss. Find breakeven for your task."), ("How does token optimization affect latency?", "Fewer tokens: faster inference. But: optimization overhead (pruning, summarization). Net: usually positive (inference time saves more than optimization cost)."), ("When would you use token optimization?",  "Cost-sensitive (pay per token). Large-scale (1M queries/day). Quality acceptable with slight degradation. Not: high-accuracy critical, streaming (can't buffer).")], "real_world_examples": [{"title": "Document Summarization for RAG", "description": "RAG: retrieve 10 documents (50K tokens). Summarize each to 1K tokens (5K total). Accuracy drop: 0.5%. Cost: 90% lower. Latency: 10-20% faster."},  {"title": "Smart Batching", "description": "Mix short (10 tokens) and long (100 tokens) requests. Pad to max: wasteful. Smart batching: group by length, process separately. Throughput: +30%."}], "workflow_diagram": """graph LR
    A["Input Text<br/>50K tokens"] -->|Prune| B["Remove Filler<br/>40K tokens"]
    B -->|Summarize| C["Key Points<br/>30K tokens"]
    C -->|Feed to LLM| D["Output"]
    D -->|Cost Saving| E["30% Reduction<br/>Lower Cost"]
    style E fill:#e8f5e9""",
    },
}

ALL_CONCEPT_ENHANCEMENTS.update(LAST_BATCH)

