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

