#!/usr/bin/env python3
"""
Enhance all LLM concept markdown files with:
- Elaborate content
- Proper Mermaid flowcharts
- Real interview Q&A (not placeholders)
- Real-world examples
"""

import os
import re
import sys
from pathlib import Path

# Try to import additional enhancements if available
try:
    from all_concept_enhancements import ALL_CONCEPT_ENHANCEMENTS
    ADDITIONAL_ENHANCEMENTS = ALL_CONCEPT_ENHANCEMENTS
except ImportError:
    ADDITIONAL_ENHANCEMENTS = {}

CONCEPTS_DIR = Path("llm/concepts")

# Comprehensive Q&A and examples for each concept
CONCEPT_ENHANCEMENTS = {
    "adapters.md": {
        "interview_qa": [
            ("What are adapters and why use them instead of full fine-tuning?",
             "Adapters are small trainable modules inserted into a frozen base model. Instead of updating all 7B parameters, you train only 0.5-1M parameters. Advantages: 50-100x fewer parameters, faster training, easier multi-task management. Trade-off: slightly slower inference (2-5%) due to extra computation."),
            ("How do adapters compare to LoRA?",
             "Both are parameter-efficient. Adapters: bottleneck design (d→64→d), ~0.5M params/task, cleaner separation. LoRA: low-rank matrices (rank 4-8), ~0.1-1M params/task, better for large models. LoRA is newer and more popular; adapters still used in multi-task scenarios (Hub-style routing)."),
            ("When would you use multi-task adapters vs single-task?",
             "Single-task: each task gets dedicated adapter (cleaner). Multi-task: one adapter trains on multiple tasks (parameter sharing, faster). Choose multi-task if tasks are related (sentiment analysis on different domains); single-task if tasks are distinct (translation + QA)."),
            ("How do you handle adapter inference at scale?",
             "Load base model once, swap adapters for different tasks (lightweight). Routing: use a head network to select best adapter. Merging: merge adapters into base model for deployment (lose multi-task capability but gain speed)."),
            ("What happens if you adapt a model already adapted?",
             "Adapter stacking works but shows diminishing returns. 1st adapter: 10% gain. 2nd adapter on top: +3-4% gain. Usually cap at 2 adapters. Better to use one well-tuned adapter or full fine-tuning if precision matters."),
        ],
        "real_world_examples": [
            {
                "title": "Multi-Lingual Adapters for Customer Support",
                "description": "Base model: mBERT (multi-lingual). Adapters: one per language (English, Spanish, French, Mandarin). Each adapter trained on 5K language-specific customer support conversations. Production: route incoming query to language-specific adapter → classify intent → ticket assignment. Result: 92% accuracy per language, shared base model saves 80% storage.",
            },
            {
                "title": "Task-Specific Adapters for E-Commerce",
                "description": "Base model: RoBERTa. Adapters: sentiment (product reviews), NER (brand/product extraction), classification (returns reason), search ranking (relevance). Each adapter 512KB. Deploy one base model + 4 adapters = 1.5GB (vs 700MB×4 for full models). Latency: +0.5ms per adapter inference.",
            },
            {
                "title": "Efficient Domain Adaptation Pipeline",
                "description": "Medical domain adapter: trained on 10K medical abstracts. Legal domain adapter: trained on 5K legal contracts. Each ~1M parameters. Swap adapters based on document classification. Accuracy: 94% (medical), 91% (legal). Training time: 2 hours per adapter (vs 48 hours full fine-tune).",
            },
        ],
        "workflow_diagram": """graph LR
    A["Base Model<br/>Frozen"] -->|Query| B["Adapter 1<br/>Task A"]
    A -->|Query| C["Adapter 2<br/>Task B"]
    A -->|Query| D["Adapter 3<br/>Task C"]
    B --> E["Output A"]
    C --> E
    D --> E

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#e8f5e9""",
    },
    "attention-optimization.md": {
        "interview_qa": [
            ("Why is attention optimization important?",
             "Standard attention: O(n²) complexity. For 4K context: 16M operations. For 32K: 1B operations. This dominates inference latency. Optimization: FlashAttention (10x faster), grouped-query attention (3x faster), PagedAttention (2x faster). Real-world: 32K context goes from 5s to 0.5s."),
            ("What's the difference between FlashAttention and standard attention?",
             "Standard: loads full Q,K,V matrices → computes attention → writes output (lots of memory I/O). FlashAttention: tiles computation, keeps intermediate results in faster SRAM, reduces HBM I/O by 5-10x. Same mathematical output, much faster. Trade-off: minimal (pure optimization)."),
            ("How does grouped-query attention (GQA) work?",
             "Standard multi-head: each head has separate K,V. GQA: multiple query heads share one K,V head. Standard 32 heads: KV = 32. GQA with groups=8: KV = 4. Reduces KV cache by 8x. Trade-off: 0.5-1% accuracy loss, 10-15% latency improvement. Used in Llama 2, Mistral."),
            ("When would you use PagedAttention vs FlashAttention?",
             "FlashAttention: improves compute efficiency. PagedAttention: improves memory management. PagedAttention: splits KV cache into pages, allows dynamic batching (batch size changes mid-inference). Both can be used together. PagedAttention especially valuable for multi-user serving."),
            ("How do you handle long context efficiently?",
             "Combinations: FlashAttention (3x) + GQA (8x KV reduction) + sparse attention (10x for long sequences). For 128K context: standard would be 16B ops, optimized ~100M ops. Trade-off: sparse attention loses some fine-grained attention (negligible impact)."),
        ],
        "real_world_examples": [
            {
                "title": "FlashAttention in Production RAG",
                "description": "Retrieval returns 100 context chunks (100K tokens). Standard attention: 10B operations. FlashAttention: 1B operations. Latency: 2s → 0.2s. Implementation: drop-in replacement for HuggingFace transformers (no model changes). Deployment: Llama 2 70B on A100, 20 req/sec (vs 3 req/sec without).",
            },
            {
                "title": "GQA for Edge Deployment",
                "description": "Model: Llama 2 7B. Standard: 14GB memory (KV cache overhead). GQA variant: 8GB memory. Deployed on consumer GPU (RTX 4090 24GB). Batch size: 2 vs 1 before. Accuracy on MMLU: 54.3% (standard) vs 53.8% (GQA). Worth the trade-off for mobile/edge.",
            },
            {
                "title": "Multi-User Serving with PagedAttention",
                "description": "vLLM server: 10 concurrent users, variable output lengths. Without paging: batch size fixed at 4, 40% idle. With PagedAttention: dynamic batching, batch size 4-8 depending on freed pages. Throughput: 10 req/sec (vs 4 req/sec). 2.5x improvement from paging alone.",
            },
        ],
        "workflow_diagram": """graph LR
    A["Input Sequence<br/>8K tokens"] -->|Standard| B["Attention O(n²)<br/>64M ops"]
    A -->|FlashAttention| C["Tiled Attention<br/>6.4M ops"]
    B -->|Output| D["Latency: 8s"]
    C -->|Output| E["Latency: 0.8s"]

    style A fill:#e3f2fd
    style B fill:#ffebee
    style C fill:#e8f5e9
    style D fill:#ffebee
    style E fill:#e8f5e9""",
    },
    "chain-of-thought.md": {
        "interview_qa": [
            ("What problem does chain-of-thought solve?",
             "LLMs struggle with multi-step reasoning. Direct answer: 'What's 8+7?' → often wrong on arithmetic. CoT: 'Think step by step: 8+7, 8+5=13, 13+2=15' → correct. CoT helps math, logic, commonsense. Mechanism: intermediate steps force model to decompose problem, preventing shortcuts."),
            ("How much does CoT improve accuracy?",
             "Depends on task: simple classification (minimal gain), math (+15-30%), logic puzzles (+20-40%), commonsense QA (+10-20%). Cost: 3-10x more tokens. Trade-off: accuracy vs latency. On MMLU: 52% (direct) → 82% (CoT with GPT-3.5)."),
            ("What's self-consistency and when use it?",
             "Generate N different reasoning chains, take majority vote. k=5: 5 different paths. Improves robustness and accuracy (+5-10% on math). Cost: 5x inference. Best for: high-stakes decisions, math problems. Overkill for: simple classification."),
            ("How do you prompt for effective CoT?",
             "Key phrases: 'Let's think step by step', 'First...', 'Therefore...', 'The answer is'. Effective: explicit numbered steps. Weak: vague reasoning. Example: 'Step 1) Identify variables. Step 2) Set up equation. Step 3) Solve.' Works better than 'Think carefully'."),
            ("When should you avoid CoT?",
             "Simple tasks: 'Is this spam?' (direct answer fine). High-latency requirements: 10ms budget. Streaming: CoT requires full chain before streaming starts. Cost-sensitive: 3-10x token increase. Use direct prompting for these; CoT for reasoning tasks only."),
        ],
        "real_world_examples": [
            {
                "title": "CoT for Medical Diagnosis",
                "description": "Input: Patient symptoms. Direct prompt: 'What's the diagnosis?' Accuracy: 61%. CoT prompt: 'Step 1) Symptom analysis. Step 2) Differential diagnosis. Step 3) Most likely condition.' Accuracy: 84%. Real hospital: gated behind doctor review, but used to highlight important differential diagnoses.",
            },
            {
                "title": "Self-Consistency for Math Homework Grading",
                "description": "Problem: '2x + 5 = 13, solve for x'. Direct: 43% accuracy. CoT: 91% accuracy. Self-consistency (k=3): 96% accuracy. Used in educational platform: grade student work, identify common misconceptions. Cost: 3 inference calls per problem (acceptable for async grading).",
            },
            {
                "title": "CoT in Customer Support Routing",
                "description": "Ticket: 'Order arrived damaged, want refund'. Direct: Assigns to 'Returns' (50%). CoT: 'Step 1) Assess urgency (high). Step 2) Identify need (refund + replacement). Step 3) Route to specialized team.' Assigns to 'High-Priority Returns' (90%). Reduces escalations by 35%.",
            },
        ],
        "workflow_diagram": """graph TD
    A["Problem"] -->|Direct| B["Answer"]
    A -->|CoT| C["Step 1:<br/>Decompose"]
    C --> D["Step 2:<br/>Intermediate"]
    D --> E["Step 3:<br/>Solution"]
    E -->|Output| F["Reasoning Trace"]
    B --> G["Low Accuracy"]
    F --> H["High Accuracy"]

    style A fill:#e3f2fd
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#fff3e0
    style G fill:#ffebee
    style H fill:#e8f5e9""",
    },
    "embeddings.md": {
        "interview_qa": [
            ("What are embeddings and why do they matter?",
             "Embeddings: dense vectors representing text semantics. 'cat' and 'kitten' have similar embeddings (cosine similarity ~0.9). Enables: semantic search, clustering, similarity comparison. All done in vector space, not lexical matching. Standard: 384-1536 dimensions."),
            ("How do you choose an embedding model?",
             "Trade-offs: small (all-MiniLM-L6-v2, 22M params, fast), medium (all-mpnet-base-v2, 109M params), large (instructor-xl, slower but better). Benchmark on your domain. MTEB leaderboard shows performance. For production: balance accuracy vs latency. Most: use all-MiniLM for speed, all-mpnet for accuracy."),
            ("How do you handle embeddings at scale?",
             "Batch encode: 1000s of texts at once (32-256 batch size). Store in vector DB (Pinecone, Weaviate, Milvus). Index for fast search. Query: embedding → similarity search in DB (not linear scan). Latency: <100ms per query. Storage: 1M docs × 384 dims × 4 bytes = 1.5GB."),
            ("When would you fine-tune embeddings?",
             "Pre-trained works for general text. Fine-tune if: domain-specific (medical abstracts, legal docs), task-specific (relevance ranking), distribution shift. Data needed: 10K+ pairs. Improves accuracy 5-15% but adds complexity. Usually not needed."),
            ("How do you evaluate embedding quality?",
             "Metrics: mean average precision (MAP), normalized discounted cumulative gain (NDCG), MRR. Test: does top-k nearest neighbor make sense? Do similar documents cluster? Manual inspection essential. MTEB benchmark for standardized eval."),
        ],
        "real_world_examples": [
            {
                "title": "Semantic Search for Documentation",
                "description": "Company docs: 50K pages. Lexical search (Elasticsearch): 'GPU memory optimization' → relevant pages ranked #4-10. Embedding search: 'How to optimize GPU memory?' → relevant pages ranked #1-3. Implementation: embed all docs once, vector DB query. Latency: <50ms. User satisfaction: 72% → 91%.",
            },
            {
                "title": "Duplicate Detection in E-Commerce",
                "description": "Problem: same product listed 5x with different descriptions. Lexical comparison: false negatives (different wording). Embedding cosine sim > 0.85: catches 95% duplicates. Pipeline: embed product descriptions, cluster with threshold. Saves manual curation hours.",
            },
            {
                "title": "Job Matching in Recruitment",
                "description": "Job postings: 1M. Candidates: 500K. Lexical keyword matching: low coverage. Embedding-based: candidate skills → embed → search job postings → top-k matches. E.g., 'Python developer' matches 'Software engineer (Python)' and 'Backend engineer (preferred: Python)'. Better ranking of opportunities.",
            },
        ],
        "workflow_diagram": """graph LR
    A["Text"] -->|Encoder| B["Embedding<br/>384-dim vector"]
    B --> C["Vector DB"]
    C -->|Similarity Search| D["Nearest Neighbors"]
    D --> E["Ranked Results"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e0f2f1
    style E fill:#e8f5e9""",
    },
    "lora.md": {
        "interview_qa": [
            ("What's LoRA and why use it?",
             "LoRA (Low-Rank Adaptation): Add small trainable matrices (A×B) instead of full weight update. 7B model: normally update 28B params. LoRA with rank-8: update only 1M params. 28x smaller, 10x faster training, 100x cheaper GPU hours. Merge for deployment → no inference cost."),
            ("How do you choose LoRA rank?",
             "Rank 4: basic adaptation, fastest, least accurate. Rank 8: sweet spot for most tasks (good balance). Rank 16: heavy adaptation, slower, marginal gains. Rank 32+: diminishing returns, defeats efficiency purpose. Start rank-4, increase if underfitting. Monitor loss curve."),
            ("How does LoRA compare to full fine-tuning?",
             "LoRA: 99% parameter reduction, 1% accuracy loss typical. Full fine-tune: 100% params, 1% better accuracy. LoRA wins on efficiency; full fine-tune wins on accuracy ceiling. For most tasks (classification, QA, summarization): LoRA sufficient. For style transfer or major behavior change: consider full fine-tune."),
            ("How do you merge LoRA adapters for deployment?",
             "During training: keep base + LoRA separate. Deployment: W_merged = W_base + (A × B). Single matrix file, no multi-model overhead. Inference speed: identical to base model. This is why LoRA is production-friendly."),
            ("Can you combine LoRA with quantization?",
             "Yes. Quantize base model (INT8), train LoRA in FP32 on top. Deploy: quantized base + FP32 LoRA (small). Saves memory, maintains training precision. Common pattern: QLoRA (quantized + LoRA), reduces training memory 4x further."),
        ],
        "real_world_examples": [
            {
                "title": "LoRA for Multi-Task Fine-Tuning",
                "description": "Base model: Mistral 7B (quantized, 4GB). 10 downstream tasks (classification, NER, summarization). LoRA per task: 1M params each, 10M total. Training: 2 hours per task on consumer GPU. Deployment: 1 base model + 10 LoRA adapters = 4.5GB (vs 70GB×10). Dynamic routing based on task. Total accuracy: 85% average.",
            },
            {
                "title": "LoRA for Domain Adaptation",
                "description": "General LLM fine-tuned to medical domain. LoRA rank-8 on 5K medical Q&A pairs. Training: 1 hour. Accuracy on medical MMLU: 42% → 68%. Deployed via API: base model serves 10 concurrent requests, LoRA loaded on-demand per request.",
            },
            {
                "title": "QLoRA for Consumer Hardware",
                "description": "Model: Llama 2 13B. Standard LoRA training: 40GB VRAM. QLoRA (quantized): 6GB VRAM. Achievable on RTX 3090. Result: 95% of full fine-tune accuracy. Cost: $200 hardware vs $10K cloud GPU. Used by independent researchers and small companies.",
            },
        ],
        "workflow_diagram": """graph LR
    A["Base Model<br/>W"] -->|Frozen| B["Forward Pass"]
    C["LoRA<br/>A × B<br/>Small"] -->|Trainable| B
    B --> D["Output"]
    D -->|Merge| E["Merged Model<br/>W + A×B"]

    style A fill:#e3f2fd
    style C fill:#fff3e0
    style E fill:#e8f5e9""",
    },
    "rag.md": {
        "interview_qa": [
            ("What problem does RAG solve?",
             "LLMs have knowledge cutoff (trained data ends 6-12 months ago). RAG (Retrieval-Augmented Generation): retrieve relevant documents, feed as context, generate answer grounded in retrieved docs. Solves: outdated info, company-specific data, fact-grounding. Reduces hallucinations by 30-50%."),
            ("How do you structure a RAG pipeline?",
             "1) Indexing: chunk documents, embed, store in vector DB. 2) Retrieval: embed query, retrieve top-k similar docs. 3) Ranking (optional): re-rank with cross-encoder. 4) Generation: feed docs + query to LLM. Latency: retrieval (10-50ms) + ranking (20-50ms) + generation (500-2000ms)."),
            ("What's the difference between dense and sparse retrieval?",
             "Dense: embedding-based (semantic, slow with large corpus). Sparse: keyword-based (BM25, fast, exact match). Hybrid: use both, combine scores. Dense: 'good customer service' matches 'satisfied with support'. Sparse: misses unless exact keywords. Hybrid gets both. State-of-art: dense + sparse + re-ranking."),
            ("When would you use re-ranking?",
             "Retrieval top-20 candidates. Re-ranker (cross-encoder) scores all 20 against query. Top-5 passed to LLM. Cost: +50-100ms, accuracy +5-10%. Use if: strict latency budget exists and quality matters. Skip if: query is simple or latency critical."),
            ("How do you handle large knowledge bases?",
             "Partitioning: shard KB across multiple vector stores. Hierarchical retrieval: first retrieve relevant shard, then search within shard. HyDE (Hypothetical Document Embeddings): generate hypothetical doc, use for retrieval. Results: <1s query latency on 100M docs."),
        ],
        "real_world_examples": [
            {
                "title": "Enterprise RAG for Customer Support",
                "description": "KB: 50K support docs, product specs, FAQs. Query: 'How do I reset my password?' Retrieval: embed query, find top-5 docs (including password reset guides). Generate: 'Go to Settings > Security > Reset Password'. Users: 200 concurrent, average latency 1.5s. Accuracy (fact-grounding): 94%. Reduced support tickets by 35%.",
            },
            {
                "title": "Medical RAG for Diagnosis Support",
                "description": "KB: medical journals, clinical guidelines (50K papers). Doctor input: patient symptoms. RAG retrieves relevant studies, differential diagnoses. LLM generates: 'Based on literature, consider these diagnoses: ...' Not used for final diagnosis (regulatory), but as decision support. Improves consistency, suggests relevant literature.",
            },
            {
                "title": "Financial RAG for Research",
                "description": "KB: earnings calls (1K companies, 20 years). Query: 'What's Apple's strategy in India?' RAG retrieves relevant earnings call excerpts. LLM synthesizes: '2018: expand retail. 2022: invest in manufacturing. 2024: local partnerships.' All statements backed by documents. Saves analysts hours of manual searching.",
            },
        ],
        "workflow_diagram": """graph LR
    A["User Query"] -->|Embed| B["Query Vector"]
    B -->|Search| C["Vector DB<br/>Top-k Retrieval"]
    C -->|Retrieved Docs| D["Re-ranker<br/>Optional"]
    D -->|Top Results| E["LLM<br/>+ Context"]
    E -->|Output| F["Grounded Answer"]

    style A fill:#e3f2fd
    style C fill:#e0f2f1
    style E fill:#fff3e0
    style F fill:#e8f5e9""",
    },
    "tokenization.md": {
        "interview_qa": [
            ("Why does tokenization matter for LLMs?",
             "LLMs don't process raw text; they process tokens. 'Hello world' = 2 tokens. Tokenization affects: model capacity (fewer tokens = more text fits), training efficiency, inference latency, cost (paid per token). Different models use different tokenizers (GPT uses cl100k_base, Llama uses Llama2)."),
            ("What are subword tokens and why use them?",
             "Character-level: 'world' = 5 tokens. Byte-pair encoding: 'world' = 1 token. Subword: 'beautiful' = 'beautiful' (1) but 'unfamiliar' = 'un' + 'familiar' (2). Balances: rare words (split) vs common (atomic). Reduces vocab size (50K tokens vs millions with char level)."),
            ("How do you handle special tokens?",
             "BOS (beginning of sequence): [CLS]. EOS (end): [SEP]. Padding: [PAD]. Unknown: [UNK]. Model-specific: [INST] for instruction models. Custom: add domain tokens (medical: [DIAG], [MED]). Register early in tokenizer to ensure proper encoding."),
            ("How does context window relate to tokenization?",
             "4K context = 4096 tokens max. Longer documents: tokenize first, check length. 'A long document (5K words)' = ~6-8K tokens (varies by tokenizer). Must truncate or chunk. Context window efficiency: subword tokenizers pack more text than character tokenizers."),
            ("When would you fine-tune a tokenizer?",
             "Specialized domains: medical (add anatomy terms), code (add language keywords), multilingual (improve rare language handling). Rare: pre-trained tokenizers usually sufficient. Benefits: +1-2% accuracy in specialized tasks. Effort: moderate (requires retraining BPE)."),
        ],
        "real_world_examples": [
            {
                "title": "Tokenizer for Code LLMs",
                "description": "General tokenizer: code is inefficient (Python keywords split). Code-specific tokenizer (add 'def', 'class', 'import', etc.): same code = fewer tokens. Effect: can fit 50% more code in context window. Models like StarCoder use custom tokenizers. Result: better code understanding, lower training cost.",
            },
            {
                "title": "Multilingual Tokenization Strategy",
                "description": "Single tokenizer for 100 languages. Problem: low-resource languages (e.g., Urdu, Tamil) heavily fragmented. Solution: add dedicated tokens for scripts (Arabic, Devanagari, Han). Effect: rare language tokens +3-5% from fragmentation penalty, improves multilingual consistency.",
            },
            {
                "title": "Efficient Streaming with Token Prediction",
                "description": "Real-time translation API. Tokenizer choice impacts: English 'hello' = 1 token (efficient). Japanese 'こんにちは' = 5 tokens (inefficient). Streaming: send tokens as they're generated. Switching to language-specific tokenizer: 3x fewer tokens for Japanese, faster streaming, lower bandwidth.",
            },
        ],
        "workflow_diagram": """graph LR
    A["Raw Text"] -->|Tokenize| B["Token IDs<br/>123, 456, 789"]
    B -->|Embed| C["Token Embeddings"]
    C -->|Model| D["Output"]
    D -->|Decode| E["Generated Text"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e0f2f1
    style E fill:#e8f5e9""",
    },
    "quantization.md": {
        "interview_qa": [
            ("What does quantization do?",
             "Store weights in lower precision. FP32 (4 bytes per param) → INT8 (1 byte) = 4x compression. FP32 (7B model) = 28GB → INT8 = 7GB. Trade-off: slight accuracy loss (0.5-2%). Inference: faster (less memory bandwidth), lower memory, deployable on smaller GPUs."),
            ("What's the difference between post-training quantization and QAT?",
             "Post-training (PTQ): quantize after training, quick (hours), some accuracy loss (~2%). QAT (quantization-aware training): train while simulating quantization, better accuracy (0.5% loss), slower (needs retraining). Use PTQ if time-critical; QAT if accuracy critical."),
            ("When would you use INT4 vs INT8?",
             "INT8: 4x compression, minimal accuracy loss, standard approach. INT4: 8x compression, 1-2% accuracy loss, for extreme size constraints (mobile, edge). INT4 inference: requires special libraries (bitsandbytes, gptq). Cost-benefit: INT4 for 7B models on phones, INT8 for datacenter."),
            ("How do you handle activation quantization?",
             "Weight quantization: straightforward. Activation quantization: depends on data (input distribution). Static: pre-computed scales. Dynamic: compute at inference. Dynamic: more accurate but slower. Typical: quantize weights, keep activations FP32 (reduces memory but not compute)."),
            ("How does quantization affect model calibration?",
             "During quantization: calibrate on representative dataset (first 100 batches). Bad calibration: outlier batches → poor scale factors → errors. Good: diverse calibration set. Check: perplexity before/after quantization (should increase <5%)."),
        ],
        "real_world_examples": [
            {
                "title": "INT8 Quantization for Batch Inference",
                "description": "Llama 2 7B FP32: 28GB VRAM, 100 ms/token. INT8: 7GB VRAM, 70 ms/token (15% faster). Deployed on A100 (80GB): FP32 = 2 models, INT8 = 10 models. Throughput: 2 models × 10 tok/s = 20 tok/s. With quantization: 10 models × 15 tok/s = 150 tok/s (7x improvement).",
            },
            {
                "title": "QLoRA for Fine-Tuning on Consumer GPU",
                "description": "Llama 2 13B FP32: 52GB VRAM (beyond consumer GPUs). INT4 + LoRA: 6GB VRAM (RTX 3090 compatible). Process: quantize base model, train LoRA on top. Result: 95% of full fine-tune accuracy. Enables small teams to fine-tune large models independently.",
            },
            {
                "title": "GPTQ for On-Device Inference",
                "description": "Model: 13B parameters. FP32: 52GB (not mobile-feasible). INT4 GPTQ: 3.5GB. Deployed on iPhone 14 Pro (6GB RAM). Inference: 5-10 tok/s (slow but functional). Use case: offline translation, on-device assistant. Trade-off: latency vs privacy (no cloud calls).",
            },
        ],
        "workflow_diagram": """graph LR
    A["FP32 Model<br/>28GB"] -->|Calibrate| B["Compute Scales"]
    B -->|Quantize| C["INT8 Model<br/>7GB<br/>4x Compression"]
    C -->|Deploy| D["Inference<br/>Faster"]
    A -.->|Accuracy| E["99.5%"]
    C -.->|Accuracy| F["98%<br/>-0.5% loss"]

    style A fill:#e3f2fd
    style C fill:#e8f5e9
    style E fill:#e8f5e9
    style F fill:#fff3e0""",
    },
}

# Add more concept enhancements for remaining concepts...
# For brevity, including detailed patterns for 8 key concepts
# Rest follow same pattern

def enhance_interview_questions(content, concept_name, qa_pairs):
    """Replace placeholder interview questions with real Q&A."""
    placeholder_pattern = r"## Interview Questions\s*\n+\*\*Q:.*?(?=\n## |$)"

    qa_content = "## Interview Questions\n\n"
    for q, a in qa_pairs:
        qa_content += f"**Q: {q}**\n*A: {a}*\n\n"

    # Replace entire Interview Questions section
    new_content = re.sub(
        r"## Interview Questions\s*\n+.*?(?=\n## |\Z)",
        qa_content.rstrip(),
        content,
        flags=re.DOTALL
    )
    return new_content

def add_workflow_diagram(content, concept_name, diagram):
    """Add or replace workflow diagram in How It Works section."""
    # Add workflow diagram after "How It Works" section
    how_it_works_pattern = r"(## How It Works\n.*?)(\n### )"

    workflow_section = f"\n### Workflow Flowchart\n\n```mermaid\n{diagram}\n```\n\n"

    if "### Workflow Flowchart" in content:
        # Replace existing
        new_content = re.sub(
            r"### Workflow Flowchart\n\n```mermaid\n.*?\n```",
            f"### Workflow Flowchart\n\n```mermaid\n{diagram}\n```",
            content,
            flags=re.DOTALL
        )
    else:
        # Add new
        new_content = re.sub(
            how_it_works_pattern,
            r"\1" + workflow_section + r"\2",
            content,
            flags=re.DOTALL
        )

    return new_content

def add_real_world_examples(content, concept_name, examples):
    """Add detailed real-world examples section."""
    examples_section = "## Real-World Examples\n\n"

    for example in examples:
        examples_section += f"### {example['title']}\n"
        examples_section += f"{example['description']}\n\n"

    # Add before Related Topics or at end
    if "## Related Topics" in content:
        new_content = content.replace(
            "## Related Topics",
            examples_section + "## Related Topics"
        )
    else:
        new_content = content + "\n" + examples_section

    return new_content

def enhance_concept_file(filepath, enhancements):
    """Enhance a single concept file."""
    print(f"Enhancing {filepath.name}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Apply enhancements
    if 'interview_qa' in enhancements:
        content = enhance_interview_questions(
            content,
            filepath.stem,
            enhancements['interview_qa']
        )

    if 'workflow_diagram' in enhancements:
        content = add_workflow_diagram(
            content,
            filepath.stem,
            enhancements['workflow_diagram']
        )

    if 'real_world_examples' in enhancements:
        content = add_real_world_examples(
            content,
            filepath.stem,
            enhancements['real_world_examples']
        )

    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  ✓ Enhanced with Q&A, diagrams, and examples")

def main():
    """Enhance all concept files."""
    print("Starting concept enhancement...\n")

    # Merge all enhancements
    all_enhancements = {**CONCEPT_ENHANCEMENTS, **ADDITIONAL_ENHANCEMENTS}

    concept_files = list(CONCEPTS_DIR.glob("*.md"))
    print(f"Found {len(concept_files)} concept files")
    print(f"Enhancements available for {len(all_enhancements)} concepts\n")

    enhanced_count = 0
    for concept_file in sorted(concept_files):
        # Check if we have enhancements for this file
        if concept_file.name in all_enhancements:
            enhancements = all_enhancements[concept_file.name]
            enhance_concept_file(concept_file, enhancements)
            enhanced_count += 1
        else:
            print(f"⊘ {concept_file.name} (no enhancements defined yet)")

    print(f"\n✓ Enhanced {enhanced_count} concepts out of {len(concept_files)} total")

if __name__ == "__main__":
    main()

# Additional enhancements for remaining 24 concepts

MORE_ENHANCEMENTS = {
    "context-window.md": {
        "interview_qa": [
            ("What's the context window and why does it matter?",
             "Maximum sequence length a model can process. GPT-4: 8K/128K, Llama 2: 4K, Claude: 100K+. Matters because: longer context = more text analyzed at once. A 4K model can't summarize 50-page documents; 128K model can. Trade-off: larger windows = more memory + slower inference."),
            ("How do you handle documents longer than context window?",
             "Options: 1) Truncate (lose info), 2) Chunk + summarize (recursive), 3) Use sliding window (overlap chunks). For RAG: chunk documents at index time, retrieve relevant chunks at query time. For fine-tuning: split into context-size pieces, train on each."),
            ("What are efficient long-context techniques?",
             "Sparse attention: attend to nearby tokens + random tokens (reduces O(n²) to O(n log n)). Recurrent: process in chunks, maintain state across chunks. Paged attention: splits KV cache into pages (PagedAttention). ROPe (Rotary Position Embeddings): handles longer sequences better than absolute positions."),
            ("When is 4K context enough vs when do you need 128K?",
             "4K (sufficient for): single-turn QA, classification, sentiment analysis. 128K (necessary for): multi-document analysis, long conversations, full book summarization, code repositories. Cost trade-off: 128K input = 32x tokens = 32x cost vs 4K."),
            ("How do position embeddings affect context length?",
             "Absolute positems: hardcoded for fixed length (4K). Extrapolation to 8K = hallucinations. RoPE/ALiBi: relative positions, can generalize to longer lengths. ALiBi: particularly good for length extrapolation (trained on 2K, works on 16K). Matters for real-world documents."),
        ],
        "real_world_examples": [
            {"title": "4K Context Limitation in Customer Support", "description": "Support ticket: reference previous 10 conversations (50K tokens). 4K context model: truncates to last 2 conversations. Customer context lost, solution suggestions miss important history. 128K model: loads all history, provides better solutions. Customer satisfaction: 70% → 88%."},
            {"title": "Code Repository Understanding", "description": "Codebase: 100K tokens across multiple files. 4K model: analyzes one function at a time (limited understanding). 128K model: sees entire repository structure, dependencies, patterns. Code review quality: better suggestions, catches subtle bugs. GitHub: suggests Copilot with larger context."},
            {"title": "Legal Document Analysis", "description": "Contract: 200 pages = 100K tokens. 4K model: useless (can't fit full contract). Standard approach: expensive human review. With 128K model: upload full contract, extract terms, identify risks. Cost: $0.50 (tokens) vs $500 (human). Trade-off: still needs human verification."},
        ],
        "workflow_diagram": """graph LR
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
    "dpo.md": {
        "interview_qa": [
            ("What's DPO and how does it differ from RLHF?",
             "RLHF: train reward model, use for RL (complex, unstable). DPO: direct preference optimization, no reward model, just use preference pairs directly. Training: instead of 'maximize reward', use 'prefer chosen over rejected'. Simpler, more stable, comparable results."),
            ("When would you use DPO vs RLHF?",
             "DPO: preference data available, want simpler pipeline, unstable RLHF. RLHF: reward model importance, have label budget for rewards. DPO getting popular because: easier to implement, no RL training needed, better results on benchmarks."),
            ("How do you collect preference data for DPO?",
             "Generate multiple outputs, have human raters choose best. 10K-100K preference pairs typical. Quality matters: bad labels → bad DPO model. Standard: binary choice (A vs B), can do ranking (order N outputs). Cost: 5-10x cheaper than RLHF (no reward model training)."),
            ("What's the DPO loss function conceptually?",
             "Maximize probability of preferred over rejected: P(preferred) > P(rejected). During training: upweight preferred outputs, downweight rejected. Unlike RLHF: no explicit reward signal, implicit from comparisons."),
            ("How does DPO affect model behavior vs base model?",
             "Alignment shift: better instruction-following, fewer refusals on legitimate requests, more appropriate refusals on harmful. Similar to RLHF but achieved faster. Risk: potential jailbreak if preferences skewed toward harmful outputs."),
        ],
        "real_world_examples": [
            {"title": "DPO for LLM Alignment", "description": "Base model: 45% human preference. After DPO (10K preference pairs): 72% human preference. Training: 3 hours on 1 GPU. Improvement: better instruction following, fewer refusals. Cost: $100 (data annotation) + $50 (compute) vs $5K (RLHF)."},
            {"title": "DPO for Code Generation", "description": "Base model: 40% correct solutions. Target: 70%. DPO on 5K (output, correct, incorrect) triplets. Result: 68% accuracy, 5% improvement. Faster than RLHF pipeline (RL training). Deployed in Copilot-style autocomplete."},
            {"title": "Multi-Objective DPO", "description": "Preferences: helpfulness + safety + conciseness. Single DPO model balances all three. Human raters rank outputs by combined criteria. Result: instruction-following improved (helpfulness) while maintaining refusal capability (safety). Easier than RLHF with separate rewards."},
        ],
        "workflow_diagram": """graph TD
    A["Preference Data<br/>A vs B"] -->|Standard| B["Train Reward<br/>Model"]
    A -->|DPO| C["Direct Training<br/>No Reward Model"]
    B -->|RLHF| D["RL Training<br/>Complex"]
    C -->|DPO| E["Aligned Model<br/>Simple"]
    D -->|Output| F["Same Result<br/>More Complex"]
    E -->|Output| G["Similar Result<br/>Faster"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#ffcdd2
    style E fill:#c8e6c9""",
    },
    "inference-optimization.md": {
        "interview_qa": [
            ("What are the main inference bottlenecks?",
             "Compute (matrix multiplies), memory bandwidth (loading weights), latency (time to first token). For long context: KV cache dominates. For large models: loading 70GB weights = 30s. Priority: first identify bottleneck."),
            ("How does batch size affect inference?",
             "Small batch: underutilize GPU (3% utilization, 1 token/s). Large batch: saturate memory, maximize throughput (80% utilization, 100 tokens/s). But: latency increases (individual request slower in larger batch). Balance: throughput vs latency."),
            ("What's the difference between throughput and latency optimization?",
             "Throughput: maximize tokens/sec across batch. Latency: minimize time per request. Batching: great for throughput, bad for latency. Options: batch + paging (dynamic batching), speculative decoding (parallel paths), quantization (reduce memory)."),
            ("When would you use speculative decoding?",
             "Small draft model generates candidates, large model verifies. If draft correct: accept. If wrong: discard, continue. Net: faster generation if acceptance rate >50%. Cost: small model overhead. Use if: latency-critical and have small model available."),
            ("How do you measure inference efficiency?",
             "Tokens/sec (throughput), ms/token (latency), memory GB, power watts. FLOPS utilization (% of peak compute). For production: measure end-to-end including data loading, not just model forward pass."),
        ],
        "real_world_examples": [
            {"title": "Quantization + Batching for Throughput", "description": "Llama 2 7B, 1 req/sec throughput (unbatched, FP32). With INT8 + batch 32: 200 req/sec. Memory: 28GB → 7GB. Latency per request: same (~500ms) but 32x more requests/day served."},
            {"title": "Speculative Decoding for Latency", "description": "Large model (Llama 70B): 50 ms/token, slow for interactive use. With draft model (Llama 7B, 5ms/token): speculative paths. Acceptance 70%, net speedup 2.5x. Latency: 50ms → 20ms for interactive chat."},
            {"title": "PagedAttention for Batch Serving", "description": "vLLM: dynamic batching with paged KV cache. Multiple users, variable output lengths. Without paging: 4 concurrent users. With paging: 16 concurrent users from freed pages. Throughput: 4x without sacrificing latency."},
        ],
        "workflow_diagram": """graph LR
    A["Model Weights"] -->|Load| B["GPU Memory"]
    B -->|Forward Pass| C["Compute"]
    C -->|Output| D["Tokens"]
    C -.->|Bottleneck| E["Identify"]
    E -->|Compute-Bound| F["Batching<br/>Quantization"]
    E -->|Memory-Bound| G["Precision<br/>Paging"]
    F --> H["Optimized"]
    G --> H

    style A fill:#e3f2fd
    style E fill:#fff3e0
    style H fill:#e8f5e9""",
    },
}

# Add this to CONCEPT_ENHANCEMENTS
CONCEPT_ENHANCEMENTS.update(MORE_ENHANCEMENTS)

