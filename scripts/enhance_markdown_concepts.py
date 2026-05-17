"""
Enhance markdown concept files with:
- Interview questions and answers
- Real-world implementation examples
- More elaborate explanations
- Best practices and common pitfalls
- Additional resources
"""

import json
from pathlib import Path

CONCEPTS_DIR = Path("llm/concepts")
MAPPING_FILE = Path("data/concepts_mapping.json")

# Detailed enhancements for each concept
ENHANCEMENTS = {
    "adapters": {
        "interview_qa": [
            {
                "q": "What's the main advantage of using adapters over full fine-tuning?",
                "a": "Adapters reduce trainable parameters from millions to thousands (often 0.5-5% of original), enabling fast adaptation to multiple tasks while keeping the base model frozen. This allows sharing a single large base model across many specialized tasks."
            },
            {
                "q": "How do adapters maintain performance while reducing parameters?",
                "a": "They leverage the fact that fine-tuning operates in a much lower-dimensional subspace. Adapters use bottleneck architectures (down-project, activate, up-project) that capture task-specific information efficiently without modifying pre-trained knowledge."
            },
            {
                "q": "Can you explain the adapter architecture in detail?",
                "a": "Adapters follow a bottleneck design: input → linear down-projection (to small dim like 64) → activation (GELU) → linear up-projection (back to original dim) → residual connection. This 2-layer MLP is much smaller than the original layer but learns task-specific transformations."
            },
            {
                "q": "What are the trade-offs between LoRA and Adapters?",
                "a": "LoRA uses low-rank matrix decomposition added to weights (multiplicative). Adapters are bottleneck MLP modules (additive). LoRA is often more parameter-efficient but harder to interpret. Adapters are modular and easier to compose but use slightly more parameters."
            },
        ],
        "real_world_examples": [
            {
                "company": "Meta (Facebook)",
                "use_case": "Efficient model specialization",
                "details": "Uses adapters to quickly adapt LLAMA models to specific languages and domains without full retraining. Reduces deployment footprint significantly."
            },
            {
                "company": "Microsoft",
                "use_case": "Multi-tenant serving",
                "details": "Deploys adapters for different enterprise customers on shared base models. Each customer gets a specialized model via lightweight adapters."
            },
            {
                "company": "Google",
                "use_case": "Task-specific models",
                "details": "Uses adapter-like modules in T5 and FLAN to efficiently adapt to new tasks without retraining massive models."
            },
        ],
        "best_practices": [
            "Keep bottleneck dimension between 8-64. Too small loses expressiveness, too large wastes parameters.",
            "Initialize down-projection with uniform distribution, up-projection with zeros to start as identity.",
            "Add layer normalization after residual connection for training stability.",
            "Use adapters in every transformer layer for consistent improvement across depths.",
        ],
        "pitfalls": [
            "Adapter bottleneck too small: insufficient capacity for complex task shifts",
            "Placing adapters only in middle layers: misses specialization opportunities in early/late layers",
            "Not freezing base model: defeats the purpose of parameter efficiency",
            "Insufficient adapter training data: small adaptation sets may overfit to adapters",
        ]
    },
    "embeddings": {
        "interview_qa": [
            {
                "q": "Why are embeddings important in modern NLP?",
                "a": "Embeddings convert discrete text into continuous vectors that capture semantic meaning. This enables: similarity computation, retrieval, clustering, and transfer learning. They're foundational for RAG, semantic search, and similarity-based tasks."
            },
            {
                "q": "What's the difference between word embeddings and sentence embeddings?",
                "a": "Word embeddings (Word2Vec, GloVe) represent single words. Sentence embeddings (Sentence-BERT, Universal Sentence Encoder) represent entire sentences/documents by pooling or attending over word embeddings, capturing full semantic context."
            },
            {
                "q": "How do you choose an embedding model?",
                "a": "Consider: task (semantic similarity, clustering, retrieval), domain (general vs. specialized), embedding dimension (128 for speed, 384+ for quality), model size (computational budget), and benchmark performance on similar tasks."
            },
            {
                "q": "What's the computational cost of generating embeddings at scale?",
                "a": "Inference cost is relatively low (ms per example). Main costs are: initial embedding generation for large corpora (hours to days), storage (embedding_dim * num_examples bytes), and similarity computation (O(n²) for exact search)."
            },
        ],
        "real_world_examples": [
            {
                "company": "Pinecone",
                "use_case": "Vector search infrastructure",
                "details": "Provides vector databases for storing and searching embeddings at scale. Used by Uber, Slack, DuckDuckGo for semantic search and recommendations."
            },
            {
                "company": "Airbnb",
                "use_case": "Listing search and recommendations",
                "details": "Uses embeddings to compute similarity between listings and user preferences, enabling semantic search beyond keyword matching."
            },
            {
                "company": "Spotify",
                "use_case": "Music recommendations",
                "details": "Generates embeddings for songs and user preferences. Uses cosine similarity to recommend songs that are semantically similar in taste space."
            },
        ],
        "best_practices": [
            "Normalize embeddings to unit vectors for efficient cosine similarity (dot product = cosine similarity).",
            "Use contrastive training (sentence-BERT) for better semantic representations than vanilla transformers.",
            "Dimension reduction (PCA, UMAP) can improve efficiency for downstream tasks without much quality loss.",
            "Fine-tune embeddings on domain-specific data for better domain relevance.",
        ],
        "pitfalls": [
            "Using overly generic embeddings for specialized domains: task-specific fine-tuning usually helps",
            "Not normalizing embeddings: breaks cosine similarity assumptions",
            "Using too-large models for simple tasks: waste of compute; smaller models often sufficient",
            "Outdated embedding models: new models (Sentence-BERT v2) have much better quality",
        ]
    },
    "lora": {
        "interview_qa": [
            {
                "q": "Why does LoRA work so well despite using so few parameters?",
                "a": "LoRA is based on the hypothesis that weight updates during fine-tuning have low intrinsic dimensionality. Instead of updating all parameters, LoRA learns low-rank factors A (in_dim × r) and B (r × out_dim) where r << both dimensions. The update is W ← W + AB^T."
            },
            {
                "q": "How do you choose the LoRA rank?",
                "a": "Rank is task and model dependent. Common values: r=4-8 (10-20% of LoRA params), r=16-32 (40-60%), r=64 (heavy adaptation). Start with r=8 and tune based on performance. Diminishing returns after rank ≈ model_dim/4."
            },
            {
                "q": "Can you stack multiple LoRA adapters?",
                "a": "Yes! LoRA is composable. You can have separate LoRA modules for different tasks and combine them (e.g., task1_weights + task2_weights). This enables multi-task adaptation with minimal additional parameters."
            },
            {
                "q": "What's the relationship between LoRA and matrix factorization?",
                "a": "LoRA decomposes large weight matrices into products of smaller matrices. This is essentially low-rank matrix factorization. It works because fine-tuning updates are approximately low-rank (empirically verified)."
            },
        ],
        "real_world_examples": [
            {
                "company": "Microsoft",
                "use_case": "Efficient LLM fine-tuning",
                "details": "Original LoRA paper. Used for fine-tuning LLAMA and other large models with 99% parameter reduction while maintaining performance."
            },
            {
                "company": "Hugging Face",
                "use_case": "PEFT library",
                "details": "Provides production-ready LoRA implementation. Used by 100k+ developers for fine-tuning models on consumer hardware."
            },
            {
                "company": "OpenAI",
                "use_case": "Model customization",
                "details": "Uses LoRA-like techniques for efficient fine-tuning of GPT models for enterprise customers."
            },
        ],
        "best_practices": [
            "Use alpha/rank scaling: multiply LoRA updates by alpha/rank for stable training across different ranks.",
            "Apply LoRA to both Q and V projections in attention; less critical but helps for heavy adaptation.",
            "Start with frozen base model + LoRA. Only unfreeze base if significant gains plateau.",
            "Use moderate learning rates (1e-4 to 5e-4). LoRA is more sensitive than full fine-tuning.",
        ],
        "pitfalls": [
            "Too small rank: insufficient capacity; performance plateaus quickly",
            "Too large rank: loses efficiency benefits; approaching full fine-tuning cost",
            "High learning rate: LoRA can diverge quickly if not careful",
            "Not scaling updates: unstable training when switching between ranks",
        ]
    },
    "rag": {
        "interview_qa": [
            {
                "q": "Why is RAG necessary if LLMs are large?",
                "a": "LLMs have knowledge cutoff dates, can hallucinate, and have limited context windows. RAG retrieves fresh, relevant information at query time, grounding generation in actual documents. This reduces hallucination and enables current information access."
            },
            {
                "q": "What are the three main components of RAG?",
                "a": "1) Retriever: finds relevant documents via semantic/BM25 search. 2) Reader/LLM: generates answer using retrieved documents. 3) Ranking: orders retrieved docs by relevance. Often uses dense retrievers (embeddings) + re-rankers."
            },
            {
                "q": "How do you evaluate RAG quality?",
                "a": "Retrieval metrics: MRR, NDCG, Recall@k (is relevant doc in top-k?). Generation metrics: BLEU, ROUGE (similarity to reference). End-to-end: EM, F1 on QA datasets. Human evaluation for factuality and relevance."
            },
            {
                "q": "What's the trade-off between retrieval and generation in RAG?",
                "a": "Better retrieval → better context → better generation. But retrieval is expensive (vector similarity over millions of docs). Need to balance: retrieve more docs (higher latency) vs. fewer docs (lower quality). Sweet spot: top-5 to top-20."
            },
        ],
        "real_world_examples": [
            {
                "company": "OpenAI",
                "use_case": "ChatGPT plugins and browsing",
                "details": "Uses RAG-like approach to fetch real-time information from web and plugins, grounding responses in live data."
            },
            {
                "company": "LinkedIn",
                "use_case": "Search and Q&A",
                "details": "Uses RAG for enterprise Q&A over company knowledge bases, enabling employees to ask natural questions."
            },
            {
                "company": "Amazon",
                "use_case": "Customer service automation",
                "details": "Retrieves from FAQs and product documentation to answer customer questions accurately without hallucination."
            },
        ],
        "best_practices": [
            "Hybrid retrieval: combine dense (semantic) + sparse (BM25) search for robustness.",
            "Re-ranking: use cross-encoder to re-rank retrieved documents by relevance before generation.",
            "Chunk documents carefully: too small → multiple docs with partial info, too large → noise.",
            "Cache embeddings: pre-compute and store document embeddings for fast retrieval.",
        ],
        "pitfalls": [
            "Retrieving irrelevant documents: LLM can't fix bad retrieval, garbage in = garbage out",
            "Too many documents: overwhelms context window and confuses model",
            "Poor document indexing: missing relevant documents makes retrieval impossible",
            "Outdated embeddings: if documents change, embeddings become stale",
        ]
    },
    "quantization": {
        "interview_qa": [
            {
                "q": "What's the intuition behind quantization?",
                "a": "Neural networks can operate at lower precision without significant accuracy loss. Quantization reduces precision (float32 → int8/int4), shrinking model size by 4-8x and speeding inference. Trade-off: slight accuracy loss, faster speed, less memory."
            },
            {
                "q": "What's the difference between post-training quantization and QAT?",
                "a": "PTQ: quantize pre-trained weights directly, fast but may lose accuracy. QAT (Quantization-Aware Training): simulate quantization during training, learn optimal scaling factors, better accuracy but requires retraining."
            },
            {
                "q": "How do you choose quantization bits?",
                "a": "8-bit: standard, minimal loss, 4x compression. 4-bit: aggressive, noticeable but acceptable loss for many tasks, 8x compression. 2-bit: extreme, only for specific models. Research shows 4-8 bit is sweet spot."
            },
            {
                "q": "What's the relationship between quantization and hardware?",
                "a": "Modern GPUs have int8 operations but less common for int4. CPUs have int8 support. Specialized hardware (TPUs, Qualcomm Snapdragon) have excellent int8 performance. Quantization choice should match target hardware."
            },
        ],
        "real_world_examples": [
            {
                "company": "NVIDIA",
                "use_case": "TensorRT optimization",
                "details": "Provides automated quantization and optimization for deep learning models on GPUs, enabling 4-8x speedup."
            },
            {
                "company": "Apple",
                "use_case": "On-device ML",
                "details": "Uses INT8 and mixed-precision quantization to run models efficiently on iPhones and Macs without cloud computing."
            },
            {
                "company": "Meta",
                "use_case": "Efficient inference",
                "details": "Uses post-training quantization for LLAMA models to serve billions of requests with reduced latency and cost."
            },
        ],
        "best_practices": [
            "Start with 8-bit symmetric quantization, most tools support it well.",
            "Use calibration data: quantization needs representative samples to determine good scaling factors.",
            "Per-channel quantization: quantize each filter differently for better accuracy than per-layer.",
            "Test on actual hardware: simulated quantization != real hardware performance.",
        ],
        "pitfalls": [
            "Naive uniform quantization: loses precision for outlier weights; use asymmetric or per-channel instead",
            "Calibration on unrepresentative data: scaling factors won't generalize to real data",
            "Extreme quantization (2-bit) for all layers: some layers are sensitive, need higher precision",
            "Ignoring hardware: int4 speedup varies wildly by hardware; may not justify complexity",
        ]
    },
}

# Generic templates for concepts without specific enhancements
GENERIC_INTERVIEW_QA = [
    {
        "q": "What's the core problem this concept solves?",
        "a": "See the 'Core Intuition' section above for the fundamental problem and how this concept addresses it."
    },
    {
        "q": "What are the main advantages and disadvantages?",
        "a": "See 'Key Properties / Trade-offs' section for detailed comparison with alternatives."
    },
    {
        "q": "How do you implement this in practice?",
        "a": "Refer to the corresponding Jupyter notebook in `llm/notebooks/` for working Python implementations and examples."
    },
]

def add_section_if_missing(content: str, section_title: str, section_content: str) -> str:
    """Add section to markdown if not already present"""
    if section_title in content:
        return content  # Section already exists
    return content.rstrip() + "\n\n" + section_content

def format_interview_qa(qa_pairs):
    """Format Q&A pairs as markdown"""
    text = "## Interview Questions\n\n"
    for i, pair in enumerate(qa_pairs, 1):
        text += f"**Q: {pair['q']}**\n"
        text += f"*A: {pair['a']}*\n\n"
    return text

def format_real_world(examples):
    """Format real-world examples as markdown"""
    text = "## Real-World Applications\n\n"
    for example in examples:
        text += f"### {example['company']}: {example['use_case']}\n"
        text += f"{example['details']}\n\n"
    return text

def format_best_practices(practices):
    """Format best practices as markdown"""
    text = "## Best Practices\n\n"
    for practice in practices:
        text += f"- {practice}\n"
    text += "\n"
    return text

def format_pitfalls(pitfalls):
    """Format common pitfalls as markdown"""
    text = "## Common Pitfalls to Avoid\n\n"
    for pitfall in pitfalls:
        text += f"- **{pitfall.split(':')[0]}**: {pitfall}\n"
    text += "\n"
    return text

def enhance_markdown(concept_key: str, content: str) -> str:
    """Enhance markdown file with additional content"""

    if concept_key in ENHANCEMENTS:
        enhancements = ENHANCEMENTS[concept_key]

        # Add interview Q&A
        if "interview_qa" in enhancements:
            qa_text = format_interview_qa(enhancements["interview_qa"])
            content = add_section_if_missing(content, "## Interview Questions", qa_text)

        # Add real-world examples
        if "real_world_examples" in enhancements:
            examples_text = format_real_world(enhancements["real_world_examples"])
            content = add_section_if_missing(content, "## Real-World Applications", examples_text)

        # Add best practices
        if "best_practices" in enhancements:
            practices_text = format_best_practices(enhancements["best_practices"])
            content = add_section_if_missing(content, "## Best Practices", practices_text)

        # Add pitfalls
        if "pitfalls" in enhancements:
            pitfalls_text = format_pitfalls(enhancements["pitfalls"])
            content = add_section_if_missing(content, "## Common Pitfalls", pitfalls_text)
    else:
        # Add generic Q&A for concepts without specific enhancements
        qa_text = format_interview_qa(GENERIC_INTERVIEW_QA)
        content = add_section_if_missing(content, "## Interview Questions", qa_text)

    return content

def main():
    """Enhance all markdown concept files"""
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    concepts = mapping.get("concepts", {})
    enhanced_count = 0

    for concept_key, concept_data in concepts.items():
        source_file = concept_data.get("source_file")
        markdown_path = CONCEPTS_DIR / source_file.split("/")[-1]

        if not markdown_path.exists():
            print(f"Warning: File not found: {markdown_path}")
            continue

        # Read original content
        with open(markdown_path) as f:
            content = f.read()

        # Enhance
        print(f"Enhancing: {concept_key}...")
        enhanced_content = enhance_markdown(concept_key, content)

        # Write back
        with open(markdown_path, 'w') as f:
            f.write(enhanced_content)

        enhanced_count += 1
        print(f"  ✓ Enhanced {markdown_path.name}")

    print(f"\n✓ Enhanced {enhanced_count} markdown files")

if __name__ == "__main__":
    main()
