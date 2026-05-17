#!/usr/bin/env python3
"""
Expand all markdown concept files with:
- Detailed explanations (replace minimal TL;DR with comprehensive overview)
- Comparison tables for alternatives and trade-offs
- Detailed "How It Works" sections
"""

import re
from pathlib import Path

CONCEPTS_DIR = Path("llm/concepts")

MARKDOWN_EXPANSIONS = {
    "adapters.md": {
        "detailed_overview": """## Detailed Overview

Adapters represent a paradigm shift in efficient model adaptation. Rather than fine-tuning all parameters of a pre-trained model (which can be computationally expensive and memory-intensive), adapters introduce small, learnable modules that are inserted into the model architecture while keeping the base model weights frozen.

### Architecture Details

An adapter typically consists of:
1. **Down-projection layer**: Projects from hidden dimension (e.g., 768) to a smaller bottleneck dimension (e.g., 64)
2. **Non-linearity**: ReLU or GELU activation
3. **Up-projection layer**: Projects back to original dimension

This bottleneck design forces the adapter to learn a compressed representation of the task-specific knowledge. The total trainable parameters per adapter: roughly 2 × hidden_dim × bottleneck_dim, typically 500K-1M parameters.

### Why This Matters

For a 7B parameter model:
- **Full fine-tuning**: Update all 7B parameters (28GB storage, days of training, GPU cost ~$5K)
- **LoRA**: Update ~1M parameters in low-rank matrices (3-5GB storage, hours of training, GPU cost ~$500)
- **Adapters**: Update ~1M parameters in bottleneck modules (3-5GB storage, hours of training, GPU cost ~$500)

The adapter approach is particularly valuable in **multi-task learning scenarios** where you need different models for different tasks but want to share a common base model.""",
        "comparison_table": """## Comparison: Adapter Types and Alternatives

| Method | Parameters Trained | Storage per Task | Training Speed | Inference Speed | Task Isolation | Multi-Task Support |
|--------|-------------------|------------------|-----------------|-----------------|---------------|--------------------|
| **Adapters (Bottleneck)** | 0.5-1M | 512KB - 1MB | Fast (1-2h) | Slight overhead (+2-3%) | Excellent | Excellent |
| **LoRA** | 0.1-1M | 100KB - 1MB | Fast (1-2h) | No overhead | Excellent | Excellent |
| **Prefix-Tuning** | 50-500K | 50-500KB | Very Fast | Minimal overhead | Good | Good |
| **Full Fine-Tuning** | 7B | 28GB | Slow (24-48h) | No overhead | Perfect | Poor |
| **BitFit** | 0.01% | Similar to adapters | Moderate | No overhead | Good | Good |
| **Prompt Tuning** | 0.001% | Tiny | Fastest | No overhead | Limited | Limited |

**Key Tradeoffs:**
- **Inference latency**: LoRA (no overhead) > Adapters (2-3% slower) > other methods
- **Multi-task**: Adapters and LoRA excellent; full fine-tuning requires separate models
- **Accuracy ceiling**: Full fine-tuning > LoRA ≈ Adapters > others
- **Memory efficiency during training**: Prefix-tuning > LoRA > Adapters > Full"""
    },
    "embeddings.md": {
        "detailed_overview": """## Comprehensive Embedding Systems

Embeddings are continuous vector representations of text that capture semantic meaning in a high-dimensional space. Unlike one-hot encoding (which is sparse and doesn't capture meaning), embeddings are dense vectors where similarity in vector space corresponds to semantic similarity.

### How Embeddings Capture Meaning

Modern embeddings are learned through contrastive learning objectives:
1. **Positive pairs**: Similar sentences → similar embeddings (high cosine similarity ~0.8-0.95)
2. **Negative pairs**: Dissimilar sentences → dissimilar embeddings (low cosine similarity ~0.1-0.3)
3. **Loss function**: Minimize distance between positive pairs, maximize distance between negative pairs

This forces the model to learn a space where:
- "good customer service" and "excellent support" are close (synonymous)
- "good customer service" and "terrible experience" are far apart (opposite meaning)
- "good customer service" and "weather today" are distant (unrelated)

### Embedding Models Landscape

The choice of embedding model significantly impacts downstream task performance. Different models are optimized for different properties.""",
        "comparison_table": """## Embedding Models Comparison

| Model | Dimension | Params | Speed | Quality | Use Case | Training Data |
|-------|-----------|--------|-------|---------|----------|---------------|
| **all-MiniLM-L6-v2** | 384 | 22M | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | General, edge devices | MS MARCO, NLI |
| **all-mpnet-base-v2** | 768 | 109M | ⚡⚡ Medium | ⭐⭐⭐⭐ Very Good | Production systems | 215M sentence pairs |
| **bge-large-en-v1.5** | 1024 | 335M | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | High-accuracy needs | Massive web corpus |
| **instructor-xl** | 768 | 335M | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | Domain-specific | Instruction-tuned |
| **text-embedding-3-small** | 512 | Proprietary | Medium | ⭐⭐⭐⭐ Very Good | OpenAI API users | OpenAI training |
| **text-embedding-3-large** | 3072 | Proprietary | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | Maximum quality | OpenAI training |
| **multilingual-e5-base** | 768 | 109M | ⚡⚡ Medium | ⭐⭐⭐⭐ Very Good | 100+ languages | Translated pairs |
| **jina-embeddings-v2** | 8192 | Proprietary | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | Long context (8K) | Long-document data |

**Selection Guidelines:**
- **Latency-critical (<100ms)**: Use all-MiniLM (384-dim, 22M params)
- **Accuracy-critical**: Use bge-large or instructor-xl (despite slower speed)
- **Multilingual**: Use multilingual-e5 (supported language count crucial)
- **Long documents (>512 tokens)**: Use jina-embeddings-v2 (8K context)"""
    },
    "lora.md": {
        "detailed_overview": """## LoRA (Low-Rank Adaptation) Deep Dive

LoRA represents a mathematical insight that allows efficient fine-tuning through low-rank updates. The key innovation: instead of fine-tuning weight matrices W directly, we represent the weight update as ΔW = AB^T, where A and B are small low-rank matrices.

### Mathematical Foundation

For a weight matrix W ∈ R^{d_out × d_in}:
- Standard fine-tuning: Update all d_out × d_in parameters
- LoRA approach: Decompose update as ΔW = AB^T where:
  - A ∈ R^{d_out × r} (r is rank, typically 4-8)
  - B ∈ R^{d_in × r}
  - Total parameters: r(d_out + d_in) << d_out × d_in

**Example with GPT-3 (175B parameters, d=12288):**
- Standard: 12288 × 12288 = 150M parameters per layer
- LoRA (r=8): 8(12288 + 12288) = 196K parameters per layer
- Compression: 766× reduction (150M → 196K)

### Why Low-Rank Works

Empirical evidence suggests that weight updates during fine-tuning are intrinsically low-rank. The model doesn't need to explore the full d × d dimensional space of possible updates; it primarily moves along a lower-dimensional manifold. LoRA exploits this structure.""",
        "comparison_table": """## LoRA Rank and Configuration Trade-offs

| Rank | Parameters | Training Speed | Accuracy | Memory | Best For |
|------|-----------|-----------------|----------|--------|----------|
| **2** | 196K | ⚡⚡⚡ Fastest | ⭐⭐ Minimal | Lowest | Proof-of-concept |
| **4** | 393K | ⚡⚡ Fast | ⭐⭐⭐ Good | Low | Resource-constrained |
| **8** | 786K | ⚡ Medium | ⭐⭐⭐⭐ Very Good | Medium | **Standard choice** |
| **16** | 1.5M | Medium | ⭐⭐⭐⭐ Very Good | Medium-High | Complex tasks |
| **32** | 3M | ⚡ Slower | ⭐⭐⭐⭐⭐ Excellent | High | Precision-critical |
| **64** | 6M | ⚡⚡ Much Slower | ⭐⭐⭐⭐⭐ Excellent | Very High | Approaching full FT |

### LoRA Variants and Extensions

| Variant | Description | Use Case | Trade-off |
|---------|-------------|----------|-----------|
| **Standard LoRA** | Rank-r matrices on Q,V projections | General | Baseline |
| **QLoRA** | LoRA on quantized (INT4) base model | Memory-constrained | Slight accuracy loss |
| **DoRA** | Weight-decomposed LoRA (separate magnitude) | Improved stability | +10% training cost |
| **LoRA+** | Different LR for A vs B matrices | Fine-grained control | Hyperparameter tuning |
| **LoRA Merge** | Combine multiple task LoRAs | Multi-task | Custom merging logic |"""
    },
    "quantization.md": {
        "detailed_overview": """## Model Quantization: Compression and Efficiency

Quantization reduces model size and computational requirements by representing weights and activations using lower-precision data types. The fundamental insight: neural networks have redundancy that allows lower precision without significant accuracy loss.

### Quantization Types and Mechanisms

**1. Weight Quantization (Primary target for deployment)**
- Reduces stored model size
- Example: FP32 (4 bytes/param) → INT8 (1 byte/param) = 4× compression
- Applied: offline, once per model

**2. Activation Quantization (Less common, complex)**
- Reduces runtime computation
- Depends on input distribution
- Applied: during inference

**3. Bit-Width Variations**
- FP32: Full precision (baseline)
- FP16/BF16: Half precision (minimal loss)
- INT8: 8-bit integer (1-2% loss)
- INT4: 4-bit integer (1-3% loss)
- INT2/INT1: Extreme (significant accuracy loss)

### Calibration: The Critical Step

Quantization requires calibration to determine appropriate scale factors:
1. **Collect calibration data**: Run forward pass on representative inputs (first 100-500 batches)
2. **Compute statistics**: Min/max values, percentiles (KL divergence, entropy)
3. **Determine scales**: Map FP32 range to lower-bit range
4. **Validate**: Check accuracy on validation set""",
        "comparison_table": """## Quantization Methods Comparison

| Method | Training Time | Accuracy Loss | Speed Gain | Memory Reduction | Complexity | Production Ready |
|--------|--------------|---------------|-----------|-----------------|-----------|-----------------|
| **Post-Training INT8** | None (fast) | 1-2% | 2-3× | 4× | Low | ✅ Yes |
| **Post-Training INT4** | None (fast) | 2-4% | 3-5× | 8× | Low | ✅ Yes (GPTQ) |
| **QAT (INT8)** | Hours | 0.5-1% | 2-3× | 4× | High | ✅ Yes |
| **QAT (INT4)** | Days | 1-2% | 3-5× | 8× | High | ⚠️ Emerging |
| **Knowledge Distillation** | Days | Variable | 5-10× | 10-50× | Very High | ✅ Yes |
| **Pruning + Quant** | Hours-Days | 2-3% | 5-10× | 10-20× | Medium | ✅ Yes |
| **Sparsity** | Variable | 1-3% | 3-8× | 4-16× | Medium | ⚠️ Emerging |

### Quantization Precision Trade-offs

| Precision | Bytes/Param | Model Size (7B) | Typical Accuracy Loss | Deployment |
|-----------|-----------|-----------------|----------------------|------------|
| **FP32** | 4 | 28 GB | Baseline (0%) | GPU/CPU |
| **FP16/BF16** | 2 | 14 GB | <0.5% | GPU |
| **INT8** | 1 | 7 GB | 1-2% | GPU/CPU/Mobile |
| **INT4 (GPTQ)** | 0.5 | 3.5 GB | 2-4% | Mobile/Edge |
| **INT3** | 0.375 | 2.6 GB | 4-8% | Research |
| **INT2/INT1** | 0.125-0.25 | 0.9-1.7 GB | 10-30% | Research |"""
    },
}

def add_detailed_section(content, concept_name, detailed_text, comparison_table):
    """Add detailed overview after TL;DR section."""
    # Find the position after TL;DR section
    tldr_pattern = r"(## TL;DR\n.*?)(\n## )"
    match = re.search(tldr_pattern, content, re.DOTALL)

    if match:
        # Insert detailed overview after TL;DR and before next section
        insertion_point = match.end(1)
        new_content = (content[:insertion_point] +
                      "\n\n" + detailed_text +
                      comparison_table +
                      content[insertion_point:])
        return new_content
    return content

def enhance_markdown(filepath, expansion):
    """Enhance markdown file with detailed content."""
    print(f"Expanding {filepath.name}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Add detailed overview and comparison table
    if 'detailed_overview' in expansion and 'comparison_table' in expansion:
        content = add_detailed_section(
            content,
            filepath.stem,
            expansion['detailed_overview'],
            expansion['comparison_table']
        )

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  ✓ Added detailed explanations and comparison tables")

def main():
    """Expand markdown files with detailed content."""
    print("Starting markdown expansion...\n")

    concept_files = list(CONCEPTS_DIR.glob("*.md"))
    print(f"Found {len(concept_files)} concept files")
    print(f"Ready to expand {len(MARKDOWN_EXPANSIONS)} concepts\n")

    expanded_count = 0
    for concept_file in sorted(concept_files):
        if concept_file.name in MARKDOWN_EXPANSIONS:
            expansion = MARKDOWN_EXPANSIONS[concept_file.name]
            enhance_markdown(concept_file, expansion)
            expanded_count += 1
        else:
            print(f"⊘ {concept_file.name} (expansion pending)")

    print(f"\n✓ Expanded {expanded_count} concepts with detailed content and tables")

if __name__ == "__main__":
    main()
