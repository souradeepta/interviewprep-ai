# LLM Concept Interactive Notebooks

32 interactive Jupyter notebooks covering essential LLM concepts, from foundation (pretraining, tokenization) to advanced optimization and specialized topics.

## Quick Navigation

**Start here:** [00-concept-map.ipynb](00-concept-map.ipynb) — Master overview of all concepts and relationships.

## Notebook Categories

### Foundation (3)
- [01-adapters.ipynb](01-adapters.ipynb) — Lightweight fine-tuning modules
- [19-pretraining.ipynb](19-pretraining.ipynb) — How LLMs are initially trained
- [30-tokenization.ipynb](30-tokenization.ipynb) — Converting text to tokens

### Fine-tuning Ecosystem (8)
- [10-finetuning.ipynb](10-finetuning.ipynb) — Full parameter updates
- [15-lora.ipynb](15-lora.ipynb) — Low-rank adaptation (efficient fine-tuning)
- [17-parameter-efficient-finetuning.ipynb](17-parameter-efficient-finetuning.ipynb) — Reduce trainable params
- [18-prefix-tuning.ipynb](18-prefix-tuning.ipynb) — Prefix-based fine-tuning
- [13-instruction-tuning.ipynb](13-instruction-tuning.ipynb) — Optimize for instructions
- [25-rlhf.ipynb](25-rlhf.ipynb) — Reinforcement learning from human feedback
- [6-dpo.ipynb](06-dpo.ipynb) — Direct preference optimization

### Prompting & In-Context Learning (6)
- [21-prompting.ipynb](21-prompting.ipynb) — Basic LLM interaction
- [11-in-context-learning.ipynb](11-in-context-learning.ipynb) — Learn from examples
- [9-few-shot-learning.ipynb](09-few-shot-learning.ipynb) — Multiple examples
- [32-zero-shot-learning.ipynb](32-zero-shot-learning.ipynb) — No examples needed
- [3-chain-of-thought.ipynb](03-chain-of-thought.ipynb) — Reasoning step-by-step
- [20-prompt-optimization.ipynb](20-prompt-optimization.ipynb) — Craft effective prompts

### Knowledge & Retrieval (5)
- [7-embeddings.ipynb](07-embeddings.ipynb) — Dense text vectors
- [27-semantic-search.ipynb](27-semantic-search.ipynb) — Find similar content
- [31-vector-databases.ipynb](31-vector-databases.ipynb) — Store/query embeddings
- [23-rag.ipynb](23-rag.ipynb) — Retrieval-augmented generation
- [26-semantic-caching.ipynb](26-semantic-caching.ipynb) — Cache by semantic meaning

### Optimization & Efficiency (8)
- [22-quantization.ipynb](22-quantization.ipynb) — Reduce model size
- [12-inference-optimization.ipynb](12-inference-optimization.ipynb) — Faster generation
- [14-kv-cache.ipynb](14-kv-cache.ipynb) — Memory-efficient caching
- [2-attention-optimization.ipynb](02-attention-optimization.ipynb) — Efficient attention
- [5-continuous-batching.ipynb](05-continuous-batching.ipynb) — Better throughput
- [28-speculative-decoding.ipynb](28-speculative-decoding.ipynb) — Parallel decoding
- [29-token-optimization.ipynb](29-token-optimization.ipynb) — Minimize tokens
- [4-context-window.ipynb](04-context-window.ipynb) — Maximum input length

### Specialized Topics (2)
- [16-multimodal.ipynb](16-multimodal.ipynb) — Text + vision + audio
- [8-evaluation.ipynb](08-evaluation.ipynb) — Assessment & metrics

## Notebook Structure

Each notebook contains:

1. **Metadata** — Concept tags, prerequisites, related concepts
2. **TL;DR** — Quick summary and use cases
3. **Core Intuition** — Plain-language explanation
4. **How It Works** — Step-by-step process + workflow flowchart
5. **Key Properties** — Trade-offs and comparisons
6. **Code** — Real Python examples and pseudocode
7. **Related Concepts** — Relationship flowchart showing connections
8. **Interview Q&A** — Common questions and concise answers
9. **References** — Papers, docs, and implementations

## Learning Paths

### For Interview Prep
1. Start with [00-concept-map.ipynb](00-concept-map.ipynb)
2. Pick 3-5 concepts relevant to your target role
3. Read the TL;DR and Interview Q&A sections
4. Review code examples if relevant

**Recommended:** RAG, LoRA, Quantization, Attention Optimization, Chain-of-Thought

### For Building RAG Applications
1. [7-embeddings.ipynb](07-embeddings.ipynb) — Understand embeddings
2. [31-vector-databases.ipynb](31-vector-databases.ipynb) — Learn storage
3. [27-semantic-search.ipynb](27-semantic-search.ipynb) — Implement search
4. [23-rag.ipynb](23-rag.ipynb) — Integrate into generation

### For Fine-tuning a Model
1. [10-finetuning.ipynb](10-finetuning.ipynb) — Basics
2. [15-lora.ipynb](15-lora.ipynb) — Efficient approach (recommended)
3. [13-instruction-tuning.ipynb](13-instruction-tuning.ipynb) — Optimize for tasks
4. [25-rlhf.ipynb](25-rlhf.ipynb) or [6-dpo.ipynb](06-dpo.ipynb) — Align with preferences

### For Model Optimization
1. [22-quantization.ipynb](22-quantization.ipynb) — Reduce size
2. [2-attention-optimization.ipynb](02-attention-optimization.ipynb) — Faster attention
3. [14-kv-cache.ipynb](14-kv-cache.ipynb) — Memory efficiency
4. [12-inference-optimization.ipynb](12-inference-optimization.ipynb) — Overall optimization

## How to Use

### Local Jupyter
```bash
jupyter notebook llm/notebooks/
# Open any notebook and run cells interactively
```

### In Claude Code
- Open any `.ipynb` file in Claude Code
- Run code cells directly
- Modify and experiment

### For Reference
- Each notebook is self-contained
- Code examples are copy-paste ready
- Interview Q&A sections are study material

## Source Material

All notebooks are derived from detailed markdown documents in `/llm/concepts/`. Each notebook links to its source file for deeper reading.

## Maintenance

Notebooks are generated from `data/concepts_mapping.json` and markdown files. To regenerate:

```bash
python3 scripts/generate_llm_notebooks.py
python3 scripts/enrich_notebooks.py
python3 scripts/generate_concept_map.py
```

## Contributing

To improve notebooks:
1. Edit the source markdown in `/llm/concepts/`
2. Regenerate notebooks using scripts above
3. Run validation tests: `pytest tests/test_llm_notebooks.py -v`
4. Commit both markdown changes and regenerated notebooks

## Statistics

- **Total Concepts:** 32
- **Total Notebooks:** 33 (32 concepts + 1 master map)
- **Cells per Notebook:** 9
- **Concept Relationships:** 80+
- **Interview Questions:** 150+
- **Code Examples:** 32+
- **Mermaid Flowcharts:** 65+
- **Test Coverage:** 290 automated tests

---

**Generated:** May 2026  
**Status:** ✅ Complete and validated  
**Ready for:** Interview Prep, Learning, Reference
