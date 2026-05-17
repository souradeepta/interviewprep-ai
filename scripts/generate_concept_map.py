"""
Generate a master concept map showing relationships among all LLM concepts.
Creates a single comprehensive flowchart notebook.
"""

import json
import nbformat as nbf
from pathlib import Path

MAPPING_FILE = Path("data/concepts_mapping.json")
NOTEBOOKS_DIR = Path("llm/notebooks")

def generate_concept_map_notebook() -> nbf.NotebookNode:
    """Create a master concept map notebook."""

    nb = nbf.v4.new_notebook()

    # Title cell
    title_cell = nbf.v4.new_markdown_cell("""# LLM Concepts Map

Master reference showing relationships among all 32 LLM concepts covered in this repository.
Use this to understand prerequisites, dependencies, and concept families.

## Navigation

- **Foundational:** Pretraining, Tokenization, Prompting
- **Fine-tuning Family:** Fine-tuning, LoRA, Adapters, Prefix-tuning, Parameter-Efficient Fine-tuning, Instruction-tuning, RLHF, DPO
- **Prompting & In-Context:** In-Context Learning, Few-Shot Learning, Zero-Shot Learning, Chain-of-Thought, Prompt Optimization
- **Retrieval & Knowledge:** RAG, Embeddings, Vector Databases, Semantic Search, Semantic Caching
- **Optimization:** Quantization, KV Cache, Attention Optimization, Speculative Decoding, Continuous Batching, Inference Optimization, Token Optimization
- **Specialized:** Multimodal, Evaluation, Context Window
""")
    nb.cells.append(title_cell)

    # Load concept mapping
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    concepts = mapping.get("concepts", {})

    # Create comprehensive relationship mermaid diagram
    mermaid_lines = ["graph TD"]
    mermaid_lines.append("")

    # Add all concepts as nodes
    for concept_key, concept_data in sorted(concepts.items(), key=lambda x: x[1].get("order", 999)):
        title = concept_data.get("title", concept_key)
        order = concept_data.get("order", 0)
        mermaid_lines.append(f'    {order}["{title}"]')

    mermaid_lines.append("")

    # Add relationships
    relationship_count = 0
    for concept_key, concept_data in concepts.items():
        order = concept_data.get("order", 0)
        related = concept_data.get("related", {})

        # Add prerequisite edges
        for prereq_key in concept_data.get("prerequisites", []):
            prereq_data = concepts.get(prereq_key, {})
            prereq_order = prereq_data.get("order", 0)
            mermaid_lines.append(f'    {prereq_order} -->|prerequisite| {order}')
            relationship_count += 1

        # Add "used with" edges (limit to avoid clutter)
        for related_concept in related.get("used_with", [])[:2]:
            related_data = concepts.get(related_concept, {})
            related_order = related_data.get("order", 0)
            if related_order > order:  # Only add one direction to avoid duplicates
                mermaid_lines.append(f'    {order} -.->|used with| {related_order}')
                relationship_count += 1

    mermaid_diagram = "\n".join(mermaid_lines)

    # Add diagram cell
    diagram_cell = nbf.v4.new_markdown_cell(f"""## Full Concept Relationship Graph

```mermaid
{mermaid_diagram}
```

**Legend:**
- Solid arrow (→): prerequisite relationship
- Dotted arrow (-.->): often used together
- Node number matches the concept order in this repo
""")
    nb.cells.append(diagram_cell)

    # Add concept families summary
    families_cell = nbf.v4.new_markdown_cell("""## Concept Families

### Foundation & Training
- **Pretraining** → Base model development
- **Tokenization** → Converting text to tokens
- **Evaluation** → Assessing model performance

### Fine-tuning Ecosystem
- **Fine-tuning** → Full parameter updates
  - **Parameter-Efficient Fine-tuning** → Reduce trainable parameters
    - **LoRA** → Low-rank updates
    - **Adapters** → Lightweight modules
    - **Prefix-tuning** → Prefix tokens only
- **Instruction-tuning** → Optimize for instructions
- **RLHF** → Human feedback alignment
- **DPO** → Direct preference optimization (alternative to RLHF)

### Prompting & In-Context Learning
- **Prompting** → Basic interaction
  - **In-Context Learning** → Learning from examples
    - **Few-Shot Learning** → Multiple examples
    - **Zero-Shot Learning** → No examples
    - **Chain-of-Thought** → Step-by-step reasoning
  - **Prompt Optimization** → Crafting effective prompts

### Knowledge & Retrieval
- **Embeddings** → Dense vector representations
  - **Semantic Search** → Find similar content
    - **Vector Databases** → Store/query embeddings
    - **Semantic Caching** → Cache by meaning
  - **RAG** → Augment generation with retrieval

### Optimization & Efficiency
- **Quantization** → Reduce precision/size
- **Inference Optimization** → Faster generation
  - **KV Cache** → Store key-value pairs
  - **Attention Optimization** → Efficient attention
  - **Continuous Batching** → Better throughput
  - **Speculative Decoding** → Parallel decoding
  - **Token Optimization** → Minimize tokens

### Advanced Topics
- **Multimodal** → Text + vision + audio
- **Context Window** → Maximum input length
- **Retrieval-Augmented Generation** → Advanced RAG patterns
""")
    nb.cells.append(families_cell)

    # Add usage guide
    usage_cell = nbf.v4.new_markdown_cell("""## How to Use This Repository

1. **Start with foundations:** Pretraining → Tokenization → Prompting
2. **Pick your path:**
   - **Interview prep?** Browse any concept notebook for Q&A and code
   - **Build RAG app?** Follow: Embeddings → Vector Databases → Semantic Search → RAG
   - **Fine-tune a model?** Follow: Fine-tuning → (Pick one: LoRA / Adapters / Prefix-tuning)
3. **Each notebook has:**
   - TL;DR & intuition
   - Workflow diagram
   - Code examples
   - Related concepts
   - Interview Q&A

## Quick Links

| Concept | Purpose | Learn Time |
|---------|---------|-----------|
| [RAG](23-rag.ipynb) | Add knowledge to LLMs | 15 min |
| [LoRA](15-lora.ipynb) | Efficient fine-tuning | 10 min |
| [Quantization](22-quantization.ipynb) | Make models smaller | 10 min |
| [Chain-of-Thought](03-chain-of-thought.ipynb) | Improve reasoning | 10 min |
| [Embeddings](07-embeddings.ipynb) | Text vectors | 10 min |
""")
    nb.cells.append(usage_cell)

    return nb

def main():
    """Generate and save concept map notebook."""
    print("Generating master concept map...")

    nb = generate_concept_map_notebook()

    concept_map_path = NOTEBOOKS_DIR / "00-concept-map.ipynb"
    with open(concept_map_path, 'w') as f:
        nbf.write(nb, f)

    print(f"✓ Concept map created: {concept_map_path}")

if __name__ == "__main__":
    main()
