"""
Generate implementation-focused Jupyter notebooks for LLM concepts.
Each notebook contains practical Python code demonstrating the concept.
"""

import json
from pathlib import Path
import nbformat as nbf

MAPPING_FILE = Path("data/concepts_mapping.json")
NOTEBOOKS_DIR = Path("llm/notebooks")
CONCEPTS_DIR = Path("llm/concepts")

# Implementation templates for each concept
IMPLEMENTATIONS = {
    "adapters": {
        "title": "Adapters Implementation",
        "description": "Lightweight parameter-efficient fine-tuning using adapter modules",
        "code": '''# Adapter modules: lightweight parameter-efficient fine-tuning
import torch
import torch.nn as nn

class AdapterModule(nn.Module):
    """Adapter: add trainable layers to a frozen base model"""
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Residual connection + adapter
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        x = self.layer_norm(x + residual)
        return x

# Usage example
input_dim = 768  # BERT embedding dimension
adapter = AdapterModule(input_dim, bottleneck_dim=64)

# Test forward pass
x = torch.randn(2, 10, input_dim)  # batch_size=2, seq_len=10
output = adapter(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Trainable params: {sum(p.numel() for p in adapter.parameters())}")'''
    },
    "embeddings": {
        "title": "Text Embeddings Implementation",
        "description": "Convert text to dense vector representations",
        "code": '''# Text embeddings: convert text to dense vectors
from sentence_transformers import SentenceTransformer
import numpy as np

# Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example texts
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps over a sleepy dog",
    "Machine learning is powerful"
]

# Get embeddings
embeddings = model.encode(texts, convert_to_tensor=True)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# Compute similarity between texts
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings)
print(f"\\nSimilarity matrix:")
for i, text in enumerate(texts):
    for j in range(i+1, len(texts)):
        print(f"  '{text[:30]}...' <-> '{texts[j][:30]}...': {similarity[i][j]:.3f}")'''
    },
    "lora": {
        "title": "LoRA (Low-Rank Adaptation) Implementation",
        "description": "Parameter-efficient fine-tuning using low-rank matrices",
        "code": '''# LoRA: Low-Rank Adaptation for parameter-efficient fine-tuning
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.base_linear = nn.Linear(in_features, out_features)

        # LoRA: low-rank update matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_alpha = 1.0
        self.rank = rank

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Base output + LoRA update
        base_out = self.base_linear(x)
        lora_out = self.lora_B(self.lora_A(x)) * (self.lora_alpha / self.rank)
        return base_out + lora_out

# Usage
in_dim, out_dim = 768, 3072
lora_layer = LoRALinear(in_dim, out_dim, rank=8)

# Compare parameters
base_params = in_dim * out_dim + out_dim
lora_params = in_dim * 8 + 8 * out_dim
print(f"Base linear layer params: {base_params:,}")
print(f"LoRA added params: {lora_params:,}")
print(f"Compression ratio: {base_params / lora_params:.1f}x")

# Forward pass
x = torch.randn(2, 10, in_dim)
output = lora_layer(x)
print(f"\\nOutput shape: {output.shape}")'''
    },
    "quantization": {
        "title": "Model Quantization Implementation",
        "description": "Reduce model size by quantizing weights",
        "code": '''# Quantization: reduce model size by converting to lower precision
import torch
import torch.nn as nn

class QuantizedLinear(nn.Module):
    """Linear layer with weight quantization"""
    def __init__(self, in_features, out_features, bits=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.bits = bits
        self.scale = None

    def quantize(self):
        """Quantize weights to lower precision"""
        if self.bits == 8:
            # INT8 quantization
            min_val = self.weight.min()
            max_val = self.weight.max()
            self.scale = (max_val - min_val) / 255
            self.weight_quant = ((self.weight - min_val) / self.scale).to(torch.uint8)
        elif self.bits == 4:
            # INT4 quantization
            min_val = self.weight.min()
            max_val = self.weight.max()
            self.scale = (max_val - min_val) / 15
            self.weight_quant = ((self.weight - min_val) / self.scale).to(torch.uint8)

    def dequantize(self):
        """Convert back to float32"""
        if self.scale is not None:
            return self.weight_quant.float() * self.scale
        return self.weight

    def forward(self, x):
        weight = self.dequantize()
        return torch.nn.functional.linear(x, weight, self.bias)

# Usage
original = QuantizedLinear(768, 3072, bits=8)
original.quantize()

# Size comparison
original_size = (768 * 3072 + 3072) * 4  # float32: 4 bytes
quantized_size = (768 * 3072 + 3072) * 1  # int8: 1 byte
print(f"Original size: {original_size / 1e6:.1f} MB")
print(f"Quantized size: {quantized_size / 1e6:.1f} MB")
print(f"Compression: {original_size / quantized_size:.1f}x")'''
    },
    "rag": {
        "title": "RAG (Retrieval-Augmented Generation) Implementation",
        "description": "Retrieve documents and augment LLM generation",
        "code": '''# RAG: Retrieve documents and augment generation with context
from sentence_transformers import SentenceTransformer, util
import torch

class SimpleRAG:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None

    def add_documents(self, docs):
        """Index documents for retrieval"""
        self.documents = docs
        self.embeddings = self.embedder.encode(docs, convert_to_tensor=True)

    def retrieve(self, query, top_k=3):
        """Retrieve top-k relevant documents"""
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(similarities, k=min(top_k, len(self.documents)))

        retrieved = []
        for idx in top_results.indices:
            retrieved.append(self.documents[idx])
        return retrieved

    def generate_with_context(self, query):
        """Generate answer using retrieved context"""
        context_docs = self.retrieve(query, top_k=3)
        context = "\\n".join(context_docs)

        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        return prompt

# Usage
rag = SimpleRAG()
documents = [
    "Python is a high-level programming language.",
    "Machine learning uses algorithms to learn from data.",
    "Neural networks are inspired by biological neurons.",
    "Transformers are state-of-the-art for NLP tasks.",
]
rag.add_documents(documents)

# Test retrieval
query = "What is machine learning?"
retrieved = rag.retrieve(query, top_k=2)
print(f"Query: {query}")
print(f"Retrieved documents:")
for i, doc in enumerate(retrieved, 1):
    print(f"  {i}. {doc}")

# Generate with context
prompt = rag.generate_with_context(query)
print(f"\\nGeneration prompt:\\n{prompt}")'''
    },
    "tokenization": {
        "title": "Tokenization Implementation",
        "description": "Convert text to tokens for LLM input",
        "code": '''# Tokenization: convert text to tokens for LLM processing
from transformers import AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example text
text = "Hello, this is a tokenization example!"

# Tokenize
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"\\nTokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(f"Num tokens: {len(tokens)}")

# Decode back to text
decoded = tokenizer.decode(token_ids)
print(f"\\nDecoded: {decoded}")

# Batch tokenization
texts = [
    "Hello world",
    "This is tokenization",
    "Multiple examples"
]

encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(f"\\nBatch encoding:")
print(f"  Input IDs shape: {encoded['input_ids'].shape}")
print(f"  Attention mask shape: {encoded['attention_mask'].shape}")'''
    },
    "chain-of-thought": {
        "title": "Chain-of-Thought Prompting Implementation",
        "description": "Generate step-by-step reasoning",
        "code": '''# Chain-of-Thought: structure prompts for step-by-step reasoning
from typing import List

class ChainOfThoughtPrompt:
    """Generate prompts that encourage step-by-step reasoning"""

    @staticmethod
    def create_cot_prompt(question: str) -> str:
        """Create a chain-of-thought prompt"""
        return f"""Let's think about this step by step.

Question: {question}

Step 1: Understand the problem
Step 2: Break it down
Step 3: Work through each part
Step 4: Combine results

Answer:"""

    @staticmethod
    def create_few_shot_cot(examples: List[dict]) -> str:
        """Create few-shot chain-of-thought examples"""
        prompt = "Let's think about this step by step.\\n\\n"

        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\\n"
            prompt += f"Question: {example['question']}\\n"
            prompt += f"Reasoning: {example['reasoning']}\\n"
            prompt += f"Answer: {example['answer']}\\n\\n"

        prompt += "Now your turn:\\n"
        return prompt

# Usage examples
cot = ChainOfThoughtPrompt()

# Single CoT prompt
prompt1 = cot.create_cot_prompt("If a train travels 60 mph for 2.5 hours, how far does it go?")
print("Basic CoT Prompt:")
print(prompt1)
print("\\n" + "="*50 + "\\n")

# Few-shot CoT
examples = [
    {
        "question": "A book costs $20. If you buy 3 books and get 10% discount, how much do you pay?",
        "reasoning": "Original cost = 20 * 3 = 60. Discount = 60 * 0.1 = 6. Final cost = 60 - 6 = 54",
        "answer": "$54"
    }
]
prompt2 = cot.create_few_shot_cot(examples)
print("Few-Shot CoT Prompt:")
print(prompt2)'''
    },
    "few-shot-learning": {
        "title": "Few-Shot Learning Implementation",
        "description": "Learn from few examples in context",
        "code": '''# Few-shot learning: learning from limited examples
from typing import List, Dict

class FewShotLearner:
    """Few-shot learning with in-context examples"""

    def __init__(self):
        self.examples = []

    def add_example(self, input_text: str, output_text: str):
        """Add training example"""
        self.examples.append({"input": input_text, "output": output_text})

    def create_prompt(self, test_input: str) -> str:
        """Create prompt with few examples"""
        prompt = ""

        # Add examples
        for i, example in enumerate(self.examples):
            prompt += f"Example {i+1}:\\n"
            prompt += f"Input: {example['input']}\\n"
            prompt += f"Output: {example['output']}\\n\\n"

        # Add test case
        prompt += f"Test:\\n"
        prompt += f"Input: {test_input}\\n"
        prompt += f"Output: "

        return prompt

# Sentiment classification example
learner = FewShotLearner()
learner.add_example("I love this movie!", "Positive")
learner.add_example("This is terrible", "Negative")
learner.add_example("It's okay, nothing special", "Neutral")

prompt = learner.create_prompt("This product is amazing!")
print("Few-shot Learning Prompt:")
print(prompt)
print("\\n" + "="*50 + "\\n")

# Named entity recognition example
learner2 = FewShotLearner()
learner2.add_example("John works at Google", "John (PERSON), Google (ORG)")
learner2.add_example("Apple is in Cupertino", "Apple (ORG), Cupertino (LOC)")

prompt2 = learner2.create_prompt("Microsoft is in Seattle")
print("NER Few-shot Example:")
print(prompt2)'''
    },
    "prompt-optimization": {
        "title": "Prompt Optimization Implementation",
        "description": "Craft effective prompts for LLMs",
        "code": '''# Prompt optimization: techniques for better prompts
from typing import List

class PromptOptimizer:
    """Techniques for crafting effective prompts"""

    @staticmethod
    def add_context(prompt: str, context: str) -> str:
        """Add relevant context"""
        return f"Context: {context}\\n\\nPrompt: {prompt}"

    @staticmethod
    def add_role(prompt: str, role: str) -> str:
        """Specify a role"""
        return f"You are {role}.\\n\\n{prompt}"

    @staticmethod
    def add_format(prompt: str, format_spec: str) -> str:
        """Specify output format"""
        return f"{prompt}\\n\\nReturn format: {format_spec}"

    @staticmethod
    def chain_prompts(prompts: List[str]) -> str:
        """Chain multiple prompts"""
        return "\\n\\n".join([f"Step {i+1}: {p}" for i, p in enumerate(prompts)])

optimizer = PromptOptimizer()

# Example 1: Basic prompt vs. optimized
basic = "Write a poem"
print("Basic prompt:")
print(f"  {basic}")
print()

optimized = optimizer.add_role(
    optimizer.add_format(
        "Write a short poem about nature",
        "JSON with 'title', 'lines' array, 'theme'"
    ),
    "a creative writer"
)
print("Optimized prompt:")
print(f"  {optimized}")
print()
print("="*50)
print()

# Example 2: Multi-step prompting
steps = [
    "Read the following article",
    "Identify the main points",
    "Summarize in 3 sentences",
    "Extract key concepts"
]
chained = optimizer.chain_prompts(steps)
print("Chained prompts:")
print(chained)'''
    },
}

def get_implementation_code(concept_key: str, concept_title: str) -> str:
    """Get implementation code for a concept"""
    if concept_key in IMPLEMENTATIONS:
        impl = IMPLEMENTATIONS[concept_key]
        return impl["code"]

    # Generic template
    return f'''# {concept_title} Implementation

import torch
import torch.nn as nn

# TODO: Implement {concept_title}
# Add your implementation here

# Example usage
if __name__ == "__main__":
    print("Implement {concept_title} here")'''

def generate_notebook(concept_key: str, concept_data: dict) -> nbf.NotebookNode:
    """Generate implementation-focused notebook"""

    nb = nbf.v4.new_notebook()

    title = concept_data.get("title", concept_key)

    # Cell 1: Title and overview
    overview = f"""# {title} - Implementation

This notebook demonstrates a practical implementation of **{title}** in Python.

**Concept:** {concept_data.get("tags", [])}
**Source:** `llm/concepts/{concept_key}.md`

## What you'll learn:
- How to implement {title} from scratch
- Key components and their interactions
- Practical usage examples
- Performance considerations"""

    nb.cells.append(nbf.v4.new_markdown_cell(overview))

    # Cell 2: Imports and setup
    nb.cells.append(nbf.v4.new_markdown_cell("## Setup"))
    nb.cells.append(nbf.v4.new_code_cell('''# Install required packages (if needed)
# !pip install torch transformers sentence-transformers numpy scikit-learn'''))

    # Cell 3: Implementation
    nb.cells.append(nbf.v4.new_markdown_cell("## Implementation"))
    code = get_implementation_code(concept_key, title)
    nb.cells.append(nbf.v4.new_code_cell(code))

    # Cell 4: Testing/Verification
    nb.cells.append(nbf.v4.new_markdown_cell("## Testing"))
    nb.cells.append(nbf.v4.new_code_cell('''# Run the implementation above to test
# The code includes example usage and verification'''))

    # Cell 5: Key insights
    insights = f"""## Key Insights

### Performance
- Profile the implementation
- Compare with production libraries
- Identify bottlenecks

### Further Reading
- See `llm/concepts/{concept_key}.md` for detailed explanation
- Review the concept relationships to understand prerequisites
- Check implementations in HuggingFace, PyTorch, etc.

### Next Steps
1. Experiment with hyperparameters
2. Combine with other concepts
3. Apply to your own use case"""

    nb.cells.append(nbf.v4.new_markdown_cell(insights))

    return nb

def main():
    """Generate all implementation notebooks"""
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    concepts = mapping.get("concepts", {})

    for concept_key, concept_data in sorted(concepts.items(), key=lambda x: x[1].get("order")):
        order = concept_data.get("order", 0)
        title = concept_data.get("title", concept_key)

        print(f"Generating implementation notebook: {title} ({order}/32)...")

        nb = generate_notebook(concept_key, concept_data)

        notebook_path = NOTEBOOKS_DIR / f"{order:02d}-{concept_key}.ipynb"
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)

        print(f"  ✓ Generated: {notebook_path.name}")

    print(f"\n✓ Generated {len(concepts)} implementation notebooks")

if __name__ == "__main__":
    main()
