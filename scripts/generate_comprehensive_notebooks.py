"""
Generate comprehensive implementation notebooks with:
- Basic implementations
- Advanced implementations
- Real-world examples
- Production-ready code patterns
- Performance benchmarks
- Integration examples
"""

import json
from pathlib import Path
import nbformat as nbf

MAPPING_FILE = Path("data/concepts_mapping.json")
NOTEBOOKS_DIR = Path("llm/notebooks")

# Comprehensive implementations for each concept
COMPREHENSIVE_IMPLEMENTATIONS = {
    "adapters": {
        "basic": '''# Basic Adapter Module Implementation
import torch
import torch.nn as nn

class SimpleAdapter(nn.Module):
    """Minimal adapter implementation: down-project -> activate -> up-project"""

    def __init__(self, hidden_size=768, bottleneck_size=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        down = self.down(x)
        activated = self.activation(down)
        up = self.up(activated)
        return x + up  # Residual connection

# Test
adapter = SimpleAdapter(hidden_size=768, bottleneck_size=64)
x = torch.randn(2, 10, 768)  # batch=2, seq_len=10, hidden=768
output = adapter(x)
print(f"Input: {x.shape}, Output: {output.shape}")
print(f"Params: {sum(p.numel() for p in adapter.parameters()):,}")''',
        "advanced": '''# Production Adapter with Layer Norm and Initialization
import torch
import torch.nn as nn
import math

class ProductionAdapter(nn.Module):
    """Production-ready adapter with layer norm and proper initialization"""

    def __init__(self, hidden_size=768, bottleneck_size=64, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(bottleneck_size)

        # Proper initialization
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up(x) * self.scale
        return residual + x

# Usage with transformer
adapter = ProductionAdapter(hidden_size=768, bottleneck_size=64)
x = torch.randn(2, 10, 768)
output = adapter(x)
print(f"Params: {sum(p.numel() for p in adapter.parameters()):,}")
print(f"Trainable: {sum(p.numel() for p in adapter.parameters() if p.requires_grad):,}")''',
        "realworld_huggingface": '''# Real-World: Using HuggingFace PEFT Library
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure adapters (actually using LoRA from PEFT for production)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # rank
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

# Wrap model with adapters
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# Training setup
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-4)

# Dummy batch
input_ids = torch.randint(0, 50257, (2, 10))
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")''',
        "realworld_multiTask": '''# Real-World: Multi-Task Adapter Sharing
class MultiTaskAdapterModel(nn.Module):
    """Shared base model with task-specific adapters"""

    def __init__(self, base_model, num_tasks=3, adapter_dim=64):
        super().__init__()
        self.base = base_model

        # One adapter per task
        self.adapters = nn.ModuleDict({
            f"task_{i}": ProductionAdapter(768, adapter_dim)
            for i in range(num_tasks)
        })

        # Task-specific classifiers
        self.classifiers = nn.ModuleDict({
            f"task_{i}": nn.Linear(768, 2)  # Binary classification
            for i in range(num_tasks)
        })

    def forward(self, input_ids, task_id, attention_mask=None):
        # Shared base
        hidden = self.base(input_ids, attention_mask).last_hidden_state

        # Task-specific adapter
        adapted = self.adapters[f"task_{task_id}"](hidden[:, 0])

        # Task-specific classifier
        logits = self.classifiers[f"task_{task_id}"](adapted)
        return logits

# Usage: same base model, different adapters for different tasks
# Total params = base + (num_tasks * adapter_params) << (base + num_tasks * full_params)'''
    },
    "embeddings": {
        "basic": '''# Basic Embedding from Scratch
import torch
import torch.nn as nn

class SimpleEmbedding(nn.Module):
    """Basic embedding: transform text tokens to vectors"""

    def __init__(self, vocab_size=10000, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids):
        return self.embedding(input_ids)

# Test
embed = SimpleEmbedding(vocab_size=10000, embedding_dim=128)
input_ids = torch.randint(0, 10000, (2, 10))  # batch=2, seq_len=10
embeddings = embed(input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Embedding shape: {embeddings.shape}")  # (2, 10, 128)''',
        "advanced": '''# Advanced: Sentence Embeddings with Pooling
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentenceEmbedding(nn.Module):
    """Convert sentences to fixed-size embeddings"""

    def __init__(self, hidden_size=768, embedding_dim=384):
        super().__init__()
        self.transform = nn.Linear(hidden_size, embedding_dim)

    def forward(self, hidden_states, attention_mask=None):
        # Mean pooling: average over sequence length
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * attention_mask
            sum_embeddings = torch.sum(hidden_states, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1)
            embeddings = sum_embeddings / sum_mask
        else:
            embeddings = torch.mean(hidden_states, dim=1)

        # Project to embedding dim
        embeddings = self.transform(embeddings)
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

# Usage
pooler = SentenceEmbedding(hidden_size=768, embedding_dim=384)
hidden = torch.randn(2, 10, 768)  # 2 sentences, 10 tokens each
embeddings = pooler(hidden)
print(f"Sentence embeddings shape: {embeddings.shape}")  # (2, 384)

# Compute similarity
similarity = torch.mm(embeddings, embeddings.t())
print(f"Similarity matrix:\\n{similarity}")''',
        "realworld_sentencebert": '''# Real-World: Using Sentence-BERT
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "The cat is on the mat",
    "A feline rests on a rug",
    "Python is a programming language",
    "Java is also a programming language"
]

embeddings = model.encode(sentences, convert_to_tensor=True)

# Compute similarity
similarities = util.pytorch_cos_sim(embeddings, embeddings)

# Find most similar pairs
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = similarities[i][j].item()
        print(f"{sentences[i]} <-> {sentences[j]}: {sim:.3f}")''',
        "realworld_production": '''# Real-World: Production Embedding Pipeline
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EmbeddingService:
    """Production embedding service for semantic search"""

    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.document_cache = {}

    def embed_documents(self, documents):
        """Embed documents and cache them"""
        embeddings = self.model.encode(documents, batch_size=self.batch_size)
        for doc, emb in zip(documents, embeddings):
            self.document_cache[hash(doc)] = emb
        return embeddings

    def search(self, query, documents, top_k=5):
        """Find top-k most similar documents"""
        query_emb = self.model.encode([query])[0]
        doc_embs = self.embed_documents(documents)

        similarities = cosine_similarity([query_emb], doc_embs)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            {"doc": documents[i], "score": similarities[i]}
            for i in top_indices
        ]
        return results

# Usage
service = EmbeddingService()
docs = [
    "Machine learning uses algorithms to learn from data",
    "Deep learning uses neural networks",
    "Python is popular for ML"
]
results = service.search("What is machine learning?", docs)
for r in results:
    print(f"{r['doc']}: {r['score']:.3f}")'''
    },
    "lora": {
        "basic": '''# Basic LoRA Implementation
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Linear layer with LoRA: W = W0 + A @ B^T"""

    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        # LoRA weights
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora = self.lora_B(self.lora_A(x))
        return base + lora

# Test
layer = LoRALinear(768, 3072, rank=8)
x = torch.randn(2, 10, 768)
output = layer(x)
print(f"Output shape: {output.shape}")
print(f"Params: {sum(p.numel() for p in layer.parameters()):,}")''',
        "advanced": '''# Advanced LoRA with Scaling and Merging
import torch
import torch.nn as nn
import math

class ScaledLoRALinear(nn.Module):
    """LoRA with alpha scaling and merge capability"""

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.alpha = alpha
        self.rank = rank
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora = self.lora_B(self.lora_A(x)) * self.scaling
        return base + lora

    def merge(self):
        """Merge LoRA into base weight for inference"""
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.weight.data += delta_w.t()
        # Disable LoRA after merge
        self.lora_A.weight.requires_grad = False
        self.lora_B.weight.requires_grad = False

# Usage
layer = ScaledLoRALinear(768, 3072, rank=8, alpha=16)
x = torch.randn(2, 768)
output = layer(x)
print(f"Training mode output: {output.shape}")
layer.merge()
output_merged = layer(x)
print(f"Merged mode (no LoRA computation)")''',
        "realworld_peft": '''# Real-World: Using HuggingFace PEFT for LoRA
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.optim import AdamW

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # Which modules to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training
optimizer = AdamW(model.parameters(), lr=1e-4)
model.train()

# Forward pass
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs, labels=inputs.input_ids)
loss = outputs.loss

loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"Loss: {loss.item():.4f}")''',
        "realworld_multilora": '''# Real-World: Multi-Task LoRA
class MultiTaskLoRA:
    """Manage multiple LoRA adapters for different tasks"""

    def __init__(self, base_model, model_name):
        self.base_model = base_model
        self.loras = {}
        self.active_lora = None

    def add_task(self, task_name, lora_config):
        """Add a LoRA adapter for a new task"""
        from peft import get_peft_model
        model = get_peft_model(self.base_model, lora_config)
        self.loras[task_name] = model

    def switch_task(self, task_name):
        """Switch active task"""
        if task_name not in self.loras:
            raise ValueError(f"Task {task_name} not found")
        self.active_lora = task_name

    def forward(self, *args, **kwargs):
        """Forward pass with active LoRA"""
        if self.active_lora is None:
            return self.base_model(*args, **kwargs)
        return self.loras[self.active_lora](*args, **kwargs)

# Usage: same base model, different LoRA for sentiment, NER, QA
# Each task-specific LoRA is ~0.5% of base model size'''
    },
    "rag": {
        "basic": '''# Basic RAG Pipeline
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BasicRAG:
    """Simple retrieval-augmented generation"""

    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None

    def add_documents(self, docs):
        self.documents = docs
        self.embeddings = self.embedder.encode(docs)

    def retrieve(self, query, top_k=3):
        query_emb = self.embedder.encode([query])[0]
        scores = cosine_similarity([query_emb], self.embeddings)[0]
        top_idx = scores.argsort()[::-1][:top_k]
        return [self.documents[i] for i in top_idx]

    def generate_prompt(self, query):
        context = "\\n".join(self.retrieve(query, top_k=3))
        prompt = f"""Context: {context}

Question: {query}

Answer:"""
        return prompt

# Usage
rag = BasicRAG()
docs = [
    "Python is a programming language",
    "Machine learning uses data",
    "Neural networks learn patterns"
]
rag.add_documents(docs)
prompt = rag.generate_prompt("What is ML?")
print(prompt)''',
        "advanced": '''# Advanced RAG with Re-ranking
from sentence_transformers import CrossEncoder

class AdvancedRAG:
    """RAG with dense + sparse retrieval and re-ranking"""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.dense_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.documents = []

    def add_documents(self, docs):
        self.documents = docs

    def dense_retrieve(self, query, top_k=10):
        """Dense retrieval: semantic similarity"""
        query_emb = self.dense_embedder.encode([query])[0]
        doc_embs = self.dense_embedder.encode(self.documents)

        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([query_emb], doc_embs)[0]
        top_idx = scores.argsort()[::-1][:top_k]
        return [self.documents[i] for i in top_idx]

    def rerank(self, query, candidates, top_k=3):
        """Re-rank candidates using cross-encoder"""
        pairs = [[query, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)
        sorted_idx = scores.argsort()[::-1][:top_k]
        return [candidates[i] for i in sorted_idx]

    def retrieve_best(self, query, top_k=3):
        candidates = self.dense_retrieve(query, top_k=10)
        return self.rerank(query, candidates, top_k=top_k)

# Usage
rag = AdvancedRAG()
docs = ["Python code", "Java code", "C++ code", "Learning models"]
rag.add_documents(docs)
best = rag.retrieve_best("programming language")
print(f"Best retrieved: {best}")''',
        "realworld_llm": '''# Real-World: RAG with LLM Generation
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class ProductionRAG:
    """Production RAG with LLM for answer generation"""

    def __init__(self):
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = pipeline("text-generation", model="gpt2")
        self.documents = []

    def add_documents(self, docs):
        self.documents = docs

    def retrieve_context(self, query, top_k=3):
        query_emb = self.retriever.encode(query)
        doc_embs = self.retriever.encode(self.documents)

        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([query_emb], doc_embs)[0]
        top_idx = scores.argsort()[::-1][:top_k]
        return [self.documents[i] for i in top_idx]

    def answer(self, query):
        context = "\\n".join(self.retrieve_context(query))
        prompt = f"Context: {context}\\n\\nQuestion: {query}\\n\\nAnswer:"

        response = self.llm(prompt, max_length=100)
        return response[0]['generated_text']

# Usage
rag = ProductionRAG()
rag.add_documents([
    "Paris is the capital of France",
    "Tokyo is the capital of Japan",
    "Berlin is the capital of Germany"
])
answer = rag.answer("What is the capital of France?")
print(f"Answer: {answer}")''',
        "realworld_vectordb": '''# Real-World: RAG with Vector Database
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

class VectorDBRAG:
    """RAG using vector database for scalability"""

    def __init__(self, api_key):
        # Note: requires Pinecone account
        # self.pc = Pinecone(api_key=api_key)
        # self.index = self.pc.Index("llm-index")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def index_documents(self, documents):
        """Index documents in vector DB"""
        embeddings = self.embedder.encode(documents)

        # Upsert to Pinecone (requires setup)
        # self.index.upsert([(str(i), emb.tolist()) for i, emb in enumerate(embeddings)])

    def retrieve(self, query, top_k=5):
        """Query vector database"""
        query_emb = self.embedder.encode(query)

        # results = self.index.query(query_emb.tolist(), top_k=top_k)
        # return results

# For production: use Pinecone, Weaviate, or Milvus
# Handles millions of documents with millisecond latency'''
    },
    "tokenization": {
        "basic": '''# Basic Tokenization
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize single text
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

# Decode back
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")''',
        "advanced": '''# Advanced Tokenization with Batching
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Batch processing
texts = [
    "Hello world",
    "This is a longer sentence",
    "Short text"
]

# Tokenize with padding and truncation
batch = tokenizer(
    texts,
    padding=True,  # Pad to same length
    truncation=True,  # Truncate long sequences
    max_length=10,
    return_tensors="pt"  # Return as PyTorch tensors
)

print(f"Input IDs shape: {batch['input_ids'].shape}")
print(f"Attention mask shape: {batch['attention_mask'].shape}")

# Special tokens handling
print(f"[CLS] token ID: {tokenizer.cls_token_id}")
print(f"[SEP] token ID: {tokenizer.sep_token_id}")
print(f"[PAD] token ID: {tokenizer.pad_token_id}")''',
        "realworld_production": '''# Real-World: Production Tokenization Pipeline
class ProductionTokenizer:
    """Production tokenization with error handling and logging"""

    def __init__(self, model_name="bert-base-uncased", max_length=512):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize_batch(self, texts, batch_size=32):
        """Tokenize large batches efficiently"""
        all_tokens = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            all_tokens.append(tokens)

        return all_tokens

    def handle_special_cases(self, text):
        """Handle edge cases"""
        text = text.strip()
        if len(text) == 0:
            text = "[UNK]"
        if len(text) > 10000:  # Very long text
            text = text[:10000] + "..."
        return text

# Usage
tokenizer = ProductionTokenizer(max_length=128)
texts = ["Sample text 1", "Sample text 2"]
batches = tokenizer.tokenize_batch(texts)'''
    },
    "chain-of-thought": {
        "basic": '''# Basic Chain-of-Thought Prompting
def create_cot_prompt(question):
    """Create prompt encouraging step-by-step reasoning"""
    return f"""Let's solve this step by step.

Question: {question}

Step 1: What is the problem asking?
Step 2: What do we know?
Step 3: What steps do we need?
Step 4: Let's solve it

Answer:"""

# Example
q = "If a train travels 60 mph for 2.5 hours, how far does it go?"
prompt = create_cot_prompt(q)
print(prompt)''',
        "advanced": '''# Advanced: Few-Shot Chain-of-Thought
def create_fewshot_cot(examples, test_question):
    """Few-shot CoT with multiple examples"""
    prompt = "Let's solve these problems step by step.\\n\\n"

    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\\n"
        prompt += f"Question: {example['question']}\\n"
        prompt += f"Reasoning:\\n{example['reasoning']}\\n"
        prompt += f"Answer: {example['answer']}\\n\\n"

    prompt += f"Now solve:\\n"
    prompt += f"Question: {test_question}\\n"
    prompt += f"Reasoning:\\n"
    return prompt

# Examples
examples = [
    {
        "question": "A book costs $20. You buy 3 and get 10% off. Total?",
        "reasoning": "1. Total before discount: 20 * 3 = $60\\n2. Discount amount: 60 * 0.1 = $6\\n3. Final cost: 60 - 6 = $54",
        "answer": "$54"
    }
]

prompt = create_fewshot_cot(examples, "A pen costs $5. You buy 10 and get 20% off. Total?")
print(prompt)''',
        "realworld_reasoning": '''# Real-World: Chain-of-Thought for Complex Reasoning
def self_consistency_cot(question, num_samples=5):
    """Self-consistency: generate multiple reasoning paths"""

    prompts = []
    for i in range(num_samples):
        # Vary the prompt slightly to get different reasoning paths
        prompt = f"""Question: {question}

Think through this carefully step by step.
Your approach {i+1}:"""
        prompts.append(prompt)

    return prompts

# Usage: generate multiple CoT paths and vote on answer
question = "A ball is dropped from 100m. How long to hit ground? (ignore air resistance)"
prompts = self_consistency_cot(question, num_samples=3)

# In practice: send to LLM, get multiple answers, vote on correct one
for i, p in enumerate(prompts):
    print(f"Prompt {i+1}:")
    print(p)
    print()'''
    }
}

def create_comprehensive_notebook(concept_key: str, concept_data: dict) -> nbf.NotebookNode:
    """Create comprehensive notebook with multiple implementations"""

    nb = nbf.v4.new_notebook()
    title = concept_data.get("title", concept_key)

    # Cell 1: Overview
    nb.cells.append(nbf.v4.new_markdown_cell(f"""# {title} - Comprehensive Implementation Guide

This notebook covers:
- **Basic Implementation**: Simple, educational version
- **Advanced Implementation**: Production-ready patterns
- **Real-World Examples**: How companies use this in production
- **Integration**: Using popular libraries

Source: `llm/concepts/{concept_key}.md`"""))

    # Cell 2: Imports
    nb.cells.append(nbf.v4.new_markdown_cell("## Setup & Imports"))
    nb.cells.append(nbf.v4.new_code_cell('''import torch
import torch.nn as nn
import numpy as np

print("Libraries loaded successfully")'''))

    # Get implementations
    if concept_key in COMPREHENSIVE_IMPLEMENTATIONS:
        impls = COMPREHENSIVE_IMPLEMENTATIONS[concept_key]

        # Basic implementation
        if "basic" in impls:
            nb.cells.append(nbf.v4.new_markdown_cell("## 1. Basic Implementation\n\nSimple, educational version to understand core concepts."))
            nb.cells.append(nbf.v4.new_code_cell(impls["basic"]))

        # Advanced implementation
        if "advanced" in impls:
            nb.cells.append(nbf.v4.new_markdown_cell("## 2. Advanced Implementation\n\nProduction-ready patterns with optimization and error handling."))
            nb.cells.append(nbf.v4.new_code_cell(impls["advanced"]))

        # Real-world examples
        for key, code in impls.items():
            if key.startswith("realworld_"):
                example_name = key.replace("realworld_", "").replace("_", " ").title()
                nb.cells.append(nbf.v4.new_markdown_cell(f"## Real-World: {example_name}\n\nHow this is used in production systems."))
                nb.cells.append(nbf.v4.new_code_cell(code))
    else:
        # Generic implementations
        nb.cells.append(nbf.v4.new_markdown_cell("## Implementation Examples"))
        nb.cells.append(nbf.v4.new_code_cell(f'''# {title} Implementation
# TODO: Add implementation examples

# Basic pattern:
class {title.replace(" ", "")}Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize layers

    def forward(self, x):
        # Forward pass
        return x

# See corresponding markdown file for detailed explanation
'''))

    # Final cell: Learning resources
    nb.cells.append(nbf.v4.new_markdown_cell(f"""## Resources & Next Steps

- **Detailed Explanation**: See `llm/concepts/{concept_key}.md`
- **Interview Questions**: Review Q&A in markdown file
- **Real-World Examples**: See companies section in markdown
- **Experiment**: Modify the code above and run cells

### Concepts to explore next:
- Related concepts (see markdown file)
- Cross-concept combinations
- Integration with other techniques"""))

    return nb

def main():
    """Generate comprehensive notebooks"""
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    concepts = mapping.get("concepts", {})

    for concept_key, concept_data in sorted(concepts.items(), key=lambda x: x[1].get("order")):
        order = concept_data.get("order", 0)
        title = concept_data.get("title", concept_key)

        print(f"Generating comprehensive notebook: {title} ({order}/32)...")

        nb = create_comprehensive_notebook(concept_key, concept_data)

        notebook_path = NOTEBOOKS_DIR / f"{order:02d}-{concept_key}.ipynb"
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)

        print(f"  ✓ Generated: {notebook_path.name}")

    print(f"\n✓ Generated {len(concepts)} comprehensive notebooks")

if __name__ == "__main__":
    main()
