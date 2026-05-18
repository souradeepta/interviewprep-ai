#!/usr/bin/env python3
"""Generate complete LLM notebooks following the ai/notebooks pattern (12 cells, 3 levels)."""

import json
import os
import re
from pathlib import Path

BASE = "/home/sbisw/github/interviewprep-ml"

NOTEBOOK_TEMPLATE = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Concept-specific implementations
IMPLEMENTATIONS = {
    "01-tokenization": {
        "level1": """import numpy as np
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Basic tokenization
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(f"Decoded: {tokenizer.decode(token_ids)}")

# Special tokens
print(f"\\nSpecial tokens:")
print(f"  BOS: {tokenizer.bos_token_id}")
print(f"  EOS: {tokenizer.eos_token_id}")
print(f"  PAD: {tokenizer.pad_token_id}")
print(f"  UNK: {tokenizer.unk_token_id}")""",
        "level2": """import numpy as np
from transformers import AutoTokenizer
import torch

class TokenizationPipeline:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(self, texts, max_length=128):
        # Tokenize with padding and truncation
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encodings

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_token_frequency(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        return dict(zip(tokens, token_ids))

# Usage
pipeline = TokenizationPipeline()

texts = ["Hello world", "How are you?", "Great to see you"]
encodings = pipeline.tokenize_batch(texts)

print(f"Vocabulary size: {pipeline.get_vocab_size()}")
print(f"Input IDs shape: {encodings['input_ids'].shape}")
print(f"Attention mask shape: {encodings['attention_mask'].shape}")

# Show one example
print(f"\\nExample tokens: {pipeline.tokenizer.convert_ids_to_tokens(encodings['input_ids'][0])}")""",
        "example1": """# Real-World: Multi-language Tokenization
from transformers import AutoTokenizer

# Different tokenizers for different languages
tokenizers = {
    "english": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "german": AutoTokenizer.from_pretrained("bert-base-german-uncased"),
    "multilingual": AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
}

texts = {
    "en": "How are you doing today?",
    "de": "Wie geht es dir heute?",
    "multi": "Hello مرحبا Привет"
}

for lang, text in texts.items():
    tokenizer = tokenizers["multilingual"]  # Use multilingual for all
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    print(f"{lang:5} | {text:30} | Tokens: {len(tokens)}")""",
        "example2": """# Real-World: Streaming Tokenization
from transformers import AutoTokenizer

class StreamingTokenizer:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.buffer = ""

    def process_chunk(self, text_chunk):
        self.buffer += text_chunk
        # Tokenize buffer when we have complete words
        last_space = self.buffer.rfind(" ")
        if last_space > 0:
            to_tokenize = self.buffer[:last_space]
            self.buffer = self.buffer[last_space+1:]
            return self.tokenizer.encode(to_tokenize, add_special_tokens=False)
        return []

    def flush(self):
        if self.buffer:
            token_ids = self.tokenizer.encode(self.buffer, add_special_tokens=False)
            self.buffer = ""
            return token_ids
        return []

# Simulate streaming input
streamer = StreamingTokenizer()
stream = ["Hello ", "world, ", "this is ", "streaming tokenization"]

all_tokens = []
for chunk in stream:
    tokens = streamer.process_chunk(chunk)
    all_tokens.extend(tokens)

final = streamer.flush()
all_tokens.extend(final)

print(f"Total tokens from stream: {len(all_tokens)}")
print(f"Token IDs: {all_tokens[:20]}")  # First 20 tokens""",
        "example3": """# Real-World: Custom Tokenizer Training
from transformers import AutoTokenizer
import numpy as np

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Analyze token distribution on a corpus
texts = [
    "Machine learning is transforming industries",
    "Transformers have revolutionized NLP",
    "BERT and GPT are foundational models",
    "Tokenization splits text into tokens",
] * 100  # Repeat for analysis

# Encode all texts
all_token_ids = []
for text in texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    all_token_ids.extend(ids)

# Token frequency analysis
unique, counts = np.unique(all_token_ids, return_counts=True)
top_tokens = sorted(zip(counts, unique), reverse=True)[:10]

print("Top 10 tokens in corpus:")
for count, token_id in top_tokens:
    token_text = tokenizer.decode([token_id])
    print(f"  Token {token_id:5d}: '{token_text:15s}' (freq: {count})")"""
    },

    "02-embeddings": {
        "level1": """import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained embeddings
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=False)

# Tokenize and get embeddings
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

print(f"Text: {text}")
print(f"Embeddings shape: {embeddings.shape}")  # (1, seq_len, 768)
print(f"Embedding dimension: {embeddings.shape[-1]}")
print(f"First token embedding (first 10 dims): {embeddings[0, 0, :10].numpy()}")""",
        "level2": """import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize

class EmbeddingService:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_text(self, text, pool_method="mean"):
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

        # Pool tokens into single embedding
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        if pool_method == "mean":
            pooled = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif pool_method == "cls":
            pooled = embeddings[:, 0, :]  # Use [CLS] token
        else:
            raise ValueError(f"Unknown pooling method: {pool_method}")

        # Normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled.numpy()

    def embed_batch(self, texts, pool_method="mean"):
        embeddings = []
        for text in texts:
            emb = self.embed_text(text, pool_method)
            embeddings.append(emb[0])
        return np.array(embeddings)

    def similarity(self, text1, text2):
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        return np.dot(emb1[0], emb2[0].T)

# Usage
service = EmbeddingService()

# Single embedding
text = "The quick brown fox"
emb = service.embed_text(text)
print(f"Single embedding shape: {emb.shape}")

# Batch embeddings
texts = ["Hello world", "How are you", "Great to see you"]
batch_embs = service.embed_batch(texts)
print(f"Batch embeddings shape: {batch_embs.shape}")

# Similarity
sim = service.similarity("I love this", "I really like it")
print(f"Similarity: {sim:.4f}")""",
        "example1": """# Real-World: Semantic Search
from sentence_transformers import SentenceTransformer
import numpy as np

# Load sentence embeddings model (faster than BERT)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Documents to search
documents = [
    "Python is a programming language",
    "Machine learning models predict patterns",
    "Cats and dogs are pets",
    "Natural language processing handles text",
    "Deep learning uses neural networks"
]

# Index documents
doc_embeddings = model.encode(documents)

# Query
query = "What can predict patterns?"
query_embedding = model.encode(query)

# Find most similar documents
scores = np.dot(query_embedding, doc_embeddings.T)
top_indices = np.argsort(scores)[::-1][:3]

print(f"Query: {query}")
print(f"\\nTop 3 results:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank}. ({scores[idx]:.3f}) {documents[idx]}")""",
        "example2": """# Real-World: Clustering with Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
import numpy as np

# Get embeddings for texts
def get_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = model(**inputs)
            # Mean pooling
            embeddings.append(output.last_hidden_state.mean(dim=1).numpy()[0])
    return np.array(embeddings)

# Texts to cluster
texts = [
    "Python programming tutorial",
    "JavaScript coding guide",
    "Machine learning basics",
    "Deep neural networks",
    "Frontend web development",
    "Backend API design"
]

embeddings = get_embeddings(texts)

# Cluster
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings)

print("Clustering results:")
for i, (text, cluster) in enumerate(zip(texts, clusters)):
    print(f"  Cluster {cluster}: {text}")""",
        "example3": """# Real-World: Fine-tuning Embeddings
import torch
from torch.nn import CosineSimilarity
from transformers import AutoTokenizer, AutoModel

# Contrastive learning: similar pairs should be close, dissimilar far
class EmbeddingFinetuner:
    def __init__(self, model_name="distilbert-base-uncased", lr=1e-5):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state.mean(dim=1)

    def train_step(self, anchor, positive, negative):
        # Get embeddings
        emb_anchor = self.embed(anchor)
        emb_pos = self.embed(positive)
        emb_neg = self.embed(negative)

        # Contrastive loss: minimize distance to positive, maximize to negative
        cos_sim = torch.nn.CosineSimilarity(dim=1)
        pos_sim = cos_sim(emb_anchor, emb_pos)
        neg_sim = cos_sim(emb_anchor, emb_neg)

        loss = torch.nn.functional.relu(neg_sim - pos_sim + 0.5).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# Example training triplets
triplets = [
    ("I love this product", "This is great", "This is awful"),
    ("Good performance", "Fast results", "Slow and bad"),
]

print("Training embedding model with contrastive loss...")
model = EmbeddingFinetuner()
for anchor, pos, neg in triplets * 10:
    loss = model.train_step(anchor, pos, neg)
print(f"Final loss: {loss:.4f}")"""
    },

    "03-pretraining": {
        "level1": """from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, Trainer, TrainingArguments
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Simple generation
prompt = "Machine learning is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate
output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")""",
        "level2": """from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.inputs = tokenizer(texts, truncation=True, max_length=max_length,
                                 padding="max_length", return_tensors="pt")

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.inputs['input_ids'][idx]
        }

# Mock pretraining data
texts = [
    "Natural language processing enables machines to understand text",
    "Deep learning models learn patterns from large datasets",
    "Transfer learning reuses knowledge from one task to another",
] * 100

# Setup training
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
dataset = SimpleTextDataset(texts, tokenizer)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# NOTE: Actual training would run with: trainer.train()
print(f"Dataset size: {len(dataset)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print("(Training would begin with: trainer.train())")""",
        "example1": """# Real-World: Continued Pretraining on Domain Data
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a base pretrained model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Domain-specific texts (e.g., medical domain)
domain_texts = [
    "The patient presents with symptoms of diabetes",
    "Diagnosis requires blood glucose testing",
    "Treatment options include medication and lifestyle changes",
] * 50

# Encode domain data
inputs = tokenizer(domain_texts, return_tensors="pt", padding=True, truncation=True)

# Continued pretraining setup
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop (simplified)
for epoch in range(2):
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

print(f"\\nDomain-adapted model ready for fine-tuning")""",
        "example2": """# Real-World: Distributed Pretraining Setup
from transformers import GPT2Config, GPT2LMHeadModel
import torch

# Create custom model architecture
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
)

model = GPT2LMHeadModel(config)

# Would use distributed training in practice
def setup_distributed_training():
    '''
    Production setup:
    1. Use torch.distributed.launch
    2. Split data across GPUs
    3. Use DistributedDataParallel
    4. Sync gradients
    '''
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (distributed training on CPU not recommended)")

setup_distributed_training()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")""",
        "example3": """# Real-World: Pretraining Curriculum Learning
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Curriculum learning: easy data first, then harder
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Stage 1: Simple English (high-quality data)
stage1_data = [
    "The cat sat on the mat",
    "Dogs are loyal animals",
] * 50

# Stage 2: Mixed language (code + text)
stage2_data = [
    "def hello(): return 'world'",
    "Machine learning uses data",
] * 50

# Stage 3: Noisy/domain-specific data
stage3_data = [
    "@user awesome #ML post",
    "C++ programming langauge",  # Intentional typo
] * 50

stages = [
    ("Stage 1: Simple", stage1_data),
    ("Stage 2: Mixed", stage2_data),
    ("Stage 3: Noisy", stage3_data),
]

print("Curriculum pretraining stages:")
for stage_name, data in stages:
    inputs = tokenizer(data[:10], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    print(f"  {stage_name}: Loss = {outputs.loss:.4f} (Data points: {len(data)})")"""
    }
}

def create_cell(cell_type, source, cell_id=None):
    """Create a notebook cell."""
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]

    cell = {
        "cell_type": cell_type,
        "execution_count": None if cell_type == "markdown" else None,
        "metadata": {},
        "outputs": [] if cell_type == "code" else None,
        "source": source
    }
    if cell_id:
        cell['id'] = cell_id
    return cell

def get_default_code(level_num, concept_slug):
    """Get default code template when no specific implementation exists."""
    templates = {
        "level1": f'''# {concept_slug.replace("-", " ").title()}
# Level 1: Basic implementation

import torch
from transformers import AutoTokenizer, AutoModel

# Load model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Basic usage
text = "Example text for {concept_slug}"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("Model loaded and inference completed")
print(f"Output shape: {{outputs.last_hidden_state.shape}}")''',
        "level2": f'''import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

class {concept_slug.replace("-", " ").title().replace(" ", "")}Pipeline:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def process(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def batch_process(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

# Usage
pipeline = {concept_slug.replace("-", " ").title().replace(" ", "")}Pipeline()
result = pipeline.process("Example text")
print(f"Result shape: {{result.shape}}")''',
    }
    return templates.get(f"level{level_num}", "# Placeholder code")

def generate_notebook(concept_slug, concept_number):
    """Generate a complete notebook for a concept."""

    # Read concept markdown for title and context
    concept_path = f"{BASE}/llm/concepts/{concept_number:02d}-{concept_slug}.md"

    if not os.path.exists(concept_path):
        print(f"  ✗ Concept file not found: {concept_path}")
        return None

    with open(concept_path, 'r') as f:
        content = f.read()

    # Extract title
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else concept_slug

    # Get implementations or use defaults
    impl = IMPLEMENTATIONS.get(f"{concept_number:02d}-{concept_slug}", {})

    level1_code = impl.get("level1", get_default_code(1, concept_slug))
    level2_code = impl.get("level2", get_default_code(2, concept_slug))
    example1_code = impl.get("example1", f"# Example 1 for {title}\n# See concept file for details")
    example2_code = impl.get("example2", f"# Example 2 for {title}\n# See concept file for details")
    example3_code = impl.get("example3", f"# Example 3 for {title}\n# See concept file for details")

    # Build notebook
    nb = json.loads(json.dumps(NOTEBOOK_TEMPLATE))

    # Cell 0: Title + objectives
    nb['cells'].append(create_cell("markdown", f"""# {title}

## Learning Objectives
1. Understand core concepts and applications of {title.lower()}
2. Implement {title.lower()} with HuggingFace Transformers
3. Apply to real-world NLP tasks
4. Optimize for production use cases

See also: `llm/concepts/{concept_number:02d}-{concept_slug}.md` for theory and interview Q&A"""))

    # Cell 1: Level 1 intro
    nb['cells'].append(create_cell("markdown", f"## Level 1: Basic Implementation"))

    # Cell 2: Level 1 code
    nb['cells'].append(create_cell("code", level1_code))

    # Cell 3: Level 2 intro
    nb['cells'].append(create_cell("markdown", f"## Level 2: Advanced Implementation"))

    # Cell 4: Level 2 code
    nb['cells'].append(create_cell("code", level2_code))

    # Cell 5: Example 1
    nb['cells'].append(create_cell("markdown", f"## Real-World Example 1: Production Pattern"))

    # Cell 6: Example 1 code
    nb['cells'].append(create_cell("code", example1_code))

    # Cell 7: Example 2
    nb['cells'].append(create_cell("markdown", f"## Real-World Example 2: Advanced Usage"))

    # Cell 8: Example 2 code
    nb['cells'].append(create_cell("code", example2_code))

    # Cell 9: Example 3
    nb['cells'].append(create_cell("markdown", f"## Real-World Example 3: Optimization"))

    # Cell 10: Example 3 code
    nb['cells'].append(create_cell("code", example3_code))

    # Cell 11: Takeaways
    nb['cells'].append(create_cell("markdown", f"""## Key Takeaways

**When to use {title.lower()}:**
- For NLP tasks with sequence data
- When transfer learning from pre-trained models saves time
- In production systems requiring both accuracy and speed

**Related Concepts:**
- Review the concept markdown for theory
- See `04-optimization-algorithms.ipynb` for training techniques
- Check `05-learning-rate-scheduling.ipynb` for hyperparameter tuning"""))

    return nb

def main():
    """Generate notebooks for all LLM concepts."""

    notebooks_dir = f"{BASE}/llm/notebooks"
    concepts_dir = f"{BASE}/llm/concepts"

    print("=== Generating LLM Notebooks ===\n")

    # Get all concept files
    concept_files = sorted([f for f in os.listdir(concepts_dir) if f.endswith('.md')])

    generated = 0
    skipped = 0

    for concept_file in concept_files:
        # Parse concept number and slug
        match = re.match(r'(\d+)-(.+)\.md', concept_file)
        if not match:
            continue

        concept_num = int(match.group(1))
        concept_slug = match.group(2)
        notebook_name = f"{concept_num:02d}-{concept_slug}.ipynb"
        notebook_path = os.path.join(notebooks_dir, notebook_name)

        # Check if already has real content (12 cells)
        if os.path.exists(notebook_path):
            with open(notebook_path, 'r') as f:
                nb = json.load(f)
            if len(nb.get('cells', [])) >= 12:
                print(f"  - {notebook_name} (already complete, {len(nb['cells'])} cells)")
                skipped += 1
                continue

        # Generate notebook
        nb = generate_notebook(concept_slug, concept_num)
        if nb is None:
            continue

        # Write notebook
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)

        print(f"  ✓ {notebook_name} ({len(nb['cells'])} cells)")
        generated += 1

    print(f"\n✅ Generated {generated} notebooks, skipped {skipped} complete ones")

if __name__ == "__main__":
    main()
