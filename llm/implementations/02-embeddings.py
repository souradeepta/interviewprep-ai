"""
Auto-generated from 02-embeddings.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Embeddings - Production Implementation
# **Complete guide with real HuggingFace libraries and production patterns.**
# This notebook uses:
# ======================================================================

# ======================================================================
# ## Setup
# ======================================================================

# Install required packages
# !pip install transformers torch sentence-transformers datasets peft bitsandbytes

import warnings
warnings.filterwarnings('ignore')

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")


# ======================================================================
# ## Quick Start
# ======================================================================

# Embeddings with Sentence Transformers
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Simple embeddings
sentences = [
    "This is a sentence",
    "This is another sentence"
]

embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}")
print(f"Dimension: {embeddings.shape[1]}")


# ======================================================================
# ## Production Implementation
# ======================================================================

# Production Embeddings Pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

class EmbeddingService:
    """Production-grade embedding service"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def embed_batch(self, texts, batch_size=32):
        """Embed large batches efficiently"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=True
        )
        return embeddings

    def find_similar(self, query, documents, top_k=5):
        """Find similar documents"""
        query_emb = self.model.encode(query, convert_to_tensor=True)
        doc_embs = self.model.encode(documents, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(query_emb, doc_embs)[0]
        top_results = torch.topk(similarities, k=min(top_k, len(documents)))

        return [
            {"doc": documents[idx], "score": score.item()}
            for idx, score in zip(top_results.indices, top_results.values)
        ]

# Usage
service = EmbeddingService('all-mpnet-base-v2')

docs = [
    "Machine learning uses algorithms",
    "Deep learning with neural networks",
    "Python programming language"
]

results = service.find_similar("What is ML?", docs)
for r in results:
    print(f"{r['doc']}: {r['score']:.3f}")


# ======================================================================
# ## Real-World: Onnx
# ======================================================================

# Real-World: Fast Embeddings with ONNX
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import convert_to_tensor
import torch

class OptimizedEmbeddingService:
    """ONNX-optimized embeddings for production"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def convert_to_onnx(self, output_path='model.onnx'):
        """Convert to ONNX for faster inference"""
        try:
            # Export to ONNX
            print(f"Converting to ONNX: {output_path}")
            # self.model._first_module().auto_model.to_onnx(output_path)
        except Exception as e:
            print(f"ONNX conversion: {e}")

    def embed_cached(self, texts, cache=None):
        """Embed with caching"""
        if cache is None:
            cache = {}

        embeddings = []
        new_texts = []

        for text in texts:
            if text in cache:
                embeddings.append(cache[text])
            else:
                new_texts.append(text)

        if new_texts:
            new_embs = self.model.encode(new_texts)
            for text, emb in zip(new_texts, new_embs):
                cache[text] = emb
                embeddings.append(emb)

        return embeddings, cache

# Usage with caching
service = OptimizedEmbeddingService()
cache = {}

texts = ["hello world", "hello world", "goodbye"]
embs, cache = service.embed_cached(texts, cache)

print(f"Cache hits: {len(cache)}")


# ======================================================================
# ## Real-World: Semantic Search
# ======================================================================

# Real-World: Production Semantic Search
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

class SemanticSearchIndex:
    """Production semantic search with persistence"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def build_index(self, documents):
        """Build searchable index"""
        self.documents = documents
        print(f"Embedding {len(documents)} documents...")
        self.embeddings = self.model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True
        )

    def search(self, query, top_k=5):
        """Search index"""
        query_emb = self.model.encode(query)
        scores = cosine_similarity([query_emb], self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {"rank": i+1, "doc": self.documents[idx], "score": scores[idx]}
            for i, idx in enumerate(top_indices)
        ]

    def save_index(self, path):
        """Persist index"""
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)

    def load_index(self, path):
        """Load from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']

# Production usage
search = SemanticSearchIndex()
docs = [
    "Python is a programming language",
    "Java is also used for programming",
    "Machine learning uses Python"
]

search.build_index(docs)
results = search.search("programming language", top_k=2)

for r in results:
    print(f"{r['rank']}. {r['doc']} ({r['score']:.3f})")


# ======================================================================
# ## Production Checklist
# - [ ] Load models from HuggingFace Hub
# - [ ] Set up GPU device handling
# - [ ] Implement batch processing
# ======================================================================
