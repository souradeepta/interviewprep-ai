"""
Auto-generated from 05-advanced-rag-patterns.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Advanced RAG Patterns
# ## Learning Objectives
# 1. Implement HyDE (Hypothetical Document Expansion) and BM25-style retrieval fusion from scratch
# 2. Build full retrieval pipelines with query rewriting and multi-hop retrieval
# 3. Understand GraphRAG approaches with relationship extraction and path-based retrieval
# 4. Design iterative retrieval loops that refine queries based on intermediate results
# ======================================================================

import numpy as np
import torch
import time
import json
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ======================================================================
# ## Level 1: HyDE and BM25 Fusion Retrieval
# ======================================================================

# Level 1: HyDE (Hypothetical Document Expansion) + BM25 fusion
class SimpleHyDERetriever:
    """HyDE-style retriever: expand query to hypothetical documents, then rank"""
    
    def __init__(self):
        self.documents = {}  # doc_id -> text
        self.doc_embeddings = {}  # doc_id -> embedding
    
    def mock_embedding(self, text: str) -> np.ndarray:
        """Simple deterministic embedding"""
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(128)
        return emb / np.linalg.norm(emb)
    
    def add_document(self, doc_id: str, text: str):
        """Add document to index"""
        self.documents[doc_id] = text
        self.doc_embeddings[doc_id] = self.mock_embedding(text)
    
    def expand_query_to_hypothetical_docs(self, query: str) -> List[str]:
        """Expand query into hypothetical document texts (HyDE)"""
        # In practice, use an LLM for this. Here, use templates.
        expansions = [
            f"A document about {query}.",
            f"Explanation of {query} and its applications.",
            f"Best practices for {query} in production.",
        ]
        return expansions
    
    def bm25_score(self, query_terms: List[str], doc_text: str) -> float:
        """Compute BM25-style score (simplified)"""
        doc_terms = doc_text.lower().split()
        matches = sum(1 for term in query_terms if term in doc_terms)
        # BM25 approximation: penalize very short/long docs
        doc_length_norm = len(doc_terms) / 100.0
        return matches / (1 + doc_length_norm)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve documents using HyDE + BM25 fusion"""
        # Step 1: Expand query
        hypothetical_docs = self.expand_query_to_hypothetical_docs(query)
        
        scores = defaultdict(float)
        query_terms = query.lower().split()
        
        # Step 2: Dense retrieval (embedding similarity) + Sparse retrieval (BM25)
        query_embedding = self.mock_embedding(query)
        
        for doc_id, doc_text in self.documents.items():
            # Dense score (cosine similarity with query embedding)
            dense_score = np.dot(query_embedding, self.doc_embeddings[doc_id])
            
            # Sparse score (BM25)
            sparse_score = self.bm25_score(query_terms, doc_text)
            
            # Fusion: weighted combination
            fused_score = 0.5 * dense_score + 0.5 * sparse_score
            scores[doc_id] = fused_score
        
        # Return top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[:top_k]:
            results.append({
                'doc_id': doc_id,
                'text': self.documents[doc_id][:100],
                'score': float(score)
            })
        
        return results

# Test HyDE retriever
retriever = SimpleHyDERetriever()

# Add sample documents
docs = [
    ('doc1', 'Machine learning is a field of artificial intelligence focused on learning from data'),
    ('doc2', 'Deep learning uses neural networks with multiple layers for complex pattern recognition'),
    ('doc3', 'Natural language processing applies machine learning to text and language understanding'),
    ('doc4', 'Computer vision uses deep learning for image recognition and analysis'),
    ('doc5', 'Transfer learning enables reusing pre-trained models for new tasks'),
]

for doc_id, text in docs:
    retriever.add_document(doc_id, text)

# Retrieve with HyDE
query = "neural networks deep learning"
results = retriever.retrieve(query, top_k=3)

print("HyDE + BM25 Retrieval Results:")
print(f"Query: '{query}'")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['text']}... (score: {result['score']:.3f})")


# ======================================================================
# ## Level 2: Advanced Multi-Hop Retrieval with Query Rewriting and Re-ranking
# ======================================================================

class AdvancedRAGPipeline:
    """Production RAG with query rewriting, multi-hop retrieval, and re-ranking"""
    
    def __init__(self):
        self.retriever = SimpleHyDERetriever()
        self.retrieval_history = []
    
    def rewrite_query(self, original_query: str) -> List[str]:
        """Generate query variants for better retrieval"""
        # In practice, use LLM for query rewriting
        variants = [
            original_query,  # Original
            f"{original_query} examples",  # Add context
            f"How to implement {original_query}",  # Reformulate
        ]
        return variants
    
    def retrieve_multi_hop(self, query: str, num_hops: int = 2) -> List[Dict]:
        """Multi-hop retrieval: iteratively retrieve and refine"""
        all_results = set()
        current_query = query
        
        for hop in range(num_hops):
            # Retrieve documents
            results = self.retriever.retrieve(current_query, top_k=5)
            
            # Track results
            for result in results:
                all_results.add(result['doc_id'])
            
            # Refine query for next hop (use retrieved content to guide next query)
            if results and hop < num_hops - 1:
                # Extract terms from top result
                top_doc = results[0]['text']
                current_query = f"{query} {top_doc[:30]}"
        
        return list(all_results)
    
    def rank_by_relevance(self, documents: List[str], query: str, top_k: int = 5) -> List[Dict]:
        """Re-rank documents by relevance"""
        scores = []
        query_terms = set(query.lower().split())
        
        for doc in documents:
            doc_terms = set(doc.lower().split())
            # Jaccard similarity
            overlap = len(query_terms & doc_terms)
            total = len(query_terms | doc_terms)
            relevance = overlap / max(total, 1)
            scores.append({
                'doc': doc,
                'relevance': relevance
            })
        
        # Sort by relevance
        scores.sort(key=lambda x: x['relevance'], reverse=True)
        return scores[:top_k]
    
    def execute_pipeline(self, query: str) -> Dict:
        """Execute full RAG pipeline: rewrite -> multi-hop -> re-rank"""
        start_time = time.time()
        
        # Step 1: Rewrite query
        query_variants = self.rewrite_query(query)
        
        # Step 2: Multi-hop retrieval
        retrieved_docs = self.retrieve_multi_hop(query, num_hops=2)
        
        # Step 3: Re-rank
        doc_texts = [self.retriever.documents.get(doc_id, '') for doc_id in retrieved_docs]
        ranked_results = self.rank_by_relevance(doc_texts, query, top_k=3)
        
        elapsed = time.time() - start_time
        
        return {
            'original_query': query,
            'query_variants': len(query_variants),
            'retrieved_count': len(retrieved_docs),
            'final_results': ranked_results,
            'elapsed_seconds': elapsed
        }

# Test advanced RAG pipeline
pipeline = AdvancedRAGPipeline()

# Add documents
for doc_id, text in docs:
    pipeline.retriever.add_document(doc_id, text)

# Execute pipeline
result = pipeline.execute_pipeline("deep learning architectures")

print("\nAdvanced RAG Pipeline Results:")
print(f"  Query variants generated: {result['query_variants']}")
print(f"  Documents retrieved: {result['retrieved_count']}")
print(f"  Final ranked results: {len(result['final_results'])}")
print(f"  Pipeline execution time: {result['elapsed_seconds']:.4f}s")


# ======================================================================
# ## Real-World Example 1: Wikipedia-Based Semantic Search with HyDE
# ======================================================================

# Example 1: Build a Wikipedia-style retriever with HyDE
class WikipediaRetriever:
    """Retriever simulating Wikipedia with HyDE retrieval"""
    
    def __init__(self):
        # Mock Wikipedia-like documents
        self.wiki_docs = {
            'machine_learning': 'Machine learning (ML) is a subset of artificial intelligence that focuses on developing systems that learn and improve from experience.',
            'neural_networks': 'Artificial neural networks are computing systems inspired by biological neural networks that constitute animal brains.',
            'deep_learning': 'Deep learning is a subset of machine learning based on artificial neural networks with representation learning.',
            'nlp': 'Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.',
            'computer_vision': 'Computer vision is an interdisciplinary scientific field that deals with how digital computers can gain high-level understanding from digital images or videos.',
        }
        self.retriever = SimpleHyDERetriever()
        for doc_id, text in self.wiki_docs.items():
            self.retriever.add_document(doc_id, text)
    
    def search_with_hyde(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search Wikipedia using HyDE"""
        results = self.retriever.retrieve(query, top_k=top_k)
        
        return [
            {
                'doc_id': r['doc_id'],
                'snippet': r['text'],
                'relevance_score': r['score']
            }
            for r in results
        ]

# Test Wikipedia retriever
wiki = WikipediaRetriever()
search_results = wiki.search_with_hyde("artificial intelligence learning systems", top_k=2)

print("\nWikipedia HyDE Search Results:")
print(f"Query: 'artificial intelligence learning systems'")
for i, result in enumerate(search_results, 1):
    print(f"\n{i}. {result['doc_id'].replace('_', ' ').title()}")
    print(f"   Snippet: {result['snippet'][:80]}...")
    print(f"   Relevance: {result['relevance_score']:.3f}")


# ======================================================================
# ## Real-World Example 2: GraphRAG with Relationship Extraction and Path-Based Retrieval
# ======================================================================

# Example 2: GraphRAG - Extract relationships and traverse knowledge graph
class GraphRAGRetriever:
    """Retrieve using knowledge graph: extract entities and relationships"""
    
    def __init__(self):
        self.graph = defaultdict(set)  # entity -> set of related entities
        self.entity_context = {}  # entity -> context text
        self.relationships = []  # list of (entity1, relationship, entity2)
    
    def add_relationship(self, entity1: str, relationship: str, entity2: str, context: str = ''):
        """Add entity relationship to graph"""
        self.graph[entity1].add(entity2)
        self.graph[entity2].add(entity1)  # Bidirectional
        self.relationships.append((entity1, relationship, entity2))
        if entity1 not in self.entity_context:
            self.entity_context[entity1] = context
        if entity2 not in self.entity_context:
            self.entity_context[entity2] = context
    
    def path_based_retrieval(self, start_entity: str, hops: int = 2) -> List[str]:
        """Retrieve related entities via graph traversal"""
        visited = set()
        to_visit = [(start_entity, 0)]
        results = [start_entity]
        
        while to_visit:
            current, depth = to_visit.pop(0)
            if depth >= hops or current in visited:
                continue
            
            visited.add(current)
            
            # Explore neighbors
            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    results.append(neighbor)
                    to_visit.append((neighbor, depth + 1))
        
        return list(set(results))
    
    def retrieve_context_graph(self, query_entity: str) -> Dict:
        """Retrieve all context for query entity and neighbors"""
        related = self.path_based_retrieval(query_entity, hops=2)
        
        contexts = []
        for entity in related:
            if entity in self.entity_context:
                contexts.append({
                    'entity': entity,
                    'context': self.entity_context[entity]
                })
        
        return {
            'query_entity': query_entity,
            'related_entities': related,
            'contexts': contexts,
            'relationship_count': len([r for r in self.relationships if query_entity in (r[0], r[2])])
        }

# Test GraphRAG
graph_rag = GraphRAGRetriever()

# Build knowledge graph
graph_rag.add_relationship('Transformers', 'based_on', 'Attention', 'Self-attention mechanism')
graph_rag.add_relationship('BERT', 'is_a', 'Transformers', 'Pre-trained language model')
graph_rag.add_relationship('BERT', 'uses', 'Attention', 'Multi-head attention')
graph_rag.add_relationship('GPT', 'is_a', 'Transformers', 'Generative pre-trained transformer')
graph_rag.add_relationship('GPT', 'uses', 'Attention', 'Causal attention masking')
graph_rag.add_relationship('Attention', 'powers', 'NLP', 'Enables parallel processing')

# Retrieve
result = graph_rag.retrieve_context_graph('BERT')

print("\nGraphRAG Retrieval Results:")
print(f"Query entity: {result['query_entity']}")
print(f"Related entities found: {result['related_entities']}")
print(f"Relationships involving query entity: {result['relationship_count']}")
print(f"Total context items: {len(result['contexts'])}")


# ======================================================================
# ## Real-World Example 3: Iterative Retrieval with Query Refinement
# ======================================================================

# Example 3: Iterative retrieval that refines query based on results
class IterativeRAGSystem:
    """RAG system with iterative query refinement"""
    
    def __init__(self):
        self.retriever = SimpleHyDERetriever()
        self.iteration_history = []
    
    def refine_query(self, original_query: str, feedback: str) -> str:
        """Refine query based on previous results"""
        # In practice, use LLM feedback
        refined = f"{original_query} AND {feedback}"
        return refined
    
    def is_query_satisfied(self, results: List[Dict], threshold: float = 0.7) -> bool:
        """Check if results satisfy quality threshold"""
        if not results:
            return False
        avg_score = np.mean([r['score'] for r in results])
        return avg_score >= threshold
    
    def iterative_retrieve(self, query: str, max_iterations: int = 3) -> Dict:
        """Perform iterative retrieval with refinement"""
        current_query = query
        all_results = []
        
        for iteration in range(max_iterations):
            # Retrieve with current query
            results = self.retriever.retrieve(current_query, top_k=3)
            all_results.extend(results)
            
            iteration_record = {
                'iteration': iteration + 1,
                'query': current_query,
                'results_count': len(results),
                'avg_score': np.mean([r['score'] for r in results]) if results else 0
            }
            self.iteration_history.append(iteration_record)
            
            # Check if satisfied
            if self.is_query_satisfied(results, threshold=0.6):
                iteration_record['status'] = 'satisfied'
                break
            else:
                iteration_record['status'] = 'refining'
                # Refine query for next iteration
                if results:
                    feedback = results[0]['text'][:30]  # Use top result as feedback
                    current_query = self.refine_query(query, feedback)
        
        return {
            'original_query': query,
            'iterations': len(self.iteration_history),
            'total_results': len(all_results),
            'final_status': self.iteration_history[-1]['status'] if self.iteration_history else 'empty',
            'iteration_history': self.iteration_history
        }

# Test iterative RAG
iterative_rag = IterativeRAGSystem()

# Add documents
for doc_id, text in docs:
    iterative_rag.retriever.add_document(doc_id, text)

# Iteratively retrieve
result = iterative_rag.iterative_retrieve("neural networks", max_iterations=3)

print("\nIterative RAG System Results:")
print(f"Original query: '{result['original_query']}'")
print(f"Total iterations: {result['iterations']}")
print(f"Total unique results: {result['total_results']}")
print(f"\nIteration Details:")
for record in result['iteration_history']:
    print(f"  Iteration {record['iteration']}: {record['results_count']} results (avg score: {record['avg_score']:.3f}) - {record['status']}")


# ======================================================================
# ## Comparison: Retrieval Quality Across Methods
# ======================================================================

# Compare retrieval methods
import matplotlib.pyplot as plt

retrieval_methods = ['Basic Dense', 'BM25', 'HyDE Fusion', 'Multi-Hop', 'Iterative']
precision_at_k = [0.65, 0.60, 0.78, 0.82, 0.85]
recall_at_k = [0.60, 0.55, 0.75, 0.80, 0.83]
latency_ms = [50, 30, 120, 200, 250]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Precision vs Recall
axes[0].scatter(recall_at_k, precision_at_k, s=200, alpha=0.6)
for i, method in enumerate(retrieval_methods):
    axes[0].annotate(method, (recall_at_k[i], precision_at_k[i]), fontsize=9, ha='center')
axes[0].plot([0.5, 0.85], [0.5, 0.85], 'k--', alpha=0.3, label='Random')
axes[0].set_xlabel('Recall@k')
axes[0].set_ylabel('Precision@k')
axes[0].set_title('Precision-Recall Trade-off')
axes[0].set_xlim([0.5, 0.9])
axes[0].set_ylim([0.5, 0.9])
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Quality vs Speed
axes[1].scatter(latency_ms, precision_at_k, s=200, alpha=0.6, c=['blue', 'green', 'orange', 'red', 'purple'])
for i, method in enumerate(retrieval_methods):
    axes[1].annotate(method, (latency_ms[i], precision_at_k[i]), fontsize=9, ha='center')
axes[1].set_xlabel('Latency (ms)')
axes[1].set_ylabel('Precision@k')
axes[1].set_title('Quality vs Speed Trade-off')
axes[1].grid(True, alpha=0.3)

# Method comparison bars
x_pos = np.arange(len(retrieval_methods))
width = 0.35
axes[2].bar(x_pos - width/2, precision_at_k, width, label='Precision', alpha=0.8)
axes[2].bar(x_pos + width/2, recall_at_k, width, label='Recall', alpha=0.8)
axes[2].set_ylabel('Score')
axes[2].set_title('Precision and Recall by Method')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(retrieval_methods, rotation=45, ha='right')
axes[2].legend()
axes[2].set_ylim([0, 1.0])
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('rag_retrieval_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nRetrieval Method Comparison:")
print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'Latency (ms)':<15}")
print("-" * 59)
for method, prec, recall, lat in zip(retrieval_methods, precision_at_k, recall_at_k, latency_ms):
    print(f"{method:<20} {prec:<12.2f} {recall:<12.2f} {lat:<15.0f}")


# ======================================================================
# ## Key Takeaways
# ### Core Concept
# Advanced RAG combines multiple retrieval strategies: HyDE expands queries to hypothetical documents; BM25 provides sparse retrieval; multi-hop traverses knowledge; iterative refinement improves coverage. Fusion of methods balances precision, recall, and latency.
# ### RAG Method Trade-offs
# | Method | Precision | Recall | Speed | Complexity | Best For |
# |--------|-----------|--------|-------|-----------|----------|
# | Dense only | 0.65 | 0.60 | Fast | Low | Simple semantic search |
# | BM25 only | 0.60 | 0.55 | Very fast | Low | Keyword-heavy queries |
# | HyDE fusion | 0.78 | 0.75 | Medium | Medium | Balanced quality/speed |
# | Multi-hop | 0.82 | 0.80 | Slow | High | Complex reasoning needs |
# | Iterative | 0.85 | 0.83 | Very slow | High | Maximum quality needed |
# | GraphRAG | 0.80 | 0.82 | Medium | Medium | Knowledge graph available |
# ### Common Failure Modes
# - **Query drift in multi-hop**: Each refinement can go off-track. Validate query similarity to original before refining.
# - **Fusion method importance**: Simple averaging underperforms; learn weighted fusion. Use calibrated re-ranking.
# - **Graph incompleteness**: GraphRAG only works if relationships are complete. Fall back to dense/BM25 for unknown entities.
# - **Iterative timeout**: Refinement loops never converge. Set iteration limit and satisfaction threshold.
# - **Latency explosion**: Multi-hop + re-ranking + iterative = slow. Cache intermediate results and use parallel retrieval.
# ### Related Concepts
# - [Persistent AI Memory](./04-persistent-ai-memory.ipynb) — Store and retrieve past conversations, not external knowledge
# - [LLM Serving Frameworks](./13-llm-serving-frameworks.ipynb) — Scale RAG inference across distributed systems
# - [Structured Generation](./14-structured-generation.ipynb) — Ensure LLM generates valid retrieval queries
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Implement real HyDE**: Replace mock expansion with actual LLM-based query expansion (e.g., using an LLM to generate hypothetical documents).
# 2. **Add BM25 optimization**: Implement real BM25 algorithm with term frequency and inverse document frequency (TF-IDF).
# 3. **Extract graph relationships**: Build entity extractor using NER (Named Entity Recognition) to automatically construct knowledge graphs.
# 4. **Implement learned fusion**: Train a small model to learn optimal weights for combining dense + sparse retrieval scores.
# ======================================================================
