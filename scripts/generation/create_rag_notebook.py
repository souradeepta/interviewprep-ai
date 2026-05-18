import nbformat as nbf

nb = nbf.v4.new_notebook()

# Title
nb.cells.append(nbf.v4.new_markdown_cell("# Retrieval-Augmented Generation\n\nObjectives: Vector search, document retrieval, context augmentation, ranking, reranking, citation"))

# Level 1: Basic RAG
code1 = """import numpy as np
from typing import List, Tuple

# Level 1: Basic Retrieval-Augmented Generation

class BasicRAG:
    def __init__(self, documents: List[str]):
        self.documents = documents
        # Mock embeddings (in practice: use real embedding model)
        self.embeddings = self._generate_embeddings(documents)

    def _generate_embeddings(self, docs):
        '''Create mock embeddings (real: use Sentence-BERT or OpenAI).'''
        np.random.seed(42)
        return [np.random.randn(128) for _ in docs]

    def _cosine_similarity(self, a, b):
        '''Compute cosine similarity between two vectors.'''
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        '''Retrieve top-K most relevant documents.'''
        query_embedding = np.random.randn(128)  # Mock: real would embed query

        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            score = self._cosine_similarity(query_embedding, doc_emb)
            similarities.append((self.documents[i], score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def build_context(self, query: str, top_k: int = 3) -> str:
        '''Build augmented context from retrieved documents.'''
        retrieved = self.retrieve(query, top_k)
        context = f"Query: {query}\\n\\nRetrieved Documents:\\n"

        for i, (doc, score) in enumerate(retrieved, 1):
            context += f"{i}. [Relevance: {score:.2f}] {doc}\\n"

        return context

# Test Level 1
print('Level 1 - Basic RAG:\\n')

documents = [
    "Python is a programming language.",
    "Machine learning uses neural networks.",
    "Paris is capital of France.",
    "Climate change affects weather patterns."
]

rag = BasicRAG(documents)
context = rag.build_context("What is Python?", top_k=2)
print(context)
print()"""

nb.cells.append(nbf.v4.new_code_cell(code1))
nb.cells.append(nbf.v4.new_markdown_cell("**Key Points:** Embed query and documents. Compute similarity. Retrieve top-K most similar. Build prompt with query + docs. Simple but effective baseline."))

# Level 2: RAG with Reranking
code2 = """# Level 2: RAG with Reranking

class RerankingRAG(BasicRAG):
    def retrieve_and_rerank(self, query: str, initial_k: int = 5, final_k: int = 2) -> List[Tuple[str, float]]:
        '''Retrieve more documents (recall), then rerank (precision).'''
        # Stage 1: Initial retrieval (fast, approximate)
        initial = self.retrieve(query, initial_k)

        # Stage 2: Rerank using query-document relevance
        reranked = []
        for doc, initial_score in initial:
            relevance = self._rerank_score(query, doc)
            reranked.append((doc, relevance))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:final_k]

    def _rerank_score(self, query: str, document: str) -> float:
        '''Score relevance of document to query.'''
        # Simple heuristic: keyword overlap
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        overlap = len(query_words & doc_words) / (len(query_words) + 0.1)
        return overlap

# Test Level 2
print('Level 2 - Reranking:\\n')

# Large retrieval set (simulated)
large_docs = [
    "Python language programming loops",
    "Python snake species dangerous",
    "Programming is fun and challenging",
    "Java also programming language",
    "Python used for data science"
]

rag = RerankingRAG(large_docs)
results = rag.retrieve_and_rerank("Python programming language", initial_k=4, final_k=2)

print("After reranking:")
for doc, score in results:
    print(f"  Relevance {score:.2f}: {doc}")
print()"""

nb.cells.append(nbf.v4.new_code_cell(code2))
nb.cells.append(nbf.v4.new_markdown_cell("**Key Takeaways:** Two-stage retrieval improves quality. Initial retrieval is fast (high recall). Reranking is expensive but accurate (high precision). Classic information retrieval pattern."))

# Example 1: Metadata Filtering
code3 = """# Example 1: RAG with Metadata Filtering

from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class DocumentWithMetadata:
    id: str
    content: str
    source: str
    date: datetime
    category: str

class MetadataAwareRAG:
    def __init__(self, documents: List[DocumentWithMetadata]):
        self.documents = documents

    def retrieve_with_filters(self, query: str, top_k: int = 3, category: str = None, days_back: int = 30) -> List[DocumentWithMetadata]:
        '''Retrieve with metadata filtering.'''
        candidates = self.documents

        # Filter by category
        if category:
            candidates = [d for d in candidates if d.category == category]

        # Filter by recency
        cutoff = datetime.now() - timedelta(days=days_back)
        candidates = [d for d in candidates if d.date > cutoff]

        # Return top-K (in practice: rank by relevance)
        return candidates[:top_k]

    def format_response_with_citations(self, query: str, top_k: int = 3) -> str:
        '''Format retrieved docs with citations.'''
        retrieved = self.retrieve_with_filters(query, top_k)

        response = f"Based on these sources:\\n"
        for i, doc in enumerate(retrieved, 1):
            response += f"\\n[{i}] {doc.source} ({doc.date.strftime('%Y-%m-%d')})\\n"
            response += f"     Category: {doc.category}\\n"
            response += f"     Content: {doc.content[:80]}...\\n"

        return response

# Example documents
docs = [
    DocumentWithMetadata("1", "Python 3.12 released with performance improvements", "Python Blog", datetime.now() - timedelta(days=5), "news"),
    DocumentWithMetadata("2", "AI safety guidelines updated", "OpenAI", datetime.now() - timedelta(days=10), "research"),
    DocumentWithMetadata("3", "Python history and evolution", "Wikipedia", datetime.now() - timedelta(days=365), "reference"),
]

rag = MetadataAwareRAG(docs)
response = rag.retrieve_with_filters("Python updates", category="news", days_back=30)

print("Example 1 - Metadata Filtering:\\n")
for doc in response:
    print(f"Source: {doc.source} | Category: {doc.category} | {doc.content}")
print()"""

nb.cells.append(nbf.v4.new_code_cell(code3))
nb.cells.append(nbf.v4.new_markdown_cell("**Example 1 Key Points:** Filter by category, date, or custom metadata. Enables precise document selection. Reduces noise in retrieval. Essential for production systems."))

# Example 2: Retrieval Evaluation
code4 = """# Example 2: Evaluating Retrieval Quality

class RAGEvaluator:
    def __init__(self, rag_system):
        self.rag = rag_system

    def evaluate_precision_recall(self, test_queries: List[dict]) -> dict:
        '''Evaluate precision and recall of retrieval.'''
        # test_queries: [{"query": "...", "relevant_docs": [doc_ids]}, ...]

        total_precision = 0
        total_recall = 0

        for test in test_queries:
            query = test["query"]
            relevant = set(test["relevant_docs"])

            retrieved = self.rag.retrieve(query, top_k=5)
            retrieved_ids = set(range(len(retrieved)))  # Mock IDs

            # Precision: of retrieved, how many are relevant?
            precision = len(relevant & retrieved_ids) / (len(retrieved_ids) + 0.1)

            # Recall: of relevant, how many did we retrieve?
            recall = len(relevant & retrieved_ids) / (len(relevant) + 0.1)

            total_precision += precision
            total_recall += recall

        avg_precision = total_precision / len(test_queries)
        avg_recall = total_recall / len(test_queries)

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 0.01)
        }

# Test evaluation
print("Example 2 - Retrieval Evaluation:\\n")

test_cases = [
    {"query": "What is Python?", "relevant_docs": {0}},
    {"query": "Machine learning", "relevant_docs": {1}},
    {"query": "Paris capital", "relevant_docs": {2}},
]

evaluator = RAGEvaluator(rag)
metrics = evaluator.evaluate_precision_recall(test_cases)

print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"F1 Score: {metrics['f1']:.2%}")
print()"""

nb.cells.append(nbf.v4.new_code_cell(code4))
nb.cells.append(nbf.v4.new_markdown_cell("**Example 2 Key Points:** Measure retrieval quality independently of generation. Precision (are results relevant?). Recall (are all relevant docs retrieved?). F1 combines both."))

# Example 3: Citation and Source Tracking
code5 = """# Example 3: Citation and Source Tracking

class CitingRAG(BasicRAG):
    def __init__(self, documents: List[dict]):  # documents: {"id": ..., "content": ..., "source": ...}
        self.doc_list = documents
        super().__init__([d["content"] for d in documents])

    def retrieve_with_metadata(self, query: str, top_k: int = 3) -> List[dict]:
        '''Retrieve documents with source information.'''
        retrieved = self.retrieve(query, top_k)

        results = []
        for doc_content, score in retrieved:
            # Find original document with metadata
            for doc in self.doc_list:
                if doc["content"] == doc_content:
                    results.append({
                        "id": doc["id"],
                        "content": doc["content"],
                        "source": doc["source"],
                        "relevance": score
                    })
                    break

        return results

    def generate_response_with_citations(self, query: str) -> str:
        '''Generate response that cites sources.'''
        retrieved = self.retrieve_with_metadata(query, top_k=3)

        response = f"Based on the following sources:\\n"
        for i, doc in enumerate(retrieved, 1):
            response += f"\\n[{i}] {doc['source']} (relevance: {doc['relevance']:.2f})\\n"
            response += f"     {doc['content'][:100]}\\n"

        return response

# Usage with sources
docs_with_sources = [
    {"id": "1", "content": "Python is a high-level programming language.", "source": "Python.org"},
    {"id": "2", "content": "Machine learning is AI using data patterns.", "source": "ML Textbook"},
    {"id": "3", "content": "Paris is in France.", "source": "Geography Guide"},
]

citing_rag = CitingRAG(docs_with_sources)

print("Example 3 - Citation:\\n")
response = citing_rag.generate_response_with_citations("Python")
print(response)
"""

nb.cells.append(nbf.v4.new_code_cell(code5))
nb.cells.append(nbf.v4.new_markdown_cell("**Example 3 Key Points:** Track source of each retrieved document. Include citations in response. Build trust with users. Enable fact-checking. Essential for professional systems."))

# Key Takeaways
nb.cells.append(nbf.v4.new_markdown_cell("""## Key Takeaways

**RAG Pattern:**
1. Embed query
2. Search knowledge base (vector DB, BM25, hybrid)
3. Retrieve top-K documents
4. Add to prompt context
5. LLM generates response conditioned on docs

**Design Choices:**
- Dense (semantic) vs sparse (keyword) retrieval
- Top-K size (tradeoff: context vs latency)
- Reranking stage (improve quality at cost of latency)
- Metadata filtering (category, date, author)
- Citation requirements

**Evaluation:**
- Precision: of retrieved, how many relevant?
- Recall: of relevant, how many retrieved?
- Latency: retrieval + generation time
- User satisfaction: A/B testing

**Production Considerations:**
- Document freshness (TTL, version management)
- Caching frequent queries
- Monitoring retrieval quality
- Handling retrieval failures
- Scaling vector DB for large collections

**Related Concepts:** [[knowledge-graphs]], [[embedding-models]], [[agent-loops]], [[context-window-management]]"""))

# Save notebook
nbf.write(nb, '/home/sbisw/github/interviewprep-ml/agentic-ai/notebooks/retrieval-augmented-generation.ipynb')
print("✓ Notebook created successfully")
