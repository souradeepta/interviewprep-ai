# AI Semantic Search Engine - System Architecture

```mermaid
graph TD
    subgraph QueryPath["Query Path"]
        SearchAPI["Search API<br/>(REST/GraphQL)"]
        QueryEncoder["Query Encoder<br/>(Embedding Model)"]
    end

    subgraph VectorSearch["Vector Search Layer"]
        VectorStore["Vector Store<br/>(FAISS/Pinecone)"]
        Reranker["Re-ranker<br/>(Cross-Encoder)"]
        ResultFilter["Result Filter<br/>(Business Rules)"]
    end

    subgraph IndexingPipeline["Indexing Pipeline"]
        DocIngestor["Document Ingestor<br/>(Batch/Stream)"]
        Chunker["Text Chunker<br/>(Overlap Split)"]
        Embedder["Embedder<br/>(text-embedding-3)"]
    end

    subgraph Storage["Storage"]
        VectorDB["Vector DB<br/>(Persistent Index)"]
        MetadataDB["Metadata DB<br/>(PostgreSQL)"]
    end

    subgraph Observability["Observability"]
        SearchMetrics["Search Metrics<br/>(P50/P99 Latency)"]
        QualityMonitor["Quality Monitor<br/>(NDCG Tracking)"]
    end

    SearchAPI --> QueryEncoder
    QueryEncoder --> VectorStore
    VectorStore --> Reranker
    Reranker --> ResultFilter
    ResultFilter --> SearchMetrics

    DocIngestor --> Chunker
    Chunker --> Embedder
    Embedder --> VectorDB
    Embedder --> MetadataDB
    VectorDB --> VectorStore
    SearchMetrics --> QualityMonitor
```

**Infrastructure Components:**
- **Query Encoder**: Bi-encoder model converts query to dense vector for ANN search
- **Vector Store**: FAISS (local) or Pinecone (managed) for approximate nearest neighbor search
- **Re-ranker**: Cross-encoder model scores query-document pairs for precision improvement
- **Result Filter**: Business logic filters (access control, content policy, date recency)
- **Indexing Pipeline**: Async document ingestion with chunking, embedding, and index update
- **Quality Monitor**: NDCG@10 tracking to detect search quality regressions over time
