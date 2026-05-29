# AI Semantic Search Engine - Application Architecture

```mermaid
graph TD
    subgraph APILayer["API Layer"]
        SearchAPI["Search API<br/>(FastAPI)"]
        QueryNorm["Query Normalizer<br/>(Lowercase, Trim)"]
    end

    subgraph EmbeddingLayer["Embedding Layer"]
        EmbeddingSvc["Embedding Service<br/>(Bi-Encoder)"]
        EmbedCache["Embedding Cache<br/>(Redis)"]
    end

    subgraph RetrievalLayer["Retrieval Layer"]
        ANNIndex["ANN Index<br/>(FAISS/Pinecone)"]
        RerankerModel["Re-ranking Model<br/>(Cross-Encoder)"]
    end

    subgraph FilterLayer["Filter Layer"]
        BusinessFilter["Business Logic Filter<br/>(ACL, Policy)"]
        RespFormatter["Response Formatter<br/>(JSON)"]
    end

    subgraph IndexLayer["Index Management"]
        ChunkSvc["Chunking Service<br/>(Overlap Split)"]
        IndexUpdater["Index Updater<br/>(Batch/Incremental)"]
    end

    SearchAPI --> QueryNorm
    QueryNorm --> EmbedCache
    EmbedCache -->|Miss| EmbeddingSvc
    EmbeddingSvc --> EmbedCache
    EmbedCache -->|Hit| ANNIndex
    EmbeddingSvc --> ANNIndex
    ANNIndex --> RerankerModel
    RerankerModel --> BusinessFilter
    BusinessFilter --> RespFormatter

    ChunkSvc --> EmbeddingSvc
    EmbeddingSvc --> IndexUpdater
    IndexUpdater --> ANNIndex
```

**Layer Breakdown:**
- **API Layer**: FastAPI with query normalization (lowercase, whitespace trim, spell correction)
- **Embedding Layer**: Bi-encoder embedding service with Redis cache for repeated queries
- **Retrieval Layer**: ANN search returns top-100 candidates, cross-encoder re-ranks to top-10
- **Filter Layer**: ACL enforcement, content policy filtering, final response formatting
- **Index Management**: Async chunking and embedding pipeline for document index updates
