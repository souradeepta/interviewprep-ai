## Application Architecture (Components & Layers)

```mermaid
graph TB
    subgraph Presentation["Presentation Layer"]
        REST["REST API<br/>(FastAPI)"]
        WS_APP["WebSocket Handler<br/>(Async)"]
        VALIDATION["Input Validator<br/>(Pydantic)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        ROUTER["Request Router<br/>(Intent → Pipeline)"]
        PIPELINE["Pipeline Executor<br/>(DAG)"]
        CONTEXT["Context Manager<br/>(Session State)"]
    end

    subgraph NLPServices["NLP Services"]
        INTENT["Intent Classifier<br/>(Fine-tuned Model)"]
        EMBEDDING["Embedding Generator<br/>(text-embedding-3)"]
        SENTIMENT["Sentiment Analyzer<br/>(Transformer)"]
    end

    subgraph RAGServices["RAG Services"]
        RETRIEVER["Semantic Retriever<br/>(Vector Search)"]
        RANKER["Result Ranker<br/>(Relevance Scoring)"]
        AUGMENTOR["Context Augmentor<br/>(Inject into Prompt)"]
    end

    subgraph LLMServices["LLM Services"]
        LLMCACHE["LLM Cache<br/>(Prompt Dedup)"]
        GENERATOR["Response Generator<br/>(GPT-4 Wrapper)"]
        VALIDATOR["Output Validator<br/>(Safety Check)"]
    end

    subgraph DataServices["Data Services"]
        CONVERSATION_DB["Conversation Manager<br/>(PostgreSQL ORM)"]
        VECTOR_CLIENT["Vector Client<br/>(Pinecone SDK)"]
        CACHE_CLIENT["Cache Client<br/>(Redis)"]
        KB_LOADER["KB Loader<br/>(S3 Sync)"]
    end

    subgraph Infrastructure["Infrastructure/Utilities"]
        LOGGER["Logger<br/>(Structured Logging)"]
        METRICS["Metrics Exporter<br/>(Prometheus)"]
        TRACER["Tracer<br/>(OpenTelemetry)"]
        ALERTER["Alerter<br/>(PagerDuty)"]
    end

    REST --> VALIDATION
    WS_APP --> VALIDATION
    VALIDATION --> ROUTER

    ROUTER --> CONTEXT
    CONTEXT --> PIPELINE

    PIPELINE --> INTENT
    PIPELINE --> RETRIEVER
    PIPELINE --> GENERATOR

    INTENT --> SENTIMENT
    EMBEDDING --> RETRIEVER
    RETRIEVER --> RANKER
    RANKER --> AUGMENTOR

    AUGMENTOR --> LLMCACHE
    LLMCACHE --> GENERATOR
    GENERATOR --> VALIDATOR

    INTENT --> CONVERSATION_DB
    VALIDATOR --> CONVERSATION_DB
    RETRIEVER --> VECTOR_CLIENT
    CONTEXT --> CACHE_CLIENT
    KB_LOADER --> VECTOR_CLIENT

    PIPELINE --> LOGGER
    GENERATOR --> METRICS
    INTENT --> TRACER
    METRICS --> ALERTER
```

**Layer Breakdown:**
- **Presentation**: REST API + WebSocket with async I/O
- **Orchestration**: Request routing, pipeline execution, context management
- **NLP Services**: Intent classification, embeddings, sentiment analysis
- **RAG Services**: Vector retrieval, ranking, context augmentation
- **LLM Services**: Caching, generation, safety validation
- **Data Services**: Database access, vector store, cache, KB management
- **Infrastructure**: Logging, metrics, tracing, alerting