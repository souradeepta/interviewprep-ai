#!/usr/bin/env python3
"""
Generate System, Application, and Process Flow Architecture Diagrams
Creates three types of Mermaid diagrams for each AI/ML system architecture.

Diagram Types:
1. System Architecture - Infrastructure, deployment, scaling, cloud services
2. Application Architecture - Components, modules, layers, internal structure
3. Process Flow - Business workflows, request pipelines, decision flows
"""

import os
from pathlib import Path


def generate_customer_service_diagrams():
    """Generate all three diagram types for customer service platform."""

    diagrams = {
        "01-system-architecture": """## System Architecture (Infrastructure & Deployment)

```mermaid
graph TB
    subgraph CDN["CDN & Load Balancing"]
        LB1["Load Balancer<br/>(AWS ALB)"]
        CDN1["CloudFront CDN<br/>(Static Assets)"]
    end

    subgraph API["API Gateway Layer"]
        APIGW["API Gateway<br/>(Rate Limiting, Auth)"]
        WS["WebSocket Server<br/>(10K concurrent)"]
    end

    subgraph Processing["Processing Cluster (Kubernetes)"]
        IC["Intent Classifier<br/>Pod × 10"]
        RAG["RAG Pipeline<br/>Pod × 15"]
        LLM["LLM Generator<br/>Pod × 5"]
        SA["Sentiment Analyzer<br/>Pod × 8"]
    end

    subgraph DataLayer["Data Layer"]
        PG["PostgreSQL<br/>(Conversations, Logs)"]
        REDIS["Redis Cluster<br/>(Session Cache)"]
        PINECONE["Pinecone<br/>(Vector DB)"]
        S3["S3 Bucket<br/>(KB Articles)"]
    end

    subgraph External["External Services"]
        OPENAI["OpenAI API<br/>(GPT-4-Turbo)"]
        EMAIL["Email Service<br/>(SendGrid)"]
        SMS["SMS Service<br/>(Twilio)"]
    end

    subgraph Monitoring["Monitoring & Observability"]
        PROM["Prometheus"]
        LOGS["ELK Stack<br/>(Logs)"]
        TRACE["Jaeger<br/>(Tracing)"]
    end

    LB1 --> APIGW
    LB1 --> WS
    CDN1 --> LB1

    APIGW --> MQ["Message Queue<br/>(RabbitMQ)"]
    WS --> MQ

    MQ --> IC
    MQ --> RAG
    MQ --> LLM
    MQ --> SA

    IC --> REDIS
    RAG --> PINECONE
    RAG --> S3
    LLM --> OPENAI
    SA --> REDIS

    IC --> PG
    LLM --> PG
    SA --> PG

    IC --> PROM
    RAG --> PROM
    LLM --> PROM
    IC --> LOGS
    LLM --> TRACE

    APIGW --> EMAIL
    APIGW --> SMS
```

**Infrastructure Components:**
- **Compute**: Kubernetes cluster (auto-scaling 5-50 pods based on load)
- **Storage**: PostgreSQL (conversations), Redis (cache), Pinecone (vectors), S3 (KB)
- **External APIs**: OpenAI (LLM), SendGrid (email), Twilio (SMS)
- **Monitoring**: Prometheus (metrics), ELK (logs), Jaeger (distributed tracing)
- **CDN**: CloudFront for static asset caching
- **Load Balancing**: AWS ALB with health checks and auto-scaling policies""",

        "02-application-architecture": """## Application Architecture (Components & Layers)

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
- **Infrastructure**: Logging, metrics, tracing, alerting""",

        "03-process-flow": """## Process Flow (Request to Response)

```mermaid
graph TD
    START["User Message Arrives<br/>(Chat/Email/API)"] --> ENQUEUE["Enqueue in Message<br/>Queue"]

    ENQUEUE --> VALIDATE["Validate Input<br/>(Length, Format)"]
    VALIDATE -->|Invalid| REJECT["Reject & Error<br/>Response"]
    VALIDATE -->|Valid| DEDUP["Check Duplicate<br/>Request Cache"]

    DEDUP -->|Cache Hit| SEND_CACHED["Send Cached<br/>Response"]
    DEDUP -->|Cache Miss| CLASSIFY["Classify Intent<br/>(Multi-label LLM)"]

    CLASSIFY --> CONF_CHECK{{"Confidence<br/>> 0.8?"}}
    CONF_CHECK -->|No| ESCALATE1["→ Human Queue<br/>(Low Confidence)"]
    CONF_CHECK -->|Yes| ROUTE{{"Intent<br/>Type?"}}

    ROUTE -->|FAQ/Knowledge| RETRIEVE["Retrieve KB<br/>(Vector Search)"]
    ROUTE -->|Complex| AGENT["Agent Orchestrator<br/>(Tool Planning)"]

    RETRIEVE --> AUGMENT["Augment Context<br/>(Add KB to Prompt)"]
    AGENT --> AUGMENT

    AUGMENT --> GENERATE["Generate Response<br/>(LLM Inference)"]
    GENERATE --> VALIDATE_OUTPUT{{"Response<br/>Valid?"}}

    VALIDATE_OUTPUT -->|Invalid| RETRY{{"Retries<br/>< 3?"}}
    RETRY -->|Yes| GENERATE
    RETRY -->|No| ESCALATE2["→ Escalation Queue"]

    VALIDATE_OUTPUT -->|Valid| SENTIMENT["Analyze Sentiment<br/>(Customer Response)"]
    SENTIMENT --> SENTIMENT_CHECK{{"Sentiment<br/>< -0.5?"}}

    SENTIMENT_CHECK -->|Negative| ESCALATE3["→ Priority Queue<br/>(Negative Sentiment)"]
    SENTIMENT_CHECK -->|Neutral/Positive| FORMAT["Format Response<br/>(Channel-specific)"]

    FORMAT --> SAVE["Save to History<br/>(PostgreSQL)"]
    SAVE --> ROUTE_CHANNEL{{"Delivery<br/>Channel?"}}

    ROUTE_CHANNEL -->|Chat| WEBSOCKET["Send via WebSocket<br/>(Real-time)"]
    ROUTE_CHANNEL -->|Email| EMAIL_SEND["Send via Email<br/>(Async)"]
    ROUTE_CHANNEL -->|SMS| SMS_SEND["Send via SMS<br/>(Twilio)"]

    WEBSOCKET --> LOG["Log Request Metrics<br/>(Latency, Tokens)"]
    EMAIL_SEND --> LOG
    SMS_SEND --> LOG

    LOG --> END["Complete ✓"]

    REJECT --> END
    SEND_CACHED --> END
    ESCALATE1 --> HUMAN["Human Agent<br/>Takes Over"]
    ESCALATE2 --> HUMAN
    ESCALATE3 --> HUMAN
    HUMAN --> END
```

**Key Decision Points:**
1. **Confidence Check**: If intent confidence < 80%, escalate to human
2. **Intent Type**: FAQ/knowledge → RAG path, complex → agent path
3. **Output Validation**: If response invalid after retries, escalate
4. **Sentiment Analysis**: If negative sentiment, prioritize human pickup
5. **Channel Routing**: Deliver via original channel (Chat/Email/SMS)

**Error Paths:**
- Invalid input → reject with error
- Low confidence classification → escalate
- Generation failure → retry (max 3), then escalate
- Negative sentiment → prioritize escalation queue

**Optimization Points:**
- Request deduplication (cache hits avoid full pipeline)
- Batch processing during peak hours
- Async email/SMS (don't block response)
- Context caching (avoid re-retrieving same KB articles)"""
    }

    return diagrams


def generate_rag_system_diagrams():
    """Generate diagrams for RAG document QA system."""

    diagrams = {
        "01-system-architecture": """## System Architecture (RAG Document QA)

```mermaid
graph TB
    subgraph Ingestion["Ingestion Pipeline"]
        UPLOAD["File Upload<br/>(REST)"]
        VALIDATE_DOC["Validate Document<br/>(PDF/DOCX)"]
        CHUNK["Chunk & Parse<br/>(Fixed Size)"]
        EMBED["Generate Embeddings<br/>(batch)"]
    end

    subgraph Retrieval["Retrieval Layer"]
        QUERY_EMBED["Query Embedding<br/>(text-embedding-3)"]
        VECTOR_SEARCH["Vector Search<br/>(Pinecone)"]
        BM25["BM25 Search<br/>(Elasticsearch)"]
        FUSION["Fusion Retriever<br/>(Rank Fusion)"]
    end

    subgraph Generation["Generation Layer"]
        CONTEXT_BUILDER["Context Builder<br/>(Top-K Docs)"]
        PROMPT_BUILDER["Prompt Template<br/>(In-context Examples)"]
        LLM_GEN["LLM Response<br/>(Claude/GPT)"]
    end

    subgraph Storage["Storage & Indexing"]
        VECTOR_DB["Vector DB<br/>(Pinecone/Weaviate)"]
        ELASTIC["Elasticsearch<br/>(BM25 Index)"]
        DOCS_STORE["Document Store<br/>(S3/PostgreSQL)"]
        METADATA["Metadata Index<br/>(SQLite/PG)"]
    end

    subgraph Monitoring["Quality & Monitoring"]
        RETRIEVAL_EVAL["Retrieval Evaluator<br/>(NDCG, MRR)"]
        GENERATION_EVAL["Generation Evaluator<br/>(ROUGE, BERTScore)"]
        METRICS["Metrics Store<br/>(Prometheus)"]
    end

    UPLOAD --> VALIDATE_DOC
    VALIDATE_DOC --> CHUNK
    CHUNK --> EMBED
    EMBED --> VECTOR_DB
    CHUNK --> DOCS_STORE
    CHUNK --> METADATA

    QUERY_EMBED --> VECTOR_SEARCH
    QUERY_EMBED --> BM25
    VECTOR_SEARCH --> FUSION
    BM25 --> FUSION

    FUSION --> CONTEXT_BUILDER
    DOCS_STORE --> CONTEXT_BUILDER
    CONTEXT_BUILDER --> PROMPT_BUILDER
    PROMPT_BUILDER --> LLM_GEN

    LLM_GEN --> GENERATION_EVAL
    FUSION --> RETRIEVAL_EVAL
    RETRIEVAL_EVAL --> METRICS
    GENERATION_EVAL --> METRICS
```""",

        "02-application-architecture": """## Application Architecture (RAG System)

```mermaid
graph TB
    subgraph API["API Layer"]
        UPLOAD_EP["POST /documents<br/>(Upload)"]
        QUERY_EP["POST /query<br/>(Ask)"]
        SEARCH_EP["GET /search<br/>(Retrieval)"]
    end

    subgraph Pipeline["Processing Pipeline"]
        DOC_PROCESSOR["Document Processor<br/>(Chunker, Cleaner)"]
        EMBEDDING_SVC["Embedding Service<br/>(Batch)"]
        INDEX_MGR["Index Manager<br/>(Sync Pinecone)"]
    end

    subgraph Query["Query Pipeline"]
        QA_ENGINE["QA Engine<br/>(Orchestrator)"]
        RETRIEVER["Retriever<br/>(Dual BM25+Vector)"]
        RE_RANKER["Re-ranker<br/>(Cross-encoder)"]
        GENERATOR["Generator<br/>(LLM Wrapper)"]
    end

    subgraph Cache["Caching & Storage"]
        CACHE_LAYER["Cache Layer<br/>(Redis)"]
        VECTOR_CLIENT["Vector Client<br/>(Pinecone SDK)"]
        SEARCH_CLIENT["Search Client<br/>(ES Client)"]
        PERSIST["Persistence<br/>(PostgreSQL)"]
    end

    UPLOAD_EP --> DOC_PROCESSOR
    DOC_PROCESSOR --> EMBEDDING_SVC
    EMBEDDING_SVC --> INDEX_MGR

    INDEX_MGR --> VECTOR_CLIENT
    PERSIST --> INDEX_MGR

    QUERY_EP --> QA_ENGINE
    QA_ENGINE --> RETRIEVER
    RETRIEVER --> VECTOR_CLIENT
    RETRIEVER --> SEARCH_CLIENT

    VECTOR_CLIENT --> CACHE_LAYER
    SEARCH_CLIENT --> CACHE_LAYER

    RETRIEVER --> RE_RANKER
    RE_RANKER --> GENERATOR
    GENERATOR --> CACHE_LAYER
    GENERATOR --> PERSIST
```""",

        "03-process-flow": """## Process Flow (RAG Query to Answer)

```mermaid
graph TD
    USER_QUERY["User Submits Query"] --> CLEAN["Clean & Normalize<br/>Query Text"]

    CLEAN --> EMBED_QUERY["Embed Query<br/>(Dense Vector)"]
    EMBED_QUERY --> CACHE_CHECK{{"Cache<br/>Hit?"}}

    CACHE_CHECK -->|Yes| SERVE_CACHED["Return Cached<br/>Answer"]
    CACHE_CHECK -->|No| VECTOR_SEARCH["Vector Search<br/>(Pinecone)"]

    VECTOR_SEARCH --> BM25_SEARCH["BM25 Search<br/>(Elasticsearch)"]
    BM25_SEARCH --> FUSION["Fusion Retriever<br/>(Hybrid Ranking)"]

    FUSION --> RERANK["Re-rank Results<br/>(Cross-encoder)"]
    RERANK --> SELECT["Select Top-K<br/>(usually 3-5)"]

    SELECT --> BUILD_CONTEXT["Build Context<br/>(Concatenate Docs)"]
    BUILD_CONTEXT --> CHECK_LENGTH{{"Context<br/>Fits Token<br/>Limit?"}}

    CHECK_LENGTH -->|No| COMPRESS["Compress Context<br/>(Summarize)"]
    CHECK_LENGTH -->|Yes| BUILD_PROMPT["Build Prompt<br/>(Template + Context)"]

    COMPRESS --> BUILD_PROMPT
    BUILD_PROMPT --> GENERATE["Generate Answer<br/>(LLM)"]

    GENERATE --> POST_PROCESS["Post-process<br/>(Format, Validate)"]
    POST_PROCESS --> CITE["Add Citations<br/>(Source URLs)"]

    CITE --> CACHE_STORE["Store in Cache<br/>(Redis)"]
    CACHE_STORE --> PERSIST_STORE["Persist Query<br/>(Logging)"]

    PERSIST_STORE --> RETURN["Return Answer<br/>+ Sources + Confidence"]

    RETURN --> END["Complete"]
    SERVE_CACHED --> END
```"""
    }

    return diagrams


def generate_agent_system_diagrams():
    """Generate diagrams for AI agent system."""

    diagrams = {
        "01-system-architecture": """## System Architecture (Agent System)

```mermaid
graph TB
    subgraph Orchestration["Agent Orchestration"]
        AGENT_LOOP["Agent Loop<br/>(Observe-Think-Act)"]
        PLANNER["Planner<br/>(Goal Decomposition)"]
        REASONER["Reasoner<br/>(Chain-of-Thought)"]
    end

    subgraph Tools["Tool/Action Layer"]
        REGISTRY["Tool Registry<br/>(Discovery)"]
        TOOL1["Calculator Tool"]
        TOOL2["Web Search Tool"]
        TOOL3["Code Execution<br/>(Sandbox)"]
    end

    subgraph Memory["Memory Systems"]
        WORKING["Working Memory<br/>(Current State)"]
        LONGTERM["Long-term Memory<br/>(Learned Facts)"]
        EPISODIC["Episodic Memory<br/>(History)"]
    end

    subgraph Models["AI Models"]
        LLM["Language Model<br/>(Reasoning)"]
        EMBEDDING["Embedding Model<br/>(Similarity)"]
        CLASSIFIER["Intent Classifier<br/>(Tool Selection)"]
    end

    subgraph Execution["Execution Engine"]
        EXECUTOR["Tool Executor<br/>(Safe Execution)"]
        MONITOR["Monitor<br/>(Resource Limits)"]
        SANDBOX["Sandbox<br/>(Code Isolation)"]
    end

    subgraph Storage["State & Storage"]
        STATE_STORE["State Store<br/>(Redis)"]
        VECTOR_INDEX["Vector Index<br/>(Embeddings)"]
        HISTORY["History DB<br/>(PostgreSQL)"]
    end

    AGENT_LOOP --> PLANNER
    PLANNER --> REASONER
    REASONER --> LLM

    LLM --> CLASSIFIER
    CLASSIFIER --> REGISTRY
    REGISTRY --> TOOL1
    REGISTRY --> TOOL2
    REGISTRY --> TOOL3

    TOOL1 --> EXECUTOR
    TOOL2 --> EXECUTOR
    TOOL3 --> EXECUTOR

    EXECUTOR --> MONITOR
    EXECUTOR --> SANDBOX

    AGENT_LOOP --> WORKING
    REASONER --> EPISODIC
    EMBEDDING --> VECTOR_INDEX
    AGENT_LOOP --> STATE_STORE

    WORKING --> STATE_STORE
    EPISODIC --> HISTORY
    VECTOR_INDEX --> LONGTERM
```""",

        "02-application-architecture": """## Application Architecture (Agent)

```mermaid
graph TB
    subgraph Core["Core Agent"]
        AGENT["Agent Class<br/>(Main Loop)"]
        PLANNING["Planning Module<br/>(Goal Tree)"]
        REFLECTION["Reflection Module<br/>(Self-Critique)"]
    end

    subgraph Processing["Processing"]
        INPUT_HANDLER["Input Handler<br/>(User Intent)"]
        OUTPUT_FORMATTER["Output Formatter<br/>(Agent Response)"]
        PARSER["Output Parser<br/>(Extract Actions)"]
    end

    subgraph Models["Model Wrappers"]
        LLM_WRAPPER["LLM Wrapper<br/>(API Client)"]
        EMBEDDING_WRAPPER["Embedding Wrapper<br/>(Batch Processing)"]
        TOOL_SELECTOR["Tool Selector<br/>(Classification)"]
    end

    subgraph Tools["Tool Management"]
        TOOL_MANAGER["Tool Manager<br/>(Registry)"]
        TOOL_VALIDATOR["Tool Validator<br/>(Schema)"]
        TOOL_EXECUTOR["Tool Executor<br/>(Sandboxed)"]
    end

    subgraph State["State Management"]
        STATE_MGR["State Manager<br/>(Current Context)"]
        MEMORY_MGR["Memory Manager<br/>(Multi-level)"]
        HISTORY_TRACKER["History Tracker<br/>(Audit Log)"]
    end

    subgraph Storage["Data Access"]
        VECTOR_STORE["Vector Store<br/>(Retrieval)"]
        DOCUMENT_STORE["Document Store<br/>(Files)"]
        CACHE["Cache Layer<br/>(Redis)"]
    end

    AGENT --> PLANNING
    AGENT --> REFLECTION
    INPUT_HANDLER --> AGENT
    AGENT --> PARSER
    PARSER --> TOOL_SELECTOR

    TOOL_SELECTOR --> TOOL_MANAGER
    TOOL_MANAGER --> TOOL_VALIDATOR
    TOOL_VALIDATOR --> TOOL_EXECUTOR

    AGENT --> LLM_WRAPPER
    PLANNING --> LLM_WRAPPER
    REFLECTION --> LLM_WRAPPER

    TOOL_EXECUTOR --> OUTPUT_FORMATTER
    AGENT --> STATE_MGR
    STATE_MGR --> MEMORY_MGR

    MEMORY_MGR --> VECTOR_STORE
    MEMORY_MGR --> DOCUMENT_STORE
    MEMORY_MGR --> CACHE
    AGENT --> HISTORY_TRACKER
```""",

        "03-process-flow": """## Process Flow (Agent Execution)

```mermaid
graph TD
    START["User Request"] --> PARSE["Parse User Input<br/>(Intent + Context)"]

    PARSE --> DECOMPOSE["Decompose Goal<br/>(Sub-tasks)"]
    DECOMPOSE --> PLAN["Create Plan<br/>(Action Sequence)"]

    PLAN --> LOOP["Agent Loop Iteration"]

    LOOP --> OBSERVE["Observe Current State<br/>(Context, History)"]
    OBSERVE --> THINK["Think/Reason<br/>(Generate Next Action)"]

    THINK --> LLM_CALL["Call LLM<br/>(Chain-of-Thought)"]
    LLM_CALL --> PARSE_ACTION["Parse Action<br/>(Tool + Args)"]

    PARSE_ACTION --> VALIDATE{{"Valid<br/>Action?"}}
    VALIDATE -->|No| ERROR_HANDLING["Error Handling<br/>(Retry/Replan)"]
    ERROR_HANDLING --> LOOP

    VALIDATE -->|Yes| SELECT_TOOL["Select Tool<br/>(from Registry)"]
    SELECT_TOOL --> EXECUTE["Execute Tool<br/>(Sandboxed)"]

    EXECUTE --> CAPTURE["Capture Output<br/>(Result)"]
    CAPTURE --> UPDATE_STATE["Update State<br/>(Memory + History)"]

    UPDATE_STATE --> CHECK_DONE{{"Goal<br/>Achieved?"}}
    CHECK_DONE -->|No| LOOP
    CHECK_DONE -->|Yes| REFLECT["Reflection<br/>(Self-Critique)"]

    REFLECT --> IMPROVE{{"Should<br/>Retry?"}}
    IMPROVE -->|Yes| LOOP
    IMPROVE -->|No| FORMAT["Format Final<br/>Answer"]

    FORMAT --> EXPLAIN["Add Explanation<br/>(Reasoning Chain)"]
    EXPLAIN --> RETURN["Return to User<br/>(With Trace)"]

    RETURN --> END["Complete"]
```

**Agent Execution Characteristics:**
- **Observe**: Get current context (user request, history, state)
- **Think**: Generate reasoning and next action using LLM
- **Act**: Execute selected tool with validated arguments
- **Reflect**: Self-critique and decide if refinement needed
- **Loop**: Continue until goal achieved or max iterations reached"""
    }

    return diagrams


def create_diagram_file(system_name: str, diagram_type: str, mermaid_content: str):
    """Create or update a markdown file with architecture diagram."""
    filename = f"{system_name}-{diagram_type}.md"
    filepath = Path("arch-review/diagrams") / filename

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(mermaid_content)
    return filepath


def main():
    """Generate all architecture diagrams."""
    os.makedirs("arch-review/diagrams", exist_ok=True)

    # Generate diagram sets
    all_diagrams = {
        "01-customer-service": generate_customer_service_diagrams(),
        "02-rag-system": generate_rag_system_diagrams(),
        "05-code-review-agent": generate_agent_system_diagrams(),
    }

    created = 0
    for system_name, diagrams in all_diagrams.items():
        for diagram_type, content in diagrams.items():
            filepath = create_diagram_file(system_name, diagram_type, content)
            print(f"✅ Created {filepath}")
            created += 1

    # Create index
    index_content = """# Architecture Diagrams Library

This directory contains three types of architecture diagrams for each system:

1. **System Architecture** - Infrastructure, deployment, cloud services, scaling
2. **Application Architecture** - Components, modules, layers, internal structure
3. **Process Flow** - Request pipelines, business workflows, decision flows

## Systems with Diagrams

### Customer Service Platform
- [System Architecture](01-customer-service-01-system-architecture.md)
- [Application Architecture](01-customer-service-02-application-architecture.md)
- [Process Flow](01-customer-service-03-process-flow.md)

### RAG Document QA System
- [System Architecture](02-rag-system-01-system-architecture.md)
- [Application Architecture](02-rag-system-02-application-architecture.md)
- [Process Flow](02-rag-system-03-process-flow.md)

### AI Code Review Agent
- [System Architecture](05-code-review-agent-01-system-architecture.md)
- [Application Architecture](05-code-review-agent-02-application-architecture.md)
- [Process Flow](05-code-review-agent-03-process-flow.md)

## How to Use

### Integrate into Architecture Files
Include diagrams in your system architecture markdown:

```markdown
## System Architecture

[System-level infrastructure diagram]

## Application Architecture

[Component-level internal structure diagram]

## Process Flow

[Request pipeline and decision flow diagram]
```

### Create New Diagrams
Run the generator script:
```bash
python3 scripts/generate_architecture_diagrams.py
```

## Diagram Types Explained

### System Architecture
Shows infrastructure, deployment, and operational concerns:
- Kubernetes clusters, auto-scaling
- Databases, caching layers
- External APIs and services
- Monitoring and observability tools
- CDN, load balancers, geographic distribution

### Application Architecture
Shows component design and internal structure:
- API layers, handlers
- Service layers and modules
- Data access patterns
- Internal dependencies
- Configuration and utilities

### Process Flow
Shows request pipelines and decision workflows:
- Request → processing → response flow
- Decision points (if/then branches)
- Error handling paths
- Optimization points (caching, deduplication)
- Async operations

## Benefits

- **Clarity**: Different levels of detail for different audiences
- **Scalability**: Shows how system scales with load
- **Operations**: Shows what happens at deployment
- **Debugging**: Shows where errors can occur
- **Design**: Shows architectural patterns and trade-offs
"""

    Path("arch-review/diagrams/README.md").write_text(index_content)
    print(f"\n✅ Created index: arch-review/diagrams/README.md")
    print(f"✅ Generated {created + 1} files total")


if __name__ == "__main__":
    main()
