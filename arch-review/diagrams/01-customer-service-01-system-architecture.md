## System Architecture (Infrastructure & Deployment)

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
- **Load Balancing**: AWS ALB with health checks and auto-scaling policies