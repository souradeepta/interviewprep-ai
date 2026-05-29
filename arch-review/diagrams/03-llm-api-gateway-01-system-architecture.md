# LLM API Gateway - System Architecture

```mermaid
graph TD
    subgraph Clients["Client Layer"]
        WebApp["Web Application"]
        MobileApp["Mobile App"]
        BackendSvc["Backend Service"]
    end

    subgraph GatewayCore["API Gateway Core"]
        AuthService["Auth and Rate<br/>Limit Service"]
        Router["Request Router<br/>(Model Selector)"]
        Cache["Response Cache<br/>(Redis)"]
        CostTracker["Cost Tracker<br/>(Usage Meter)"]
    end

    subgraph ProviderPool["LLM Provider Pool"]
        OpenAI["OpenAI<br/>(GPT-4)"]
        Anthropic["Anthropic<br/>(Claude)"]
        Cohere["Cohere<br/>(Command)"]
        Mistral["Mistral<br/>(Mistral-7B)"]
    end

    subgraph Observability["Observability"]
        Prometheus["Prometheus<br/>(Metrics)"]
        ELK["ELK Stack<br/>(Logs)"]
    end

    WebApp --> AuthService
    MobileApp --> AuthService
    BackendSvc --> AuthService

    AuthService -->|Allowed| Router
    AuthService -->|Blocked| RejectionLog["Rejection<br/>Log"]

    Router --> Cache
    Cache -->|Miss| OpenAI
    Cache -->|Miss| Anthropic
    Cache -->|Miss| Cohere
    Cache -->|Miss| Mistral
    Cache -->|Hit| ResponseAgg["Response<br/>Aggregator"]

    OpenAI --> ResponseAgg
    Anthropic --> ResponseAgg
    Cohere --> ResponseAgg
    Mistral --> ResponseAgg

    ResponseAgg --> CostTracker
    CostTracker --> Prometheus
    Router --> ELK
```

**Infrastructure Components:**
- **Auth and Rate Limiting**: JWT validation, per-key quota enforcement, IP allowlist
- **Request Router**: Model selection based on cost, latency, capability requirements
- **Provider Pool**: OpenAI, Anthropic, Cohere, Mistral with fallback routing
- **Response Cache**: Redis for deduplicating identical prompts (cache key = hash of prompt)
- **Cost Tracker**: Per-token usage metering, budget alerts, chargeback reporting
- **Observability**: Prometheus metrics, ELK logs for audit and debugging
