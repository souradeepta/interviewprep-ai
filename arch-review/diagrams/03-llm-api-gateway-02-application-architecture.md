# LLM API Gateway - Application Architecture

```mermaid
graph TD
    subgraph APILayer["API Layer"]
        RestAPI["REST API<br/>(FastAPI)"]
        InputVal["Input Validator<br/>(Pydantic)"]
    end

    subgraph Middleware["Middleware Layer"]
        AuthMW["Auth Middleware<br/>(JWT / API Key)"]
        RateLimiter["Rate Limiter<br/>(Token Bucket)"]
        ReqNorm["Request Normalizer<br/>(Schema Mapper)"]
    end

    subgraph RoutingLayer["Routing Layer"]
        CacheCheck["Cache Lookup<br/>(Redis)"]
        RoutingEngine["Routing Engine<br/>(Policy Rules)"]
    end

    subgraph ProviderLayer["Provider Adapters"]
        OpenAIAdapter["OpenAI<br/>Adapter"]
        AnthropicAdapter["Anthropic<br/>Adapter"]
        CohereAdapter["Cohere<br/>Adapter"]
        MistralAdapter["Mistral<br/>Adapter"]
    end

    subgraph PostProcess["Post-Processing"]
        RespNorm["Response Normalizer<br/>(Unified Schema)"]
        UsageTracker["Usage Tracker<br/>(DB Write)"]
    end

    RestAPI --> InputVal
    InputVal --> AuthMW
    AuthMW --> RateLimiter
    RateLimiter --> ReqNorm
    ReqNorm --> CacheCheck
    CacheCheck -->|Hit| RespNorm
    CacheCheck -->|Miss| RoutingEngine
    RoutingEngine --> OpenAIAdapter
    RoutingEngine --> AnthropicAdapter
    RoutingEngine --> CohereAdapter
    RoutingEngine --> MistralAdapter
    OpenAIAdapter --> RespNorm
    AnthropicAdapter --> RespNorm
    CohereAdapter --> RespNorm
    MistralAdapter --> RespNorm
    RespNorm --> UsageTracker
```

**Layer Breakdown:**
- **API Layer**: FastAPI REST endpoints with Pydantic request/response validation
- **Middleware**: JWT/API key auth, token-bucket rate limiting, schema normalization
- **Routing Layer**: Redis cache check first, then rule-based provider selection
- **Provider Adapters**: Thin adapters mapping unified schema to each provider API
- **Post-Processing**: Normalize provider responses to unified schema, write usage metrics
