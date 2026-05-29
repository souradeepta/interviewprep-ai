# LLM API Gateway - Process Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant GW as Gateway
    participant Auth as Auth Service
    participant Redis as Redis Cache
    participant Router as Routing Engine
    participant Provider as LLM Provider

    C->>GW: POST /v1/completions
    GW->>Auth: Validate API key and quota
    alt Auth failed or quota exceeded
        Auth-->>GW: 401 or 429
        GW-->>C: Error response
    else Auth passed
        Auth-->>GW: OK
        GW->>Redis: Lookup cache key (hash of prompt)
        alt Cache hit
            Redis-->>GW: Cached response
            GW-->>C: Return cached response
        else Cache miss
            Redis-->>GW: Not found
            GW->>Router: Select provider (cost, latency, model)
            Router->>Provider: Forward normalized request
            Provider-->>Router: Raw completion response
            Router-->>GW: Provider response
            GW->>Redis: Store response (TTL 1 hour)
            GW->>GW: Record token usage and cost
            GW-->>C: Normalized response
        end
    end
```

**Key Decision Points:**
1. **Auth Check**: API key validity and per-key quota enforcement before any processing
2. **Cache Lookup**: Identical prompts return cached responses, avoiding duplicate LLM calls
3. **Provider Selection**: Router picks provider based on model requirements and cost policy
4. **Error Handling**: Auth failures return 401, quota exceeded returns 429, provider errors trigger fallback

**Optimization Points:**
- Cache hit rate target 20-40% for repeated queries (FAQ, similar user requests)
- Provider fallback: if primary provider times out, retry on secondary within 500ms
- Async usage recording so billing writes do not block the response path
