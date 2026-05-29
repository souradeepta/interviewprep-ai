# Realtime Translation Service - Process Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant GW as Input Gateway
    participant LangID as Language Detector
    participant ASR as ASR Service
    participant Trans as Translation Engine
    participant PostProc as Post-Processor
    participant Cache as Redis Cache

    C->>GW: Send input (audio or text)
    alt Audio input
        GW->>ASR: Stream audio chunks
        ASR-->>GW: Transcribed text
        GW->>LangID: Detect language of transcription
    else Text input
        GW->>LangID: Detect language of text
    end
    LangID-->>GW: Source language detected
    GW->>Cache: Lookup (source text + target lang)
    alt Cache hit
        Cache-->>GW: Cached translation
        GW-->>C: Return cached translation
    else Cache miss
        Cache-->>GW: Not found
        GW->>Trans: Translate (src lang + target lang + text)
        Trans-->>GW: Raw translation
        GW->>PostProc: Grammar correction pass
        PostProc-->>GW: Cleaned translation
        GW->>Cache: Store translation (TTL 24h)
        GW-->>C: Stream translated text
    end
```

**Key Decision Points:**
1. **Input Type Routing**: Audio goes through ASR first, text goes directly to language detection
2. **Language Detection**: Determines source language before cache lookup or translation
3. **Cache Check**: Common phrases and sentences served from Redis without NMT model call
4. **Post-Processing**: Grammar correction improves fluency of raw NMT output

**Optimization Points:**
- Chunk audio at 500ms intervals for streaming translation with low perceived latency
- Cache TTL 24 hours for common business phrases; shorter TTL for dynamic content
- Batch short segments together to improve NMT throughput at cost of minor latency
