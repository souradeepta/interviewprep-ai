# Realtime Translation Service - Application Architecture

```mermaid
graph TD
    subgraph Gateway["Input Gateway"]
        InputGW["Input Gateway<br/>(FastAPI/WS)"]
        InputRouter["Input Router<br/>(Audio vs Text)"]
    end

    subgraph LanguageServices["Language Services"]
        LangIDSvc["Language ID Service<br/>(FastText)"]
        ASRSvc["ASR Service<br/>(Whisper API)"]
    end

    subgraph TranslationServices["Translation Services"]
        TransSvc["Translation Service<br/>(NMT Engine)"]
        GrammarSvc["Grammar Corrector<br/>(T5 Model)"]
    end

    subgraph CacheAndStore["Cache and Storage"]
        TransCache["Translation Cache<br/>(Redis)"]
        RequestLog["Request Logger<br/>(PostgreSQL)"]
    end

    subgraph DeliveryLayer["Delivery Layer"]
        Formatter["Response Formatter<br/>(JSON/SSE)"]
        WSDelivery["WebSocket Delivery<br/>(Real-time)"]
    end

    InputGW --> InputRouter
    InputRouter -->|Audio| ASRSvc
    InputRouter -->|Text| LangIDSvc
    ASRSvc --> LangIDSvc
    LangIDSvc --> TransCache
    TransCache -->|Miss| TransSvc
    TransSvc --> GrammarSvc
    GrammarSvc --> TransCache
    GrammarSvc --> Formatter
    TransCache -->|Hit| Formatter
    Formatter --> WSDelivery
    Formatter --> RequestLog
```

**Layer Breakdown:**
- **Input Gateway**: WebSocket and REST endpoints, routes audio vs text to correct pipeline
- **Language Services**: FastText language detection and Whisper ASR for audio transcription
- **Translation Services**: NMT model for core translation, T5-based grammar correction pass
- **Cache and Storage**: Redis for translation cache, PostgreSQL for audit logging
- **Delivery Layer**: Server-Sent Events or WebSocket streaming for low-latency delivery
