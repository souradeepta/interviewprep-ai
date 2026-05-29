# LLM Content Moderation - System Architecture

```mermaid
graph TD
    subgraph Ingestion["Content Ingestion"]
        ContentAPI["Content API<br/>(REST/Stream)"]
        MsgQueue["Message Queue<br/>(Kafka)"]
    end

    subgraph Classifiers["Classifier Pipeline"]
        RuleEngine["Rule-Based Filter<br/>(less than 10ms)"]
        MLClassifier["ML Classifier<br/>(less than 100ms)"]
        LLMReviewer["LLM Reviewer<br/>(Hard Cases)"]
    end

    subgraph Actions["Action Engine"]
        ActionEngine["Action Engine<br/>(Allow/Flag/Block)"]
        AppealQueue["Appeal Queue<br/>(Human Review)"]
    end

    subgraph Logging["Decision Logging"]
        DecisionLog["Decision Logger<br/>(PostgreSQL)"]
        AuditStore["Audit Store<br/>(S3)"]
    end

    subgraph Monitoring["Monitoring"]
        Prometheus["Prometheus<br/>(Throughput)"]
        AlertManager["Alert Manager<br/>(Policy Breach)"]
    end

    ContentAPI --> MsgQueue
    MsgQueue --> RuleEngine
    RuleEngine -->|Clear violation| ActionEngine
    RuleEngine -->|Pass to ML| MLClassifier
    MLClassifier -->|High confidence| ActionEngine
    MLClassifier -->|Low confidence| LLMReviewer
    LLMReviewer --> ActionEngine
    ActionEngine --> DecisionLog
    ActionEngine --> AuditStore
    ActionEngine --> AppealQueue
    DecisionLog --> Prometheus
    Prometheus --> AlertManager
```

**Infrastructure Components:**
- **Rule-Based Filter**: Regex and keyword blocklist, sub-10ms, catches obvious violations
- **ML Classifier**: Fine-tuned transformer (toxicity, hate speech, NSFW), sub-100ms
- **LLM Reviewer**: Full LLM review for ambiguous or borderline content, 1-5 seconds
- **Action Engine**: Enforces policy decisions (allow, soft filter, block, shadowban)
- **Appeal Queue**: Human review interface for contested moderation decisions
- **Audit Store**: Immutable S3 log of all moderation decisions for compliance
