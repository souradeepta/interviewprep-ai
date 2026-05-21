## Process Flow (Request to Response)

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
- Context caching (avoid re-retrieving same KB articles)