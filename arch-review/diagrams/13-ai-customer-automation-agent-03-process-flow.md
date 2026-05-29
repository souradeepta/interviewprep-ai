## Process Flow (Ticket to Resolution)

```mermaid
sequenceDiagram
    participant User as Customer
    participant Gateway as API Gateway
    participant Classifier as Intent Classifier
    participant Router as Ticket Router
    participant Agent as Agent Core
    participant KB as Knowledge Base
    participant CRM as CRM Tool
    participant LLM as Response Generator
    participant Feedback as Feedback Collector

    User->>Gateway: Submit support message
    Gateway->>Classifier: Classify intent
    Classifier->>Classifier: Score confidence
    alt Confidence below 0.75
        Classifier->>Router: Escalate to human queue
        Router-->>User: Acknowledgement + wait time
    else Confidence above 0.75
        Classifier->>Router: Route by category
        Router->>Agent: Dispatch to agent
        Agent->>KB: Search relevant KB articles
        KB-->>Agent: Return top-3 articles
        Agent->>CRM: Fetch account context
        CRM-->>Agent: Account history, status
        Agent->>LLM: Generate response with context
        LLM->>LLM: Validate response safety
        alt Response confidence above 0.85
            LLM-->>User: Send auto-response
            User->>Feedback: Rate response
            Feedback->>Classifier: Update training signal
        else Response confidence below 0.85
            LLM->>Router: Escalate with context
            Router-->>User: Escalate to human agent
        end
    end
```

**Key Decision Points:**
1. **Confidence Check (0.75)**: Below threshold routes directly to human queue
2. **KB Search**: Retrieves top-3 articles before response generation
3. **CRM Lookup**: Fetches account context for personalized responses
4. **Response Gate (0.85)**: Auto-send only if generation confidence is high enough
5. **Feedback Loop**: User ratings flow back to retrain the classifier

**Error Paths:**
- Low intent confidence: immediate human escalation with ticket context
- Low response confidence: escalation with LLM reasoning attached
- Tool failure (CRM/KB): fallback to generic response with human review flag

**Optimization Points:**
- Cache frequent KB lookups (Redis, 1-hour TTL)
- Batch classify low-urgency tickets during off-peak hours
- Pre-warm CRM context for known high-value accounts
