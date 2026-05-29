# LLM Content Moderation - Process Flow

```mermaid
sequenceDiagram
    participant Content as Content Source
    participant Classifier as ML Classifier
    participant PolicyEng as Policy Engine
    participant Router as Escalation Router
    participant LLM as LLM Reviewer
    participant DecLog as Decision Logger
    participant ActionEng as Action Engine
    participant User as User

    Content->>Classifier: Submit content for review
    Classifier-->>PolicyEng: Toxicity score
    PolicyEng-->>Router: Policy evaluation result
    alt Clear violation (score above 0.9)
        Router->>DecLog: Log block decision
        DecLog->>ActionEng: Block content
        ActionEng-->>User: Content blocked
    else Low confidence (score 0.4 to 0.8)
        Router->>LLM: Escalate for review
        LLM-->>Router: LLM decision with reasoning
        Router->>DecLog: Log LLM decision
        DecLog->>ActionEng: Enforce LLM decision
        ActionEng-->>User: Decision applied
    else Passes all checks (score below 0.4)
        Router->>DecLog: Log allow decision
        DecLog->>ActionEng: Allow content
        ActionEng-->>User: Content approved
    end
```

**Key Decision Points:**
1. **Score Threshold**: Score above 0.9 triggers immediate block; 0.4-0.8 escalates to LLM review
2. **LLM Escalation**: Reserved for borderline cases to keep cost low; processes ~5% of content
3. **Decision Logging**: Every decision logged regardless of outcome for audit trail
4. **Action Enforcement**: Block, flag, soft-filter, or allow based on policy configuration

**Optimization Points:**
- Rule-based pre-filter eliminates 60-70% of content before ML model inference
- ML classifier handles 25-30%; only 5% reaches expensive LLM reviewer
- Async logging does not block the enforcement action response path
