# LLM Content Moderation - Application Architecture

```mermaid
graph TD
    subgraph InputLayer["Input Layer"]
        InputAPI["Input API<br/>(FastAPI)"]
        InputSanitizer["Input Sanitizer<br/>(Encoding Fix)"]
    end

    subgraph ClassificationLayer["Classification Layer"]
        ToxicityClassifier["Toxicity Classifier<br/>(Transformer)"]
        PolicyEngine["Policy Engine<br/>(Rule Evaluator)"]
        EscalationRouter["Escalation Router<br/>(Confidence Check)"]
    end

    subgraph ReviewLayer["Review Layer"]
        LLMReviewer["LLM Reviewer<br/>(GPT-4 based)"]
        ReviewContext["Review Context<br/>(Thread Builder)"]
    end

    subgraph DecisionLayer["Decision Layer"]
        DecisionLogger["Decision Logger<br/>(DB Write)"]
        ActionDispatcher["Action Dispatcher<br/>(Enforce Policy)"]
        AppealHandler["Appeal Handler<br/>(Override Logic)"]
    end

    InputAPI --> InputSanitizer
    InputSanitizer --> ToxicityClassifier
    ToxicityClassifier --> PolicyEngine
    PolicyEngine --> EscalationRouter
    EscalationRouter -->|High confidence| DecisionLogger
    EscalationRouter -->|Low confidence| ReviewContext
    ReviewContext --> LLMReviewer
    LLMReviewer --> DecisionLogger
    DecisionLogger --> ActionDispatcher
    ActionDispatcher --> AppealHandler
```

**Layer Breakdown:**
- **Input Layer**: FastAPI endpoint with sanitization (encoding normalization, length limits)
- **Classification Layer**: Transformer-based toxicity scoring, policy rule application, confidence routing
- **Review Layer**: LLM reviewer with full thread context for nuanced moderation decisions
- **Decision Layer**: Persistent logging of all decisions, policy enforcement, appeal handling
