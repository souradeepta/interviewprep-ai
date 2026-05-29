## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        RestAPI["REST API\n(FastAPI)"]
        WebSocketHandler["WebSocket Handler\n(Async)"]
        InputValidator["Input Validator\n(Pydantic)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        TicketRouter["Ticket Router\n(Intent to Pipeline)"]
        AgentOrchestrator["Agent Orchestrator\n(Tool Planner)"]
        ContextManager["Context Manager\n(Session State)"]
    end

    subgraph MLServices["ML Services"]
        IntentClassifier["Intent Classifier\n(Fine-tuned BERT)"]
        ConfidenceScorer["Confidence Scorer\n(0.0 to 1.0)"]
        EscalationDecider["Escalation Decider\n(Threshold Logic)"]
    end

    subgraph ToolServices["Tool Services"]
        CRMClient["CRM Client\n(Salesforce SDK)"]
        OrderClient["Order Client\n(Internal API)"]
        KBRetriever["KB Retriever\n(Vector Search)"]
    end

    subgraph GenerationServices["Generation Services"]
        PromptBuilder["Prompt Builder\n(Template Engine)"]
        LLMWrapper["LLM Wrapper\n(GPT-4 Turbo)"]
        ResponseValidator["Response Validator\n(Safety Check)"]
    end

    subgraph DataServices["Data Services"]
        TicketDB["Ticket Manager\n(PostgreSQL ORM)"]
        CacheClient["Cache Client\n(Redis)"]
        FeedbackCollector["Feedback Collector\n(Event Stream)"]
    end

    RestAPI --> InputValidator
    WebSocketHandler --> InputValidator
    InputValidator --> TicketRouter
    TicketRouter --> ContextManager
    ContextManager --> AgentOrchestrator
    AgentOrchestrator --> IntentClassifier
    IntentClassifier --> ConfidenceScorer
    ConfidenceScorer --> EscalationDecider
    AgentOrchestrator --> CRMClient
    AgentOrchestrator --> OrderClient
    AgentOrchestrator --> KBRetriever
    KBRetriever --> PromptBuilder
    PromptBuilder --> LLMWrapper
    LLMWrapper --> ResponseValidator
    ResponseValidator --> TicketDB
    ContextManager --> CacheClient
    ResponseValidator --> FeedbackCollector
```

**Layer Breakdown:**
- **Presentation**: REST and WebSocket APIs with input validation
- **Orchestration**: Intent-based routing, agent tool planning, session context management
- **ML Services**: Intent classification, confidence scoring, escalation decisions
- **Tool Services**: CRM, order management, and knowledge base integrations
- **Generation Services**: Prompt construction, LLM invocation, safety validation
- **Data Services**: Ticket persistence, session cache, feedback collection
