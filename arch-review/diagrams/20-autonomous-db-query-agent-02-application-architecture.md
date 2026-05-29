## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        WebUI["Web UI\n(React)"]
        RestAPI["REST API\n(FastAPI)"]
        SlackIntegration["Slack Integration\n(Bolt SDK)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        QueryOrchestrator["Query Orchestrator\n(Multi-step agent)"]
        ClarificationHandler["Clarification Handler\n(Ambiguity detection)"]
        ApprovalGate["Approval Gate\n(Cost check)"]
    end

    subgraph NLUServices["NLU Services"]
        IntentParser["Intent Parser\n(LLM, entity extract)"]
        AmbiguityDetector["Ambiguity Detector\n(Confidence score)"]
        SchemaMapper["Schema Mapper\n(Vector retrieval)"]
    end

    subgraph SQLServices["SQL Services"]
        SQLGenerator["SQL Generator\n(LLM + few-shot)"]
        QueryValidator["Query Validator\n(Plan analysis)"]
        AccessController["Access Controller\n(RBAC check)"]
    end

    subgraph ResultServices["Result Services"]
        QueryExecutor["Query Executor\n(Read-only connection)"]
        SanityChecker["Sanity Checker\n(Row count, types)"]
        ExplainGenerator["Explain Generator\n(LLM)"]
    end

    subgraph DataServices["Data Services"]
        SchemaVectorClient["Schema Vector Client\n(Pinecone)"]
        QueryCacheClient["Query Cache Client\n(Redis)"]
        AuditLogger["Audit Logger\n(PostgreSQL)"]
    end

    WebUI --> RestAPI
    SlackIntegration --> RestAPI
    RestAPI --> QueryOrchestrator
    QueryOrchestrator --> IntentParser
    IntentParser --> AmbiguityDetector
    AmbiguityDetector --> ClarificationHandler
    IntentParser --> SchemaMapper
    SchemaMapper --> SchemaVectorClient
    SchemaMapper --> SQLGenerator
    SQLGenerator --> QueryValidator
    QueryValidator --> ApprovalGate
    ApprovalGate --> AccessController
    AccessController --> QueryExecutor
    QueryExecutor --> SanityChecker
    SanityChecker --> ExplainGenerator
    ExplainGenerator --> QueryCacheClient
    QueryExecutor --> AuditLogger
```

**Layer Breakdown:**
- **Presentation**: Web UI, REST API, Slack integration for natural language queries
- **Orchestration**: Multi-step query agent, clarification handling, cost approval gate
- **NLU Services**: LLM-based intent parsing, ambiguity detection, schema vector mapping
- **SQL Services**: LLM SQL generation, query plan validation, RBAC access control
- **Result Services**: Read-only query execution, sanity checks, LLM explanations
- **Data Services**: Schema vector store, query result cache, audit log
