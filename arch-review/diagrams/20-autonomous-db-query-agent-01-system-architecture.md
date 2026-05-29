## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Ingestion["Query Ingestion Layer"]
        WebUI["Web UI\n(React)"]
        RestAPI["REST API\n(FastAPI)"]
        SlackBot["Slack Bot\n(Integration)"]
    end

    subgraph Processing["Processing Cluster (Kubernetes)"]
        NLQueryParser["NL Query Parser\nPod x 5"]
        SchemaLoader["Schema Loader\nPod x 3"]
        SQLGenerator["SQL Generator\n(LLM) Pod x 8"]
        Validator["Query Validator\nPod x 5"]
    end

    subgraph Execution["Execution Layer"]
        QueryExecutor["Query Executor\n(Read-only)"]
        ResultFormatter["Result Formatter\nPod x 5"]
        ExplainGen["Explanation Generator\n(LLM) Pod x 3"]
    end

    subgraph DataLayer["Data Layer"]
        TargetDB["Target Database\n(PostgreSQL/BigQuery)"]
        SchemaVectorDB["Schema Vector DB\n(Pinecone)"]
        QueryCache["Query Cache\n(Redis, 4h TTL)"]
        QueryLog["Query Audit Log\n(PostgreSQL)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus\n(Query Metrics)"]
        ELK["ELK Stack\n(Audit Logs)"]
        CostTracker["Cost Tracker\n(DB query cost)"]
    end

    WebUI --> RestAPI
    SlackBot --> RestAPI
    RestAPI --> NLQueryParser
    NLQueryParser --> SchemaLoader
    SchemaLoader --> SchemaVectorDB
    SchemaLoader --> SQLGenerator
    SQLGenerator --> Validator
    Validator --> QueryExecutor
    QueryExecutor --> TargetDB
    QueryExecutor --> ResultFormatter
    ResultFormatter --> ExplainGen
    ExplainGen --> QueryCache
    ExplainGen --> QueryLog
    SQLGenerator --> Prom
    QueryExecutor --> ELK
    QueryExecutor --> CostTracker
```

**Infrastructure Components:**
- **Compute**: Kubernetes cluster with separate pods for NL parsing, SQL generation, validation, execution
- **Storage**: PostgreSQL/BigQuery (target DB, read-only access), Pinecone (schema embeddings), Redis (query cache), audit log DB
- **Security**: Read-only query executor, RBAC per user role, access control layer
- **Monitoring**: Prometheus (query metrics), ELK (audit logs), per-query cost tracking
