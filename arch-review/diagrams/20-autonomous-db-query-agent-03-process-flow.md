## Process Flow (Natural Language to Query Result)

```mermaid
sequenceDiagram
    participant User as Business User
    participant API as REST API
    participant Parser as Intent Parser
    participant Schema as Schema Mapper
    participant SQLGen as SQL Generator
    participant Validator as Query Validator
    participant DB as Target Database
    participant Explainer as Explain Generator

    User->>API: "How many active users signed up last week?"
    API->>Parser: Parse natural language intent
    Parser->>Parser: Extract entities (users, active, signup, last week)
    alt Low confidence or ambiguous
        Parser-->>User: "Did you mean (a) logged in, or (b) newly registered?"
        User->>Parser: Clarify intent
    end
    Parser->>Schema: Map intent to schema tables
    Schema->>Schema: Vector search for relevant tables
    Schema-->>Parser: users table, events table
    Parser->>SQLGen: Intent + schema context
    SQLGen->>SQLGen: Generate candidate SQL (LLM + few-shot)
    SQLGen->>Validator: Validate query plan and cost
    Validator->>Validator: Check: cost estimate, access control, row count
    alt Query cost above $10
        Validator-->>User: "This query costs $12. Approve?"
        User->>Validator: Approve
    end
    Validator->>DB: Execute query (read-only)
    DB-->>Validator: Result set (N rows)
    Validator->>Validator: Sanity check: row count, data types
    alt Anomalous result
        Validator-->>User: "Got 0 rows - schema may have changed"
    end
    Validator->>Explainer: Results + query context
    Explainer->>Explainer: LLM generates English explanation
    Explainer-->>User: Result table + "Got 12,543 active users..."
```

**Key Decision Points:**
1. **Ambiguity Detection**: Clarification question if confidence below 0.85
2. **Schema Retrieval**: Vector search limits LLM context to top-5 relevant tables
3. **Cost Gate**: Queries estimated above $10 require explicit user approval
4. **Access Control**: RBAC blocks restricted tables before execution
5. **Sanity Check**: Row count and type validation before returning results

**Error Paths:**
- No schema match: ask user to rephrase or show available data domains
- Query execution timeout: suggest smaller time window or aggregation
- Access denied: show which role is required, suggest contacting data team

**Optimization Points:**
- Cache results for identical queries (Redis, 4-hour TTL)
- Pre-embed schema summaries for fast vector retrieval
- Bookmark frequent queries as saved reports for one-click re-run
