## Process Flow (Dataset to Insight Report)

```mermaid
sequenceDiagram
    participant User as Analyst
    participant API as Upload API
    participant Schema as Schema Detector
    participant Planner as Analysis Planner
    participant SQLExec as SQL Executor
    participant PyExec as Python Executor
    participant Insight as Insight Generator
    participant Chart as Chart Builder
    participant Report as Report Writer

    User->>API: Upload dataset (CSV/Parquet/DB)
    API->>Schema: Detect schema and types
    Schema->>Schema: Check data quality (nulls, dups, types)
    Schema-->>User: Quality report with flags
    alt Quality below threshold
        Schema-->>User: Warning - data quality issues found
    end
    Schema->>Planner: Schema profile and quality score
    Planner->>Planner: LLM plans analysis steps (DAG)
    Planner->>SQLExec: Dispatch aggregation queries
    Planner->>PyExec: Dispatch distribution analysis
    SQLExec-->>Planner: Aggregated statistics
    PyExec-->>Planner: Distribution and outlier results
    Planner->>Insight: Aggregated results
    Insight->>Insight: LLM ranks top-10 insights
    Insight->>Chart: Generate chart specs
    Chart-->>Insight: Chart artifacts (Plotly JSON)
    Insight->>Report: Write markdown report
    Report-->>User: Analysis report with charts
    User->>Insight: Feedback on insight quality
    Insight->>Planner: Update analysis strategy
```

**Key Decision Points:**
1. **Quality Gate**: Data quality flags shown before deep analysis begins
2. **LLM Planning**: Planner determines which SQL and Python tasks to dispatch in parallel
3. **Insight Ranking**: LLM ranks insights by business relevance (top-10 surfaced)
4. **Chart Selection**: Three chart types generated per key insight, analyst picks preferred view
5. **Feedback Loop**: Analyst feedback refines insight ranking model over time

**Error Paths:**
- File too large: stream in chunks, partition analysis
- SQL timeout: fall back to sampled analysis (10% sample with note)
- LLM insight failure: return raw statistics without narrative

**Optimization Points:**
- Cache schema profiles for repeated uploads of same dataset
- Run SQL and Python executors in parallel (DAG-scheduled)
- Batch chart generation for multi-column reports
