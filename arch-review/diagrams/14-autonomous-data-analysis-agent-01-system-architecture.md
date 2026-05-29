## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Upload["Data Ingestion Layer"]
        FileUpload["File Upload\n(REST API)"]
        StreamIngest["Stream Ingest\n(Kafka)"]
        DBConnector["DB Connector\n(JDBC/ODBC)"]
    end

    subgraph Processing["Processing Cluster (Kubernetes)"]
        SchemaDetector["Schema Detector\nPod x 5"]
        Planner["Analysis Planner\n(LLM)"]
        SQLExecutor["SQL Executor\nPod x 8"]
        PythonExecutor["Python Executor\nPod x 8"]
        InsightGen["Insight Generator\n(LLM)"]
    end

    subgraph Visualization["Visualization Services"]
        ChartBuilder["Chart Builder\n(Plotly)"]
        ReportGen["Report Generator\n(Markdown)"]
    end

    subgraph DataLayer["Data Layer"]
        DataLake["Data Lake\n(S3)"]
        MetaStore["Metadata Store\n(PostgreSQL)"]
        Redis["Result Cache\n(Redis)"]
        VectorDB["Vector DB\n(Schema Embeddings)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus\n(Job Metrics)"]
        ELK["ELK Stack\n(Logs)"]
        CostTracker["Cost Tracker\n(Per-analysis)"]
    end

    FileUpload --> SchemaDetector
    StreamIngest --> SchemaDetector
    DBConnector --> SchemaDetector
    SchemaDetector --> Planner
    SchemaDetector --> VectorDB
    Planner --> SQLExecutor
    Planner --> PythonExecutor
    SQLExecutor --> InsightGen
    PythonExecutor --> InsightGen
    InsightGen --> ChartBuilder
    InsightGen --> ReportGen
    ChartBuilder --> DataLake
    ReportGen --> MetaStore
    InsightGen --> Redis
    Planner --> Prom
    SQLExecutor --> ELK
    InsightGen --> CostTracker
```

**Infrastructure Components:**
- **Compute**: Kubernetes cluster with autoscaling executor pods for SQL and Python workloads
- **Storage**: S3 (datasets, charts, reports), PostgreSQL (metadata, analysis history), Redis (result cache), VectorDB (schema embeddings)
- **Processing**: Schema-aware planners, SQL and Python executors, LLM insight generators
- **Monitoring**: Prometheus (job metrics), ELK (execution logs), per-analysis cost tracking
