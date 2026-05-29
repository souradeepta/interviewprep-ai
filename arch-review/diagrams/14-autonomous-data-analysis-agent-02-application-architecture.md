## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        UploadAPI["Upload API\n(REST/multipart)"]
        QueryAPI["Query API\n(REST)"]
        ReportAPI["Report API\n(REST)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        AnalysisPlanner["Analysis Planner\n(LLM-driven DAG)"]
        TaskScheduler["Task Scheduler\n(Celery)"]
        ResultAggregator["Result Aggregator\n(Merge insights)"]
    end

    subgraph SchemaServices["Schema Services"]
        SchemaDetector["Schema Detector\n(Type inference)"]
        QualityChecker["Quality Checker\n(Null, dup, type)"]
        ProfileBuilder["Profile Builder\n(Stats summary)"]
    end

    subgraph ExecutionServices["Execution Services"]
        SQLExecutor["SQL Executor\n(DuckDB/Presto)"]
        PythonExecutor["Python Executor\n(Sandbox)"]
        StatEngine["Stat Engine\n(scipy, pandas)"]
    end

    subgraph InsightServices["Insight Services"]
        InsightGenerator["Insight Generator\n(LLM)"]
        ChartBuilder["Chart Builder\n(Plotly/Matplotlib)"]
        ReportWriter["Report Writer\n(Markdown)"]
    end

    subgraph DataServices["Data Services"]
        DataLakeClient["Data Lake Client\n(S3)"]
        MetaStoreClient["Meta Store Client\n(PostgreSQL ORM)"]
        CacheClient["Cache Client\n(Redis)"]
    end

    UploadAPI --> SchemaDetector
    QueryAPI --> AnalysisPlanner
    SchemaDetector --> QualityChecker
    QualityChecker --> ProfileBuilder
    ProfileBuilder --> AnalysisPlanner
    AnalysisPlanner --> TaskScheduler
    TaskScheduler --> SQLExecutor
    TaskScheduler --> PythonExecutor
    TaskScheduler --> StatEngine
    SQLExecutor --> ResultAggregator
    PythonExecutor --> ResultAggregator
    StatEngine --> ResultAggregator
    ResultAggregator --> InsightGenerator
    InsightGenerator --> ChartBuilder
    InsightGenerator --> ReportWriter
    ChartBuilder --> DataLakeClient
    ReportWriter --> MetaStoreClient
    InsightGenerator --> CacheClient
    ReportAPI --> MetaStoreClient
```

**Layer Breakdown:**
- **Presentation**: Upload, query, and report APIs
- **Orchestration**: LLM-driven analysis planning, task scheduling, result aggregation
- **Schema Services**: Type inference, data quality checks, statistical profiling
- **Execution Services**: SQL (DuckDB), sandboxed Python, statistical computation
- **Insight Services**: LLM insight generation, chart building, report writing
- **Data Services**: Data lake, metadata store, result cache
