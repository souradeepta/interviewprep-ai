# Data Pipelines

## TL;DR
ETL: Extract data, Transform (clean, join, aggregate), Load to warehouse/lake. Batch (daily) or streaming (continuous). Foundation of ML systems.

## Core Intuition
Raw data → useful features. Pipeline automates: extract from sources, transform, load.

## How It Works
```
Data sources (databases, APIs, logs)
  ↓
Extract (scheduled jobs, streaming)
  ↓
Transform (clean, join, aggregate)
  ↓
Load (to warehouse, feature store)
  ↓
Available for training, serving, analytics
```

**Batch:** nightly, daily (latency OK, cost-efficient)
**Streaming:** continuous, real-time (low latency, expensive)

## Key Properties / Trade-offs
- Latency: batch (hours), streaming (seconds)
- Cost: batch cheap, streaming expensive
- Complexity: batch simple, streaming complex

## Common Mistakes / Gotchas
- **No error handling:** bad data propagates → broken models
- **Slow transforms:** bottleneck. Optimize with distributed processing (Spark).
- **Data quality:** no validation → garbage data in
- **Lineage:** can't trace data origin → hard to debug

## Interview Quick-Reference
**Data pipeline?** Extract, transform, load. Batch or streaming. Automate, monitor, validate.

## Related Topics
- [Feature Store](feature-store.md) — stores transformed data
- [Data Governance](data-governance.md) — data quality

## Resources
- [Apache Airflow](https://airflow.apache.org/) — orchestration
- [Spark](https://spark.apache.org/) — distributed processing
