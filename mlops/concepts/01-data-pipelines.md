# Data Pipelines: Building Reliable ETL for ML

## Comprehensive Overview

Data pipelines form the backbone of ML systems, responsible for ingesting, transforming, and delivering data to models at scale. A production data pipeline must handle millions of events per second, recover from failures gracefully, and maintain data quality while transforming raw inputs into feature-ready datasets. The core challenge: ETL is not just engineering—it's critical infrastructure that determines both model training velocity and inference reliability. Unlike traditional software systems, data pipelines cannot simply drop events; they must preserve completeness while ensuring timeliness and correctness.

Data pipeline design reflects fundamental constraints: batch pipelines excel at historical analytics and model training (high throughput, days of latency acceptable) while streaming pipelines serve real-time features and online serving (millisecond latency, stateful computation). Most production systems need both. The decision between Airflow (DAG orchestration, task dependencies, scheduling), Kubeflow (Kubernetes-native ML workflows, distributed execution), and Luigi (simpler Python workflows) hinges on team expertise, existing infrastructure, and tolerance for complexity. Choosing wrong costs months in context-switching and operational overhead.

Production pipelines face four persistent challenges: data quality (garbage in, garbage out), latency (when should data be available?), cost (compute and storage at scale), and debuggability (why did a run fail at 3am?). Monitoring pipelines is harder than monitoring services—success looks like "data arrived on time," failure looks like "data arrived late" or "data arrived wrong," both requiring different responses. Modern teams implement data contracts (schema validation, completeness checks, freshness guarantees) alongside orchestration, treating data quality as equivalent to code quality.

The business impact is outsized. Netflix features trained on stale data degrade recommendation quality. Uber's surge pricing trained on hours-old traffic patterns misses real-time demand. Stripe's fraud detection trained on day-old transactions misses emerging fraud patterns. Data freshness, quality, and completeness are not engineering details—they directly determine model performance and business outcomes.

## How It Works

### Batch Pipelines

```
Source Systems (APIs, DBs)
    ↓
Ingestion (extract data)
    ↓
Transformation (clean, aggregate, enrich)
    ↓
Feature Store or Data Warehouse
    ↓
Model Training or Inference
```

**Schedule-based execution:** Run at fixed times (daily, hourly) via cron or orchestrator.  
**Latency:** Minutes to hours (acceptable for training, risky for serving).  
**Throughput:** High (can process millions of rows efficiently).  
**State management:** Stateless (idempotent operations).

### Streaming Pipelines

```
Event Stream (Kafka, Pub/Sub)
    ↓
Stateful Transformation (windowing, aggregation)
    ↓
Real-Time Feature Store
    ↓
Online Model Inference
```

**Event-driven execution:** Process data as it arrives.  
**Latency:** Milliseconds (necessary for real-time applications).  
**Throughput:** Medium (constrained by processing latency per event).  
**State management:** Stateful (maintains windows, aggregations).

### Hybrid Architecture (Batch + Streaming)

Most production systems use both:
- **Batch:** Model training (needs historical data), daily feature updates
- **Streaming:** Feature serving (needs real-time features), online inference

The integration point is typically the feature store, which maintains both batch-computed and streaming-computed features.

## Tool Comparisons

| Tool | Type | Strengths | Weaknesses | Best For |
|------|------|-----------|-----------|----------|
| **Apache Airflow** | Batch orchestration | DAG clarity, large community, great UI, many integrations, easy to debug | Stateless (bad for streaming), scheduling overhead, can be slow for large DAGs | Medium-complexity batch pipelines, Python teams, established infra |
| **Kubeflow Pipelines** | Kubernetes-native | Native Kubernetes, container-first, good for ML workflows, reproducibility | Steeper learning curve, smaller community, requires K8s | ML teams on Kubernetes, complex workflows |
| **Luigi** | Python-first orchestration | Simple, lightweight, dependency tracking, low overhead | Less powerful than Airflow, smaller ecosystem, limited integrations | Small teams, simple workflows, quick prototyping |
| **Apache Flink** | Stream processing | Low-latency processing, stateful computation, complex windowing, fault tolerance | Operational complexity, JVM overhead, steeper learning curve | Real-time feature computation, complex streaming logic |
| **Spark Streaming** | Batch-like streaming | SQL integration, Hadoop ecosystem, structured APIs, familiar to data engineers | Micro-batch model (not true streaming), higher latency than Flink | Medium-latency streaming, SQL-heavy transformations |
| **dbt** | Transformation focus | SQL-first, lineage tracking, testing, documentation, simple to learn | Limited to SQL, not for orchestration, newer tool | Analytical workflows, data warehouse transformations |

**Decision Framework:**
- **Batch pipelines:** Airflow (industry standard, safe bet) or Luigi (simplicity)
- **Streaming:** Flink (low-latency) or Spark (SQL-heavy)
- **Hybrid:** Airflow (batch) + Flink/Spark (streaming)
- **Starting out:** Luigi (simplicity), then migrate to Airflow as complexity grows
- **K8s shops:** Kubeflow Pipelines

## Interview Q&A

**Q: Design a data pipeline for a recommendation system processing 1M events/second. Walk me through your approach.**

A: I'd separate into batch and streaming. Batch (daily): ingest historical user behavior, compute user embeddings and content similarities, update training dataset. Streaming (real-time): ingest live user events, compute session features (current browsing context), cache in Redis. At inference time: combine batch features (user embeddings) with streaming features (session context) to score recommendations. Latency matters—fresh features for better recommendations.

**Q: Your batch pipeline failed at 2am, missing the 3am model training deadline. How do you debug and prevent future occurrences?**

A: Debugging: (1) Check orchestrator logs—did the job start? (2) If started, check task-level logs: data ingestion (source available?), transformation (errors?), validation (data quality?). (3) Check resources—did it timeout due to memory/CPU? (4) Check data—did it arrive but fail validation? Prevention: (1) Implement data freshness monitoring—alert if data hasn't arrived by 2:30am. (2) Automated retries for transient failures (network timeouts). (3) Fallback to cached data if fresh data unavailable. (4) Reduce batch window—if 24h window too slow, use 12h instead.

**Q: Your pipeline ingests data from 3 sources with different schemas. How do you handle schema evolution without breaking downstream systems?**

A: Schemas evolve: new fields appear, types change, fields disappear. Strategy: (1) Validation tier—validate incoming data against expected schema using tools like Great Expectations. Log schema violations, don't silently drop. (2) Transformation tier—use schema-aware libraries (Pydantic, Spark StructType) that enforce types and catch mismatches. (3) Versioning—version schemas in data contracts so consumers know what to expect. (4) Breaking changes—deprecate fields before removing (give consumers 2 weeks notice), add new fields as optional. (5) Monitoring—alert on schema violations, track schema change frequency.

**Q: Batch pipeline costs $10K/month. How do you optimize?**

A: Identify cost drivers: compute (60%), storage (30%), networking (10%). Optimize each: (1) Compute—filter unnecessary data early (reduce volume processed), better algorithms (reduce execution time), parallelize better (reduce wall-clock time), schedule during off-peak (use cheaper compute). (2) Storage—compress data (Parquet > CSV, 10x reduction), delete old snapshots, archive rarely-accessed data to cheaper storage. (3) Networking—colocate compute near data, reduce data movement. Measurement: track cost/GB processed, aim for 20% reduction without sacrificing quality.

**Q: How do you ensure pipeline reliability and recovery from failures?**

A: Reliability requires: (1) Idempotency—re-run same job 10x, get same result (no duplicates, no data loss). (2) Checkpointing—restart from failure point, not beginning. (3) Retries—transient failures (network timeouts) auto-retry 3x with backoff; permanent failures (bad code) fail fast. (4) Monitoring—alert on lateness, data quality issues, resource failures. (5) Fallback—if today's data unavailable, use yesterday's cached data. (6) Testing—unit test transformations, integration test end-to-end with sample data, chaos test failure scenarios.

**Q: Compare batch and streaming pipelines. When would you choose each?**

A: Batch: high throughput, acceptable latency (hours), stateless, cost-efficient. Use for model training (needs historical data), daily reports. Streaming: low latency (milliseconds), event-driven, stateful, higher cost. Use for real-time serving (recommendations, fraud detection), real-time alerts. Most systems need both—batch for training, streaming for serving. Integration point: feature store holds both batch and streaming features.

**Q: Your data pipeline is running slower. Debug the bottleneck.**

A: Profile each stage: which step is slow? (1) I/O bottleneck—data ingestion slow? Parallelize connections, batch fetches, increase concurrency. (2) Compute bottleneck—transformation slow? Optimize algorithm, increase compute resources, parallelize processing. (3) Storage bottleneck—writing output slow? Optimize serialization format (Parquet > CSV), increase parallelism, use better storage (SSD > HDD). (4) Memory bottleneck—out of memory? Stream data instead of loading all, increase memory, optimize data structures. Use observability: log each stage's duration, find the slowest stage, optimize that first (80/20 rule).

## Best Practices

1. **Idempotency First:** Design pipelines so re-running same job produces same result. This enables retry-on-failure and manual re-runs without data corruption.

2. **Data Quality as Code:** Use schema validation, row-level checks, and statistical validation (Great Expectations). Treat data quality like code quality—test it, alert on failures.

3. **Monitor Freshness:** Alert if data hasn't arrived by expected time. Freshness is often more critical than perfection (late good data worse than missing data).

4. **Separate Orchestration from Transformation:** Airflow orchestrates; Spark/Pandas do transformation. Don't embed complex logic in DAG definitions.

5. **Version Schemas & Transformations:** Enable rolling back to previous data or transformation logic. Document breaking changes.

6. **Cost Monitoring:** Track cost per pipeline, per stage. Optimize the most expensive pipelines first.

7. **Implement Data Contracts:** Formalize expectations (schema, freshness, completeness). Producers commit to delivering data meeting contract; consumers depend on contract.

8. **Observability & Alerts:** Log each stage's start time, duration, row count, errors. Alert on lateness, data quality failures, resource issues.

## Common Pitfalls

1. **Ignoring Skew:** Data isn't uniform. Power-law distribution means million-user dataset has hot keys. Optimize for hot keys, not average case.

2. **Tight Coupling Between Stages:** Tight coupling causes cascading failures. Build buffers, use intermediate caching, enable independent stage failure.

3. **Over-Engineering:** Don't build for 1000x scale day one. Build for current scale, optimize when measurement shows bottlenecks.

4. **Silent Failures:** Data arrives but is corrupted. Validate early, fail loud, alert on quality issues.

5. **Manual Operations:** Don't require manual intervention (restarting jobs, fixing data). Automate retries, alerts, and rollbacks.

6. **Ignoring Latency:** Batch pipelines creep from daily to weekly. Latency compounds—model quality suffers. Monitor and optimize freshness.

7. **No Observability:** Don't realize pipelines failed until models degrade. Implement comprehensive monitoring from day one.

## Real-World Examples

### Netflix: Feature Pipeline at 1B Events/Day

Netflix ingests 1B+ events/day across movies, shows, user interactions. Their batch pipeline:
1. Daily job ingests yesterday's events (100GB+ data)
2. Aggregates into viewing history (user × content × watch time)
3. Computes content-based features (genre, similar titles)
4. Streams features to Kafka
5. Recommendation models consume in real-time

**Cost optimization:** Pre-aggregate common queries, cache hot features, archive old data. Result: 50% cost reduction without quality loss.

### Uber: Real-Time Surge Pricing

Surge pricing requires real-time features: current demand, supply, travel times. Uber's pipeline:
1. Real-time events from driver/rider apps → Kafka
2. Stream processor computes features in <100ms (surge_ratio = demand / supply)
3. Real-time features served to pricing model
4. Decision returned in <100ms

**Latency is critical here:** 5-minute-old data produces wrong prices, harming both drivers and riders. Streaming is non-negotiable.

### Stripe: ML System for Fraud Detection

Stripe processes 1M+ transactions/day. Fraud pipeline:
1. Batch (daily): confirmed-fraud labels from past 5 days → feature engineering → model retraining
2. Streaming: real-time transactions → real-time features (transaction velocity, merchant risk)
3. Inference: combine batch and streaming features → fraud score in <100ms

**Key insight:** Batch for training (needs historical labels), streaming for serving (needs low latency).

## Sample Interview Questions

1. "Design a data pipeline for an e-commerce company with 100M daily transactions. The model needs fresh features hourly, but historical training data spans 5 years. Walk me through your approach."

2. "Your batch pipeline runs at 6am but sometimes finishes at 8am, breaking the 7am report deadline. How do you debug and fix this?"

3. "Design a real-time data pipeline for fraud detection that needs to flag suspicious transactions in <100ms. Cost is a constraint—you have $100K/month infrastructure budget."

## Interview Case Study

**Scenario:** You're hired at Stripe to improve their data pipeline for fraud detection.

**Current State:**
- Fraud model trained weekly on historical transactions
- Label delay: fraudulent transactions confirmed 5 days after fact
- Manual data ingestion (error-prone, slow)
- No monitoring (don't know if data pipeline broke until model quality drops)

**Problem:** Weekly training is stale (fraudsters evolve daily); manual processes are unreliable.

**Constraints:**
- Process 1M transactions/day
- Label delay: fraudulent transactions confirmed 5 days after fact
- Real-time features needed for <100ms fraud scoring
- Cost-sensitive (infrastructure budget $50K/month)

**Expected Solution:**
1. **Batch pipeline:** Daily (instead of weekly) ingestion of confirmed frauds → automated feature engineering → model retraining. Cost: optimize by sampling negative examples, caching intermediate features.
2. **Streaming pipeline:** Real-time transaction ingestion → compute velocity-based features (fast, cheap) → cache in Redis for serving.
3. **Integration:** Combine batch-trained model with real-time features for <100ms fraud scoring.
4. **Monitoring:** Alert if pipeline lateness increases, data quality degrades, or fraud detection accuracy drops.

**What Interviewers Listen For:**
- Understanding latency and freshness trade-offs
- Separating batch (training) from streaming (serving)
- Cost optimization without sacrificing quality
- Monitoring and iteration (feedback loops)
- Operational thinking (failures happen, how to recover?)

## Common Answer Patterns

**Strong Answer:**
"I'd build three components: (1) Batch pipeline—daily job on past 5 days of confirmed fraud labels. More responsive than weekly, but labels still delayed. Cost: optimize by sampling (most transactions are legitimate), early filtering. (2) Streaming—real-time transactions compute velocity-based features (fast), cached in Redis. (3) Monitoring—alert on lateness (if data not arrived by expected time), data quality (if validation fails), accuracy drop (if fraud detection rate falls below 95%). If accuracy drops, trigger retraining."

**Weak Answer:**
"I'd use Airflow for the batch pipeline. I'd use Kafka for streaming. I'd add monitoring." (No explanation of why these tools, no discussion of latency/cost trade-offs, no mention of failure recovery.)

---

## Related Concepts

- **Concept 02:** Feature Stores — Where pipeline outputs are stored and served
- **Concept 03:** Data Validation — Quality checks in pipeline
- **Concept 04:** Data Versioning — Versioning pipeline outputs
- **Concept 22:** Workflow Orchestration — Tools to run pipelines

## Resources

- Apache Airflow: https://airflow.apache.org/
- dbt: https://www.getdbt.com/
- Apache Flink: https://flink.apache.org/
- Great Expectations: https://greatexpectations.io/
