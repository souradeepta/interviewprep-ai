# Feature Store

## TL;DR
A centralized system for managing ML features: computing, storing, serving. Features are reused across many models (training and serving). A feature store avoids code duplication, ensures consistency, and scales feature serving to low-latency production systems.

## Core Intuition
Without a feature store: each team rebuilds the same features (user_age, user_purchase_history, etc.) in different code → inconsistencies → production bugs. A feature store is like a library: define once, reuse everywhere.

## How It Works

**Components:**
- **Offline store:** compute features on historical data (batch job), store in data warehouse
- **Online store:** low-latency key-value store (Redis, DynamoDB) for serving at prediction time
- **Materialization:** sync offline → online (fresh copy every hour, for example)
- **Feature registry:** catalog of all features (name, schema, lineage, owner)

**Typical flow:**
1. Engineer defines feature (e.g., "user_avg_purchase_30d")
2. Batch job runs nightly, computes on historical data
3. Values written to offline store (data warehouse)
4. Values synced to online store (Redis) for serving
5. Model at inference queries online store for user's feature values
6. Prediction returned in <50ms

## Key Properties / Trade-offs
- **Latency vs freshness:** hourly sync is fast; real-time sync adds latency
- **Cost vs coverage:** compute everything = high cost; selective computation = risk of missing important features
- **Governance:** standardized features prevent inconsistencies but slower feature iteration

## Common Mistakes / Gotchas
- **Training/serving skew:** different code for training vs serving → model degrades in production
- **Data leakage:** future information in features at training time
- **Stale features:** sync interval too long → serving with outdated data

## Examples (Concrete)
- **Tecton, Feast, Databricks Feature Store:** open/commercial feature platforms
- **Uber Michelangelo, Airbnb ML:** built internal feature stores

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Why use a feature store?" | Reusable features, avoid skew, scale to low-latency serving, governance. |
| "Offline vs online?" | Offline: historical computation. Online: real-time serving. Materialization syncs them. |

## Related Topics
- [Data Pipelines](data-pipelines.md) — [Online vs Batch Inference](online-vs-batch-inference.md)

## Resources
- [Feast: A Feature Store for ML](https://feast.dev)
- [Real-time ML: Challenges and Solutions (Tecton)](https://www.tecton.ai/blog/)
