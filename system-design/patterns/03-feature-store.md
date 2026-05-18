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

## Interview Q&A

**Q: When should you implement a feature store vs. computing features inline?**
A: Implement a feature store when: multiple models share features (avoid duplicate computation), features require expensive computation (aggregations over large datasets), online/offline feature consistency is critical (training-serving skew prevention), or you need point-in-time correct features for retraining. Compute inline when: you have one model, features are simple and cheap to compute, or you're still in early ML development. The overhead of a feature store (infrastructure, maintenance) is only justified when it solves real problems you're experiencing.

**Q: What is training-serving skew and how does a feature store prevent it?**
A: Training-serving skew: features are computed differently at training time (offline, batch) vs. serving time (online, real-time), causing the model to receive different distributions than it was trained on. Classic example: training uses the median of a column but serving uses the mean. A feature store prevents this by ensuring the exact same feature computation logic is used for both training and serving—the feature definition is single-source-of-truth.

**Q: How do you implement point-in-time correct features for model retraining?**
A: Point-in-time correctness means: when generating training data for a label that occurred at time T, use only feature values that were available at time T (no future leakage). Feature stores implement this by storing feature values with timestamps and providing time-travel queries: "what was the value of feature X for entity Y at time T?" This requires storing the full history of feature values, not just current values—a significant storage cost that justifies the feature store.

**Q: What are the latency requirements for online feature serving and how do you meet them?**
A: Real-time inference: online features must be served in <10ms (leaving budget for model inference). Requirements: in-memory store (Redis, Cassandra) for fast reads, pre-computed feature values (not computed on-the-fly at serving), co-location of feature store with inference service (avoid network latency), and bulk fetch (retrieve all features for an entity in one call, not one call per feature). Monitor feature serving latency as a primary SLA—slow feature serving can silently cause SLA violations.

**Q: How do you manage feature versioning and backward compatibility?**
A: Treat feature definitions as versioned contracts. When a feature computation changes (new data source, modified aggregation window): create a new feature version, maintain old version while downstream models migrate, use a shadow-mode period where both versions are computed and compared. Never update a feature definition in-place if it's used by production models—the change will cause training-serving skew for those models. A feature store should provide version history and deprecation workflows.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Why use a feature store?" | Reusable features, avoid skew, scale to low-latency serving, governance. |
| "Offline vs online?" | Offline: historical computation. Online: real-time serving. Materialization syncs them. |

## Related Topics
- [Data Pipelines](02-data-pipelines.md) — [Online vs Batch Inference](07-online-vs-batch-inference.md)

## Resources
- [Feast: A Feature Store for ML](https://feast.dev)
- [Real-time ML: Challenges and Solutions (Tecton)](https://www.tecton.ai/blog/)
