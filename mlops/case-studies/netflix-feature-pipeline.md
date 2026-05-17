# Case Study: Netflix Feature Pipeline

## Interview Scenario

**Setup:** You're building a feature store for Netflix's recommendation system.

**Constraints:**
- 100M+ users worldwide
- 100+ recommendation models (ranking, matching, diversity, etc.)
- 1000s of features needed across models
- <100ms serving latency (recommendation response time)
- Training uses features from 6+ months ago (data availability)

**Problem:**
- Before feature store: each team builds features independently
  - Duplicated feature engineering code (same feature implemented 5 different ways)
  - Training-serving skew (models train on v1 features, serve on v2)
  - 40% of ML engineering time spent on feature engineering
  - Inconsistent definitions (different teams, different logic)

**Goal:**
- Centralized feature store eliminates duplication
- Consistent features across all models
- Faster model iteration (reuse existing features)
- Training-serving consistency (versioning)

---

## Strong Answer Walkthrough

### 1. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│           Offline Path (Training)                     │
│──────────────────────────────────────────────────────│
│ Data Sources: Kafka, Data Warehouse, Event Logs       │
│     ↓                                                  │
│ Daily Batch Jobs (Spark):                            │
│  - User embeddings (PCA on viewing history)          │
│  - Content embeddings (genre, metadata)              │
│  - User-content interactions (co-watches)            │
│  - Genre preferences (weighted scores)               │
│     ↓                                                  │
│ Feature Registry (Versioning):                       │
│  - Feature name, owner, version (v1, v2, ...)       │
│  - Freshness SLA (daily, weekly, ...)               │
│  - Schema and validation rules                       │
│     ↓                                                  │
│ Offline Storage (Data Warehouse, S3):                │
│  - Partitioned by date, user_id                      │
│  - Parquet format (10x compression)                  │
│  - Immutable (no overwrites)                         │
│     ↓                                                  │
│ Training Pipeline:                                    │
│  - Fetch features with explicit version (v1)         │
│  - Train model, log features used                    │
│  - Enable reproducibility: same data, same model     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│          Online Path (Real-Time Serving)              │
│──────────────────────────────────────────────────────│
│ Real-Time Data: Current User Session, Trending       │
│     ↓                                                  │
│ Feature Computation (Real-Time):                     │
│  - What user is watching now?                        │
│  - What titles are trending (last hour)?             │
│  - User session context                              │
│     ↓                                                  │
│ Online Cache (Redis, DynamoDB):                      │
│  - <5ms lookup for hot features                      │
│  - TTL: 1 hour for session context                   │
│  - Fallback to default if unavailable                │
│     ↓                                                  │
│ Serving (Recommendation API):                        │
│  - Fetch batch features (50ms) + real-time (<5ms)   │
│  - Explicit version: serve on v1 (matched training) │
│  - Score with model, return ranking                  │
└──────────────────────────────────────────────────────┘
```

### 2. Feature Categories

**Batch Features (Computed Daily):**
```
User Features:
- user_id, account_age, preferred_genres, watch_history_embedding
- Genre preferences (weighted by watch time)
- User segment (power user, casual, inactive)

Content Features:
- title_id, genre, release_year, rating, language
- Content embeddings (learned from metadata)
- Popularity (watch rate, retention)

Interaction Features:
- user_title co-watch count (user watched similar content)
- Collaborative similarity (users like me watched...)
```

**Real-Time Features (Streamed):**
```
Session Context:
- user_current_watching (title_id, minutes watched)
- session_duration (minutes elapsed)
- device (mobile vs TV)

Trending:
- trending_titles_global (hourly refresh)
- trending_titles_user_region (hourly refresh)
- new_releases_today
```

### 3. Feature Registry & Versioning

**Example Feature Definition:**

```yaml
Feature: user_embedding
Version: v2  # v1 was 512-dim, v2 is 1024-dim
Owner: recommendations_team
Description: User embedding from PCA on viewing history
Type: batch
Freshness: daily (computed at 2am UTC)
Schema: array[float], shape=(1024,), range=[-1, 1]
Dependencies:
  - user_events (30 days)
  - content_metadata
CreatedAt: 2026-04-15
Status: active
Consumers: [ranking_model_v5, matching_model_v3]
```

**Version Control:**
```
Model trained on: user_embedding v1
Serving uses: user_embedding v2

Problem: different versions → training-serving skew

Solution:
1. Training explicitly requests v1
2. Serving explicitly requests v1 (matched to training)
3. A/B test: control (v1), treatment (v2)
4. Only after treatment wins, roll out v2 everywhere
```

### 4. Governance & Discovery

**Feature Catalog (SQL-like)**
```sql
SELECT * FROM feature_registry
WHERE owner = 'recommendations_team'
  AND freshness_sla < 'daily'
  AND status = 'active'
ORDER BY consumers DESC
```

Result:
```
user_embedding (v2)       → 5 consumers (ranking, matching, diversity, ...)
genre_preferences (v1)    → 8 consumers
watch_history_agg (v3)    → 3 consumers
```

**Monitoring Dashboards:**
- % of features fresh (updated within SLA)
- Feature usage: which features used by which models
- Feature quality: nulls, distribution shifts
- Serving latency: batch fetch + real-time compute

### 5. Key Design Decisions

| Decision | Why | Trade-off |
|----------|-----|-----------|
| Batch + Real-time | Different latency requirements | More complex |
| Daily batch jobs | Enough freshness for user features | 24-hour staleness |
| Real-time for session context | Need immediate signals | Can't use offline |
| Versioning all features | Prevent training-serving skew | Overhead to manage versions |
| Feature ownership | Accountability, discovery | Coordination needed |
| Online cache (Redis) | Sub-5ms lookups required | Cache invalidation, memory |
| Immutable storage | Reproducibility, no overwrites | More storage initially |

### 6. Typical Flow: Training a New Model

```
Step 1: Feature discovery
  Engineer searches catalog: "genre preferences"
  Finds: genre_preferences v3 (5 consumers already using)
  
Step 2: Request features
  Training job: 
    - Fetch user_embedding v2
    - Fetch genre_preferences v3
    - Fetch watch_history_agg v3
    - Explicit version in request: avoid skew
    
Step 3: Train
  Model version: ranking_model_v12
  Features used: [user_embedding v2, genre_preferences v3, ...]
  Accuracy: 92%
  
Step 4: Deploy
  Serving explicitly requests same features:
    - user_embedding v2 ← matched to training
    - genre_preferences v3 ← matched to training
  Serving latency: <100ms ✓
  
Step 5: Monitor
  Track: feature versions in serving
  Alert: if serving uses v2 but model trained on v1 (skew detected!)
```

### 7. Handling Challenges

**Challenge 1: Feature Explosion (10K+ features)**

Solution:
- Feature families: group related features (user_*, content_*, interaction_*)
- Discovery: search by owner, tag, freshness
- Deprecation: mark unused features, retire after 6 months
- Cost tracking: which features cost most to compute?

**Challenge 2: Feature Freshness**

Solution:
- Define SLA per feature (daily, hourly, real-time)
- Monitor: is feature updated on time?
- Alert: if stale (not updated within SLA)
- Fallback: use cached version (trade freshness for availability)

**Challenge 3: Training-Serving Skew**

Solution:
- Version everything
- Training explicitly requests version
- Serving explicitly requests same version
- A/B test: before rolling out new version
- Validation: test that serving version matches training

**Challenge 4: Data Quality**

Solution:
- Schema validation (type checks)
- Completeness checks (% nulls)
- Statistical validation (distribution monitoring)
- Data contracts: "genre_preferences must have 99% completeness"

---

## Strong vs Weak Answers

### STRONG Answer
"I'd build a centralized feature store with batch and real-time paths. **Batch:** Daily Spark jobs compute user/content embeddings, stored in data warehouse. **Real-time:** Session context streamed to Redis cache. **Registry:** Centralized catalog with feature versioning, ownership, SLA. **Governance:** explicit versioning prevents training-serving skew—training explicitly requests features v1, serving requests same v1. **Monitoring:** track freshness, usage, quality. **Benefits:** 60% reduction in feature engineering code, consistent features across 100+ models, faster iteration. Storage: compress to 10GB (100M users × 100 bytes each)."

**Why this is strong:**
- ✓ Addresses all constraints (100M users, 1000+ features, <100ms)
- ✓ Separates batch (daily, cheap) from real-time (fast, required)
- ✓ Explains versioning strategy explicitly
- ✓ Discusses governance, monitoring, challenges
- ✓ Provides concrete numbers (60% reduction, 10GB storage)
- ✓ Shows understanding of operational complexity

### WEAK Answer
"I'd use Feast to build a feature store."

**Why this is weak:**
- ✗ Names tool without explaining architecture
- ✗ Doesn't address batch vs real-time
- ✗ No discussion of versioning strategy (training-serving skew)
- ✗ No mention of governance or scaling to 1000+ features
- ✗ No latency analysis (critical constraint: <100ms)
- ✗ Doesn't explain how to prevent duplication

---

## Follow-Up Questions

**Q: How do you prevent training-serving skew with 1000+ features?**

A: Enforce versioning at every step. (1) Feature registry tracks all versions. (2) Training: specify feature version in request (v1). (3) Serving: specify same version (v1). (4) Validation: test that serving uses same version. (5) A/B testing: compare control (v1) vs treatment (v2) before rollout. (6) Monitoring: alert if mismatch detected (serving v2, model trained v1).

**Q: Model accuracy dropped from 93% to 89%. Feature store is suspected. How do you debug?**

A: Investigate: (1) Did feature definition change? Check if training uses v1 but serving uses v2. (2) Are features stale? Check freshness SLA (updated within 24h?). (3) Did data distribution shift? Compare feature distributions (median spend: was $50, now $75). (4) Did upstream data break? Check dependencies (data pipeline failures). Use feature store's versioning and lineage: compare v93 (93% accuracy) vs v89 model (89% accuracy). What features changed between versions?

**Q: You have 10K features. How do you manage them?**

A: Governance: (1) Feature families (group by domain). (2) Discovery (search by owner, tag). (3) Monitoring (track usage—unused features retired after 6 months). (4) Cost tracking (which features cost most?). (5) Ownership (who owns this feature?). (6) SLA (what's freshness requirement?). (7) Quality (validation rules). Central registry prevents duplication and ensures consistency.

**Q: Real-time serving latency degraded from 50ms to 500ms. How do you fix it?**

A: Profile each stage: (1) Batch feature fetch from warehouse: slow query? Add caching layer. (2) Real-time feature computation: slow Flink job? Reduce computation, increase resources. (3) Online cache: slow Redis? Add instances, increase throughput. (4) Feature joins: too many features? Reduce cardinality. Once bottleneck identified, optimize that path. Monitor: track latency per feature family.

**Q: Cost is too high. How would you optimize?**

A: (1) Retire unused features (monitor usage, archive after 6 months). (2) Compress features (store as int16 instead of float32). (3) Reduce batch freshness (compute weekly instead of daily for some features). (4) Sample data (validate on 10% instead of 100%). (5) Tiered cache (hot features in Redis, cold in S3). (6) Feature selection: which features have highest importance? Remove low-impact ones.

---

## Key Takeaways

**Concepts Demonstrated:**
- ✓ Batch vs streaming architecture
- ✓ Feature versioning and training-serving consistency
- ✓ Governance and discoverability at scale
- ✓ Real-time latency budgeting
- ✓ Monitoring and alerting
- ✓ Cost optimization strategies

**Why This Design Wins:**
- Directly addresses the duplication problem
- Scales to 100M+ users with 1000+ features
- Prevents training-serving skew through versioning
- Shows understanding of operational challenges
- Includes governance, monitoring, cost management

**Common Mistakes to Avoid:**
- ❌ Not addressing versioning (training-serving skew)
- ❌ Using same features for batch and real-time (different latency needs)
- ❌ Not mentioning governance (can't discover, reuse features)
- ❌ Ignoring latency budget (overshooting 100ms requirement)
- ❌ Not discussing monitoring (how do you know if it's working?)
