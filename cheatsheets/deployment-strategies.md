# Deployment Strategies Quick Reference

---

## Strategy Decision Matrix

| Strategy | Rollout Speed | Risk | Rollback | Traffic Split | Best For |
|----------|--------------|------|----------|---------------|---------|
| Blue-Green | Instant cutover | Medium (instant flip) | Instant (flip back) | 0% or 100% | Zero-downtime deploys; database schema changes |
| Canary | Gradual (hours/days) | Low (small blast radius) | Fast (route 100% to old) | 1% → 5% → 25% → 100% | High-stakes features; unknown production risk |
| Shadow / Dark Launch | Immediate (parallel) | None (no production effect) | N/A (shadow has no user impact) | 100% to old + shadow copy | Validating new model quality before any exposure |
| Rolling Update | Gradual (instance by instance) | Low-Medium | Slow (must roll back each instance) | Proportional to live instances | Stateless services; Kubernetes default strategy |
| Feature Flags | Instant per-segment | Low (granular control) | Instant (flip flag) | Any % or user-segment | Progressive user rollout; A/B experiments |
| A/B Test Deployment | Gradual | Low-Medium | Fast | Even split (50/50 or custom) | Measure business metric impact with statistical rigor |
| Multi-Armed Bandit | Dynamic / self-adjusting | Low | Automatic (poor arms dropped) | Dynamic based on reward | Real-time optimization; ad ranking; recommenders |

---

## Strategy Details

### Blue-Green Deployment
```
Production (Blue) -> v1 model serving all traffic
Staging (Green)   -> v2 model deployed and tested

When ready:
  Load Balancer -> flip from Blue to Green (instant)
  Old Blue kept warm for rollback

Rollback: flip Load Balancer back to Blue
```
**ML Considerations:**
- Both environments need identical feature stores
- Model warm-up (JIT compilation, TensorRT build) must complete BEFORE flip
- KV caches and embedding caches are NOT shared between Blue and Green; cold start on flip

### Canary Deployment
```
Traffic distribution:
  t=0h:   1% -> v2 new model; 99% -> v1
  t=4h:   5% -> v2; monitor metrics
  t=12h: 25% -> v2; monitor metrics
  t=24h: 50% -> v2; monitor metrics
  t=48h: 100% -> v2; decommission v1

Auto-promote criteria: latency p99, error rate, business KPI all within bounds
Auto-rollback trigger: error rate > threshold OR metric regression > X%
```
**ML Considerations:**
- Requires matching user cohorts (don't mix users between models mid-session)
- Log both model versions' predictions for offline comparison
- Monitor **prediction distribution shift**, not just latency/errors

### Shadow / Dark Launch
```
Incoming Request
     |
     +---> Production Model (v1) ---> User Response (only this one shown)
     |
     +---> Shadow Model (v2) ---> /dev/null (logs captured, never served)

Background: compare v1 vs v2 outputs on same inputs without user exposure
```
**ML Considerations:**
- Essential before deploying models with unknown production latency
- Captures real production traffic distribution (very different from test data)
- Reveals training-serving skew on actual production features
- Shadow traffic can warm up model's caches before going live

### Rolling Update
```
Instance pool: [v1, v1, v1, v1, v1, v1]

Step 1: [v2, v1, v1, v1, v1, v1]  -> health check
Step 2: [v2, v2, v1, v1, v1, v1]  -> health check
...
Done:   [v2, v2, v2, v2, v2, v2]

maxSurge (Kubernetes): how many extra pods allowed during update
maxUnavailable: how many pods can be down at once
```
**ML Considerations:**
- Stateful models (with KV cache or session state) need sticky routing during rollout
- Mixed fleet (v1 + v2) can cause inconsistent responses if user hits different servers

### Feature Flags
```python
# Example: gradual feature exposure by user segment
if feature_flag_enabled("new_ranker_v2", user_id=user.id, rollout_pct=10):
    score = new_ranker_v2.predict(features)
else:
    score = baseline_ranker.predict(features)
```
Tools: LaunchDarkly, Split.io, Unleash (open source), GrowthBook

**ML Considerations:**
- Allows per-user, per-region, per-segment control
- Enables instant kill switch without code deployment
- Flag state must be consistent within a user session

### A/B Test Deployment
```
Requirement: Statistical power analysis BEFORE launch
  n = z_{alpha/2}² * 2 * sigma² / delta²
  (delta = minimum detectable effect; sigma = metric std dev)

Assignment: hash(user_id + experiment_id) % 100 < 50 -> group A
Analysis: two-sample t-test or Mann-Whitney for non-normal metrics
Guardrails: automatic stop if harm metric (latency, error rate) degrades
```
**ML Considerations:**
- Avoid novelty effect: run for at least 1-2 weeks (weekly patterns matter)
- Use CUPED (variance reduction) for noisy business metrics
- Separate the A/B assignment from model deployment (don't re-hash mid-experiment)

---

## ML-Specific Deployment Considerations

### Model Warm-Up Time
| Scenario | Warm-Up Source | Mitigation |
|----------|---------------|------------|
| PyTorch JIT compilation | First batch per model | Run warmup requests before traffic flip |
| TensorRT engine build | First build on GPU | Pre-build and cache engine; or use persistent daemon |
| KV cache (LLM serving) | Cold at new instance start | Sticky routing; pre-fill cache with common system prompts |
| Java-based feature pipelines | JVM cold start (2-10s) | Keep instances warm; avoid scale-to-zero for latency-sensitive paths |

### Feature Store Refresh Lag
- Online feature stores (Redis, DynamoDB) refresh in near-real-time (<100ms)
- Offline feature stores (S3, BigQuery) may be 1-24h stale
- Mismatch between training features (batch) and serving features (online) = **training-serving skew**
- Validate: log a sample of production feature vectors; compare distribution to training set

### Model Versioning Strategies
| Strategy | Pros | Cons |
|----------|------|------|
| Date-based (v20240315) | Traceability; automatic ordering | Hard to know what changed |
| Semantic (v2.1.3) | Clear major/minor/patch | Requires discipline; subjective |
| Hash-based (model_abc1234) | Immutable, auto-generated | Not human-readable |
| Artifact registry (MLflow, W&B) | Tracks lineage, metrics, artifacts | Setup overhead |

**Best practice:** Register every model with: training run ID, dataset version, eval metrics, hyperparameters. Never deploy an unregistered model to production.

---

## Rollback Decision Matrix

| Condition | Action | Timeframe |
|-----------|--------|-----------|
| Error rate > 2x baseline | Immediate rollback | Within 15 minutes |
| P99 latency > 3x baseline | Immediate rollback or traffic reduction | Within 15 minutes |
| Business KPI down > 2% in 1h | Canary halt, do not promote | Within 1 hour |
| Prediction distribution drift | Investigate; possibly halt rollout | Within 4 hours |
| Silent model quality degradation | Rollback + investigate root cause | Within 24 hours (requires monitoring) |
