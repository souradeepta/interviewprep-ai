# Model Versioning

## Detailed Description

Semantic versioning (MAJOR.MINOR.PATCH) for trained models. MAJOR: architecture change (XGBoost→GBDT). MINOR: feature set change (add feature). PATCH: hyperparameter tuning. Enables reproducibility, rollback, clear communication.

## Core Intuition

Versioning = clear semantics. v1.2.3 says: same architecture (1), new features (2), tweaked hyperparams (3). v1.3.0 says: new features added, probably need revalidation. Without versioning: confusing (which is better, 1.1 or 1.2?). With it: clear.

## How It Works

**Versioning Scheme:**
- MAJOR (e.g., 1.0.0 → 2.0.0): architecture change (XGBoost → neural net)
- MINOR (e.g., 2.0.0 → 2.1.0): feature set change (add/remove features)
- PATCH (e.g., 2.1.0 → 2.1.1): hyperparameter tuning, bugfix

**Metadata per version:**
- Code commit hash (git): enables exact code reproduction
- Data version (DVC): which training data version
- Hyperparameters: learning_rate, batch_size, epochs
- Metrics: accuracy, F1, AUC (comparison across versions)

## Key Properties / Trade-offs

| Aspect | Semantic | Calendar |
|--------|----------|----------|
| Example | 2.1.0 | 2024-01-15.3 |
| Readability | Clear meaning (major/minor/patch) | Timestamp-based |
| Rollback | "Go back to 1.9.0" | "Go back to yesterday" |
| When to use | Code-driven models | Daily retrained models |
| Clarity | Clear breaking changes | Less clear |

## Common Mistakes / Gotchas
- No versioning at all: models scattered, impossible to reproduce
- Calendar versioning without semantic: unclear if v2 is minor or major change
- Losing code: model v1.0.0 trained on deleted code branch → can't rebuild
- Data drift ignored: v1.0.0 trained on 2024-01 data, trying to use with 2024-03 data

## Best Practices

- **Git commit hash as anchor:** model_v1.0.0_commit_abc123def456. Reproducible.
- **Data version locked:** Pin with DVC or commit hash. Same data + code = same model.
- **Breaking changes → major:** architecture change increments major version.
- **Backward compatibility:** never tag the same version twice. Each model is immutable.
- **Deprecation policy:** versions < 6 months old marked deprecated, removed after 12 months.

## Detailed Trade-off Analysis

### Versioning Scheme Comparison

| Aspect | Semantic (MAJOR.MINOR.PATCH) | Calendar (YYYY-MM-DD.N) | Git Hash | Timestamp+Hash Hybrid |
|--------|-------------------------------|--------------------------|----------|----------------------|
| **Readability** | Clear meaning (1.5.0 = new features) | Timestamp-based | Opaque (abc123def) | Both meanings |
| **Rollback clarity** | "go to v1.9.0" obvious | "go to yesterday" | Specific but unclear | Both meanings |
| **Metadata tracking** | Manual (must track meaning) | Automatic (date = data version) | Automatic (git = code) | Automatic (both) |
| **Compatibility tracking** | Clear (major = breaking) | Unclear (same date, different models) | Unclear (which is newer?) | Clear (both dimensions) |
| **Automation** | Manual semantic decisions | Automatic daily | Automatic per commit | Automatic per commit |
| **When to use** | Code-driven models (weekly+) | Daily retrained models | CI/CD continuous deployment | Hybrid (recommended) |
| **Production example** | ML pipeline models | Time-series forecasting | ML service deployments | Enterprise ML systems |

### Real-World Cost Analysis

**Semantic Versioning (code-driven):**
- Used by: Scikit-learn, PyTorch models
- Example: v2.1.0 (8 months in production)
- Cost of managing: 5 min per version (minimal)
- Rollback cost if v2.1.0 fails: 5 min to identify v2.0.0, 2 min to deploy
- Metadata overhead: ~1KB per version

**Calendar Versioning (daily retrains):**
- Used by: Time-series forecasting (daily), recommendation systems (continuous)
- Example: 2024-01-15.3 (third model trained that day)
- Cost of managing: 2 min per model (automatic)
- Rollback cost if today's model fails: 2 min (use yesterday's)
- Metadata overhead: ~100 bytes per model (lighter)

**Git Hash (CI/CD driven):**
- Used by: Continuous deployment systems, canary releases
- Example: abc123def456 (git commit hash)
- Cost of managing: 1 min per deployment (automatic)
- Rollback cost: 1 min (revert to previous commit)
- Metadata overhead: ~40 bytes per model

### Decision Matrix by Use Case

| Use Case | Recommendation | Reasoning | Typical Cadence |
|----------|----------------|-----------|-----------------|
| **Long-lived models** (e.g., fraud detection) | Semantic | Clear breaking changes, infrequent updates | Every 2-4 weeks |
| **Frequently retrained** (e.g., demand forecasting) | Calendar + Hash | Daily retrains, need to rollback to yesterday | Every 24 hours |
| **Continuous deployment** (e.g., ML API service) | Git Hash + Calendar | Automatic deployment from commits | Per commit (multiple/day) |
| **Multi-version serving** (A/B testing) | Semantic | Need to run v1.9.0 and v2.0.0 simultaneously | Weekly |
| **Compliance/auditing** (e.g., lending model) | Semantic + Calendar + Hash | Must track: code, data, approval date | Monthly |
| **Backtest/reproducibility** (e.g., research) | Semantic + Git Hash | Must rebuild exact model from code | Varies |

---

## Production Failure Scenarios

### Scenario 1: Cannot Rollback (Model Code Deleted, Lost Git History)

**What breaks:** v2.0.0 crashes in production. Need to rollback to v1.9.0, but source code for v1.9.0 has been deleted. Can't rebuild model from scratch.

**Why it happens:**
- Git branch deleted after merging
- No git commit hash stored with model metadata
- Model registry only has binary file, no code version

**Detection:**
```
Metric: model_rollback_time_minutes
Alert: if (unable to locate source code for version) → CRITICAL
Check: SELECT code_commit FROM model_registry WHERE version = 'v1.9.0'
```

**Recovery (15-30 minutes):**
1. Check git log if commit still in history: `git log --all --grep="v1.9.0"`
2. If found: checkout commit, rebuild model (5-10 min)
3. If not found: restore from backup/git reflog (10-20 min)
4. Worst case: use v1.8.0 if v1.9.0 unreproducible

**Prevention:**
- Always store git commit hash with model: `v2.0.0_commit_abc123`
- Implement model archival: pin commit hash permanently
- Regular reproducibility tests: rebuild v1.9.0 monthly, verify identical

---

### Scenario 2: Version Incompatibility (Model Expects Features Not Available)

**What breaks:** v2.0.0 was trained with new feature `user_lifetime_value` (LTV). But in production, data pipeline only computes old features. Model crashes with "KeyError: user_lifetime_value".

**Why it happens:**
- Feature engineer added LTV to training pipeline
- Model trained with LTV (minor version bump: v2.0.0)
- Feature computation hasn't been updated yet
- Deployment didn't check for feature availability

**Detection:**
```
Metric: feature_missing_for_model
Alert: if (features used by model != features available) → CRITICAL
Check: Compare model.feature_names vs available_features at serving time
```

**Recovery (10-20 minutes):**
1. Identify missing feature: `model_error: KeyError: user_lifetime_value`
2. Check feature pipeline status: Is LTV being computed?
   - If no: rollback to v1.9.0 (doesn't need LTV)
   - If yes: check why not reaching serving layer
3. Fallback: Revert to v1.9.0 (immediate), keep v2.0.0 for later deployment
4. Plan migration: Update serving to compute LTV before using v2.0.0

**Prevention:**
- Pre-deployment checklist: all features in model.feature_names available?
- Feature availability test in CI/CD: verify serving has all features before release
- Version compatibility matrix: document which features needed per version
  ```
  v2.0.0: [user_id, price, ltv, engagement]
  v1.9.0: [user_id, price, engagement]  # ltv not needed
  ```

---

### Scenario 3: Data Drift (Model Trained on Old Data, Used on New)

**What breaks:** v1.0.0 trained on 2024-01 data (distribution: mean_price=$50). Now it's 2024-03 and mean_price=$150 (different distribution). Model predictions are systematically biased.

**Why it happens:**
- Model v1.0.0 deployed 8 weeks ago, never retrained
- Data distribution shifted (seasonal, trend, one-time event)
- No monitoring of data distribution
- Model assumed stable data, but real world changed

**Detection:**
```
Metric: data_distribution_shift (KS-test vs training distribution)
Alert: if (KS_statistic > 0.1) → WARN (significant distribution shift)
Check: Compare quantiles(current_data) vs quantiles(training_data)
```

**Recovery (2-24 hours):**
1. Confirm data drift: `KS_test(current, training_data) = 0.15` (significant)
2. Impact assessment: Does shift affect predictions? (run model on old vs new data)
3. Quick fix: Adjust model thresholds/calibration for new distribution (2-4 hours)
4. Long fix: Retrain v1.1.0 on recent data (6-24 hours depending on training time)
5. Deploy v1.1.0 after validation

**Prevention:**
- Monitor data distribution: monthly KS-test vs training baseline
- Set retraining trigger: if KS > 0.05, schedule retraining
- Version data distribution with model: `v1.0.0_training_distribution_mean_50`
- Retrain on sliding window: use last 30 days of data, retrain weekly

---

### Scenario 4: Breaking Change Not Flagged (Major Version Used as Minor)

**What breaks:** Developer changes model from XGBoost to LightGBM (major architecture change), tags as v1.5.0 (minor). Clients expect same input/output format, but LightGBM output is subtly different (different feature importance ranking). Unknown impact.

**Why it happens:**
- Developer doesn't realize architecture change is breaking
- No enforcement of semantic versioning rules
- v1.5.0 deployed, clients assume backward compatible
- Subtle differences cause downstream issues

**Detection:**
```
Metric: model_compatibility_check
Alert: if (model_type_changed) → CRITICAL (should be major version bump)
Check: Compare model type: "if type(v1.5.0) != type(v1.4.0) then WARN"
```

**Recovery (30-60 minutes):**
1. Detect: Clients report unexpected behavior with v1.5.0
2. Investigate: Compare v1.5.0 vs v1.4.0 output on same data
3. Reassess: Is this truly compatible? If not, it's breaking change
4. Rollback: Revert to v1.4.0 immediately
5. Retag and redeploy: Tag LightGBM version as v2.0.0 (major), redeploy with communication

**Prevention:**
- Semantic versioning enforcement: config file specifies allowed changes per version bump
  ```python
  MAJOR: architecture_type, input_schema, output_format
  MINOR: hyperparameters, feature_set (additive only)
  PATCH: training_data, hyperparameter_tuning
  ```
- Pre-deployment validation: compare v1.5.0 vs v1.4.0 systematically
- Release notes: document all changes, flag breaking changes
- Client communication: notify of major version changes 2 weeks in advance

---

### Scenario 5: Version Mismatch (Old Model Still Running in Prod)

**What breaks:** v2.0.0 deployed, but due to slow canary rollout, some servers still running v1.9.0. A/B test results are confounded: can't tell if improvement from v2.0.0 or from 50/50 traffic split of both versions.

**Why it happens:**
- Slow rolling deployment (1 server at a time)
- No version pinning in load balancer
- Monitoring doesn't track version distribution

**Detection:**
```
Metric: model_version_distribution (histogram: % requests per version)
Alert: if (variance in version distribution > 5%) → WARN (uneven rollout)
Check: SELECT version, COUNT(*) FROM inference_logs GROUP BY version
```

**Recovery (15-30 minutes):**
1. Identify: Check inference logs — which version served each request?
2. Analyze: Separate metrics by version (v1.9.0 accuracy vs v2.0.0 accuracy)
3. Accelerate rollout: Force v2.0.0 to all servers faster, don't wait for canary
4. Mark invalid period: A/B test results between start and 100% v2.0.0 are invalid

**Prevention:**
- Version pinning: explicitly deploy v2.0.0 to 100%, not gradual default
- Health check: verify all servers running expected version
  ```python
  for server in servers:
    deployed_version = server.model.version
    assert deployed_version == "v2.0.0", f"Wrong version: {deployed_version}"
  ```
- Monitoring dashboard: show % of requests per model version, alert if drifts from expected

---

## Implementation Guidance & Gotchas

**❌ Wrong: Unversioned or ambiguous versions**
```python
model.save("model.pkl")  # What version? No git? Impossible to rollback.
```

**✅ Right: Semantic + git commit**
```python
version = "v1.2.0"
git_hash = "abc123"
tag = f"{version}_{git_hash}"
registry.register(tag, model, metadata)
# Can rebuild from git if needed
```

---

## Sophisticated Interview Q&A

**Q1: Version 2.0.0 breaks backward compatibility (new required feature). How do you deploy without crashing old clients?**

A: Multi-step deployment with version coordination:

1. **Preparation (1 week before):**
   - Announce breaking change (e.g., "v2.0.0 requires 'user_engagement' feature")
   - Update all clients: deploy code that computes new feature (still use v1.9.0)
   - Verify all clients updated (check logs: all sending new feature)

2. **Deployment:**
   - Deploy v2.0.0 to 10% canary (monitor for errors)
   - Gradually increase: 10% → 50% → 100% over 2 hours
   - Keep v1.9.0 available for instant rollback

3. **After deployment:**
   - Monitor: v2.0.0 accuracy, latency, errors
   - Communicate: "v1.9.0 deprecated, will be removed 2024-04-15"
   - Deprecation period: 3 months for clients to adapt
   - Remove: 2024-04-15, delete v1.9.0

**Follow-up:** What if a client still uses v1.9.0 after deprecation date?
   - Answer: 500 error with message "Model v1.9.0 deprecated, use v2.0.0". Forces update.

---

**Q2: Model trained on data version A, now new data version B. Keep same version or bump?**

A: Always bump version if data distribution changed:

- **Same version (1.0.0):** Only if data is backward compatible, same distribution
  - Example: Added 10% more rows, same features, same distribution
- **Bump minor (1.1.0):** If data changed in ways affecting predictions
  - Example: New feature added, feature distribution shifted, data cleaned differently
- **Bump major (2.0.0):** If data format fundamentally different
  - Example: Schema changed, new upstream data source

**Reasoning:** Different data = potentially different model behavior. Version change communicates this to stakeholders.

**Follow-up:** How detect if data is truly "different"?
   - Use KS-test or other distribution comparison: if KS > 0.05, treat as different data

---

**Q3: You have v1.0.0, v1.5.0, v2.0.0 running in production (multi-version serving for A/B test). One version has a bug. How identify and isolate?**

A: Version isolation strategy:

1. **Detect issue:** Users report incorrect predictions
2. **Identify which version:** Log model version with every prediction
   ```
   SELECT version, COUNT(*), AVG(error_rate) 
   FROM inference_logs 
   GROUP BY version
   ```
   Example result: v2.0.0 has 15% error rate vs v1.5.0 (2%) and v1.0.0 (1%)

3. **Isolate:** Stop sending traffic to v2.0.0
   ```python
   # Load balancer config
   v1.0.0: 33%
   v1.5.0: 33%
   v2.0.0: 0%   # blocked, investigate separately
   ```

4. **Root cause:** Debug v2.0.0 in sandbox (use real data, isolated environment)
5. **Fix and redeploy:** v2.0.1 (patch) after fix
6. **Resume A/B test:** v2.0.1 at 33% once verified

**Why versioning helps:** Without versions, you wouldn't know which model has bug. With versioning, you isolate instantly.

---

**Q4: Backward compatibility: v2.0.0 adds new output field. Can old clients still use it?**

A: Depends on output format handling:

**If clients parse JSON output:**
- **Backward compatible:** Old clients see `{"prediction": 0.8, "confidence": 0.95}` (v2.0 with confidence)
  - Old code: `pred = output["prediction"]` still works (ignores confidence)
  - Minor version bump (v2.0.0) is appropriate

**If clients parse CSV/positional:**
- **Breaking change:** Old code expects `[prediction]`, gets `[prediction, confidence]`
  - Output format changed, clients break
  - Major version bump (v3.0.0) required

**Rule of thumb:** Output changes that clients can ignore = minor. Changes that break parsing = major.

---

**Q5: How do you prevent accidentally deploying a v1.0.0 model after v2.0.0 has been running?**

A: Version enforcement at deployment:

```python
def deploy_model(version_to_deploy):
    current_version = get_running_version()
    
    # Prevent downgrade to older version
    if semantic_version(version_to_deploy) < semantic_version(current_version):
        raise ValueError(
            f"Cannot deploy {version_to_deploy}, "
            f"current is {current_version}. Downgrade not allowed."
        )
    
    # Allow lateral move (v2.0.0 → v2.0.1)
    # Allow upgrade (v2.0.1 → v2.1.0)
    deploy(version_to_deploy)
```

**Exception:** Allow rollback in emergencies with explicit approval:
```python
deploy(version_to_deploy, force_rollback=True, reason="P1 bug detected")
```

**Follow-up:** What if you need to rollback?
   - Answer: Use force_rollback=True, but this should be rare (1x/year max)

---

**Q6: Versioning strategy changes: currently semantic, want to switch to calendar (2024-01-15.1). How migrate?**

A: Gradual migration strategy:

1. **Phase 1 (parallel, 2 weeks):** Deploy new models with calendar version, old models keep semantic
   - v2.5.0 (old semantic)
   - 2024-01-15.1 (new calendar)

2. **Phase 2 (transition, 4 weeks):** Deprecate semantic versions
   - v2.5.0 marked deprecated (still functional)
   - New models only use calendar

3. **Phase 3 (cleanup):** Remove old semantic versions after 3 months
   - v2.4.0 and older archived
   - Only calendar versioning going forward

**Key:** Never force migration abruptly (breaks tracking). Gradual migration allows teams to adapt.

---

**Q7: A client is still using v1.0.0 from 2 years ago. What's your strategy?**

A: Version lifecycle management:

**Tier 1 (Supported, 2 years):**
- v2.5.0 (current)
- Receive bug fixes, security updates

**Tier 2 (Deprecated, 2-3 years):**
- v2.0.0
- No new features, critical fixes only
- Notify: "upgrade to v2.5.0 in next 6 months"

**Tier 3 (EOL, >3 years):**
- v1.0.0
- No support, will be shut down
- Hard deadline: "v1.0.0 will be removed 2024-04-15"

**If client still on v1.0.0 after EOL:**
- Option A: Force upgrade (500 error, message directing to v2.5.0)
- Option B: Provide limited support (charge fee for continued support on old version)
- Option C: Client maintains their own copy (no longer our responsibility)

**Communication:** At least 3 months notice before EOL.

---

**Q8: You notice that versions are being used inconsistently (some v1.5.0, some v1.50.0, some v1_5_0). How standardize?**

A: Semantic versioning enforcement:

1. **Standard:** MAJOR.MINOR.PATCH (e.g., v2.1.3)
2. **Validation:** Regex check in CI/CD
   ```python
   import re
   version = "v2.1.3"
   if not re.match(r"^v\d+\.\d+\.\d+$", version):
       raise ValueError(f"Invalid version format: {version}")
   ```

3. **Enforcement:** Block deployments that don't match format
4. **Documentation:** Clearly document rules in CONTRIBUTING.md
   - MAJOR: architecture change
   - MINOR: feature/data changes
   - PATCH: bugfixes, hyperparameter tuning

---

**Q9: How version model + code + data together for reproducibility?**

A: Reproducibility with composite versioning:

```python
model_metadata = {
    "model_version": "v2.1.0",
    "code_commit": "abc123def456",
    "data_version": "dvc_hash_xyz789",
    "training_timestamp": "2024-01-15T10:30:00Z",
    "hyperparameters": {"lr": 0.001, "epochs": 100}
}
```

**To reproduce exact model 6 months later:**
```bash
git checkout abc123def456          # get exact code
dvc checkout xyz789                # get exact training data
python train.py --config metadata  # train with same hyperparams
# Result: identical model (byte-for-byte same)
```

**Why all three:** Code changes, data changes, and hyperparams all affect output. Only versioning all three guarantees reproducibility.

---

**Q10: Models evolve (v1.0 → v2.0 → v3.0). Should you deprecate old versions or keep them?**

A: Version retention strategy:

**Keep (recommended):**
- Reasons: Rollback capability, historical analysis, regulatory compliance
- Cost: Storage (models are ~100MB-1GB typically, cheap)
- Duration: Keep at least v(current-2), e.g., if current is v4.0, keep v3.0 and v2.0

**Archive:**
- After 2 years: move to cold storage (S3 Glacier, costs 10x less)
- After 5 years: can delete if no regulatory requirement

**Regulatory/compliance:** Some industries (finance, healthcare) require keeping all versions forever
   - Example: lending model v1.0 must be available for 7 years (regulatory audit)

**Decision:** Default = keep recent versions (3-5), archive old, delete only after legal review

---

## Cost & Resource Analysis

### Operational Cost

**Versioning infrastructure:**
- Model registry storage: 1GB per model × 100 versions = 100GB (S3: $2.30/month)
- Metadata database: negligible (<1MB)
- Git storage: negligible (code < 1MB per model)
- **Total infrastructure cost: ~$2.30/month**

**Personnel cost:**
- Registration per model: 5 min (automatic in CI/CD, near-zero cost)
- Version management/cleanup: 1 hour/month = ~$150/month (1 engineer)
- Documentation/governance: 2 hours/month = ~$300/month (1 engineer)
- **Total personnel cost: ~$450/month** for organization

### Cost of NOT Versioning

**Scenario: Model failure without proper versioning**
- Time to identify which version failed: 1-2 hours (debugging, logs)
- Time to rollback: 2-4 hours (rebuild from old binary if available, else train new model)
- Downtime: 3-6 hours (revenue loss for e-commerce: $1K-$50K)
- Data loss risk: retraining from scratch loses feedback history
- **Cost of one failure: $2K-$50K+ in downtime**

**Scenario: Model failure WITH proper versioning**
- Time to identify issue: 5 min (which version failed? check logs)
- Time to rollback: 2 min (deploy v1.9.0, pre-tested)
- Downtime: <10 minutes
- **Cost of one failure: ~$100-$500 (minimal)**

### ROI Analysis

**Assumption:** 1 critical failure per quarter (3-month outage scenario)

- Without versioning: 4 failures × $25K avg cost = $100K/quarter
- With versioning: 4 failures × $500 avg cost = $2K/quarter
- **Savings: $98K/quarter = $392K/year**

**Against cost of versioning system: $450/month = $5.4K/year**

**ROI: $392K / $5.4K = 72x return** (immediate break-even in 1 week)

---

## Monitoring & Observability Patterns

### Key Metrics to Instrument

**Version Distribution Metrics:**
```
- model_version_distribution (histogram)
  Example: v2.0.0 serving 95% traffic, v1.9.0 serving 5% (canary)
  Alert: if any version < 1% OR > 99%, rollout is uneven

- version_deployment_timestamp
  Track when each version was deployed
  Alert: if version > SLA_age (e.g., 8 weeks), mark for retraining

- version_rollback_count
  How many rollbacks happened this month
  Alert: if > 1 per week, investigate code quality

- version_compatibility_status
  Are all required features available for deployed version?
  Alert: if incompatible, block deployment
```

**Data Version Metrics (for models trained on data v1.0, v1.1, etc):**
```
- training_data_version_mismatch
  Example: model trained on data v1.0, but production data now v1.5
  Alert: if mismatch, schedule retraining

- data_distribution_shift_ks_statistic
  KS test comparing current vs training data distribution
  Alert: if KS > 0.05 (significant shift), recommend retraining

- feature_missing_count
  How many features expected by model are unavailable in production?
  Alert: if count > 0, block inference
```

**Versioning System Health:**
```
- version_registry_latency
  Time to lookup and load model version
  Alert: if > 100ms, investigate registry performance

- version_storage_usage_gb
  Total storage used by all model versions
  Alert: if > 80% of quota, clean up old versions

- version_validation_failures
  How many version deployments failed validation?
  Alert: if > 5% failure rate, investigate CI/CD pipeline
```

### Alert Thresholds & Escalation

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Version age (weeks) | > 8 weeks | > 12 weeks | Recommend/require retraining |
| Version rollback count | > 1/week | > 2/week | Investigate code quality, increase testing |
| Incompatible version deployed | N/A | Yes | Block immediately, requires emergency fix |
| Data distribution shift (KS) | > 0.05 | > 0.15 | Notify data/ML team, schedule retraining |
| Feature missing count | 0 | > 0 | Block inference, escalate to feature eng |
| Version uneven deployment | > 5% variance | > 20% variance | Accelerate rollout or investigate |

### Monitoring Dashboard

```
Panel 1: Version Timeline (line chart)
  - X-axis: date
  - Y-axis: active versions
  - Shows: which versions deployed when, how long each lasted
  
Panel 2: Version Distribution (pie chart)
  - Current traffic % per version (v2.0.0: 95%, v1.9.0: 5%)
  - Alert if any version < 1% or > 99%

Panel 3: Rollback Frequency (bar chart)
  - X-axis: month
  - Y-axis: rollback count
  - Shows: trend (increasing = code quality issue)

Panel 4: Model Compatibility (table)
  - Columns: version, training_data_version, required_features, available_features
  - Color code: green (compatible), red (incompatible)

Panel 5: Data Distribution Shift (line chart)
  - X-axis: date
  - Y-axis: KS-statistic (training vs current)
  - Red line: KS > 0.05 (significant shift, schedule retraining)

Panel 6: Version Age (gauge)
  - Red: > 12 weeks (EOL soon)
  - Yellow: 8-12 weeks (should upgrade)
  - Green: < 8 weeks (recent)
```

## Code Example

```python
import json, subprocess
from pathlib import Path

class ModelVersioning:
    def register_version(self, model, version, metrics, code_commit, data_version):
        metadata = {
            "version": version,
            "code_commit": code_commit,
            "data_version": data_version,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        version_dir = Path(f"models/v{version}")
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(f"{version_dir}/model.pkl")
        
        # Save metadata
        with open(f"{version_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Registered model v{version} with code commit {code_commit}")
```

## Interview Q&A

Q: When bump MAJOR vs MINOR vs PATCH?
A: A: MAJOR: architecture change (LSTM→CNN). MINOR: feature set change (add X). PATCH: hyperparameter only. Uncertain? Use MINOR (safe).

Q: v2.0.0 worse than v1.9.0?
A: A: Registry stores both. Deploy v1.9.0 (5min). Investigate v2.0.0 bug. Fix, retrain as v2.0.1.

Q: Have v1.2.0, v1.2.1, v1.2.2. Deploy which?
A: A: Compare validation: highest accuracy? Also check latency, size, cost. Deploy best.

Q: Version in code or assigned at deployment?
A: A: Assigned at deployment. Code doesn't mention version. System increments based on what changed (prevents mismatch).

Q: v1.0.0 on data_v1, v1.1.0 on data_v2, accuracy dropped?
A: A: Track data version with model. v1.1.0={code:abc, data:v2, accuracy:0.91}. Understand: accuracy dropped due to data, not code.

Q: Two teams release v1.2.0?
A: A: Prevent with central authority. One registry, one version number generator. Teams coordinate.

Q: Maintain how many old versions?
A: A: Production + 1 prior (rollback). Latest 3 for comparison. Archive older.

Q: v1.5.0 passes validation but fails in production?
A: A: Tag versions: v1.5.0{caveat:'works on segment_A, lower on segment_B'}. Use tags: 'stable', 'experimental', 'production_ready'.
## Interview Quick-Reference

| Semver | Example | When |
|--------|---------|------|
| MAJOR | 1.0.0 → 2.0.0 | Architecture change |
| MINOR | 2.0.0 → 2.1.0 | Feature set change |
| PATCH | 2.1.0 → 2.1.1 | Bugfix/hyperparameter |

## Related Topics
- [Model Registry](04-model-registry.md) - manages all versions
- [Data Pipelines](02-data-pipelines.md) - data versioning (DVC)

## Resources
- [Semantic Versioning](https://semver.org/)
- [DVC: Data Version Control](https://dvc.org/)

