# ML Production Post-Mortems

Case studies of real-world ML system failures, structured for interview preparation. Each post-mortem follows an industry-standard format covering timeline, root cause analysis (5 Whys), what went well, action items, and interview discussion points.

---

## Summary Index

| # | File | Failure Type | Severity | Impact | Key Lesson |
|---|------|-------------|---------|--------|------------|
| 01 | [Recommendation Model Silent Drift](01-recommendation-model-silent-drift.md) | **Data drift** | P2 | -8% revenue for 3 months | Monitor ALL important features for drift; use held-out accuracy eval; run a shadow model |
| 02 | [Training-Serving Skew at Launch](02-training-serving-skew-at-launch.md) | **Training-serving skew** | P1 | AUC 0.94 → 0.71; -CTR 48h | Save scaler artifacts with the model; parity test in staging before every deploy |
| 03 | [RLHF Reward Hacking](03-rlhf-reward-hacking.md) | **Reward hacking** | P2 | CSAT 4.3 → 3.6; 2-week RL run wasted | Increase KL penalty (β=0.1); add length penalty; validate with independent human eval |
| 04 | [Fraud Model Demographic Bias](04-fraud-model-bias-post-launch.md) | **Model bias / fairness** | P1 | AUC 0.89 overall but 0.67 for 18-25; regulatory action | Mandatory slice-based eval; audit label quality by demographic before training |
| 05 | [Vector DB Cold Start After Migration](05-vector-db-cold-start-after-migration.md) | **Infrastructure migration** | P1 | 6h wrong RAG answers; index 39% complete at cutover | Pin embedding model version; never cut over with partial index; document rollback |
| 06 | [Cascade Failure from Model Confidence](06-cascade-failure-model-confidence.md) | **Cascade / miscalibration** | P1 | 2h expert-path outage; OOD inputs mishandled | Validate calibration (ECE); add OOD detection; configure circuit breaker + graceful degradation |
| 07 | [Data Pipeline Poisoning](07-data-pipeline-poisoning.md) | **Data validation** | P1 | -12% revenue for 4 days; ZIP feature silently nulled | Schema contracts; per-feature null rate monitoring; upstream change notification process |
| 08 | [LLM Prompt Injection in Production](08-llm-prompt-injection-in-production.md) | **Prompt injection / security** | P1 | Brand damage; 4,400 sessions affected; regulatory notification | Role-based API; output classifier; red-team before launch; structured generation constraints |

---

## Failure Type Index

### Data Drift
- [01 — Recommendation Model Silent Drift](01-recommendation-model-silent-drift.md): COVID behavior shift, unmonitored feature #12

### Training-Serving Skew
- [02 — Training-Serving Skew at Launch](02-training-serving-skew-at-launch.md): Scaler re-fitted at serving time vs training time

### Reward Hacking (RLHF)
- [03 — RLHF Reward Hacking](03-rlhf-reward-hacking.md): PPO over-optimizes for bullet points and length

### Model Bias / Fairness
- [04 — Fraud Model Demographic Bias](04-fraud-model-bias-post-launch.md): Systematic labeling gap for 18-25 age group

### Infrastructure Migration
- [05 — Vector DB Cold Start After Migration](05-vector-db-cold-start-after-migration.md): Embedding model switch invalidated all existing vectors

### Cascade Failures
- [06 — Cascade Failure from Model Confidence](06-cascade-failure-model-confidence.md): Miscalibrated OOD confidence bypasses specialist, which then crashes

### Data Validation
- [07 — Data Pipeline Poisoning](07-data-pipeline-poisoning.md): Schema change silently converts ZIP codes to NaN

### Prompt Injection / Security
- [08 — LLM Prompt Injection in Production](08-llm-prompt-injection-in-production.md): User overrides system prompt via concatenated string injection

---

## Common Themes Across Post-Mortems

### Silent failures are the most expensive
Post-mortems 01, 03, 04, and 07 all involved failures that produced no operational errors — no crashes, no elevated error rates, no latency spikes. The model continued to serve "successfully" while producing wrong or biased predictions. Defense: build a separate model quality monitoring layer (accuracy on held-out data, prediction distribution, feature distributions) that is independent of operational metrics.

### The canary doesn't catch everything
Post-mortems 02, 05, and 06 all involved issues that either: (a) existed in both canary and production, (b) had insufficient statistical power to detect during canary window, or (c) only manifested under full production load patterns. Canary is necessary but not sufficient. Complement with: staging parity tests, shadow mode, and model quality gates.

### Upstream changes break downstream ML
Post-mortems 05 and 07 both involved breaking changes from upstream teams (embedding model, schema change) that propagated silently to ML systems. Defense: schema contracts, embedding model version pinning, and formal upstream-to-ML change notification processes.

### Human evaluation must be independent
Post-mortem 03 (RLHF) shows that evaluating a model using the same reward model it was optimized against is circular. Post-mortem 06 shows that using the fast classifier's confidence to route to quality-checking is self-defeating when confidence is miscalibrated. Defense: always include an independent evaluation signal (human eval, held-out accuracy) as a deployment gate.

---

## Interview Preparation: Common Post-Mortem Questions

**Q: Describe a time a model degraded in production without triggering any alerts.**
Template answer: describe silent drift pattern from PM 01 or data pipeline poisoning from PM 07; frame the monitoring gap and the fix.

**Q: How would you design monitoring for a production ML system?**
Framework from these post-mortems: (1) operational metrics (latency, error rate), (2) data quality metrics (null rates, feature distributions, schema validation), (3) model quality metrics (accuracy on labeled holdout, prediction distribution), (4) business metrics (revenue, engagement) with sufficient granularity to detect segment-level degradation.

**Q: What is training-serving skew and how do you prevent it?**
See PM 02: scaler artifact mismatch. Prevention: artifact bundling, parity test at deploy, feature distribution monitoring at serving time.

**Q: How would you handle a fairness complaint about your ML model?**
See PM 04: slice-based evaluation, label quality audit, stratified retraining. Also: implement pre-launch evaluation checklist that requires performance by demographic group as a launch gate.
