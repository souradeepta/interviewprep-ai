# MLOps Interview Questions

These prompts focus on operational machine-learning systems. Scenario questions ask candidates to identify the failure boundary, gather evidence, and choose a reversible response before proposing a tool.

## Data contracts and feature reliability

### 1. Design a training-data contract
**Prompt:** Define a data contract for daily product events used to train a purchase-propensity model.
**Role/domain:** MLOps / data engineering. **Difficulty:** Medium.
**Expected answer:** Specify schema and units, event and entity keys, event-time semantics, null/range constraints, allowed lateness, ownership/versioning, privacy classification, and compatibility policy. Validate each batch and quarantine or reject violations with an auditable reason.
**Evaluation:** Semantic constraints, compatibility, failure handling, ownership.
**Follow-up:** How do you handle a backward-compatible optional field?
**Common mistake:** Defining only column names and types while ignoring meaning, time, and null behavior.

### 2. Debug a feature-store parity gap
**Prompt:** Offline validation is strong, but online predictions drop after launch. Offline and online feature values differ for recent users. How do you investigate?
**Role/domain:** MLE; debugging. **Difficulty:** Hard.
**Expected answer:** Trace a stable set of entity/time examples through source, transformation version, event-time cutoff, defaults, late data, and online materialization. Compare serialized features and freshness; fix shared transformation or parity tests, then backfill/replay and shadow before rollout.
**Evaluation:** Point-in-time correctness, reproducible trace, safe remediation.
**Follow-up:** How can you detect leakage in the offline join?
**Common mistake:** Retraining immediately or comparing aggregate distributions only.

### 3. Prevent training-serving leakage
**Prompt:** A feature is computed from the latest account state at training time. Why can this leak, and what is the correct join rule?
**Role/domain:** ML platform; data correctness. **Difficulty:** Medium.
**Expected answer:** The latest state may include information unavailable at the prediction timestamp or after the label outcome. Join features as-of each example’s prediction time using event/availability time, enforce label windows, and test with temporal holdouts and deliberately late updates.
**Evaluation:** Availability-time reasoning, point-in-time joins, validation.
**Follow-up:** Why can event time alone still be insufficient?
**Common mistake:** Splitting rows randomly after features were computed using future state.

### 4. Debug a pipeline schema change
**Prompt:** An upstream producer changes an integer feature to a string; training succeeds after coercion but quality drops. What controls should catch this?
**Role/domain:** Data platform; incident analysis. **Difficulty:** Medium.
**Expected answer:** Enforce a versioned schema contract and semantic/range checks at ingestion, fail or quarantine incompatible batches, alert the owner, and track coercion rates. Reproduce the affected dataset/model version and only backfill after compatibility and downstream behavior are validated.
**Evaluation:** Detection, blast-radius containment, lineage and recovery.
**Follow-up:** When is a schema migration safe to roll out?
**Common mistake:** Silently coercing values and treating successful job completion as data correctness.

## Evaluation, release, and serving

### 5. Design a reproducible training run
**Prompt:** What must be recorded so a model can be reproduced and its lineage audited six months later?
**Role/domain:** MLOps; reproducibility. **Difficulty:** Medium.
**Expected answer:** Record immutable code and configuration, data snapshot/manifest and feature definitions, dependency/container versions, random seeds, hardware-relevant settings, metrics and evaluation slices, artifact checksum, and parent run. Preserve access controls and retention for data/model provenance.
**Evaluation:** Complete lineage, immutability, reproducibility limits.
**Follow-up:** Which parts may still make bitwise reproduction impossible?
**Common mistake:** Saving only a model file and a random seed.

### 6. Debug offline/online metric divergence
**Prompt:** Offline AUC improves by 4 points, but the canary’s business metric is flat. What do you check before declaring success or failure?
**Role/domain:** MLE; evaluation debugging. **Difficulty:** Hard.
**Expected answer:** Verify cohort, label window, prediction/decision threshold, metric aggregation, exposure and logging, data freshness, traffic mix, and online feature parity. Compare slice-level outcomes and uncertainty; keep the canary bounded while separating metric instrumentation failure from model behavior.
**Evaluation:** Measurement validation, cohort alignment, controlled response.
**Follow-up:** How would you distinguish a delayed outcome from a true null effect?
**Common mistake:** Treating offline AUC as a direct estimate of online business impact.

### 7. Choose a deployment strategy
**Prompt:** A new fraud model has uncertain latency and false-positive impact. Propose a staged rollout.
**Role/domain:** MLOps / system design. **Difficulty:** Medium.
**Expected answer:** Validate artifact and contract, replay/shadow traffic, canary a small eligible fraction, monitor latency/errors and quality guardrails against a baseline, then ramp only when predeclared gates pass. Keep a tested rollback route and account for delayed labels.
**Evaluation:** Stages, guardrails, rollback, label delay.
**Follow-up:** What would make shadow traffic misleading?
**Common mistake:** Jumping directly from offline evaluation to full traffic.

### 8. Plan a rollback with stateful features
**Prompt:** A model rollback is requested, but online features were migrated at the same release. What must be considered?
**Role/domain:** ML platform; release engineering. **Difficulty:** Hard.
**Expected answer:** Treat model and feature schema as a compatible release unit. Determine whether the old model can consume current features, preserve backward-compatible reads or dual-write during migration, validate rollback on representative traffic, and avoid reverting state in a way that loses updates. Define a tested fallback if direct rollback is unsafe.
**Evaluation:** Compatibility, state management, tested recovery.
**Follow-up:** How would you stage a feature rename without downtime?
**Common mistake:** Assuming model artifact rollback also restores compatible feature state.

### 9. Design a drift-monitoring plan
**Prompt:** Specify monitoring for a model whose labels arrive two weeks after prediction.
**Role/domain:** Production MLE; monitoring. **Difficulty:** Hard.
**Expected answer:** Monitor service health and input/feature quality immediately, prediction and cohort shifts with context, then delayed performance when labels mature. Track freshness, calibration/error by meaningful slices, uncertainty and minimum sample; define owners, thresholds, alert actions, and a policy against automatic retraining from drift alone.
**Evaluation:** Leading/lagging signals, delayed labels, actionable alert policy.
**Follow-up:** How do you distinguish seasonal shift from harmful drift?
**Common mistake:** Monitoring only feature-distribution distance or treating every shift as a model failure.

### 10. Debug a silent data outage
**Prompt:** Prediction volume is normal, but 30% of a key feature is now defaulted to zero. The model has not triggered an error alert. What do you do?
**Role/domain:** On-call MLE; incident response. **Difficulty:** Hard.
**Expected answer:** Declare/triage impact, compare missing/default rates by source, version and slice, inspect freshness and upstream changes, and use a safe fallback or rollback if the feature is critical. Bound exposure, notify owners, preserve examples/lineage, and verify recovery plus affected-period scoring before closing.
**Evaluation:** Detection gap, containment, diagnosis, recovery evidence.
**Follow-up:** What invariant would have caught this before predictions were served?
**Common mistake:** Waiting for delayed labels while a known feature contract is violated.

### 11. Respond to a latency regression
**Prompt:** P99 latency doubles after model release while median remains stable. Outline your diagnosis.
**Role/domain:** Serving systems; debugging. **Difficulty:** Hard.
**Expected answer:** Segment latency by model version, request shape, batch size, hardware, queue and feature-fetch timing; inspect tail-specific input lengths, contention, cold starts and timeout/retry amplification. Compare against the prior release under equivalent load, cap impact, and roll back or route safely if the SLO is breached.
**Evaluation:** Tail-focused decomposition, load comparability, mitigation.
**Follow-up:** Why can retries worsen the incident?
**Common mistake:** Optimizing the average or scaling blindly without finding the tail source.

### 12. Avoid misleading drift alerts
**Prompt:** A population-stability alert fires every Monday because traffic mix changes by weekday. How do you improve it?
**Role/domain:** MLOps; monitoring design. **Difficulty:** Medium.
**Expected answer:** Establish seasonality-aware baselines and compare like cohorts/time windows; monitor feature quality and delayed outcome performance alongside distribution shifts. Set thresholds using expected variation and operational cost, require minimum samples, and make alerts trigger investigation rather than automatic retraining.
**Evaluation:** Baseline design, complementary signals, alert actionability.
**Follow-up:** How would you detect a real shift within one weekday cohort?
**Common mistake:** Raising the threshold until the alert stops without checking sensitivity.

## Reliability, governance, and cost

### 13. Handle a suspected label leak before release
**Prompt:** A feature importance review shows a post-outcome status field dominates a model. The launch is tomorrow. What is your recommendation?
**Role/domain:** MLOps; release governance. **Difficulty:** Hard.
**Expected answer:** Block promotion until temporal availability and feature lineage are verified. Remove or correctly time-bound the feature, rerun leakage-safe evaluation, compare expected performance, and document the decision. Schedule pressure does not establish that the field is safe at prediction time.
**Evaluation:** Release gate, evidence, re-evaluation plan.
**Follow-up:** What evidence could show the field is legitimately available at inference?
**Common mistake:** Shipping behind a flag while assuming a post-outcome field will somehow be absent online.

### 14. Define rollback criteria
**Prompt:** Give concrete rollback criteria for a model canary when labels are delayed.
**Role/domain:** MLOps; deployment. **Difficulty:** Medium.
**Expected answer:** Use immediate guardrails such as error/timeout rate, latency SLO, feature-contract violations, resource exhaustion, and safety/business proxies with minimum volume and duration. Define owners, thresholds, fallback behavior, and separate later quality gates once labels mature.
**Evaluation:** Observable criteria, delayed outcome handling, action ownership.
**Follow-up:** How can a proxy metric create a false sense of safety?
**Common mistake:** Making “rollback if metrics look bad” the entire release policy.

### 15. Review a proposed auto-retraining loop
**Prompt:** A team wants to retrain and deploy whenever feature drift exceeds a threshold. What risks and controls do you raise?
**Role/domain:** Production ML; governance. **Difficulty:** Hard.
**Expected answer:** Drift does not establish label degradation or that new data is valid. Require data quality and label checks, leakage-safe temporal evaluation, model approval and canary, rollback, provenance, and rate limits; separate retraining trigger from deployment decision and monitor feedback loops.
**Evaluation:** Causal meaning of drift, validation gates, governance.
**Follow-up:** What monitoring signal could justify starting a retraining job without authorizing deployment?
**Common mistake:** Equating a changed input distribution with a need for immediate promotion.

### 16. Manage a privacy deletion request
**Prompt:** A user requests deletion of their data after it has contributed to training. What system capabilities and decisions are needed?
**Role/domain:** ML governance / platform. **Difficulty:** Hard.
**Expected answer:** Trace data lineage through raw/derived stores, feature tables, snapshots, and training runs; apply retention/deletion policy and document exceptions. Determine legal/product requirements for model impact, assess whether retraining or a validated unlearning approach is needed, and verify deletion propagation without claiming the model forgot automatically.
**Evaluation:** Lineage, policy governance, verification and limits.
**Follow-up:** What evidence demonstrates deletion from active and backup systems?
**Common mistake:** Deleting the source row while ignoring cached features and retained training snapshots.

### 17. Reduce serving cost under an SLO
**Prompt:** Inference spend rose 40%; latency and quality SLOs must hold. How do you find safe savings?
**Role/domain:** MLOps / platform engineering. **Difficulty:** Medium.
**Expected answer:** Attribute cost by traffic, model/version, accelerator utilization, batching, feature I/O, retries and idle capacity. Benchmark candidate changes on representative load, evaluate tail latency and quality guardrails, then canary bounded optimizations such as right-sizing or batching with a rollback path.
**Evaluation:** Cost attribution, representative measurement, constrained rollout.
**Follow-up:** When can batching increase total cost despite better throughput?
**Common mistake:** Choosing the cheapest instance from per-request cost without measuring utilization or tail SLOs.

### 18. Design artifact promotion and governance
**Prompt:** Several teams can register models. Design controls that allow fast experiments but prevent an unreviewed artifact from reaching production.
**Role/domain:** ML platform; governance. **Difficulty:** Medium.
**Expected answer:** Separate experiment and production namespaces/permissions; require immutable artifact identity, provenance, evaluations, data/privacy checks, owner and approval records for promotion. Automate policy gates and audit events, while allowing reproducible rollback to a previously approved version.
**Evaluation:** Separation of duties, traceability, usable workflow.
**Follow-up:** How should an emergency fix move through the gate?
**Common mistake:** Relying on model names or mutable registry tags as identity and approval evidence.

### 19. Prioritize a feature-store outage
**Prompt:** Online feature lookups are timing out for 5% of requests. A default value is available, but it was not validated for every model. What response do you design?
**Role/domain:** ML systems; reliability. **Difficulty:** Hard.
**Expected answer:** Identify affected models/features and whether defaults are semantically safe; apply a per-model fallback or route to a stable baseline, bound retries/timeouts, and preserve the latency budget. Alert with impact and owner, test recovery and parity, and review fallback behavior before enabling it broadly.
**Evaluation:** Model-specific blast radius, safe fallback, SLO and recovery.
**Follow-up:** How could a default itself create a feedback loop?
**Common mistake:** Returning zero universally because it prevents request failure.

### 20. Respond to a regulated-model audit
**Prompt:** An auditor asks why a specific prediction was produced and which model/data governed it. What records should your serving path make available?
**Role/domain:** Regulated ML operations; auditability. **Difficulty:** Hard.
**Expected answer:** Link request/time and authorized entity reference to immutable model and feature-definition versions, input provenance or permitted feature snapshot, preprocessing/config, output and applicable explanation, and deployment/approval record. Apply access controls, privacy minimization and retention policy; state limits where exact inputs cannot be retained.
**Evaluation:** Reconstructability, immutable versioning, privacy-aware audit design.
**Follow-up:** How do you investigate when feature snapshots were intentionally not retained?
**Common mistake:** Logging unrestricted raw personal data indefinitely as a substitute for an audit design.
