# Model Registry & CI/CD: Automating Model Deployment Pipelines

## Definition & Why It Matters

Model registry is a centralized repository managing model lifecycle: registration → staging → approval → production → archive. CI/CD automation runs tests, builds artifacts, deploys automatically when criteria met.

**The deployment problem:** Manual model deployment is error-prone. Engineer retrains, uploads model to random S3 path, tells ops team "it's ready." Ops team forgets to test, deploys, breaks production. Solution: automated pipeline.

**Why CI/CD for models matters:**
- **Consistency**: Every model deployment follows same workflow (train → test → build → deploy)
- **Validation**: Pipeline runs tests before deploying (catches bugs)
- **Audit trail**: Every deployment logged (who deployed what when)
- **Rollback**: Previous version tracked; instant rollback if needed
- **Speed**: Automation = hours not days from training to production

Netflix has fully automated model deployment. Stripe validates with shadow testing before auto-deploying. Every high-velocity ML team automates.

---

## How It Works

### Model Lifecycle

```
Development
    ↓
Train new model
    ↓
[Model Registry] Register model, version 1.0.0
    ↓
Staging
    ↓
Validation pipeline: unit tests, integration tests, shadow test (1 week)
    ↓
[Approval Gate] Engineering lead approves (code review style)
    ↓
Production
    ↓
Canary: 5% traffic (1 day), monitor
    ↓
Rollout: 25% → 50% → 100% (over 3 days)
    ↓
Monitoring: track metrics continuously
    ↓
If degradation detected → auto-rollback to previous version
    ↓
Archive: version 1.0.0 available for history
```

```mermaid
graph TD
    A["Model Training<br/>v1.0.0"] --> B["Register<br/>Model Registry"]
    B --> C["Staging Environment<br/>Unit Tests<br/>Integration Tests"]
    C --> |"Pass?"| D{Tests OK?}
    D --> |"No"| E["Debug<br/>Retrain"]
    E --> B
    D --> |"Yes"| F["Shadow Test<br/>1 Week<br/>Production Data"]
    F --> |"Pass?"| G{Metrics OK?}
    G --> |"No"| E
    G --> |"Yes"| H["Approval Gate<br/>Manual Review"]
    H --> |"Approve?"| I{Approved?}
    I --> |"No"| E
    I --> |"Yes"| J["Canary Deploy<br/>5% Traffic<br/>1 Day"]
    J --> |"OK?"| K{Monitor<br/>Metrics}
    K --> |"Degradation"| L["Auto-Rollback<br/>to v0.9.9"]
    K --> |"Good"| M["Expand: 25%"]
    M --> |"OK?"| K
    M --> |"Good"| N["Expand: 50%"]
    N --> |"OK?"| K
    N --> |"Good"| O["Expand: 100%"]
    O --> |"OK?"| K
    O --> |"Good"| P["Production<br/>v1.0.0 Live"]
    P --> Q["Continuous Monitoring"]
    Q --> |"Degradation"| L
    L --> R["Archive: v0.9.9"]
```

### CI/CD Pipeline Stages

**Stage 1: Model Training**
- Trigger: code pushed, new data available, on-schedule
- Output: trained model, metrics (accuracy, latency)
- Validation: accuracy > threshold? Save model. Else: training failed, alert.

**Stage 2: Validation**
- Unit tests: model forward pass works?
- Integration tests: model + data pipeline together?
- Shadow test: model runs on production data, predictions compared to baseline
- Output: passed/failed

**Stage 3: Build**
- Package model: into Docker image or model artifact
- Tag version: git commit hash, semantic version
- Push to registry: model registry and container registry

**Stage 4: Staging Approval**
- Requires: 2 approvals (code review style)
- Checklist: metrics acceptable? Shadow test passed? Documentation complete?
- Can be automatic if all checks pass, or manual review

**Stage 5: Production Deployment**
- Canary: 5% traffic
- Monitor: latency, error rate, business metrics
- Expand: 25% → 50% → 100%
- Rollback: if any metric degrades > threshold, auto-rollback

### Example: Fraud Model Pipeline

```yaml
# .github/workflows/deploy-fraud-model.yml
on:
  push:
    paths: [fraud_model/**, test/**]

jobs:
  train:
    runs-on: gpu-runner
    steps:
      - train model
      - if accuracy > 0.95: save model  # validation
      - else: fail
  
  validate:
    needs: train
    steps:
      - run unit tests
      - run integration tests with real data
      - shadow test: run on 1-week production data
      - if shadow accuracy ~= baseline: pass
      - else: fail
  
  build:
    needs: validate
    steps:
      - docker build fraud-model:abc123  # tag with commit
      - docker push registry/fraud-model:abc123
  
  stage-approval:
    needs: build
    steps:
      - create merge request for approval
      - require 2 approvals
      - comment checklist (metrics, testing, docs)
  
  canary:
    needs: stage-approval
    steps:
      - deploy fraud-model:abc123 to 5% traffic
      - monitor latency, error rate, fraud detection rate
      - if degraded: auto-rollback
      - if stable 24h: proceed
  
  rollout:
    needs: canary
    steps:
      - 25% traffic (24h)
      - 50% traffic (24h)
      - 100% traffic
      - monitor continuously
```

---

## Interview Q&A: Model Registry & CI/CD

### Q1: "Model is trained. How do you safely deploy it?"
**Answer outline:** Not "push to production immediately." Pipeline:
1. **Register**: Model registry records training run, metrics, code version
2. **Validate**: Unit + integration + shadow tests (shadow = production data, no user risk)
3. **Approve**: Code review style, 2 approvals confirm metrics acceptable
4. **Canary**: 5% traffic, 24h monitoring (catch any issues)
5. **Rollout**: Gradual expansion (25% → 50% → 100%)
6. **Monitor**: Continuous monitoring (auto-rollback if degradation)

Example: Fraud model trained, registered in MLflow. Shadow test compares to baseline on last week's data. Approval gate checks: accuracy good? False positive rate acceptable? Then canary to 5% fraud detection traffic.

### Q2: "Model deployed at 10am. By 4pm, accuracy dropped from 95% → 91%. Auto-rollback?"
**Answer outline:** Yes, with caveats:
1. **Immediate**: Auto-rollback to previous version (instant)
2. **Investigation**: Why did accuracy drop? Data quality issue? Code bug? Real-world shift?
3. **Fix**: Address root cause
4. **Re-deploy**: With fix, run through pipeline again (not skipping validation)

Without auto-rollback: Someone monitors manually, notices at 6pm, tells ops team, ops team manually switches. 8 hours of bad predictions. With auto-rollback: instant, <10 seconds of bad predictions.

Threshold: auto-rollback if any metric degrades >2% (configurable).

### Q3: "How do you version models? How do you know which version is production?"
**Answer outline:** Multi-level versioning:
1. **Semantic versioning**: 1.2.3 (major.minor.patch)
   - 1.0.0 → 2.0.0: breaking change (different input format)
   - 1.0.0 → 1.1.0: new feature (new input optional)
   - 1.0.0 → 1.0.1: bug fix (same behavior)

2. **Git versioning**: Tag commit: v1.2.3-fraud-model-2024-05-17

3. **Registry versioning**: Model registry tags production versions
   ```
   myregistry/fraud-model:1.2.3 (production)
   myregistry/fraud-model:1.2.2 (previous, rollback available)
   myregistry/fraud-model:1.3.0-rc1 (release candidate, staging)
   ```

4. **Deployment**: Deploy specific version
   ```
   kubectl set image fraud-model=myregistry/fraud-model:1.2.3
   ```

### Q4: "Design approval workflow for 100 models in production."
**Answer outline:** Workflow scales with automation:
1. **Automatic approval** if all gates pass:
   - Accuracy > threshold ✓
   - Latency < SLO ✓
   - Shadow test passed ✓
   - Fairness metrics acceptable ✓
   - All tests green ✓
   → Auto-approve, deploy to canary

2. **Manual approval** if flags raised:
   - Accuracy slightly above threshold (85% vs 84% required)
   - Latency at edge of SLO (99ms vs 100ms SLO)
   → Require 1 senior engineer approval

3. **Blocked** if critical failures:
   - Accuracy below threshold
   - Fairness metrics show bias
   - Tests failed
   → Require retraining

For 100 models: auto-approve 80% (no human intervention), manual 15% (quick engineer review), block 5% (needs retrain).

### Q5: "Model pipeline takes 2 hours (training + tests). How do you speed up feedback?"
**Answer outline:** Optimize each stage:
1. **Training**: Parallel hyperparameter search (reduce 1 hour → 20 min), early stopping, smaller validation set
2. **Unit tests**: Run fast tests only (skip integration), parallelize test suites (5 tests in parallel)
3. **Integration tests**: Subset of data (1 week instead of 1 year), mock external APIs
4. **Shadow test**: Skip pre-deployment shadow, do shadow in parallel with approval (not sequential)
5. **Build**: Docker layer caching (skip rebuilds of unchanged layers)

Result: 2 hours → 30 minutes. Enables iteration: train → validate → deploy in 30 minutes.

---

## Best Practices

1. **Automate everything possible**: Manual approvals are OK, manual deployments are not.

2. **Fail fast**: If model doesn't meet accuracy threshold, fail immediately, don't proceed.

3. **Separate concerns**: Training pipeline ≠ validation ≠ deployment. Orchestrate with workflow (Airflow, GitHub Actions).

4. **Track lineage**: Which training data created which model? Which model is production? Traceable.

5. **Approve before deploying**: Code review for models (approval gate) same as code review for code.

6. **Canary before rollout**: Never deploy to 100% without testing on small traffic first.

7. **Monitor continuously**: Metrics should auto-rollback if degrading, not rely on manual detection.

8. **Version everything**: Code (git), data (DVC), model (registry), and config (git). Reproducibility.

9. **Document rollback strategy**: If current version fails, which version to rollback to? How long does rollback take?

10. **Communicate changes**: When deploying, notify stakeholders (data scientists, product, ops). Don't surprise them.

---

## Common Pitfalls

1. **No versioning**: "Latest" is ambiguous. Multiple people training simultaneously creates confusion.

2. **Manual deployment**: Engineer trains, pushes to S3 by hand, tells ops "deploy this." Error-prone, no audit trail.

3. **No validation before production**: Deploy directly to 100% traffic, break something, debug in production.

4. **Slow feedback loop**: Train → wait 4 hours for tests → results. Iteration stalled.

5. **No rollback plan**: Deployed bad version. Previous version is "somewhere." Can't quickly rollback.

6. **Approval is rubber-stamp**: Engineers approve without reading metrics. Bad models go to production.

7. **No monitoring**: Model deployed, metrics not monitored. Degradation discovered by users, not alerts.

8. **All traffic at once**: Deploy to 100% traffic immediately. Hidden bugs affect all users.

9. **Pipeline is "black box"**: Engineer doesn't understand what pipeline does. Can't debug when it fails.

10. **No reproducibility**: Can't retrain exact model from 3 months ago (data/code versions lost).

---

## Real-World Examples

### Example 1: Netflix ML Pipeline
Netflix fully automates model deployment:
- **Training**: Spark job runs nightly, trains 100+ models
- **Validation**: Each model shadow-tested on 1 week production data
- **Approval**: Auto-approval if metrics passed, no human review needed
- **Canary**: 5% Netflix members see new recommendation for 24h
- **Rollout**: Successful → 100% rollout. Failure → auto-rollback.
- **Frequency**: Deploy 50+ new models/day

### Example 2: Stripe Fraud Model Safety
Stripe CI/CD prioritizes safety:
- **Training**: Daily fraud model retraining with latest data
- **Validation**: Shadow test compares new model to production on real fraud cases
- **Approval**: Manual code review (what changed in model?) + metrics review
- **Canary**: 1% of transactions with new fraud model, compared to baseline
- **Monitoring**: Alert immediately if false-positive rate increases >1%
- **Rollback**: Auto-rollback if fraud loss increases

### Example 3: Uber ETA Model Updates
Uber continuously improves ETA model:
- **Training**: Weekly retraining with latest traffic data
- **Validation**: Test on historical data, shadow test on real requests
- **Deployment**: Canary 1% of requests (24h), expand if MAE improved
- **Monitoring**: Track p99 latency, alert if increased >10%
- **Frequency**: Deploy ~3 new versions/week

---

## Sample Interview Case Study

**Scenario:** Build CI/CD for recommendation model. 50M users, requires approval before production.

**Pipeline:**

1. **Training stage** (2 hours):
   - Engineer pushes code
   - GitHub Actions triggers training job
   - Spark job trains on 2 weeks data
   - Output: model metrics (click-through rate, diversity)

2. **Validation stage** (1 hour):
   - Unit tests: model inference works?
   - Integration tests: with feature store?
   - Shadow test: model on last week's production data
   - Comparison: click-through rate within 1% of baseline?

3. **Approval stage** (30 min):
   - Create merge request with metrics
   - Require 2 engineer approvals
   - Checklist: accuracy acceptable? Shadow test good? Documentation?
   - Auto-comment with: "Ready to canary if approved"

4. **Canary stage** (24 hours):
   - Deploy model to 5% of users
   - Monitor: click-through rate, latency, user complaints
   - Automated alert: if click-through rate drops >1%

5. **Rollout stage** (3 days):
   - Day 1: 25% of users
   - Day 2: 50% of users
   - Day 3: 100% of users
   - Continuous monitoring

**Result:** Safe, auditable, fast (3 hours from code push to canary).

**Strong answer:** "Full CI/CD pipeline: training stage produces metrics, validation (shadow test on production data), approval gate (2 engineers check metrics), canary to 5% (24h monitoring), gradual rollout. Auto-rollback if any metric degrades >1%."

---

## Key Takeaways

CI/CD transforms ML deployment from risky to routine. Automated testing + approval gates + canary deployment = safe, fast iteration.

**Pipeline stages:** Train → Validate → Approve → Canary → Rollout → Monitor → (auto-rollback if needed)

**Common interview pattern:** "Model is ready. How do you deploy?" → Answer: "Full CI/CD: shadow test on production data, approval gate, canary to 5% traffic, monitor for degradation, auto-rollback if metrics drop."

---

## Related Concepts

- **Model Registry** (this concept): Stores models
- **Containerization** (Concept 13): Package models for deployment
- **Model Serving** (Concept 14): Serving infrastructure
- **Monitoring** (Concept 18): Monitor deployment metrics

---

## Quick Reference Card

### 2-Minute Elevator Pitch
Model Registry with CI/CD is the automation layer that transforms model deployment from a risky manual process into a reliable engineering practice. The registry is the source of truth: what models exist, which version is production, what was its training lineage. CI/CD is the automation: when a model trains, tests run automatically; when tests pass, approval gates open; when approved, canary deployment runs automatically. The key insight: a model deployment pipeline should look exactly like a software deployment pipeline — with tests, approvals, staged rollouts, and automatic rollback.

### Numbers to Know
- Netflix: deploys 50+ new models/day fully automated; average time from training to 100% production: 4 days
- Stripe: fraud model CI/CD pipeline: training (2h) + shadow test (7 days) + approval (2h) + canary (24h) = ~10 days total
- Uber: ~3 new model versions/week per major model; rollback time with proper registry: <5 minutes
- Auto-approval rate: well-designed gates auto-approve ~80% of model updates; 20% require human review
- Pipeline runtime target: training + validation + build should complete in <4 hours (enables same-day iteration)
- MLflow Model Registry: supports 10K+ models, <100ms latency for metadata queries
- Rollback SLA: industry standard for ML systems is <10-minute rollback capability
- Canary duration: 24 hours minimum, 7 days for safety-critical models (fraud, pricing, medical)

### Decision Framework: Automated vs. Manual Approval Gates

```mermaid
graph TD
    A["Model promotion decision"] --> B{"Risk level?"}
    B --> |"Low risk:<br/>metric improved,<br/>no architecture change"| C["Auto-approve if:<br/>primary metric improved &gt; 0.1%<br/>AND latency within SLO<br/>AND all automated tests pass<br/>AND shadow test ≥ baseline"]
    B --> |"Medium risk:<br/>new features added,<br/>retraining trigger"| D["Auto-approve gates +<br/>1 senior engineer review<br/>Reviews model card<br/>SLA: 2 business hours"]
    B --> |"High risk:<br/>architecture change,<br/>regulated domain,<br/>business-critical"| E["Full review required:<br/>2 senior engineers + PM<br/>+ Compliance if regulated<br/>SLA: 1 business day"]
    
    C --> F["Canary 5% for 24h<br/>Auto-expand if metrics hold<br/>Auto-rollback if metrics degrade"]
    D --> F
    E --> G["Canary 1% for 7 days<br/>Manual expand decision<br/>Manual rollback trigger"]
```

---

## Strong vs Weak Answers

### Q: Your team deploys ML models manually today. It takes 3-5 days per model and often results in production incidents. Design a CI/CD pipeline that reduces this to <4 hours with fewer incidents.

**Weak Answer:** "I would automate the deployment with GitHub Actions. The pipeline would train the model, run tests, and deploy to production automatically."

**Strong Answer:** "Manual deployment has three root causes for incidents: missing tests (problems discovered in production), environment drift (works locally, fails in production), and no staged rollout (one bad deployment affects all users). I'd fix all three. The pipeline: Code push → GitHub Actions triggers training job (2 hours on GPU runner) → automated validation (30 min: unit tests, integration tests with staging feature store, backtest on 14 days historical data, fairness check). If validation passes: build Docker image tagged with git commit + version (2 min, fast because dependency layer is cached) → push to ECR → create model registry entry with all metadata (metrics, git commit, dataset version). Approval gate: if all automated checks pass and metric improved >0.5%, auto-approve and deploy to canary (5% traffic). If metric improved 0-0.5% or any warning fires, require 1 human approval. Canary monitoring: 24-hour automated monitoring with auto-rollback if any metric degrades >2%. Total: 2h training + 30min tests + 24h canary = ~26h first deployment. For 90% of subsequent deployments (minor improvements): same pipeline, same timeline. The 3-5 day manual process compresses to automated reliability. Key: the pipeline is the deployment — no manual steps, no SSHing, no 'I'll just push this once without testing.'"

---

### Q: A model deployed 6 hours ago is causing false positives to spike. You need to rollback immediately. How does a proper CI/CD registry make this 5 minutes instead of 2 hours?

**Weak Answer:** "With versioning, I can quickly identify the previous model version and redeploy it. This is faster than trying to debug and fix the current issue."

**Strong Answer:** "Five-minute rollback requires all infrastructure to be pre-positioned. With a proper registry: Step 1 (30 seconds): alert fires on false positive spike. On-call engineer opens the model registry dashboard, sees the current production model is v23 (deployed 6 hours ago). Step 2 (30 seconds): one click to view v22's metadata — it was last in production 24 hours ago with false positive rate 0.3% (vs current 2.1%). Step 3 (1 minute): trigger rollback command — `mlflow.set_registered_model_alias('fraud_model', 'production', 'v22')`. The CI/CD system detects the registry change and triggers Kubernetes to switch the deployment's container image to the v22 image (already in ECR, no rebuild needed). Step 4 (2 minutes): Kubernetes rolling-updates pods from v23 to v22. Traffic shifts gradually. Step 5 (1 minute): confirm false positive rate returning to 0.3% in Grafana. Done. Without this infrastructure: find old code (5 min), find old training run (10 min), find old weights (10 min), rebuild container (5 min), deploy (10 min), verify (10 min) = 50 minutes minimum. And that's assuming you know exactly where everything is — which you often don't during an incident."

---

### Q: How do you prevent a model trained on corrupted or biased data from reaching production?

**Weak Answer:** "I would add data validation checks before training and data quality tests as part of the CI pipeline to block bad models."

**Strong Answer:** "Defense in depth: data quality checks at four points in the pipeline. Point 1 — Before training: Great Expectations suite validates schema (no unexpected nulls, correct types), completeness (all required features present for >99.9% of training samples), distribution (no unexpected shifts vs baseline), and data version integrity (hash of training data matches registry). If any check fails, training is halted. Point 2 — During training: monitor training metrics for signals of bad data — loss divergence (NaN or infinity indicates data corruption), unexpectedly fast convergence (possible label leakage), validation/training accuracy gap >15% (possible distribution mismatch). These halt training with a diagnostic report. Point 3 — Post-training: automated fairness checks on the trained model (not the data). If model shows >3% performance gap across demographic proxies, flag for human review. Backtesting on a temporal holdout verifies model performance degrades gracefully on out-of-distribution data (not a cliff). Point 4 — Registry gate: model metadata must include `training_data_version` and `validation_run_id`. The registry checks that the validation run passed all required checks before allowing staging promotion. A model with an unvalidated training data version cannot transition to production — no override without two-engineer approval and an incident ticket."

---

## System Design: End-to-End CI/CD Pipeline for a Safety-Critical ML Model

**Question:** "You're building CI/CD for Waymo's object detection model — one of the safest autonomous driving models in the world. Failures cause accidents. How do you design a CI/CD pipeline that can deploy model improvements continuously while maintaining the highest possible safety standards?"

**Walkthrough:**

1. **Risk-stratified deployment philosophy.** Unlike e-commerce recommendations, autonomous driving model failures have physical consequences. The pipeline must be asymmetric: it's OK for deployment to be slow (days, not hours) if it means fewer failures. The CI/CD design principle: every stage reduces the probability of a bad model reaching vehicle fleets.

2. **Staged model registry with strict transition criteria.** Five stages (not the usual 2-3): development (any model can exist), lab-validated (automated tests pass), simulation-tested (passes safety benchmarks in simulation), fleet-shadow (runs on 100 vehicles but doesn't control), production (active vehicle control). Each transition requires passing the previous stage's criteria plus new criteria.

3. **Automated lab testing pipeline (2-4 hours).** Unit tests: model produces valid bounding boxes (coordinates within frame, confidence scores in [0,1]). Regression suite: 10,000 labeled scenes from historical safety-critical scenarios (pedestrians at night, obscured signs, construction zones). Must match or exceed previous version on all scenario categories, with zero regressions on scenarios that previously caused safety flags.

4. **Simulation testing (8-24 hours).** Run new model in Waymo's simulation environment (millions of simulated miles). Metrics: collision rate, red-light violations, pedestrian interactions. Threshold: must achieve simulation safety score ≥ 99.99th percentile of previous model. Any simulation scenario where the new model performs worse than the current model blocks promotion.

5. **Safety review board approval.** Required for any model transition beyond lab validation. A 5-person safety review board (ML, perception, safety engineer, legal, ops) reviews: what changed from previous version, which test scenarios improved/degraded, simulation results. Approval documented with cryptographic signature. This is the human judgment layer — cannot be automated away for safety-critical systems.

6. **Fleet shadow deployment (7+ days).** New model runs on 100 test vehicles alongside current model. Current model controls the vehicle; new model's decisions are logged but don't affect control. Comparison: do the two models agree on critical decisions >99.9% of the time? Any systematic disagreement is analyzed before production promotion.

7. **Staged fleet rollout.** Production rollout: 1% of fleet (10 vehicles) × 7 days → 10% (100 vehicles) × 14 days → 50% × 30 days → 100%. At each stage, automated metrics comparison against baseline. Any stage can be paused or rolled back based on operational data. Total time from development to 100% fleet: 2-3 months.

8. **Automatic emergency rollback triggers.** Any safety event (collision, near-miss, regulatory violation) triggers immediate pause of rollout and review. If the new model version is associated with >2x the safety event rate of the baseline, automatic rollback across the entire active fleet within 1 hour. Emergency rollback doesn't require approval — it triggers automatically.

9. **Model registry audit trail.** Every model version stores: who approved it, when, with what evidence, what simulation results, what fleet shadow results, which safety scenarios it was tested on. This trail is immutable and retained permanently. In the event of an accident, regulators can review the complete decision trail for why a specific model version was deployed.

10. **Canary feedback loop to training.** Operational data from fleet shadow and production is automatically labeled (where the model was uncertain, where incidents occurred) and fed back to the training pipeline. The CI/CD system is not just a deployment system — it's a continuous learning system. New training runs automatically incorporate recent operational data, with the safety testing pipeline validating each resulting model.

**Key decisions:**
- 5-stage registry (not 2-3): extra stages add weeks to deployment but reduce the probability of a safety regression reaching the full fleet
- Shadow deployment before production: in safety-critical applications, shadow testing is not optional — it's the only way to validate model behavior on real-world conditions without risking harm
- No fully automated promotion to production: human judgment is required for physical-world AI systems; full automation is appropriate for recommendation systems, not autonomous vehicles
