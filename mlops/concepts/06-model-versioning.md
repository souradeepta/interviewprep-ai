# Model Versioning & Registry: Managing Models in Production

## Comprehensive Overview

Model versioning tracks trained model artifacts—weights, code, dependencies, metrics—enabling teams to deploy, rollback, and compare models. Without versioning, teams deploy a model to production, but 3 months later, can't remember which code created it, what data it trained on, or why accuracy was 95%. Model registry is the central place: what models exist, which version is in production, which are candidates for deployment, what's archived.

The cost of poor model versioning is catastrophic. A production model degrades: accuracy drops from 95% to 92%. Engineers need to rollback to the previous version, but it's lost—nobody remembered to save weights or code. With versioning, they find the previous version in the registry, deploy with one command, rollback in minutes. Without it, they rebuild the model from scratch, losing days.

Modern teams implement model registries (MLflow Model Registry, Amazon SageMaker Model Registry) that track: model artifacts (weights), parameters used, training data version, code commit, metrics, and metadata. Registries enable governance: approval workflows (only approved models go to production), access control (who can deploy), and audit trails (who deployed what, when).

The operational challenge is managing lifecycle: train hundreds of models, only a few are candidates, one is in production, old ones archived. Registry enables this: transition states (staging, production, archived), approval workflows, A/B testing infrastructure, and automatic rollback on metric degradation.

## How It Works

### Model Registry Lifecycle

```
Train Experiment
    ↓ (if metrics good)
Register Model in Registry
    ├─ Version: v1
    ├─ Metrics: accuracy=0.95
    ├─ Parameters: learning_rate=0.01
    ├─ Data version: training_set_v3
    └─ Code: git commit abc123
    ↓
Model Staging
    ├─ Test in staging environment
    ├─ Run validation suite
    ├─ Check dependencies (transformers==4.30)
    └─ Compare with current production (v0)
    ↓
Approval & Promotion
    ├─ Code review: did training code change?
    ├─ Data review: what data trained this?
    ├─ Metrics review: is accuracy better than prod?
    └─ Approval: ready to go
    ↓
Production Deployment
    ├─ Blue-green: deploy v1, keep v0 active
    ├─ Monitor: track metrics in prod
    ├─ Canary: 5% traffic to v1, 95% to v0
    └─ Validation: if metrics degrade, rollback to v0
    ↓
(When new model approved)
Archive Previous Version
    ├─ Keep for compliance/audit
    ├─ Don't serve traffic
    └─ Retain weights for analysis
```

```mermaid
graph TD
    A["Train<br/>Experiment"] --> |"Metrics OK?"| B{Accept?}
    B --> |"No"| C["Iterate<br/>New params"]
    C --> A
    B --> |"Yes"| D["Register<br/>Model Registry<br/>v5"]
    
    D --> E["Staging<br/>Run tests<br/>Validate"]
    E --> |"Pass?"| F{Tests OK?}
    F --> |"No"| C
    F --> |"Yes"| G["Approval<br/>Code review<br/>Metrics review"]
    
    G --> |"Approved?"| H{Ready?}
    H --> |"No"| C
    H --> |"Yes"| I["Deploy<br/>Canary 5%<br/>Monitor 24h"]
    
    I --> |"Good?"| J{Metrics OK?}
    J --> |"No"| K["Rollback<br/>to v4"]
    J --> |"Yes"| L["Expand<br/>25% → 50%<br/>→ 100%"]
    
    L --> |"Live"| M["Production<br/>v5 Active"]
    K --> N["Archive<br/>v4 Previous"]
    M --> N
```

### Registry Entry Example

```yaml
Model: fraud_detection
Version: v5
Status: production
Created: 2026-05-16
CreatedBy: fraud_team
Metrics:
  precision: 0.98
  recall: 0.92
  f1: 0.95
Parameters:
  learning_rate: 0.001
  batch_size: 64
  model_type: XGBoost
Artifacts:
  model_weights: s3://models/fraud_v5.pkl
  model_config: s3://models/fraud_v5_config.json
  feature_importance: s3://models/fraud_v5_importance.pkl
Data:
  training_set_version: fraud_training_v12
  training_date: 2026-05-15
  rows: 1_000_000
Code:
  git_commit: abc123def456
  git_branch: main
  training_script: train_fraud_model.py
Dependencies:
  transformers: "==4.30.0"
  xgboost: "==1.7.5"
  pandas: "==1.5.3"
History:
  v4: archived, 2026-04-15
  v3: archived, 2026-03-15
  v2: archived, 2026-02-15
```

## Tool Comparisons

| Tool | Approach | Strengths | Weaknesses | Best For |
|------|----------|-----------|-----------|----------|
| **MLflow Model Registry** | Open-source, Python-first | Simple, free, integrates with MLflow tracking | Limited governance features, less polished | Small teams, startups, on-prem |
| **Amazon SageMaker** | AWS-native, enterprise | Strong governance, audit trails, model approval | AWS lock-in, steep learning curve, expensive | AWS shops, regulated industries |
| **Hugging Face Model Hub** | Community-focused | Large model library, easy sharing, versioning | Limited to NLP/vision, less governance for internal models | NLP teams, research, model sharing |
| **Databricks Model Registry** | Delta Lake integrated | Strong lineage, unity catalog, governance | Databricks ecosystem lock-in | Databricks shops, data-heavy teams |
| **Custom (S3 + metadata DB)** | Build your own | Full control, minimal cost, flexible | High maintenance, inconsistency risk | Well-resourced teams with specific needs |

**Decision Framework:**
- **Small team:** MLflow (free, simple)
- **AWS ecosystem:** SageMaker (governance, audit)
- **Databricks shop:** Databricks Model Registry (Delta integration)
- **NLP/HuggingFace:** Hugging Face Model Hub
- **Custom governance:** Build custom (S3 + metadata store)

## Interview Q&A

**Q: Design a model versioning system for a company deploying 10+ models across multiple teams. What goes in the registry?**

A: Registry entries should contain: (1) Model metadata (name, version, status, created_by), (2) Metrics (accuracy, precision, latency from testing), (3) Parameters (hyperparameters used), (4) Artifacts (model weights path, config), (5) Data (training set version, date), (6) Code (git commit, branch, training script), (7) Dependencies (library versions), (8) History (previous versions). Enables reproducibility, governance, and rollback.

**Q: A production model's accuracy dropped from 95% to 92%. How do you rollback?**

A: (1) Check registry: find previous version (v-1) with 95% accuracy. (2) Verify: same architecture, different training data? (3) Load weights: fetch from registry (S3 path). (4) Deploy: use blue-green (v-1 in parallel, gradually switch traffic). (5) Monitor: track accuracy in production. (6) Investigate: what changed between v and v-1? (code, data, hyperparams).

**Q: Model registry contains 1000 models across 10 teams. How do you organize and govern it?**

A: Organization: (1) Hierarchical (team/model_name/version). (2) Tags (model_type, status, dataset). (3) Ownership (which team owns this?). Governance: (1) Approval workflow (only approved models to prod). (2) Access control (team A can't deploy team B's models). (3) Audit trail (who deployed what, when, why). (4) SLA (how long between training and deployment?). (5) Deprecation (how to retire old models).

**Q: How do you prevent deploying a model trained on bad data?**

A: Validation in registry: (1) Capture data version with model. (2) Data validation: before training, validate that data meets quality checks (schema, completeness, distributions). (3) On deployment: cross-check—if data version is blacklisted (bad quality), block deployment. (4) Code review: who approved training data? (5) Test: compare model output on test set vs production ground truth (should match).

**Q: How do you handle model dependencies changing (pandas 1.5 → 2.0)?**

A: Dependency management: (1) Log dependencies in registry (exact versions, not ranges). (2) Create docker image with exact versions, version the image. (3) On deployment: use pinned image, not latest. (4) Upgrade strategy: test new version on canary traffic, validate metrics match, only then promote. (5) Compatibility: document breaking changes (pandas 2.0 has API changes). (6) Rollback: if new version breaks, revert to old image.

## Best Practices

1. **Semantic Versioning:** Use major.minor.patch (v1.2.3) to indicate breaking changes vs improvements.

2. **Rich Metadata:** Capture more than weights. Log code, data, dependencies, metrics, training date.

3. **Governance Workflow:** Don't allow manual deployments. Require approval before moving to production.

4. **Immutable Artifacts:** Once deployed, don't change. Create new version if changes needed.

5. **Audit Trail:** Log who deployed what, when, and why. Required for compliance.

6. **Monitor After Deployment:** Track production metrics. If degradation detected, automate rollback.

7. **Archive Old Versions:** Keep for compliance, but don't serve traffic. Saves compute.

8. **Test Before Deploy:** Staging environment with same data, traffic patterns as production.

## Common Pitfalls

1. **Lost Weights:** Trained model but forgot to save. Can't rollback.

2. **Missing Metadata:** Registered model but no data version or git commit. Can't reproduce.

3. **Version Explosion:** Too many versions clutters registry. Archive aggressively.

4. **No Audit Trail:** Can't tell who deployed what or why. Compliance risk.

5. **Dependency Hell:** Didn't pin library versions. Production breaks when dependencies upgrade.

6. **Manual Deployments:** Engineer manually copies model weights. Inconsistent, risky.

7. **No Rollback Plan:** Deployed new model, can't go back. Production broken for hours.

## Real-World Examples

### Netflix: Model Registry for Recommendations

Netflix's registry tracks 100+ recommendation models:
- Versions: daily updates, archive old after 30 days
- Governance: approval required before prod deployment
- Deployment: canary (5% traffic to new, 95% to old), monitor for 24h
- Rollback: automated if metrics degrade >1%
- Audit: logs all deployments for compliance

### Stripe: Model Registry for Fraud

Stripe's registry tracks fraud models with strict governance:
- Approval: code review + data review required
- Testing: validated on holdout test set before deployment
- Deployment: blue-green (both versions running), switch gradually
- Monitoring: false positive rate tracked in production
- Rollback: automated if false positive rate spikes >0.5%

### Uber: Multi-Model Registry

Uber tracks pricing, matching, ETA models:
- Teams: each team manages their models
- Registry: centralized (can see all models across Uber)
- Status: staging (validated), production (live), archived
- Rollback: one-click rollback to previous version
- Integration: automatic A/B test infrastructure

## Sample Interview Questions

1. "Design a model versioning system for a company with 50 models across 5 teams."

2. "Production model's accuracy dropped. Rollback is blocked. How do you troubleshoot?"

3. "How do you prevent deploying a model trained on corrupted data?"

## Interview Case Study

**Scenario:** You're deploying ML models at a large company. Design a model registry that enables: (1) Reproducibility, (2) Governance, (3) Rollback, (4) Audit trail.

**Solution Walkthrough:**

1. **Registry Metadata:**
   - Model info: name, version, status (staging/production/archived)
   - Metrics: accuracy, precision, latency from validation
   - Parameters: hyperparameters used
   - Artifacts: where model weights stored (S3 path)
   - Data: training set version, date
   - Code: git commit, branch, training script
   - Dependencies: library versions (pinned)
   - Created by, created at, approved by

2. **Workflow:**
   ```
   Train experiment → Log to tracking system (accuracy=0.95)
       ↓
   Register in model registry (v1, status=staging)
       ↓
   Validation (test on holdout set, check data quality)
       ↓
   Code review + Data review (approval gate)
       ↓
   Deploy to staging (run integration tests)
       ↓
   A/B test (canary: 5% traffic to v1, 95% to v0)
       ↓
   Monitor metrics (accuracy, latency, cost)
       ↓
   If metrics good → Promote v1 to production (v0 archived)
       ↓
   If metrics bad → Rollback to v0 (v1 archived)
   ```

3. **Governance:**
   - Approval: required before prod (code review, data review)
   - Access control: only authorized teams can deploy
   - Audit: log who deployed what, when, why

4. **Reproducibility:**
   - Fetch model: look up v5 in registry
   - Download weights: s3://models/v5.pkl
   - Check dependencies: transformers==4.30, pandas==1.5.3
   - Load code: git checkout abc123
   - Run model: should produce same output as in production

5. **Rollback:**
   - Metrics degrade in production
   - Trigger: if accuracy < v(n-1) by >1%
   - Action: deploy v(n-1), archive v(n)
   - Validation: verify accuracy returns to baseline

**Strong vs Weak Answers:**

Strong: "Registry should contain: model metadata, metrics, parameters, artifacts (S3 path), data version, git commit, pinned dependencies. Workflow: train → register (staging) → approve → deploy (canary) → monitor. Governance: approval gates, audit trail. Rollback: automated if metrics degrade."

Weak: "Store model weights in S3." (No metadata, no governance, no versioning)

---

## Related Concepts

- **Concept 05:** Experiment Tracking — Logging training runs
- **Concept 07:** Reproducibility — Reproducing exact results
- **Concept 08:** Hyperparameter Optimization — Finding best parameters

## Resources

- MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
- SageMaker Model Registry: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html
- Databricks Model Registry: https://docs.databricks.com/machine-learning/model-registry/

---

## Quick Reference Card

### 2-Minute Elevator Pitch
Model versioning is the organizational backbone that makes model deployment reliable and auditable. A model version is not just a file in S3 — it's a complete record: weights, parameters, training data version, code commit, dependencies, metrics, and deployment history. Without this record, a production incident becomes a forensic investigation: "What model is running? What data trained it? Can we roll back?" With a proper registry, the answer to all three is available in under 60 seconds. The key governance insight: every model promotion requires approval, and every model should have an explicit deprecation plan.

### Numbers to Know
- MLflow Model Registry: stores unlimited models, metadata is lightweight (~1KB per version), artifacts in S3/GCS
- SageMaker Model Registry: supports 250 model groups per account, up to 10,000 model versions per group
- Rollback time with proper versioning: <5 minutes (fetch previous version's S3 path, redeploy container)
- Without versioning: typical rollback takes 2-8 hours (find old code, retrain, redeploy)
- Netflix: versions 100+ recommendation models; each has 30 days of version history retained
- Stripe: all fraud model versions retained 7 years (regulatory requirement)
- Storage cost: 1GB model artifact × 10 versions × 100 models = 1TB = ~$23/month on S3

### Decision Framework: What to Store in a Model Registry Entry

```mermaid
graph TD
    A["Model Registry Entry"] --> B["Model Identity"]
    A --> C["Reproducibility Info"]
    A --> D["Quality Evidence"]
    A --> E["Operational Metadata"]
    A --> F["Governance Trail"]
    
    B --> B1["name, version, status<br/>staging|production|archived<br/>created_by, created_at"]
    C --> C1["git_commit hash<br/>data_version hash<br/>dependency_versions pinned<br/>random_seed"]
    D --> D1["primary_metric value<br/>guardrail metrics<br/>test_set_performance<br/>fairness metrics"]
    E --> E1["artifact_s3_path<br/>docker_image_tag<br/>inference_latency_p99<br/>model_size_bytes"]
    F --> F1["approved_by<br/>approval_date<br/>deployment_history<br/>incident_history"]
```

---

## Strong vs Weak Answers

### Q: A production fraud model's accuracy dropped from 95% to 88% after last night's deployment. Your on-call rotation needs to resolve this in 30 minutes. Walk me through the rollback process.

**Weak Answer:** "I would roll back to the previous model version. I would find the previous model weights and redeploy them."

**Strong Answer:** "With a proper model registry, this is a 5-step process that takes under 10 minutes. Step 1 (1 minute): query the registry: `mlflow.search_model_versions("name='fraud_detector' AND tags.status='production'")` — identifies the currently running version (v15) and the previous production version (v14). Step 2 (1 minute): verify v14 is still viable — check its production deployment metrics from when it was last live (accuracy 95%, latency 45ms, no known issues). Step 3 (2 minutes): issue rollback command to Kubernetes: `kubectl set image deployment/fraud-model fraud-model=registry/fraud-model:v14`. The old container is already in the registry; no rebuild needed. Step 4 (2 minutes): verify rollback — monitor accuracy metric in Grafana; should return to baseline within 2-3 minutes as traffic shifts to v14. Step 5 (4 minutes): page the model owner with incident details — which version failed, when it was deployed, what metric degraded. Simultaneously, tag v15 as `status=incident` in the registry with the incident timestamp. The post-mortem can wait until morning. The 30-minute clock is for restoration; root cause analysis is separate."

---

### Q: Design a governance process for model deployment when 10 teams are sharing a single model registry.

**Weak Answer:** "I would have a review process where engineers check the model before deploying it to production. Each team would have their own namespace in the registry."

**Strong Answer:** "Governance at 10-team scale requires both technical controls and process controls. Technical: namespace isolation (each team's models are in `{team_name}/{model_name}` — only team members can write, anyone can read), deployment locks (only CI/CD service account can transition models to production status, not individual engineers), and audit logging (every status change, approval, and deployment logged immutably). Process: a four-gate approval workflow for production promotion. Gate 1 (automated): accuracy threshold check, latency benchmark, fairness metrics within bounds — automated, no human needed. Gate 2 (automated): shadow test on last 7 days of production traffic — model's predictions are compared to the incumbent; if primary metric is within 2% of incumbent, automatic gate pass. Gate 3 (human): senior ML engineer from the team reviews the model card (what changed from previous version, why, expected impact) and approves. Gate 4 (optional, for high-risk models): security review for models touching PII or financial decisions. This process: automatically approves ~60% of models (low-risk, clear improvement), requires 1 human review for ~35%, and blocks ~5% for rework. Average time from training to production: 4 hours."

---

### Q: How do you handle the situation where a dependency update (PyTorch 2.0 → 2.1) breaks an older model version you might need to roll back to?

**Weak Answer:** "I would pin the dependencies in the requirements file so the model always uses the same library versions."

**Strong Answer:** "Pinning dependencies is necessary but not sufficient — the problem is that production environments evolve, and a model registered months ago with PyTorch 2.0 may fail to run in a PyTorch 2.1 environment if you need to roll back. The solution has three components. First, container-level versioning: every model version is associated with a specific Docker image tag (e.g., `fraud-model-base:pytorch2.0.1`), and the registry stores this image tag alongside the model weights. Rollback means deploying the exact same container, not just the same weights. Second, base image retention policy: base images are retained for 2 years minimum (or as long as the model that depends on them is retained), regardless of whether they're the 'current' base image. Third, compatibility testing: when upgrading the production environment (new PyTorch version), run compatibility tests on all currently-registered production models and the previous 2 versions. Any model that breaks gets flagged; the team either upgrades it or accepts that rollback to that version would require a container rebuild. This is how Stripe manages fraud model compatibility — they discovered a PyTorch 1.11 → 1.12 change that broke a serialized model format, and the container registry policy prevented a production incident."

---

## System Design: Model Registry for Enterprise Multi-Team ML Platform

**Question:** "You're the ML platform team at an insurance company (think Allstate). The company has 40 ML models across 8 teams: claims fraud, pricing, underwriting, churn prediction, customer segmentation, document classification, image damage assessment, and chatbot. Models affect real financial decisions and are subject to state insurance regulations. Design the model registry and governance system."

**Walkthrough:**

1. **Registry architecture.** Use MLflow Model Registry with PostgreSQL backend (for audit-grade logging) and AWS S3 for artifact storage. Separate registries for: experiment tracking (all models, all versions, lightweight), and production registry (only approved models, immutable, full audit trail). The separation prevents experiment clutter from affecting production operations.

2. **Model card as a required artifact.** Every model version must include a standardized model card: model purpose, training data description, known limitations, fairness evaluation results, performance on protected class subgroups (required for insurance), expected drift schedule, and responsible engineer contact. The registry validates model card presence before allowing staging-to-production transition.

3. **Four-environment lifecycle.** Development (experiment tracking, no governance, full team access) → Staging (automated tests pass, model card complete) → Shadow (model runs on production traffic without affecting decisions, 7-day minimum) → Production (approved, deployed, monitored). Each transition is a status change in the registry with a required approver log.

4. **Regulatory compliance fields.** Insurance models are regulated by state insurance departments. Registry stores: (a) regulatory jurisdiction (which state laws apply), (b) disparate impact test results (model performance must not differ by race, gender, geography beyond legal thresholds), (c) model explainability artifacts (SHAP values for underwriting decisions — required for adverse action notices), (d) audit trail for 7 years (regulatory requirement).

5. **Automated quality gates.** Before any model can transition to staging: primary metric must exceed a team-defined threshold, latency p99 must be below SLO, fairness metrics must pass disparate impact tests, model card must be complete. These run automatically on every registered model. Gate failures generate issues in Jira, not just notifications — they require resolution, not just acknowledgment.

6. **Shadow testing infrastructure.** For insurance models, shadow testing is mandatory (not optional). During shadow phase: the new model runs alongside production, both models score the same inputs, but only the production model's output affects decisions. Shadow metrics are tracked in a dashboard; a 7-day minimum shadow period catches weekly behavioral patterns.

7. **Rollback SLA.** Registry design requirement: any model must be rollback-able within 10 minutes by any on-call engineer (not just the model owner). This means: rollback procedure is documented in the registry entry, previous version's container is retained and ready to deploy, and rollback requires one-click approval rather than a deployment pipeline run.

8. **Access control.** Role-based access: data scientists (read all, write own team's staging), senior data scientists (write all staging, approve own team's production), ML platform engineers (write all production, manage infrastructure), compliance officer (read all production metadata, no write access). Access audit logs stored for 7 years.

9. **Model deprecation workflow.** Every model in the registry must have a deprecation plan: when will it be replaced, what will replace it, who is responsible? Models without deprecation plans after 18 months trigger automatic alerts. Before archiving, the compliance team must confirm regulatory obligations are met (some jurisdictions require models to be available for 5 years after last use).

10. **Integration with incident management.** Every production incident that involves an ML model creates an automatic registry annotation: incident ID, timestamp, metric degradation magnitude, root cause (once identified), and time to resolution. Over time, this builds a model reliability history that informs deployment risk scoring for future versions.

**Key decisions:**
- Separate experiment and production registries: prevents experiment noise from cluttering production governance; different access control requirements
- Model cards as deployment blockers: without enforcement, model cards don't get written; making them a gate ensures compliance documentation exists before regulatory examination
- Shadow testing as mandatory for regulated models: in insurance, "it worked in testing" is not sufficient justification — shadow testing on real data is the only evidence regulators accept
