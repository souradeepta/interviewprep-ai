"""
Auto-generated from 15-model-registry.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Model Registry & CI/CD: Automating Safe Model Deployment
# ## Learning Objectives
# - Design automated model deployment pipelines
# - Implement approval gates and safety checks
# ======================================================================

# ======================================================================
# ## Basic: Model Registry + Versioning
# ======================================================================

registry_example = '''
Model Registry Workflow:

STEP 1: Register model
  mlflow.models.log(model)
  Result: MLflow creates entry
    - Model ID: fraud-detector
    - Version: 1.0.0
    - Metrics: accuracy=0.95, precision=0.98
    - Status: staging

STEP 2: Validate in staging
  - Run shadow test on production data (1 week)
  - Compare predictions to baseline
  - If accuracy good: mark as ready

STEP 3: Request approval
  - Create review request
  - Require 2 senior engineers
  - Check: metrics passing? Documentation complete? Risks identified?

STEP 4: Approve and promote
  - Status: staging → approved
  - Schedule canary deployment

STEP 5: Canary
  - 5% traffic on new model
  - Monitor for 24h
  - If good: expand. If bad: rollback instantly.

STEP 6: Archive previous version
  - Status: production → archived
  - Keep for history and rollback
'''

print("MODEL REGISTRY WORKFLOW")
print()
print(registry_example)


# ======================================================================
# ## Advanced: CI/CD Pipeline with Approval Gates
# ======================================================================

cicd_pipeline = '''
GitHub Actions CI/CD Pipeline for Model Deployment

on: [push]

jobs:
  train:
    runs-on: gpu-runner
    steps:
      - name: Train model
        run: python train.py
      - name: Validate accuracy
        run: python validate.py
        # Fail if accuracy < threshold
  
  validate:
    needs: train
    steps:
      - name: Unit tests
        run: pytest tests/
      - name: Integration tests
        run: python integration_test.py
      - name: Shadow test
        run: python shadow_test.py
        # Compare to baseline on 1 week production data
  
  build:
    needs: validate
    steps:
      - name: Create Docker image
        run: docker build -t fraud:abc123 .
      - name: Push to registry
        run: docker push myregistry.com/fraud:abc123
  
  approval:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Create review request
        run: gh pr create --body "Ready for approval"
      - name: Wait for approval
        run: gh pr status --checks
        # Blocks until 2 approvals
  
  canary:
    needs: approval
    steps:
      - name: Deploy to 5% traffic
        run: kubectl set image fraud=myregistry/fraud:abc123
      - name: Monitor for 24h
        run: python monitor.py
      - name: If metrics good, expand
        run: kubectl apply -f 25-percent-rollout.yaml
'''

print("CI/CD PIPELINE WITH APPROVAL GATES")
print()
print(cicd_pipeline)
print()
print("Key features:")
print("- Train: automated, fails on bad accuracy")
print("- Validate: unit + integration + shadow tests")
print("- Build: Docker image tagged with commit hash")
print("- Approval: requires 2 human reviews")
print("- Canary: 5% traffic, 24h monitoring")


# ======================================================================
# ## Real-World Examples: Netflix, Stripe, Uber
# ======================================================================

def netflix_model_registry():
    """Manage model versions in central registry"""

    print("NETFLIX: Model Registry")
    print("=" * 60)

    print("\nVERSION LIFECYCLE:")
    print("  Development: training in progress")
    print("  Staging: deployed to staging, tested")
    print("  Production: serving live traffic (90%)")
    print("  Archived: kept for 30 days (rollback)")
    print("  Deleted: removed after 30 days")

    print("\nMODEL APPROVAL GATES:")
    print("  ✓ Code review (2 approvals)")
    print("  ✓ Unit tests (100% coverage)")
    print("  ✓ Integration tests (1000+ test cases)")
    print("  ✓ Accuracy > baseline + 0.5%")
    print("  ✓ Fairness: <2% variance across regions")
    print("  ✓ Latency: p99 < 50ms")

    print("\nAUTO-ROLLBACK TRIGGERS:")
    print("  Accuracy drops >1%: ROLLBACK")
    print("  Latency p99 >100ms: ROLLBACK")
    print("  Error rate >5%: ROLLBACK")
    print("  OOM events: ROLLBACK")

def stripe_model_registry():
    """Track fraud models across environments"""

    print("\nSTRIPE: Fraud Model Registry")
    print("=" * 60)

    print("\nREGISTRY CONTENTS:")
    print("  Model ID: fraud_v3")
    print("  Type: XGBoost")
    print("  Training date: 2026-05-16")
    print("  Training size: 500M transactions")
    print("  Validation accuracy: 94.5%")
    print("  Fairness check: PASS")

    print("\nSTAGING DEPLOYMENT:")
    print("  Status: Running")
    print("  Traffic: 1% (canary)")
    print("  Metrics: accuracy=94.2%, latency=35ms")
    print("  Duration: 3 days")
    print("  Decision: Promote to production (metrics stable)")

    print("\nPRODUCTION DEPLOYMENT:")
    print("  Model: fraud_v2 (current)")
    print("  Traffic: 99%")
    print("  Accuracy: 93.2%")
    print("  Latency: p99=40ms")

def uber_model_registry():
    """Multi-model registry (ETA, matching, pricing)"""

    print("\nUBER: Multi-Model Registry")
    print("=" * 60)

    print("\nMODELS TRACKED:")
    print("  1. ETA Model (20 versions)")
    print("     Current: eta_v12 (production)")
    print("     Latest: eta_v13 (staging)")
    print()
    print("  2. Matching Model (15 versions)")
    print("     Current: matching_v8 (production)")
    print("     Latest: matching_v9 (staging)")
    print()
    print("  3. Pricing Model (25 versions)")
    print("     Current: pricing_v20 (production)")
    print("     Latest: pricing_v21 (development)")

    print("\nCOORDINATION:")
    print("  Weekly: Review all models for promotion")
    print("  Daily: Monitor production metrics")
    print("  Monthly: Retraining on latest data")
    print("  Rollback: <5 minutes if metrics degrade")

netflix_model_registry()
stripe_model_registry()
uber_model_registry()



# ======================================================================
# ## Interview Case Study: Designing Registry & CI/CD
# ======================================================================

case_study = '''
SCENARIO: Fraud model deployment for payment processor
- Current model: 96% detection rate
- New model: 97% detection rate (1% improvement)
- Risk: higher detection might increase false declines

DESIGN SAFE DEPLOYMENT:

1. TRAINING & VALIDATION
   - Train new model
   - Validate: accuracy > 96% ✓
   - Shadow test: run on 1 month production fraud data
     - Compare: detection 97%, false decline +0.5% (acceptable)
     - Result: passed

2. REGISTER & APPROVE
   - Register in MLflow: fraud-detector:v1.1.0
   - Create approval request with checklist:
     ✓ Accuracy improved
     ✓ Shadow test passed (detection 97%)
     ✓ False decline rate acceptable (+0.5% < +1% threshold)
     ✓ Risk assessment: medium (operational, not critical)
   - Require 2 approvals from senior engineers

3. CANARY DEPLOYMENT
   - Deploy to 1% of transaction processing
   - Monitor for 24 hours:
     - Detection rate: 97% (good)
     - False decline rate: +0.4% (within +1% threshold)
     - No errors
   - Auto-rollback threshold: detection < 96% or decline > +1%

4. GRADUAL ROLLOUT
   - 1% → 5% (24h) if metrics stable
   - 5% → 25% (24h) if metrics stable
   - 25% → 100% (after 72h total, all metrics stable)

5. POST-DEPLOYMENT
   - Continue monitoring
   - Archive previous version (v1.0.0) for rollback
   - Weekly review: detection rate trend, false decline trend

RESULT: Safe, auditable, reversible deployment
'''

print(case_study)


# ======================================================================
# ## Key Takeaways
# **CI/CD transforms ML:** Automated testing + approval gates = safe iteration.
# **Approval gates prevent bad models:** Not every model that passes tests should deploy.
# ======================================================================
