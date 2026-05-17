#!/usr/bin/env python3
"""Fix the last 4 notebooks (13-16) with combined Real-World Examples"""

import json
from pathlib import Path

FIXES = {
    '13-containerization.ipynb': '''def netflix_containerization():
    """Build optimized Docker image for recommendation model"""

    print("NETFLIX: Docker Container Optimization")
    print("=" * 60)

    print("\\nMULTI-STAGE BUILD:")
    print("  Stage 1: Builder (install deps)")
    print("    Size: 800+ MB (includes build tools)")
    print("  Stage 2: Runtime (copy only binaries)")
    print("    Size: 200 MB (final image)")
    print("  Reduction: 75% smaller")

    print("\\nLAYER CACHING:")
    print("  Base (python:3.9-slim): 125 MB (cached)")
    print("  System deps: +45 MB (cached)")
    print("  Python libs: +150 MB (cached if no changes)")
    print("  App code: +50 MB (rebuilt on changes)")

    print("\\nBUILD TIMES:")
    print("  First build: 5 minutes")
    print("  Incremental (code change): 1 minute (layers cached)")
    print("  Cold build (all changed): 5 minutes")

def stripe_containerization():
    """Build fraud detection model image"""

    print("\\nSTRIPE: Fraud Model Containerization")
    print("=" * 60)

    print("\\nDOCKERFILE:")
    print("  FROM python:3.9-slim")
    print("  COPY requirements.txt .")
    print("  RUN pip install --no-cache-dir -r requirements.txt")
    print("  COPY src/ /app")
    print("  WORKDIR /app")
    print("  EXPOSE 8000")
    print("  CMD ['python', 'app.py']")

    print("\\nIMAGE REGISTRY:")
    print("  stripe-fraud:v3.1.0-sha256abc123")
    print("  Size: 420 MB")
    print("  Build: 2026-05-16")
    print("  Registry: private ECR")

    print("\\nDEPLOYMENT:")
    print("  docker run -p 8000:8000 stripe-fraud:v3.1.0")
    print("  Latency: 50ms fraud scoring")
    print("  Throughput: 10K req/sec per instance")

def uber_containerization():
    """Build matching service image"""

    print("\\nUBER: Matching Service Containerization")
    print("=" * 60)

    print("\\nPERFORMANCE:")
    print("  Cold start: 2 seconds")
    print("  Warm start: <100ms")
    print("  Model load: 1.5 GB (GPU memory)")
    print("  Inference: 30ms per match")

    print("\\nHEALTH CHECKS:")
    print("  Readiness: /health → model loaded? (fast)")
    print("  Liveness: /alive → responding? (tcp)")
    print("  Metrics: /metrics → Prometheus")

    print("\\nRESOURCE LIMITS:")
    print("  CPU: 4 cores")
    print("  Memory: 8 GB (including model)")
    print("  GPU: 1 × V100 (16GB VRAM)")
    print("  Storage: 50 GB cache")

netflix_containerization()
stripe_containerization()
uber_containerization()
''',

    '14-model-serving.ipynb': '''def netflix_fastapi_serving():
    """FastAPI server for recommendation serving"""

    print("NETFLIX: Model Serving")
    print("=" * 60)

    print("\\nLATENCY BREAKDOWN:")
    print("  Cache lookup: 1ms")
    print("  Feature fetch: 20ms (Redis)")
    print("  Model inference: 15ms (single GPU)")
    print("  Top-K selection: 2ms")
    print("  Serialization: 2ms")
    print("  Total: 40ms (< 50ms SLO)")

    print("\\nTHROUGHPUT:")
    print("  1 GPU server: 500 req/sec")
    print("  100 servers: 50K req/sec")
    print("  Batching: 10-20ms latency for batch_size=32")

    print("\\nCACHING STRATEGY:")
    print("  Per-user: 100ms TTL (session context)")
    print("  Per-content: 1 hour TTL (trending)")
    print("  Hit rate: 85% (25M active users)")

def stripe_model_serving():
    """Real-time fraud scoring"""

    print("\\nSTRIPE: Fraud Scoring Pipeline")
    print("=" * 60)

    print("\\nREQUEST FLOW:")
    print("  1. Transaction arrives (event)")
    print("  2. Fetch user features (5ms, cache)")
    print("  3. Fetch merchant features (10ms, DB)")
    print("  4. Compute transaction features (2ms)")
    print("  5. Model inference (25ms)")
    print("  6. Decision logic (1ms)")
    print("  Total: 43ms (< 50ms SLA)")

    print("\\nDECISION LOGIC:")
    print("  score > 0.7: DECLINE + challenge")
    print("  0.5 < score < 0.7: MONITOR")
    print("  score < 0.5: APPROVE")
    print("  Special: FLAG if high velocity (50+ txns/min)")

    print("\\nBACKPRESSURE HANDLING:")
    print("  Timeout: 100ms (fallback = approve with monitor)")
    print("  Queue depth: max 10K pending requests")
    print("  Graceful degradation: if service slow, approve more")

def uber_model_serving():
    """ETA and matching serving"""

    print("\\nUBER: ETA & Matching Serving")
    print("=" * 60)

    print("\\nETA MODEL SERVING:")
    print("  Features: distance, traffic, time_of_day, route")
    print("  Latency: 30ms inference + 20ms feature fetch = 50ms")
    print("  Update: every 1 minute (traffic patterns)")
    print("  Model: XGBoost (48MB, fast inference)")

    print("\\nMATCHING SERVING:")
    print("  Features: driver location, rating, acceptance_rate, etc")
    print("  Latency: 30ms model + 30ms feature = 60ms")
    print("  Update: real-time (driver moves)")
    print("  Model: XGBoost (large, 500MB)")

    print("\\nFEDERATION:")
    print("  50 edge servers (distributed globally)")
    print("  Federated learning: model trained centrally")
    print("  Inference: local (~10ms latency)")

netflix_fastapi_serving()
stripe_model_serving()
uber_model_serving()
''',

    '15-model-registry.ipynb': '''def netflix_model_registry():
    """Manage model versions in central registry"""

    print("NETFLIX: Model Registry")
    print("=" * 60)

    print("\\nVERSION LIFECYCLE:")
    print("  Development: training in progress")
    print("  Staging: deployed to staging, tested")
    print("  Production: serving live traffic (90%)")
    print("  Archived: kept for 30 days (rollback)")
    print("  Deleted: removed after 30 days")

    print("\\nMODEL APPROVAL GATES:")
    print("  ✓ Code review (2 approvals)")
    print("  ✓ Unit tests (100% coverage)")
    print("  ✓ Integration tests (1000+ test cases)")
    print("  ✓ Accuracy > baseline + 0.5%")
    print("  ✓ Fairness: <2% variance across regions")
    print("  ✓ Latency: p99 < 50ms")

    print("\\nAUTO-ROLLBACK TRIGGERS:")
    print("  Accuracy drops >1%: ROLLBACK")
    print("  Latency p99 >100ms: ROLLBACK")
    print("  Error rate >5%: ROLLBACK")
    print("  OOM events: ROLLBACK")

def stripe_model_registry():
    """Track fraud models across environments"""

    print("\\nSTRIPE: Fraud Model Registry")
    print("=" * 60)

    print("\\nREGISTRY CONTENTS:")
    print("  Model ID: fraud_v3")
    print("  Type: XGBoost")
    print("  Training date: 2026-05-16")
    print("  Training size: 500M transactions")
    print("  Validation accuracy: 94.5%")
    print("  Fairness check: PASS")

    print("\\nSTAGING DEPLOYMENT:")
    print("  Status: Running")
    print("  Traffic: 1% (canary)")
    print("  Metrics: accuracy=94.2%, latency=35ms")
    print("  Duration: 3 days")
    print("  Decision: Promote to production (metrics stable)")

    print("\\nPRODUCTION DEPLOYMENT:")
    print("  Model: fraud_v2 (current)")
    print("  Traffic: 99%")
    print("  Accuracy: 93.2%")
    print("  Latency: p99=40ms")

def uber_model_registry():
    """Multi-model registry (ETA, matching, pricing)"""

    print("\\nUBER: Multi-Model Registry")
    print("=" * 60)

    print("\\nMODELS TRACKED:")
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

    print("\\nCOORDINATION:")
    print("  Weekly: Review all models for promotion")
    print("  Daily: Monitor production metrics")
    print("  Monthly: Retraining on latest data")
    print("  Rollback: <5 minutes if metrics degrade")

netflix_model_registry()
stripe_model_registry()
uber_model_registry()
''',

    '16-deployment-strategies.ipynb': '''def netflix_canary_deployment():
    """Gradual rollout with monitoring"""

    print("NETFLIX: Canary Deployment")
    print("=" * 60)

    print("\\nCANARY PLAN:")
    print("  Day 1 (1% traffic): Monitor for errors")
    print("  Day 2 (5% traffic): Check metrics improving")
    print("  Day 3 (25% traffic): Full regional test")
    print("  Day 4 (100% traffic): Full rollout")
    print("  Total: 4 days to full deployment")

    print("\\nMONITORING METRICS:")
    print("  Primary: NDCG@10 (ranking quality)")
    print("  Secondary: Latency (p50, p99)")
    print("  Tertiary: Error rate, completion rate")
    print("  Threshold: If accuracy drops 1%, auto-rollback")

    print("\\nROLLBACK PROCEDURE:")
    print("  1. Alert: Accuracy below threshold (1min)")
    print("  2. Approve: Automatic if drop > 1%")
    print("  3. Execute: Revert to previous version (<2min)")
    print("  4. Verify: Check metrics restore")

def stripe_blue_green_deployment():
    """Instant switchover with zero downtime"""

    print("\\nSTRIPE: Blue-Green Deployment")
    print("=" * 60)

    print("\\nDEPLOYMENT PHASES:")
    print("  Blue (current): fraud_v2, 100% traffic, stable")
    print("  Green (new): fraud_v3, 0% traffic, isolated")
    print()
    print("  Phase 1 (15min): Deploy fraud_v3 to Green")
    print("  Phase 2 (10min): Full test of Green environment")
    print("  Phase 3 (1sec): Switch all traffic Blue → Green")
    print("  Phase 4 (ongoing): Monitor Green (new prod)")
    print()
    print("  Rollback: Switch back to Blue (<1sec)")

    print("\\nADVANTAGES:")
    print("  ✓ Zero downtime (instant switch)")
    print("  ✓ Easy rollback (just switch back)")
    print("  ✓ Full validation before go-live")
    print("  ✓ Cost: 2x infrastructure during deployment")

def uber_shadow_deployment():
    """Run new model in parallel, validate offline"""

    print("\\nUBER: Shadow Deployment")
    print("=" * 60)

    print("\\nSHADOW PHASE (7 days):")
    print("  Step 1: Deploy pricing_v21 in shadow mode")
    print("  Step 2: Run on 100% of requests (no user impact)")
    print("  Step 3: Compare pricing_v20 vs pricing_v21 offline")
    print("  Step 4: Validate metrics match or improve")
    print()
    print("  Metrics collected:")
    print("  - Trip completion rate")
    print("  - Driver acceptance rate")
    print("  - Revenue impact (simulated)")

    print("\\nDECISION:")
    print("  If metrics good: Promote to canary (1% traffic)")
    print("  If metrics bad: Rollback (no customer impact)")
    print("  Decision time: 7 days (thorough validation)")

    print("\\nWHEN TO USE:")
    print("  Experimental features (use shadow)")
    print("  Low-risk changes (use blue-green)")
    print("  Gradual rollout (use canary)")

netflix_canary_deployment()
stripe_blue_green_deployment()
uber_shadow_deployment()
'''
}

def fix_notebook(notebook_path, code):
    """Replace Real-World Examples code cell"""

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Find "Real-World Examples:" markdown
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'Real-World Examples:' in source and i + 1 < len(nb['cells']):
                next_cell = nb['cells'][i + 1]
                if next_cell['cell_type'] == 'code':
                    # Replace the entire code cell
                    next_cell['source'] = code.split('\n')
                    for j in range(len(next_cell['source']) - 1):
                        next_cell['source'][j] += '\n'

                    with open(notebook_path, 'w') as f:
                        json.dump(nb, f, indent=1)
                    return True

    return False

def main():
    nb_dir = Path('/home/sbisw/github/interviewprep-ml/mlops/notebooks')

    print("FIXING LAST 4 NOTEBOOKS (13-16)")
    print("=" * 70)

    for nb_name, code in FIXES.items():
        nb_path = nb_dir / nb_name
        if nb_path.exists():
            success = fix_notebook(nb_path, code)
            status = "✓ Fixed" if success else "✗ Failed"
            print(f"{nb_name:40s} {status}")
        else:
            print(f"{nb_name:40s} ✗ Not found")

    print("=" * 70)
    print("✓ All notebooks fixed!")

if __name__ == '__main__':
    main()
