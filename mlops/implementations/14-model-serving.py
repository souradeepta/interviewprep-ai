"""
Auto-generated from 14-model-serving.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Model Serving: Making Models Available for Real-Time Predictions
# ## Learning Objectives
# - Understand serving patterns and frameworks
# - Optimize for latency and throughput
# ======================================================================

# ======================================================================
# ## Basic: FastAPI Inference Server
# ======================================================================

fastapi_example = '''
from fastapi import FastAPI
import joblib
import numpy as np
from typing import List

app = FastAPI()

# Load model once
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: List[float]):
    """Single prediction"""
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)[0]
    return {"prediction": prediction}

@app.post("/predict-batch")
def predict_batch(batch: List[List[float]]):
    """Batch predictions (faster)"""
    X = np.array(batch)
    predictions = model.predict(X)
    return {"predictions": predictions.tolist()}

@app.get("/health")
def health():
    """Health check for Kubernetes"""
    return {"status": "healthy"}
'''

print("FASTAPI INFERENCE SERVER")
print()
print(fastapi_example)
print()
print("Usage:")
print("  uvicorn app:app --host 0.0.0.0 --port 8000")
print()
print("Testing:")
print("  curl -X POST http://localhost:8000/predict -d '{\"features\": [1,2,3]}'")
print()
print("Characteristics:")
print("  ✓ Simple (100 lines)")
print("  ✓ Fast (no overhead)")
print("  ✗ No versioning")
print("  ✗ No auto-scaling")
print("  ✗ No monitoring")


# ======================================================================
# ## Advanced: Feature Server + Latency Optimization
# ======================================================================

feature_serving = '''
Design: Parallel feature fetching

Request Flow:
  1. Request arrives with user_id
  2. Spawn 3 async tasks (parallel):
     - Fetch user features from Redis (10ms)
     - Call real-time API (20ms)
     - Fetch batch features from S3 (50ms)
  3. Wait for all 3 to complete: max(10, 20, 50) = 50ms (not 80ms!)
  4. Model inference: 50ms
  5. Total latency: 100ms (within SLO!)

Without parallelism:
  Sequential: 10 + 20 + 50 + 50 = 130ms (exceeds SLO!)
'''

print("FEATURE SERVING FOR LATENCY OPTIMIZATION")
print()
print(feature_serving)
print()

latency_breakdown = '''
Latency budget: 100ms

Without optimization (sequential):
- Feature fetch (Redis): 10ms
- Feature fetch (API): 20ms
- Feature fetch (S3): 50ms
- Model inference: 50ms
- Total: 130ms ✗ (exceeds SLO)

With caching:
- Redis (hot features): 10ms
- API (cached): 5ms (if cache hit)
- S3 (prefetched): 30ms
- Model inference: 40ms (quantized)
- Total: 85ms ✓

Optimization techniques:
1. Cache frequently-accessed features (Redis)
2. Pre-compute batch features offline
3. Quantize model (INT8)
4. Parallelize feature fetch
'''

print(latency_breakdown)


# ======================================================================
# ## Real-World Examples: Netflix, Stripe, Uber
# ======================================================================

def netflix_fastapi_serving():
    """FastAPI server for recommendation serving"""

    print("NETFLIX: Model Serving")
    print("=" * 60)

    print("\nLATENCY BREAKDOWN:")
    print("  Cache lookup: 1ms")
    print("  Feature fetch: 20ms (Redis)")
    print("  Model inference: 15ms (single GPU)")
    print("  Top-K selection: 2ms")
    print("  Serialization: 2ms")
    print("  Total: 40ms (< 50ms SLO)")

    print("\nTHROUGHPUT:")
    print("  1 GPU server: 500 req/sec")
    print("  100 servers: 50K req/sec")
    print("  Batching: 10-20ms latency for batch_size=32")

    print("\nCACHING STRATEGY:")
    print("  Per-user: 100ms TTL (session context)")
    print("  Per-content: 1 hour TTL (trending)")
    print("  Hit rate: 85% (25M active users)")

def stripe_model_serving():
    """Real-time fraud scoring"""

    print("\nSTRIPE: Fraud Scoring Pipeline")
    print("=" * 60)

    print("\nREQUEST FLOW:")
    print("  1. Transaction arrives (event)")
    print("  2. Fetch user features (5ms, cache)")
    print("  3. Fetch merchant features (10ms, DB)")
    print("  4. Compute transaction features (2ms)")
    print("  5. Model inference (25ms)")
    print("  6. Decision logic (1ms)")
    print("  Total: 43ms (< 50ms SLA)")

    print("\nDECISION LOGIC:")
    print("  score > 0.7: DECLINE + challenge")
    print("  0.5 < score < 0.7: MONITOR")
    print("  score < 0.5: APPROVE")
    print("  Special: FLAG if high velocity (50+ txns/min)")

    print("\nBACKPRESSURE HANDLING:")
    print("  Timeout: 100ms (fallback = approve with monitor)")
    print("  Queue depth: max 10K pending requests")
    print("  Graceful degradation: if service slow, approve more")

def uber_model_serving():
    """ETA and matching serving"""

    print("\nUBER: ETA & Matching Serving")
    print("=" * 60)

    print("\nETA MODEL SERVING:")
    print("  Features: distance, traffic, time_of_day, route")
    print("  Latency: 30ms inference + 20ms feature fetch = 50ms")
    print("  Update: every 1 minute (traffic patterns)")
    print("  Model: XGBoost (48MB, fast inference)")

    print("\nMATCHING SERVING:")
    print("  Features: driver location, rating, acceptance_rate, etc")
    print("  Latency: 30ms model + 30ms feature = 60ms")
    print("  Update: real-time (driver moves)")
    print("  Model: XGBoost (large, 500MB)")

    print("\nFEDERATION:")
    print("  50 edge servers (distributed globally)")
    print("  Federated learning: model trained centrally")
    print("  Inference: local (~10ms latency)")

netflix_fastapi_serving()
stripe_model_serving()
uber_model_serving()



# ======================================================================
# ## Interview Case Study: Designing ML Serving
# ======================================================================

case_study = '''
SCENARIO: Airbnb price prediction
- 50M searches/day
- Must return price in <500ms
- Model inference: 50ms
- Feature requirements: user features + listing features + real-time signals

DESIGN:

1. Framework: KServe on Kubernetes
   - Handles versioning, scaling, monitoring
   - Support for GPU inference

2. Feature Architecture:
   - User features (cached in Redis, <5ms)
   - Listing features (cached in DynamoDB, <10ms)
   - Real-time signals (fetch inline, <10ms)
   - Total feature latency: max(5,10,10) = 10ms (parallel)

3. Model Optimization:
   - Quantized: INT8 (50ms inference)
   - GPU batching: dynamic batch size 64 (better throughput)
   - Result: 10ms features + 50ms inference = 60ms (well within 500ms)

4. Versioning:
   - Old model (v1.0): handles 80% traffic
   - New model (v1.1): handles 20% traffic (canary)
   - Monitor both separately
   - If v1.1 good after 24h: switch to 50% → 100%

5. Monitoring:
   - Latency p50/p99 (should stay <500ms)
   - Model accuracy (compare predictions to bookings)
   - Feature staleness (ensure features <1h old)

RESULT: Serves 50M searches, <500ms latency, safe rollout
'''

print(case_study)


# ======================================================================
# ## Key Takeaways
# **Serving != Training:** Optimize for latency/throughput, not just accuracy.
# **Feature serving is critical:** Pre-compute when possible, cache aggressively.
# ======================================================================
