# Model Serving

## TL;DR
Deploy trained models to production infrastructure to serve predictions. Patterns: batch serving (offline, computed in bulk), online serving (real-time, per-request), streaming (continuous updates). Trade-offs: latency vs cost, simplicity vs flexibility.

## Core Intuition
Training produces a model artifact. Serving puts it to work: accept requests, return predictions in sub-second latency, at scale. Model serving is about reliability, speed, and efficiency.

## How It Works

**Serving Patterns:**

**1. Batch Serving (Offline):**
```
Data → Model → Precomputed Predictions → Storage → Retrieve
  
Timeline:
  - Nightly: run batch job on 1M users → predict churn scores
  - Store in database/cache
  - Next day: app queries cached predictions (instant lookup)
```
- **Pros:** cheap (compute once), high throughput, offline optimization (can use expensive model)
- **Cons:** stale predictions (not fresh), latency for new users, requires storage

**2. Online Serving (Real-time):**
```
Request → Load Model → Extract Features → Predict → Response
  
Timeline:
  - User visits app
  - Request hits prediction service
  - Service loads model, runs inference
  - Returns prediction in <100ms
  - Next user gets different prediction (real-time)
```
- **Pros:** fresh predictions, no storage, personalizable
- **Cons:** high latency (compute on-demand), expensive (scale with requests), requires fast inference

**3. Streaming:**
```
Event Stream → Model → Update Predictions → External System
  
Timeline:
  - Kafka topic: user interactions
  - Stream processor: apply model continuously
  - Output: updated scores (e.g., ranking scores for feed)
```
- **Pros:** always up-to-date, high volume
- **Cons:** complex infrastructure, higher latency

**Serving Frameworks:**

| Framework | Best For | Latency | Throughput |
|-----------|----------|---------|-----------|
| TensorFlow Serving | Large scale, TF models | 10-50ms | 1000+ req/sec |
| TorchServe | PyTorch models | 10-50ms | 1000+ req/sec |
| ONNX Runtime | Any model (via ONNX) | 5-30ms | 1000+ req/sec |
| FastAPI | Custom Python | 5-20ms | 500+ req/sec |
| Seldon Core | Kubernetes native | 10-100ms | 100-1000 req/sec |
| vLLM | LLMs specifically | 20-500ms | 10-100 req/sec (token generation) |

**Architecture Example (Online Serving at Scale):**
```
                    Load Balancer
                        |
          _______________+_______________
         /               |               \
      Replica1        Replica2        Replica3
      (GPU)           (GPU)           (GPU)
        |               |               |
        +-------+-------+-------+-------+
                |
        Feature Store (Cache)
                |
        Database / Redis
```

## Key Properties / Trade-offs

| Aspect | Batch | Online |
|--------|-------|--------|
| Latency | Minutes-hours | <100ms |
| Freshness | Stale | Real-time |
| Cost | Low (compute once) | High (per-request) |
| Scalability | Linear with data | Linear with requests |
| Storage | High (precomputed) | None |
| Personalization | Limited | Full |

**Compute Resource Selection:**
```
CPU:
  - Good for: tree models (XGBoost), simple models
  - Cost: cheap
  - Latency: 10-100ms
  
GPU:
  - Good for: deep learning, large models
  - Cost: expensive
  - Latency: 5-50ms (with good utilization)
  
TPU/Accelerator:
  - Good for: very high throughput (LLMs, large batch)
  - Cost: moderate-high
  - Latency: <10ms for optimized models
```

## Common Mistakes / Gotchas

- **Training/serving skew:** Model trained differently than served (different preprocessing, libraries). Use same code for both.
- **Feature mismatch:** Features at serving time ≠ features at training. Validate schema.
- **Not handling latency:** Model takes 200ms but SLA is 50ms. Needs optimization (quantization, caching, approximate inference).
- **Cold start:** Model not loaded → first request times out. Preload models, use autoscaling warmup.
- **Resource exhaustion:** No request limiting → requests pile up → system crashes. Add queue, rate limiting, circuit breaker.
- **No versioning:** Can't rollback bad model. Tag versions, serve multiple versions, switch via config.
- **Ignoring dependency versions:** Model trained with TF 2.8, served with TF 2.10 → incompatible. Pin dependencies.
- **Single replica:** One replica dies → 0% availability. Always have redundancy, health checks, failover.

## Code Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List

app = FastAPI()

# Load model (once, at startup)
model = joblib.load("model.pkl")  # Trained model
scaler = joblib.load("scaler.pkl")  # Feature scaler

class PredictionRequest(BaseModel):
    features: List[float]  # Input features

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.on_event("startup")
async def startup():
    """Startup event: validate model is loaded."""
    print("Model loaded successfully")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for a single request."""
    try:
        # 1. Validate input
        if not request.features or len(request.features) != 10:
            raise HTTPException(status_code=400, detail="Expected 10 features")
        
        # 2. Preprocess (must match training)
        X = np.array(request.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # 3. Predict
        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled)[0].max()
        
        return PredictionResponse(prediction=float(prediction), confidence=float(confidence))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch predictions for multiple requests."""
    try:
        # Collect all feature vectors
        X_list = [np.array(req.features) for req in requests]
        X = np.vstack(X_list)
        X_scaled = scaler.transform(X)
        
        # Predict all
        predictions = model.predict(X_scaled)
        confidences = model.predict_proba(X_scaled).max(axis=1)
        
        return [
            PredictionResponse(prediction=float(p), confidence=float(c))
            for p, c in zip(predictions, confidences)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "Batch vs online?" | Batch: cheap, stale. Online: fresh, expensive. Pick based on latency SLA and personalization needs. |
| "Latency optimization?" | Caching (avoid recompute), quantization (faster inference), approximate inference, early exit (stop if confident). |
| "Scaling?" | Replicas + load balancer for horizontal scale. Use autoscaling based on latency or QPS. |
| "Cold start?" | Preload models at startup. Use orchestration (K8s) to warm up replicas. Set request timeout > inference time. |
| "Training/serving skew?" | Use same preprocessing code. Version dependencies. Test serving code with training data. |
| "Rollback?" | Version all models, serve multiple versions. Switch via config (no redeployment). Always have fallback. |

## Related Topics
- [Online vs Batch Inference](online-vs-batch-inference.md) — choosing the right serving pattern
- [Inference Caching](inference-caching.md) — speed up serving via caching
- [Request Batching](request-batching.md) — optimize throughput
- [Load Balancing](load-balancing.md) — distribute requests across replicas
- [A/B Testing](ab-testing.md) — compare models in production

## Resources
- [TensorFlow Serving Architecture](https://www.tensorflow.org/tfx/serving/architecture)
- [Model Serving with FastAPI](https://github.com/tiangolo/fastapi/issues/26)
- [Seldon Core: Production ML Model Server](https://www.seldon.io/)
- [vLLM: High-Throughput and Memory-Efficient LLM Serving](https://github.com/lm-sys/vllm)
- [Papers: Clipper (Berkeley), KubeFlow, TFServing](https://www.usenix.org/system/files/nsdi17-crankshaw.pdf)
