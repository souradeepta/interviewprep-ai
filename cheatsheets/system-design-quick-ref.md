# ML System Design Quick Reference

---

## Capacity Estimation Examples

### 1M QPS Embedding Service

```
Target: 1,000,000 queries per second (1M QPS)

Assumptions:
  - Model: sentence-transformers all-MiniLM-L6 (22M params, 384-dim output)
  - Batch size: 64 on A100 GPU
  - GPU throughput: ~20,000 sentences/sec (64 batch, ~3ms latency)
  - Model loaded in memory: 22M * 4 bytes (fp32) = 88 MB

GPU count:
  1M QPS / 20,000 QPS/GPU = 50 GPUs

Memory per GPU: 88 MB model + 64 * 512 * 4 bytes (batch) = ~100 MB -> fits easily

Network: 1M * 512 bytes (avg query) = 512 MB/s inbound
         1M * 384 * 4 bytes (embedding) = 1.5 GB/s outbound
  -> need 10Gbps+ NIC or load balancing

Cost (A100 cloud): 50 GPUs * $3/hr = $150/hr = $3,600/day

Optimization: use fp16 -> doubles throughput, halves memory
  -> 25 GPUs, $75/hr at same QPS
```

### LLM Serving: 10K Users, 100 tokens/response

```
Assumptions:
  - Model: 70B params in fp16
  - Model memory: 70B * 2 bytes = 140 GB -> needs 4x A100 (80GB each) per replica
  - Throughput: ~1,000 tokens/sec per 4-GPU node (vLLM continuous batching)
  - Each user: 100 output tokens, assume 200ms prefill + 2s generation = 2.2s total
  - Concurrent users at peak: 10,000 / 2.2s * average session = ~5,000 concurrent

Token demand: 5,000 concurrent users * ~50 tokens in flight = 250,000 tokens/sec
Node throughput: 1,000 tokens/sec
Nodes needed: 250,000 / 1,000 = 250 nodes (1,000 GPUs)

With 8-bit quantization: ~70 GB model -> fits 2x A100 80GB, double the throughput
  -> 125 nodes (500 GPUs)
```

---

## Latency Budgets (Typical Production Numbers)

| Operation | P50 Latency | P99 Latency | Notes |
|-----------|-------------|-------------|-------|
| Dense embedding (small model) | 2–5 ms | 10 ms | Batch of 32–64, GPU |
| Dense embedding (large model) | 10–20 ms | 50 ms | E5-large class, GPU |
| BM25 / keyword search | <5 ms | 20 ms | Elasticsearch / OpenSearch |
| ANN vector search (FAISS, Qdrant) | 5–15 ms | 30 ms | Top-100, 100M vectors |
| Reranker (cross-encoder) | 30–100 ms | 200 ms | Top-50 docs, GPU |
| LLM TTFT (Time to First Token) | 100–300 ms | 500 ms | 70B model, continuous batch |
| LLM token generation | 15–30 ms/token | 50 ms/token | Depends on KV cache, batch |
| Feature store read (Redis) | <1 ms | 5 ms | Single key lookup |
| Feature store read (DynamoDB) | 3–10 ms | 20 ms | Single item get |
| ML model inference (tabular, XGBoost) | <1 ms | 5 ms | Single sample |
| ML model inference (tabular, neural net) | 1–5 ms | 15 ms | Batch of 1, CPU/GPU |

### End-to-End RAG Latency Budget Example
```
Total SLA: 3 seconds

  Query embedding:          20 ms  (GPU, large model)
  ANN vector search:        15 ms  (100M vectors, top-50)
  Reranker (top-50->top-5): 80 ms  (cross-encoder, GPU)
  LLM TTFT:                300 ms  (70B, continuous batch)
  LLM generation (200 tok): 4000 ms  <- PROBLEM

Resolution:
  Switch to 8B model for generation: 500 ms (fits budget)
  Or use streaming + speculative decoding: show tokens as generated
```

---

## ML System Components Checklist

### Data Layer
- [ ] **Feature store** (online: Redis/DynamoDB; offline: S3/BigQuery)
- [ ] **Data pipeline** (ingestion, transformation, feature computation)
- [ ] **Data versioning** (DVC, Delta Lake, or Iceberg snapshots)
- [ ] **Schema contracts** (Great Expectations, Pandera)

### Model Layer
- [ ] **Model registry** (MLflow, W&B, SageMaker, Vertex AI)
- [ ] **Experiment tracking** (metric logging, hyperparameter tracking)
- [ ] **Model versioning** (immutable artifact storage: S3 + hash)
- [ ] **A/B testing framework** (assignment, logging, analysis)

### Serving Layer
- [ ] **Inference server** (vLLM for LLMs, Triton for ONNX/TensorRT, TorchServe)
- [ ] **Load balancer** + auto-scaling (HPA in Kubernetes)
- [ ] **Caching** (KV cache for LLMs; embedding cache; prediction cache)
- [ ] **Rate limiting + circuit breaker** (protect downstream models)

### Monitoring Layer
- [ ] **Data drift monitoring** (PSI, KS test on feature distributions)
- [ ] **Model performance monitoring** (accuracy on labeled sample, business KPI)
- [ ] **Operational metrics** (latency P50/P99/P999, error rate, throughput)
- [ ] **Alerting** (PagerDuty, Slack hooks; tiered P1/P2/P3)
- [ ] **Logging** (structured logs: request ID, model version, feature values, prediction)

---

## Common ML System Patterns

### Two-Tower (Dual Encoder) Architecture
```
Query Tower              Item Tower
    |                        |
[User embeddings]       [Item embeddings]
    |                        |
[Query vector]          [Item vectors] (pre-computed, indexed in ANN)
         \              /
          \            /
       Dot product similarity score
               |
           Top-K candidates
               |
           Reranker (optional)
```
Use case: recommendation, search, semantic retrieval.
Key insight: item vectors pre-computed and indexed offline; query vector computed at request time.

### Cascade (Multi-Stage) Architecture
```
10M candidates
    |
[Cheap retrieval] (BM25 or ANN): 10M -> 1,000 candidates  (< 20ms)
    |
[Fast ranker] (LightGBM or small NN): 1,000 -> 100 candidates  (< 50ms)
    |
[Slow precise reranker] (cross-encoder or LLM): 100 -> 10 results  (< 200ms)
    |
[Business logic] (diversity, dedup, policy): 10 -> final list
```
Design principle: each stage reduces candidates while increasing compute budget per item.

### RAG (Retrieval Augmented Generation)
```
User Query
    |
[Query Embedding]
    |
[ANN Search] -> Top-K chunks from vector DB
    |
[Reranker] (optional) -> Top-M most relevant chunks
    |
[Prompt Assembly: system + context chunks + query]
    |
[LLM Generation]
    |
Response
```
Key levers: embedding model quality, chunk size (512–1024 tokens), K (50–200), reranker.

### Ensemble Pattern
```
Input
 |     \     \
[M1]  [M2]  [M3]    (diverse models: XGBoost, NN, LightGBM)
 |     |     |
[Stacking layer / averaging / voting]
 |
Final prediction
```
Diversity is key: models should have different failure modes. Correlation < 0.9 between model errors.

---

## Back-of-Envelope: Model Memory

### Calculating Model Memory

```
Model parameters:
  Bytes = num_params * bytes_per_param

bytes_per_param by precision:
  fp32 (full precision):  4 bytes
  fp16 / bf16:            2 bytes
  int8:                   1 byte
  int4:                   0.5 bytes

Examples:
  7B model in fp16:   7B * 2 = 14 GB
  13B model in fp16:  13B * 2 = 26 GB
  70B model in fp16:  70B * 2 = 140 GB  -> 2x A100 80GB (with overhead)
  70B model in int4:  70B * 0.5 = 35 GB -> 1x A100 80GB (just fits)
```

### KV Cache Memory (LLM)

```
KV cache per token = 2 * num_layers * num_heads * head_dim * bytes_per_element

For LLaMA 3.1 70B (fp16):
  layers=80, heads=8 (GQA kv_heads), head_dim=128
  Per token: 2 * 80 * 8 * 128 * 2 = 327,680 bytes = 320 KB

For sequence length 4096:
  4096 * 320 KB = 1.28 GB per sequence

For batch of 8 sequences: 10.24 GB just for KV cache
```

### Batch Size for GPU

```
Rule of thumb: fill GPU memory to 70-80% utilization

Available memory = GPU memory - model memory - KV cache
Batch memory = batch_size * seq_len * hidden_dim * bytes_per_element * (activations factor ~2-3)

Example (A100 80GB, 7B model fp16):
  Model: 14 GB
  Remaining: 66 GB
  Per sample (512 tokens, dim=4096, fp16, activations=3x): 512 * 4096 * 2 * 3 = 12 MB
  Max batch: 66 GB / 12 MB = ~5,500  (use with gradient checkpointing; otherwise ~1,000)
```

### Throughput Formula
```
Throughput (tokens/sec) = batch_size * seq_len / latency_per_batch

For a 7B model on A100:
  Batch of 32, seq_len=512, latency=1.5s
  Throughput = 32 * 512 / 1.5 = ~10,900 tokens/sec
```
