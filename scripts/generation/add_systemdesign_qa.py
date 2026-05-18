#!/usr/bin/env python3
"""
Add '## Interview Q&A' sections to all 31 markdown files in
/home/sbisw/github/interviewprep-ml/system-design/patterns/
Each Q&A is inserted BEFORE the existing '## Interview Quick-Reference' section.
"""

import os

PATTERNS_DIR = "/home/sbisw/github/interviewprep-ml/system-design/patterns"

# Q&A content keyed by file slug (filename without .md extension)
QA_CONTENT = {

"01-mlops-overview": """\
## Interview Q&A

**Q: What is the most common reason ML models fail in production?**
A: Training-serving skew: the model was trained on data preprocessed differently than production data—different scaling, encoding, or feature order. Other top causes: data drift (distribution of production inputs changes), inadequate monitoring (failures go undetected), and insufficient testing of edge cases. The infrastructure to catch these issues (feature stores, monitoring, CI/CD) is what MLOps provides.

**Q: When should a company invest in MLOps infrastructure vs. keeping ML simple?**
A: Invest when: you have 3+ models in production, models need to be retrained regularly, multiple teams contribute to ML pipelines, or model failures have significant business impact. Keep simple when: you have 1-2 models that rarely change, the models are low-stakes, or you're still validating the ML use case. Over-engineering MLOps for one model is wasteful; under-engineering for 10 models creates technical debt that slows the entire team.

**Q: How does MLOps differ from DevOps and what does that mean for tooling?**
A: DevOps handles code and infrastructure; MLOps handles code + data + models + experiments. Unique MLOps challenges: data versioning (code versioning tools don't handle large datasets), experiment tracking (many hyperparameter configurations to compare), model validation (both technical metrics and business metrics), and data drift (models degrade without code changes). Tooling must address all four: MLflow/W&B for experiments, DVC for data versioning, Evidently for drift, and custom model cards for documentation.

**Q: What does a minimal viable MLOps setup look like?**
A: Minimum: (1) version control for code and model artifacts, (2) automated retraining pipeline with data validation, (3) staging environment for model validation before production, (4) basic monitoring of prediction distribution and key business metrics. This can be built with MLflow + GitHub Actions + simple dashboards in a few weeks. Don't let perfect MLOps be the enemy of shipping—start minimal and add components as team pain points emerge.

**Q: What is the ROI calculation for investing in MLOps infrastructure?**
A: Quantify: engineer hours spent debugging production issues (monitoring reduces this), time to retrain and deploy models (CI/CD reduces this), models that failed silently before monitoring caught them (value of prevented failures). Typical improvements: 3-5x faster model deployment, 50-70% reduction in production incidents, 2-3x more models maintained per engineer. The investment pays off when the cost of infrastructure is less than the cost of the manual work it replaces, usually at 3-5 production models.

""",

"02-data-pipelines": """\
## Interview Q&A

**Q: How do you decide between batch processing and stream processing for an ML data pipeline?**
A: Batch when: features don't need to be real-time (end-of-day reports, daily retraining), data volumes are large and latency requirements are loose (hours to days), simplicity is valued over freshness. Stream when: features need to be up-to-date within seconds (fraud detection, real-time recommendations), you're processing continuously arriving events, or decisions need the latest context. Many production systems use both: batch for expensive features (complex aggregations), stream for simple low-latency features (recent activity counts).

**Q: What are the most common causes of data pipeline failures in production?**
A: Schema changes upstream: a source system adds/removes/renames a column without notifying the ML team. Upstream system outages: the pipeline fails silently when source data is missing. Data volume spikes: pipeline times out or runs out of memory during unusually large batches. Silent data quality degradation: pipeline succeeds but produces wrong features (e.g., NULL values increase, outliers appear). Infrastructure changes: a library upgrade breaks serialization. All of these require monitoring, not just code fixes.

**Q: How do you implement data validation in an ML pipeline?**
A: Validate at every boundary: raw data ingestion (check expected columns, types, row counts), feature computation (check distributions against historical baselines), training data preparation (check class balance, feature correlations). Tools: Great Expectations, TFX Data Validation, or custom statistical tests. Set thresholds based on historical variation (flag if metric deviates >3σ from rolling average). Fail the pipeline and alert rather than silently proceeding with bad data.

**Q: What is data lineage and why does it matter for ML?**
A: Data lineage tracks how each piece of data was transformed from raw source to model input. It enables: debugging (trace a model prediction back to the raw data that produced it), compliance (prove what data was used to train a model), impact analysis (understand which models are affected when a data source changes), and reproducibility (recreate exact training datasets). Without lineage, investigating production issues is extremely difficult—you can't answer "why did this prediction change?"

**Q: How do you handle late-arriving data in time-based features?**
A: Late data is common in distributed systems: events with timestamps from yesterday that arrive today. Strategies: (1) watermarks—define a maximum lateness threshold (e.g., 6 hours), process data within the window, reject or ignore later arrivals; (2) reprocessing—allow late data to trigger recomputation of affected time windows; (3) approximate processing—treat late data as "close enough" and accept small errors in time-sensitive features. For model training, ensure your training data has the same late-arrival characteristics as production.

""",

"03-feature-store": """\
## Interview Q&A

**Q: When should you implement a feature store vs. computing features inline?**
A: Implement a feature store when: multiple models share features (avoid duplicate computation), features require expensive computation (aggregations over large datasets), online/offline feature consistency is critical (training-serving skew prevention), or you need point-in-time correct features for retraining. Compute inline when: you have one model, features are simple and cheap to compute, or you're still in early ML development. The overhead of a feature store (infrastructure, maintenance) is only justified when it solves real problems you're experiencing.

**Q: What is training-serving skew and how does a feature store prevent it?**
A: Training-serving skew: features are computed differently at training time (offline, batch) vs. serving time (online, real-time), causing the model to receive different distributions than it was trained on. Classic example: training uses the median of a column but serving uses the mean. A feature store prevents this by ensuring the exact same feature computation logic is used for both training and serving—the feature definition is single-source-of-truth.

**Q: How do you implement point-in-time correct features for model retraining?**
A: Point-in-time correctness means: when generating training data for a label that occurred at time T, use only feature values that were available at time T (no future leakage). Feature stores implement this by storing feature values with timestamps and providing time-travel queries: "what was the value of feature X for entity Y at time T?" This requires storing the full history of feature values, not just current values—a significant storage cost that justifies the feature store.

**Q: What are the latency requirements for online feature serving and how do you meet them?**
A: Real-time inference: online features must be served in <10ms (leaving budget for model inference). Requirements: in-memory store (Redis, Cassandra) for fast reads, pre-computed feature values (not computed on-the-fly at serving), co-location of feature store with inference service (avoid network latency), and bulk fetch (retrieve all features for an entity in one call, not one call per feature). Monitor feature serving latency as a primary SLA—slow feature serving can silently cause SLA violations.

**Q: How do you manage feature versioning and backward compatibility?**
A: Treat feature definitions as versioned contracts. When a feature computation changes (new data source, modified aggregation window): create a new feature version, maintain old version while downstream models migrate, use a shadow-mode period where both versions are computed and compared. Never update a feature definition in-place if it's used by production models—the change will cause training-serving skew for those models. A feature store should provide version history and deprecation workflows.

""",

"04-model-registry": """\
## Interview Q&A

**Q: What metadata should every model version in the registry include?**
A: Mandatory: training data version (or dataset hash), hyperparameters, evaluation metrics on test set, training code version (git commit), training infrastructure (GPU type, framework version), model size and inference latency benchmarks. Highly recommended: data lineage (what raw data was used), feature importance, behavioral tests results, fairness/bias metrics, and the business context (what problem this model solves). Without this metadata, debugging production issues or replicating experiments is extremely difficult.

**Q: How do you implement model promotion gates (dev to staging to production)?**
A: Each gate should have automated checks: dev to staging (unit tests pass, basic metrics above floor threshold, no regression vs. baseline), staging to production (full eval suite on held-out data, A/B test results meet statistical significance, latency/throughput benchmarks met, security scan clean). Some checks require human approval (model card review, compliance sign-off for regulated use cases). Never promote a model without automated gates—a broken model in production is more expensive than the time saved by skipping checks.

**Q: How do you handle rollback when a production model starts failing?**
A: Design for fast rollback from the start: keep the previous production model artifact registered and deployable in <5 minutes. Implement automated rollback triggers: if key metrics drop >X% in the first hour after deployment, automatically roll back. Have a manual rollback runbook that any on-call engineer can execute without ML expertise. After rollback, post-mortem to identify the root cause before attempting re-deployment. The ability to roll back quickly is more important than preventing all bad deployments.

**Q: What is the difference between a model registry and a model store?**
A: Model store: binary artifact storage (S3, GCS)—stores model files, serialized weights, ONNX files. Model registry: metadata management and lifecycle tracking—stores the version history, evaluation metrics, deployment status, and lineage of models. You need both: the registry knows about model versions and their metadata, and points to artifacts in the store. MLflow, Weights & Biases, and SageMaker Model Registry combine both; many teams use S3 + a custom metadata database separately.

**Q: How do you manage model registry access control in a multi-team environment?**
A: Implement role-based access: data scientists can register and evaluate models; ML engineers can promote to staging; deployment approvals require additional sign-off. Restrict production promotion to an automated CI/CD system (not individual humans). Maintain an audit log of all registry operations: who registered, promoted, or deprecated each model version. For regulated industries, the audit log is a compliance requirement, not just a best practice.

""",

"05-model-serving": """\
## Interview Q&A

**Q: How do you choose between REST APIs, gRPC, and streaming for model serving?**
A: REST: default choice—simple, widely supported, easy to debug. Use for: low-to-medium throughput, non-binary responses, when clients are diverse (browsers, mobile). gRPC: when performance matters—binary protocol, 3-10x faster serialization. Use for: high-throughput microservice-to-microservice, when client and server are both under your control. Streaming (Server-Sent Events, WebSocket, gRPC streaming): when responses are generated incrementally (LLMs, real-time scores). Match the protocol to actual client needs—gRPC adds complexity that's only worth it above 1000 RPS.

**Q: What health checks should a model serving endpoint implement?**
A: Liveness: is the process running? (Simple HTTP 200) Readiness: is the model loaded and ready to serve? (Run inference on a synthetic sample, check latency < threshold) Startup: has initialization completed? (Separate from readiness to prevent restart loops during model loading). Deep health: are all dependencies (feature store, database) reachable and healthy? Surface readiness and startup to your load balancer; surface deep health to your monitoring dashboard. A pod that fails readiness gets no traffic; a pod that fails liveness gets restarted.

**Q: How do you handle model loading latency in containerized deployments?**
A: Large models (GPT-2, ResNet) can take 30-120 seconds to load into GPU memory. Mitigate: use readiness probes that prevent traffic until loading completes, pre-download model artifacts in the container image (not at runtime), use model artifact caching layers in the container build, implement graceful startup (old pods keep serving while new ones load). For very large models (7B+), consider model serving frameworks that support fast model loading (TensorRT, vLLM) as a first-class feature.

**Q: What are the key trade-offs between single-model serving and multi-model serving on the same infrastructure?**
A: Single model per deployment: isolation (one model's failure doesn't affect others), simple scaling (scale based on one model's load), but higher infrastructure cost when models are underutilized. Multi-model per deployment: lower cost through resource sharing, but requires careful resource isolation and capacity planning. Use multi-model when: models are small and underutilized, models have complementary usage patterns (different peak hours), or cost reduction is critical. Keep them isolated when: models have different SLAs or different update frequencies.

**Q: How do you implement graceful model updates with zero downtime?**
A: Rolling deployment with overlap: bring up new model replicas with the new version, wait for readiness, shift traffic gradually, then terminate old replicas. Blue-green at the load balancer level: instant traffic switch after new deployment is verified. Key requirement: the serving API must be backward compatible (same request/response schema). If schema changes, version your API endpoint (/v1/predict to /v2/predict) and migrate clients independently. Never push breaking changes to a live endpoint without a migration window.

""",

"06-model-versioning": """\
## Interview Q&A

**Q: What constitutes a new model version vs. a model update?**
A: Treat as a new version: different architecture, different training data, different hyperparameters that change model behavior, schema changes to inputs or outputs. Treat as a minor update: bug fixes in serving code, infrastructure changes that don't affect predictions, documentation updates. The test: would two versions produce different predictions for the same input? If yes, it's a new version that requires evaluation and promotion gates. If no, it's an infrastructure change with different deployment procedures.

**Q: How do you version models when training data changes frequently?**
A: Version the training data as part of the model version: store a hash of the dataset or a reference to a specific data snapshot. This allows you to answer: "what data was this model trained on?" and "if we retrain on the same data, do we get the same model?" Without data versioning, model versions are not reproducible—you can't debug regressions by comparing model versions. DVC (Data Version Control) or Delta Lake are standard tools for data versioning in ML workflows.

**Q: What is the shadow registry pattern and when do you use it?**
A: Shadow registry: a model version that receives a copy of production traffic (shadow mode) without affecting real responses. Use it for: evaluating a candidate model on production data distribution before promotion, comparing predictions between versions without risk, and validating that a new version handles edge cases in production. Shadow mode requires: infrastructure to duplicate requests, logging to compare shadow vs. production predictions, and a comparison framework to identify meaningful differences.

**Q: How do you maintain multiple model versions in production simultaneously?**
A: Multiple production versions are needed for: A/B testing, gradual rollout, customer-specific model versions, and multi-tenant deployments. Implementation: version the serving endpoint (model_id or version_id as a request parameter), route requests to the appropriate version at the load balancer, maintain separate scaling policies per version. Monitor each version independently—a new version may fail on a specific segment of traffic that the aggregate monitoring misses. Have a sunset policy: versions older than N months should be deprecated.

**Q: How do you handle the model registry when models are trained with different frameworks?**
A: Use a framework-agnostic serialization format: ONNX (covers most frameworks), pickle for Python-specific models, SavedModel for TensorFlow. Store both the native format (for retraining) and a serving-optimized format (ONNX/TensorRT). Include the framework version in the model metadata—models trained with PyTorch 1.x may not load with PyTorch 2.x. Design your serving infrastructure to support multiple model formats or standardize on one (ONNX is the most portable choice).

""",

"07-online-vs-batch-inference": """\
## Interview Q&A

**Q: How do you decide whether to switch a batch inference system to online inference?**
A: Switch to online when: business logic requires real-time decisions (fraud detection, real-time recommendations), the latency between batch creation and consumption causes stale predictions, or user interactions require immediate responses. Stay with batch when: predictions can be precomputed (user-item scores computed nightly), input data isn't available in real-time (end-of-day transaction aggregations), or cost is a primary constraint (batch is 5-10x cheaper). Many systems use both: batch for expensive features, online for final scoring.

**Q: What is the optimal batch size for batch inference and how do you determine it?**
A: Batch size is a throughput vs. latency trade-off. GPU utilization peaks at batch sizes of 32-256 for most models. Test with increasing batch sizes and measure GPU utilization and throughput—the optimal batch size is where GPU utilization is >80% and latency is still acceptable. For time-constrained batch jobs, maximize batch size. For online pseudo-batch (grouping real-time requests), use dynamic batching with max-latency timeout (e.g., batch up to 32 requests or wait up to 10ms).

**Q: How do you handle stragglers in distributed batch inference?**
A: Stragglers (slow workers) in distributed batch jobs can delay the entire job. Mitigations: use speculative execution (re-run the slowest 5% of tasks on new workers), implement task timeouts (kill and retry tasks exceeding 3x median time), partition data to avoid skew (straggler is often due to uneven data distribution—sort by input complexity), and use heterogeneous hardware detection (don't assign large batches to slower machines). Monitor task completion time distribution, not just average—a bimodal distribution indicates stragglers.

**Q: What monitoring is needed for batch inference pipelines?**
A: Job-level: completion time trend (is the job getting slower?), success/failure rate, input data volume. Quality: distribution of prediction scores (compare against historical baseline), sample of predictions for manual review. Data quality: null rate in input features, count of inputs outside model's training distribution. Infrastructure: worker failures, memory usage, data read throughput. Alert on: job duration exceeding 2x historical average, prediction distribution shift >3σ, input data volume drop >20%.

**Q: How do you version and reproduce batch inference runs?**
A: For reproducibility, a batch inference run requires: model version, input data version (or snapshot), inference code version, inference configuration (batch size, preprocessing parameters). Store these provenance records in your model registry or a separate job tracking system. This enables: debugging (reproduce a specific run to trace a bad prediction), auditing (prove which model made which prediction), and regression testing (re-run a historical batch with a new model to compare).

""",

"08-inference-caching": """\
## Interview Q&A

**Q: What cache invalidation strategy should you use for model inference caching?**
A: TTL-based: set expiration based on how fast the underlying data changes—static lookup tables: 24 hours, personalized recommendations: 15 minutes, real-time fraud scores: never cache. Event-based: invalidate when the model is retrained or input data changes (use a cache version key that includes model version). Request-specific: for LLM responses, include a hash of the exact prompt as the cache key—any change in the prompt is a cache miss. Never cache responses from non-deterministic models at temperature >0 unless you explicitly want to freeze a specific response.

**Q: When does inference caching hurt more than it helps?**
A: Caching hurts when: the cache hit rate is <10% (overhead outweighs benefit), the cached data becomes stale quickly causing wrong predictions, memory pressure from the cache degrades other system performance, or the cache provides a false sense of capacity (real load spikes hit a cold cache). Measure: cache hit rate, latency reduction for hits vs. misses, stale cache rate (responses served after model retrain), and tail latency at cache misses (the worst case is what users experience when cache fails).

**Q: How do you implement caching for an LLM API to reduce cost and latency?**
A: Exact cache (hash match): store (prompt_hash to response), serve from cache for identical prompts. Works well for: FAQ answers, product descriptions, templated responses. Semantic cache (embedding similarity): embed the query, retrieve cached response if sufficiently similar exists. Works for: slightly rephrased questions with the same intent. Implement both with different thresholds: exact match first (free), semantic match second (cost of embedding). Track cache hit rate and cost savings; validate that semantic cache hits are actually equivalent answers to the original queries.

**Q: How do you handle personalized inference that can't be cached naively?**
A: Separate the personalization from the base computation. Cache the base computation (non-personalized model output), then apply a lightweight personalization layer (re-ranking, score adjustment) using cached user features. This way you cache the expensive part and keep the personalization layer fast. Alternatively, cache at a user segment level rather than individual level—users in the same segment get the same cached base results with segment-level adjustments. Segment-level caching has higher hit rates than individual-level.

**Q: What are the distributed caching considerations for multi-region inference serving?**
A: Cache locality: read from the nearest cache replica to minimize latency; write-through to all replicas for consistency. Replication lag: in a multi-region setup, a model retrain may not invalidate all regional caches simultaneously—implement version-aware cache keys (include model version in cache key). Cache stampede: when the cache expires and many requests simultaneously miss—use probabilistic early expiration or mutex-based single-flight to prevent all requests from computing simultaneously. Monitor cache hit rates per region independently.

""",

"09-request-batching": """\
## Interview Q&A

**Q: What is the difference between static batching and dynamic batching for ML inference?**
A: Static batching: fixed batch size, pad shorter sequences, wait for the batch to fill before processing. Simple to implement but inefficient when traffic is bursty or sequence lengths vary greatly. Dynamic batching: batch requests as they arrive, process when either max batch size or max latency timeout is reached. More complex but significantly better GPU utilization. Dynamic batching with a 10ms max-wait typically achieves 5-10x higher throughput than static batching at similar latency, because the GPU processes full batches more often.

**Q: How does request batching interact with SLA requirements and queue management?**
A: Larger batches improve throughput but increase latency—a request that waits for a batch to fill waits longer than a request in a singleton batch. Design: set max-batch-wait-time based on your TTFT SLA (e.g., if SLA is 500ms and model inference is 200ms, max wait time is 200ms). Implement priority queues: premium-tier requests have shorter max-wait times. Monitor queue depth: if queue consistently exceeds 10 requests, add capacity. Queue depth is a leading indicator of latency degradation before the SLA is actually violated.

**Q: What padding and masking considerations affect batching efficiency?**
A: For variable-length sequences (text, time series), shorter sequences must be padded to match the longest sequence in the batch. If one sequence is 512 tokens and all others are 50 tokens, 90% of compute is wasted on padding. Mitigate: sort sequences by length before batching (bucket by length range), use dynamic padding (pad only to the max length in each batch), implement sequence packing (concatenate multiple short sequences into one long sequence). These techniques can improve throughput 2-4x for highly variable length inputs.

**Q: How do you implement adaptive batching that responds to traffic changes?**
A: Adaptive batching adjusts batch size based on current queue depth and latency observations. Algorithm: start with max-wait-time=T. If recent P95 latency is above SLA: decrease max-wait-time (process smaller batches faster). If queue depth is growing: increase batch size to process more per cycle. If GPU utilization is low: increase batch size. Implement a PID controller or simple threshold rules. Test the adaptive controller under different traffic patterns: sudden spike, sustained high load, gradual increase.

**Q: What failure modes does batching introduce that singleton request serving doesn't have?**
A: Batch failure modes: one bad request in a batch fails the entire batch (implement per-request error isolation), head-of-line blocking (large requests delay small ones in the same batch), batch timeout cascade (when batches time out, the queue grows and subsequent batches also time out), and memory allocation failure for oversized batches. Mitigate: validate requests individually before batching, implement request size limits, use separate queues for large and small requests, and implement circuit breakers that degrade to smaller batches under memory pressure.

""",

"10-load-balancing": """\
## Interview Q&A

**Q: What load balancing strategy works best for ML model serving?**
A: Least-connections for heterogeneous models: route to the backend with fewest in-flight requests—works well when request processing times vary significantly. Round-robin for homogeneous stateless models: simple and effective when all backends have equal capacity. Latency-aware (P2C: Power of Two Choices): randomly pick 2 backends, route to the faster one—statistically approaches optimal routing with low overhead. Avoid pure round-robin for GPU serving: if one replica is saturated (slow), round-robin keeps sending it traffic, causing cascading degradation.

**Q: How do you handle sticky sessions for stateful model inference?**
A: Most ML models should be stateless (predict independently for each request). When state is needed (conversational AI, streaming generation): implement session affinity at the load balancer (route same session_id to same backend), use an external state store (Redis) that any backend can access (better—removes stickiness requirement), or use a request router that forwards state context with each request. Stateless design is strongly preferred because it simplifies scaling, failover, and deployment—use external state stores rather than in-process state.

**Q: What health check configuration prevents sending traffic to model replicas that are warming up?**
A: Configure separate liveness and readiness probes. Readiness probe: HTTP GET /health/ready with a timeout of 30s and failure threshold of 2. In the readiness endpoint, verify: model is loaded (run a synthetic inference), GPU memory is allocated, and all dependencies are reachable. Set initialDelaySeconds equal to your worst-case model loading time (e.g., 60s for a 7B model). Never route traffic to a replica that hasn't passed readiness—a partially loaded model produces unpredictable outputs.

**Q: How do you do weighted traffic routing for gradual model rollouts?**
A: Implement at the load balancer level with weighted backends: 95% traffic to stable model, 5% to new model candidate. Increment the new model's weight in stages (5% to 10% to 25% to 50% to 100%), with a hold period at each stage to validate metrics. Many load balancers (NGINX, Envoy, AWS ALB) support weighted routing natively. For A/B testing, add a request ID header to enable per-user consistency (same user always hits the same model version across requests).

**Q: What are the failure modes when a load balancer becomes a bottleneck?**
A: The load balancer itself can become a single point of failure or a throughput bottleneck. Signs: load balancer CPU usage >80%, queue depth on the load balancer increasing, latency from client to backend exceeds latency from load balancer to backend by >20ms. Mitigation: use a managed load balancer service (AWS ALB, GCP Load Balancer) that auto-scales, implement client-side load balancing for service-to-service calls (avoids the load balancer for internal traffic), and distribute traffic across multiple load balancer instances with anycast or DNS-based routing.

""",

"11-blue-green-deployment": """\
## Interview Q&A

**Q: What are the infrastructure requirements for blue-green deployment of ML models?**
A: Requires: capacity to run two full production environments simultaneously (2x cost during deployment), a load balancer that can switch traffic instantaneously, automated health checking for the green environment before traffic switch, and a rollback mechanism that can switch back in <5 minutes. For ML specifically: both environments need access to the same feature store and inference infrastructure, model artifacts must be pre-loaded in the green environment before traffic switch, and both environments must produce identical predictions for the same inputs (to validate correctness before switch).

**Q: How long should you maintain the blue environment after switching traffic to green?**
A: Maintain blue for: the time needed to detect slow-burn failures (P99 latency degradation, accuracy drift) that don't appear immediately. Minimum: 24 hours after the switch, longer for models where business impact takes time to manifest (recommendation models: 48-72 hours to see session depth changes). During this window, keep blue warm (don't scale down) so rollback is instantaneous, not another deployment. Delete blue after: deployment is considered stable AND rollback window has passed.

**Q: How do you validate the green environment before switching traffic?**
A: Automated checks: run the full model evaluation suite on the green endpoint, replay a sample of recent production requests and compare predictions to blue (should match for the same inputs unless the model deliberately changed), run performance benchmarks (latency/throughput within 10% of blue), verify all dependencies are reachable. Manual validation: have the team test the green endpoint on realistic inputs. Gate the traffic switch on all automated checks passing—never switch manually without automated validation.

**Q: When is blue-green deployment inappropriate for ML models?**
A: Inappropriate when: 2x infrastructure cost is prohibitive, the new model requires different infrastructure (different GPU type, different serving framework) making side-by-side impossible, or the model change is intentionally backward-incompatible (different input schema). Alternatives: canary deployment (route small % of traffic to new model, not 0/100% switch), shadow mode (run new model but ignore its predictions), or rolling deployment (replace replicas one by one with brief overlap period). Choose based on your risk tolerance and infrastructure constraints.

**Q: How do you handle database schema changes in conjunction with blue-green ML deployments?**
A: Database changes that affect feature computation or model metadata must be backward-compatible with both blue and green. Expand-contract pattern: (1) expand—add new column/table without removing old, both blue and green work; (2) cut over—switch traffic to green; (3) contract—remove old column/table after blue is decommissioned. Never do a one-step database migration that breaks the currently-live blue environment. Feature store schema changes require the same careful coordination.

""",

"12-canary-deployment": """\
## Interview Q&A

**Q: What percentage of traffic should you route to a canary and for how long?**
A: Start with 1-5% of traffic: enough to get statistical signal, small enough to limit blast radius if the canary fails. For models with high-volume traffic (>10K req/s), 1% gives 100+ req/s to the canary—sufficient for significance in 1-2 hours. For lower-traffic models, increase to 5-10% to accumulate enough samples. Hold at each traffic level for at least 1 hour (longer for metrics that take time to manifest: session-level metrics need multiple user sessions). Stop at 50% and validate before proceeding to 100%.

**Q: How do you decide which metrics trigger an automatic canary rollback?**
A: Rollback triggers should be: latency degradation (canary P99 > baseline P99 x 1.5), error rate increase (canary error rate > baseline x 2), prediction quality degradation (primary business metric drops > threshold), or model-specific anomaly (prediction distribution shifts significantly). Set thresholds conservatively—it's better to roll back a good model unnecessarily than to leave a bad model running. Log all automatic rollbacks and review them; false rollbacks reveal metric sensitivity issues or model quality improvements that look like regressions.

**Q: How do you compare canary and baseline performance statistically?**
A: Use statistical tests appropriate for your metric type: t-test for continuous metrics (latency, revenue), proportion test for rate metrics (click-through rate, error rate), Mann-Whitney U for non-normal distributions. Determine minimum detectable effect and required sample size before the experiment (don't run until you see significance). Correct for multiple comparisons if testing many metrics. Report effect size, not just p-value—a statistically significant 0.1% improvement may not be practically significant.

**Q: What is a canary analysis framework and how does it differ from manual monitoring?**
A: Manual monitoring: engineers watch dashboards and decide when canary looks bad. Error-prone: humans miss slow degradations, have inconsistent thresholds, and don't account for time-of-day effects. Automated canary analysis (Spinnaker, Kayenta): continuously compares canary vs. baseline metrics, accounts for confounders (traffic patterns, time of day), applies statistical tests, and produces a pass/fail/inconclusive verdict. Automated analysis is more reliable, faster, and consistent—it should gate promotion to 100% traffic without requiring manual approval for routine deployments.

**Q: How do you run canary deployments for ML models with personalization?**
A: Personalized models require user-consistent routing: the same user should always hit either canary or baseline during the experiment, not alternate between them (which would contaminate both groups). Implement user-level assignment: hash user_id % 100, route <5% to canary for the duration of the experiment. This ensures: clean A/B comparison, consistent user experience, and accurate measurement of long-term behavioral changes (which require a user to experience only one model version).

""",

"13-shadow-mode": """\
## Interview Q&A

**Q: What is shadow mode and how does it differ from canary deployment?**
A: Shadow mode: the new model receives a copy of all production traffic and generates predictions, but those predictions are discarded—production responses still come from the current model. Canary: a fraction of users receive responses from the new model. Shadow mode has zero risk to users but requires duplicate infrastructure cost. Use shadow mode for: validating new models on production data distribution before any user exposure, comparing predictions between model versions, and testing infrastructure changes. Use canary after shadow mode validates basic correctness and quality.

**Q: How do you handle the infrastructure cost of running shadow mode at scale?**
A: Shadow mode doubles inference cost. Mitigate: sample traffic (run shadow on 10-20% of requests, not 100%), use shadow in off-peak hours for CPU/memory-bound models, share GPU capacity during low-utilization periods. For very expensive models (large LLMs), shadow mode may be cost-prohibitive—use synthetic traffic replay instead (collect real requests, replay them asynchronously against the shadow model). Track shadow mode cost as a separate budget line and time-box shadow periods (1-2 weeks max).

**Q: What metrics should you compare between shadow and production during shadow mode?**
A: Prediction comparison: what fraction of inputs get different predictions? If >10% are different, investigate root cause before promoting. Quality proxy metrics: if you have labels for recent inputs, compare accuracy. Distribution metrics: are the shadow model's prediction score distributions similar to production? Performance: shadow model latency/throughput (must meet production SLAs before promotion). Silent failure detection: are there input types where the shadow model errors while production succeeds?

**Q: How do you use shadow mode to validate model changes that are intentionally different?**
A: When you expect the new model to produce different (better) outputs, shadow comparison isn't a pass/fail test—it's data collection. Collect: the distribution of differences (what types of inputs get different predictions?), human evaluations of a sample of differing predictions (which model's output is better?), and downstream metric impact simulation (if we had served the shadow model's outputs, what would engagement/conversion look like?). Shadow mode becomes a model evaluation pipeline, not just a regression detector.

**Q: What are the limitations of shadow mode for testing ML model correctness?**
A: Shadow mode can't test: user interactions that depend on model output (a recommendation model in shadow mode can't test whether users click the shadow model's recommendations), long-term behavioral effects (session quality requires users to actually experience the model), and models with side effects (if the model's output triggers other actions). For conversational AI, shadow mode is especially limited because responses depend on user reactions, creating fundamentally different conversations. In these cases, a canary with proper user assignment is required.

""",

"14-ab-testing": """\
## Interview Q&A

**Q: How do you calculate the required sample size for an A/B test on a model?**
A: Sample size depends on: minimum detectable effect (how small an improvement matters to the business), baseline metric value and variance, desired statistical power (typically 80%), and significance level (typically 5%). Use a power calculator: n = (z_alpha/2 + z_beta)^2 x 2sigma^2 / delta^2 where delta is MDE and sigma is standard deviation. For binary metrics (click rate 10%, MDE 0.5%), you need ~30K users per variant. Run the power calculation before starting—underpowered tests give inconclusive results that waste time.

**Q: What is the novelty effect and how does it affect ML A/B test results?**
A: Users initially engage more with novel features or UI changes regardless of quality, inflating the new model's metrics in the first few days. This artificial boost fades after users adapt. To control for novelty: run experiments for at least 2 weeks, look for a "novelty decay" pattern in daily metric trends (high engagement early, decreasing to steady state), analyze new vs. returning users separately, and discount first-day results in your final analysis. Declaring victory based on week-1 data is a common mistake.

**Q: How do you handle network effects in A/B tests for models that affect shared resources?**
A: Network effects (one user's experience affecting another's) violate the independence assumption of standard A/B tests. Examples: recommendation models where popular items compete, ranking models where showing an item to one user affects its ranking for others. Mitigation: use cluster-based randomization (assign entire user cohorts to treatment/control, not individuals), use time-based assignments (all users on Tuesdays get treatment, Wednesdays get control), or use holdout groups (treatment: 95% of users, holdout: 5% who never see the new model as a long-term control).

**Q: When is A/B testing not appropriate for evaluating ML models?**
A: Not appropriate when: the experiment would expose some users to a degraded experience that's ethically unacceptable (medical diagnosis, safety-critical systems), the metric takes too long to manifest relative to your release cycle (lifetime value changes that take months), the sample size required exceeds available traffic, or you can't randomize users cleanly (e.g., recommendations in a social network with strong network effects). In these cases: use shadow mode, offline evaluation on historical data, or quasi-experimental designs (before/after analysis with a holdout).

**Q: How do you avoid running too many A/B tests simultaneously?**
A: Simultaneous tests cause interaction effects if the same users are in multiple experiments. Avoid by: using an experiment mutex (users can only be in one experiment at a time for the same feature area), prioritizing experiments by expected impact and run sequentially, using a factorial design when you must test multiple things together. Track experiment coverage: what fraction of users are in at least one experiment? Above 60%, interaction effects become significant. Budget experiment slots as a limited resource.

""",

"15-drift-detection": """\
## Interview Q&A

**Q: What is the difference between data drift, concept drift, and model drift?**
A: Data drift (covariate shift): the distribution of input features changes, but the relationship between features and target stays the same—retrain with new data. Concept drift: the relationship between features and target changes (e.g., user behavior patterns shift after a major event)—need to collect new labels and retrain, not just update feature distributions. Model drift: model performance degrades over time due to either data or concept drift—the observable symptom, caused by one of the above. Diagnosing which type is occurring determines the remediation.

**Q: How do you detect drift without labeled production data?**
A: Unsupervised drift detection uses only input features (no labels needed): compare production feature distributions against training data using statistical tests (KS test for continuous, chi-square for categorical, MMD for multivariate). Population Stability Index (PSI) measures distributional shift for each feature. Reconstruction error from an autoencoder trained on training data is high for out-of-distribution inputs. These detect input drift immediately; output drift (model prediction distribution changes) can also be monitored without labels. Labeled validation (comparing predictions to outcomes) is the gold standard but requires waiting for labels.

**Q: How do you set thresholds for drift alerts without generating too many false alarms?**
A: Set thresholds based on empirical distribution of the metric in a stable period. Compute PSI or KS statistic for each day in a 3-month historical window, take the 95th percentile as the alert threshold. This accounts for natural seasonal variation and traffic patterns. Use dynamic thresholds that adjust for seasonality (a metric may look "drifted" every holiday season but it's predictable). Alert on sustained drift (3+ consecutive days above threshold) rather than single-day spikes.

**Q: How do you prioritize which features to monitor for drift?**
A: Monitor: features with high importance (top 10 by SHAP/permutation importance), features known to be operationally unstable (external data sources, calculated fields), and features that have drifted before. Don't monitor exhaustively—with 100+ features, even 5% false positive rate means 5 false alarms per day. Use feature importance to prioritize: a drift in a low-importance feature has minimal impact on model performance, while drift in a top-5 feature is likely to cause significant degradation.

**Q: What is the appropriate response to detected drift and how do you automate it?**
A: Immediate response: alert the model owner with drift details and estimated impact on model performance. Automated triage: run model quality metrics on recent predictions to quantify actual performance degradation (not just distribution shift). If degradation is significant: trigger retraining pipeline, route traffic to fallback model if retraining takes too long. Not all drift requires action: investigate whether the drift represents a permanent change (retrain) or a temporary spike (wait and monitor). Build a drift response runbook with decision criteria for each response level.

""",

"16-monitoring-and-observability": """\
## Interview Q&A

**Q: What is the difference between monitoring and observability for ML systems?**
A: Monitoring: tracking predefined metrics against thresholds—tells you when something is wrong. Observability: the ability to understand system state from its outputs (logs, metrics, traces) without knowing what to look for in advance—helps you figure out why something is wrong. ML systems need both: monitoring catches known failure modes (latency SLA violations, error rate spikes), observability helps debug novel failures (why did predictions suddenly change for one user segment?). Invest in structured logging and distributed tracing for observability—dashboards alone are insufficient.

**Q: What are the essential metrics to monitor for an ML prediction service?**
A: Infrastructure: request rate, error rate (5xx), P50/P95/P99 latency, CPU/GPU utilization. ML-specific: prediction score distribution (compare to training baseline), feature value distributions for key features, model confidence distribution (are predictions getting less confident?), prediction volume per class. Business: downstream metric that the model affects (CTR, conversion rate, fraud caught). Alert on: sudden changes in any metric (>3σ from rolling average), sustained high error rate (>1% for 5 minutes), latency exceeding SLA.

**Q: How do you implement distributed tracing for ML inference pipelines?**
A: Add a trace_id to every request at entry point (API gateway or client). Propagate trace_id through: feature retrieval, pre-processing, model inference, post-processing, and response. Each component logs: trace_id, component name, start time, duration, and relevant metadata. Use OpenTelemetry for standardized tracing instrumentation. Aggregate traces in a backend (Jaeger, Zipkin, Datadog). This enables: end-to-end latency breakdown (is the bottleneck feature retrieval or inference?), debugging per-request failures, and performance profiling.

**Q: How do you monitor model quality when labels are delayed or unavailable?**
A: Proxy metrics: if direct labels take weeks, identify a leading indicator that correlates with quality (click-through for recommendations, transaction completion for credit decisions). Prediction confidence monitoring: track the distribution of model confidence scores—a shift to lower confidence indicates the model is less certain about its predictions. Output monitoring: for classification, track prediction class distribution; for regression, track output value distribution. Human evaluation: regularly sample and manually evaluate a small fraction of predictions.

**Q: What is an ML system on-call runbook and what should it contain?**
A: The runbook should enable an on-call engineer unfamiliar with the ML model to diagnose and respond to incidents. Include: description of what the system does and what failure looks like to users, decision tree for common alert types (latency spike → check GPU utilization → check feature store latency → check upstream data), commands to run for each diagnostic step, thresholds for escalation to ML team vs. self-resolving, rollback procedure (which command to run, expected time to take effect), and contact list with escalation path.

""",

"17-model-debugging": """\
## Interview Q&A

**Q: How do you systematically debug a production model that starts producing wrong predictions?**
A: Step 1: Isolate the scope—is it all predictions or a specific subset (one feature value, one user segment, one time period)? Step 2: Check data pipeline—are features computed correctly, is there new data quality issue? Step 3: Check model inputs—are any features outside training distribution? Step 4: Check model version—did a recent deployment change anything? Step 5: Check the prediction itself—run the input through the model with logging to see the reasoning chain. Start with the simplest hypothesis (data issue) before assuming model bug.

**Q: What information should you log for model predictions to enable debugging?**
A: Log: request ID, timestamp, model version, all input features (or their hashes for PII), raw model output (logits/probabilities, not just final prediction), prediction latency, and the session/user context. For LLMs: log the full prompt, the completion, token counts, and model confidence if available. Store these logs in a queryable format (not just as text). Retention: 30 days for full logs, then aggregated statistics indefinitely. Never debug production issues without this data—it's the equivalent of application error logging for ML systems.

**Q: What are the most common root causes of sudden model performance degradation?**
A: In order of frequency: (1) data pipeline change—upstream feature changed schema, computation bug, missing data; (2) traffic distribution shift—new user segment, marketing campaign changes input distribution; (3) code deployment—serving code change introduced bug; (4) model registry issue—wrong model version deployed; (5) infrastructure change—hardware, library version change affects output; (6) label/feedback loop—model's predictions affect future training data. Check these in order before concluding "the model got worse."

**Q: How do you identify which features are causing a model to make incorrect predictions?**
A: Use SHAP values for individual predictions: compute per-feature contribution for the incorrect predictions and compare to correct ones. Look for: features with unusually high or low values in the failure cases, features where the contribution sign is reversed compared to typical predictions. Use slice-based analysis: segment predictions by feature bins and compute accuracy per slice—the slice with worst accuracy identifies the problematic feature range. A/B compare the feature distributions between your true positives, true negatives, false positives, and false negatives.

**Q: How do you debug a model that is biased toward one class in production?**
A: Check: class distribution in recent production predictions vs. training data, threshold calibration (is the decision threshold still appropriate for current traffic?), class distribution in recent training data (if model was recently retrained), and feature distributions for each class (are class distributions shifting differently?). If the class distribution in predictions has shifted without a corresponding shift in true class distribution, the model may need recalibration. If true class distribution has shifted (concept drift), retraining is needed.

""",

"18-model-explainability": """\
## Interview Q&A

**Q: What is the difference between model interpretability and model explainability?**
A: Interpretability: understanding why a model makes predictions by examining its internal structure—inherent property of simple models (linear models, decision trees) that humans can directly inspect. Explainability: post-hoc techniques applied to black-box models to approximate their behavior for specific predictions—SHAP, LIME, attention weights. Interpretable models are always explainable; explainable models may not be interpretable (SHAP provides per-prediction explanations but doesn't make a neural network "interpretable"). Use interpretability when regulatory requirements demand it; explainability when debugging or communicating decisions.

**Q: When is SHAP preferable to LIME for explaining individual predictions?**
A: SHAP is preferable when: you need globally consistent feature importance (SHAP values satisfy axioms like efficiency and symmetry that LIME doesn't guarantee), you have tree-based models (TreeSHAP is exact and fast), or you need to aggregate explanations across many predictions. LIME is preferable when: SHAP is too slow (SHAP on deep neural networks requires approximation), you want local linearity (LIME's local linear model is easier to communicate), or you need custom input perturbations for complex input types (text, images with domain-specific perturbations).

**Q: How do you use explainability in production to debug model failures?**
A: For each reported model failure (wrong prediction, customer complaint): compute SHAP values for the failing example, identify which features have unusually high attribution, compare the feature values and attributions to similar correct predictions. Build a "failure explanation dashboard": aggregate explanations for the worst predictions, identify systematic patterns (feature X has unexpectedly high negative attribution in 60% of failures). This turns individual failure analysis into systematic model improvement signal.

**Q: What are the limitations of SHAP-based explanations that you should communicate to stakeholders?**
A: SHAP explains predictions in terms of feature contributions, but it cannot: prove causality (high SHAP for a feature doesn't mean changing it would change the prediction), guarantee that the explanation reflects the model's actual decision process for deep networks (it approximates), or account for feature correlations fully (correlated features may have their importance split arbitrarily). Also: explanations are for single predictions, not the model's general behavior. Communicate these limitations explicitly when sharing explanations to prevent overconfidence in their meaning.

**Q: How do you build explainability into a model from the design phase rather than as an afterthought?**
A: Design choices: prefer simpler models when accuracy is comparable (a gradient boosted tree with 94% accuracy is more explainable than a neural network with 95%); use attention mechanisms for text models (attention weights give rough explanations); design features to be human-interpretable (one-hot encode rather than embed categorical features when explanation matters); include explanation-specific features in the architecture (auxiliary objectives that force the model to learn interpretable representations). Document the explanation strategy in the model card before training.

""",

"19-interpretability": """\
## Interview Q&A

**Q: When does a model need to be interpretable by regulatory requirement?**
A: Regulations requiring interpretability: EU GDPR Article 22 (right to explanation for automated decisions that significantly affect individuals), US Equal Credit Opportunity Act (ECOA requires adverse action notices explaining credit decisions), EU AI Act (high-risk systems must provide documentation and explanations), and various financial regulations (SR 11-7 guidance for model risk management). Healthcare (FDA SaMD): not strictly required but expected in submissions. When unsure, consult legal/compliance before choosing an opaque model for regulated decisions.

**Q: What are the trade-offs between model complexity and interpretability?**
A: Linear regression: fully interpretable, limited expressivity—use for low-dimensional, linear problems. Gradient boosted trees: partially interpretable via SHAP, high expressivity—good default for tabular data. Neural networks: black-box, highest expressivity—use when expressivity is essential and interpretability can be provided post-hoc. The interpretability-accuracy trade-off is smaller than often assumed: for many tabular tasks, a tuned gradient boosted tree matches neural network accuracy. Reach for the most interpretable model that meets accuracy requirements.

**Q: How do concept-based explanations (TCAV) differ from feature-based explanations (SHAP)?**
A: SHAP: explains predictions in terms of input feature importance—good for tabular data, less meaningful for images (pixel importance is hard to interpret). TCAV (Testing with Concept Activation Vectors): trains probes to detect human-defined concepts (e.g., "striped texture" for animal classification), then measures how much each concept influences model predictions. Better for: vision models, complex feature spaces where individual features lack semantic meaning, and testing for known biases (does the model use "woman in kitchen" as a concept for chef predictions?).

**Q: How do you evaluate whether an explanation is faithful to the model's actual decision process?**
A: Faithfulness tests: (1) feature ablation—remove the top-k features by explanation importance and measure prediction change (should change significantly); (2) roar/FRESH test—retrain with important features masked and compare accuracy drop; (3) sanity checks—perturb random (unimportant) features and verify explanation doesn't change. Beware: many explanation methods pass visual inspection but fail faithfulness tests—LIME explanations in particular can be unfaithful to the model's actual reasoning for complex models.

**Q: How do you communicate model explanations to non-technical stakeholders?**
A: Use concrete language, not statistical terms: "The model predicted high fraud risk because the transaction occurred at an unusual time (3am) and in a city different from the customer's registered address" rather than "SHAP values for time_of_day=-0.34 and location_delta=0.52." Visualize: waterfall charts for feature contributions, decision trees for simple rules. Provide action: "To reduce fraud risk, verify transactions that occur [condition]." Validate communication with target audience before using in production.

""",

"20-feature-importance-tracking": """\
## Interview Q&A

**Q: How do you use feature importance to identify when a model should be retrained?**
A: Track feature importance over time using a sliding window of recent predictions. When important features' importance ranks change significantly (feature that was always top-5 drops to top-20), investigate: has the feature distribution changed? Has its correlation with the target shifted? Use importance change as a leading indicator of model degradation. Set alerts when top-5 feature importances shift >30% from historical baseline. Importance change often precedes measurable accuracy degradation by days to weeks, enabling proactive retraining.

**Q: What are the limitations of permutation importance vs. SHAP importance?**
A: Permutation importance: shuffle one feature, measure accuracy decrease—captures total effect including correlation with other features. Limitation: correlated features split importance between them unpredictably. SHAP importance: average absolute SHAP values—captures marginal contribution while accounting for correlation. More reliable for correlated features. Use permutation importance for quick feature selection; use SHAP for debugging and communication. Both can mislead when features are highly correlated—be skeptical of any single importance metric for correlated feature sets.

**Q: How do you use feature importance to detect data leakage?**
A: Leakage indicator: a feature has suspiciously high importance (>3x the next feature), especially a feature that shouldn't logically cause the target. Examples: a transaction timestamp in a fraud model (fraud transactions may be processed later), a post-event feature in a before-event prediction. For each high-importance feature, ask: "Could a value of this feature only be known if we already knew the outcome?" If yes, it's leakage. Verify by checking whether removing the suspicious feature causes a large accuracy drop—it will if it's leaking the target.

**Q: How do feature importance values change between model versions and what do you do with that information?**
A: Changes to expect: new features introduced in a retrain may rank highly, displacing existing features. Features whose distribution has changed may rank differently. Important features that now rank low may indicate the model has learned a different solution (possibly through shortcuts). Compare feature importance between model versions before promoting: if a version has a completely different feature importance ranking, investigate even if accuracy is comparable—it may have learned a different (possibly less robust) function.

**Q: How do you use feature importance for feature selection in production models?**
A: Eliminate features with near-zero importance if: (1) they're expensive to compute (API calls, complex aggregations), (2) they add noise that reduces interpretability, or (3) you want to reduce serving complexity. Never eliminate features based on importance alone without measuring the impact on model accuracy—low-importance features may still contribute to edge-case performance. Implement a feature ablation test: retrain without the candidate features and measure accuracy on a held-out test set. Remove only features where ablation shows <1% accuracy change.

""",

"21-reproducibility": """\
## Interview Q&A

**Q: What are the minimum requirements for a reproducible ML experiment?**
A: Code: exact git commit hash. Data: exact dataset version (hash or immutable reference). Environment: exact library versions (requirements.txt or conda environment), hardware type. Random seeds: set in NumPy, PyTorch, and Python random for all operations that use randomness. Configuration: all hyperparameters stored in a config file or experiment tracker. Given these five things, you should be able to re-run the exact experiment. If you can't, identify which element is not reproducible and fix it.

**Q: How do you handle non-determinism in GPU training for reproducibility?**
A: GPU operations are non-deterministic by default (CUDA's parallel reduction algorithms are non-associative). Enable determinism: set CUDA_LAUNCH_BLOCKING=1, torch.use_deterministic_algorithms(True), and cuDNN benchmark off (torch.backends.cudnn.benchmark = False). Trade-off: deterministic mode can be 20-50% slower. Use it for: debugging experiments where you need to isolate changes, final training runs before deployment. Accept non-determinism in exploratory training but record the variance range to understand expected performance variation.

**Q: What is the difference between reproducibility and replicability in ML research?**
A: Reproducibility: re-running the same code with the same data produces the same result (computational reproducibility). Replicability: running a different implementation of the same method on different data produces similar results (scientific replicability). For production ML: focus on computational reproducibility—you need to rebuild models exactly in case of rollback or audit. Replicability matters for research claims but is a separate concern. Production model training should be reproducible within the same compute environment; replicability across environments is harder and often not required.

**Q: How do you maintain reproducibility when training data comes from a live production database?**
A: Never train directly against a live database—snapshot training data at a fixed point in time and store it immutably. Create a training data snapshot pipeline: extract data with explicit filters (date ranges, version flags), store in versioned storage (S3 with versioning, Delta Lake), log the query used and execution timestamp. If you must refresh training data, create a new dataset version rather than overwriting the old one. The training pipeline should accept a dataset version as input, not query the live database directly.

**Q: How do you audit which training data was used to produce a production model?**
A: Maintain a model registry entry that links: model artifact to training run to dataset version to raw data snapshots. Store the dataset hash alongside the model artifact. For regulated industries, this audit trail must be preserved for the lifetime of the model (often 5-7 years). Test your audit capability: given a model version in production, can you reproduce the exact training dataset? If not, your audit trail is incomplete. Tools: MLflow lineage tracking, SageMaker Experiments, or custom PostgreSQL lineage tables.

""",

"22-cost-optimization": """\
## Interview Q&A

**Q: What are the highest-ROI cost optimization strategies for ML inference workloads?**
A: In order of ROI: (1) right-sizing instances (25-40% savings: match GPU memory to model size), (2) model quantization INT8/INT4 (2-4x cost reduction with minimal quality loss), (3) auto-scaling (30-50% savings by eliminating idle capacity during off-peak), (4) spot/preemptible instances for batch inference (60-70% savings), (5) continuous batching (2-3x throughput improvement on the same hardware). Model distillation (smaller model) has highest long-term ROI but requires upfront training cost and quality validation.

**Q: How do you balance model quality and inference cost when making optimization trade-offs?**
A: Quantify the cost of quality degradation: A/B test a cheaper model (smaller, quantized, faster) and measure business metric impact. If the cheaper model costs 50% less and reduces business metric by 2%, that's a concrete trade-off decision for stakeholders, not a purely technical one. Some quality degradation is worth the cost savings; some isn't. Present the trade-off with data, don't make the decision unilaterally. Implement the optimization with a canary deployment and measure actual business impact before full rollout.

**Q: How do you identify overprovisioned ML infrastructure?**
A: Indicators of overprovisioning: GPU utilization consistently <30%, CPU utilization <20%, memory usage <50% of provisioned. Check: average vs. peak utilization (provision for peak, not average), request queue depth (if near zero, capacity is sufficient and may be excess), cost per prediction over time (should be decreasing as you optimize, not flat). Use cloud cost allocation tags to attribute GPU spend to specific models. A model that costs $10K/month with <30% GPU utilization is likely overprovisioned and worth investigating.

**Q: What is model cascading and how does it reduce inference cost?**
A: Model cascading: use a cheap, fast model for easy cases and a powerful, expensive model only for hard cases. Example: a small BERT model handles 80% of text classification requests with high confidence; uncertain cases (<70% confidence) are escalated to a larger model. Cost reduction: 80% of requests use the cheap model, reducing average cost by 70-80% with minimal quality impact (the expensive model handles the cases it's actually needed for). Design the cascade threshold by measuring the accuracy distribution of the cheap model on a validation set.

**Q: How do you estimate and forecast ML infrastructure costs?**
A: Build a cost model: cost = f(requests per day, average tokens/request, model size, GPU type, batch size efficiency). Track: actual cost per 1K predictions by model, cost trend over time, cost breakdown by component (inference vs. feature store vs. data pipeline). Forecast using request growth projections. Alert when: actual cost exceeds forecast by >20% (unexpected usage spike or efficiency regression), cost per prediction increases unexpectedly (optimization regression). Make cost a first-class metric alongside accuracy and latency.

""",

"23-production-readiness": """\
## Interview Q&A

**Q: What constitutes an ML system being "production ready" and who decides?**
A: Production readiness is multidimensional: functional (correct predictions on held-out test set meeting accuracy SLA), operational (latency/throughput SLAs met under load), reliability (99th percentile availability, graceful degradation under failure), observability (monitoring, alerting, and runbooks in place), compliance (security review, data governance, explainability documentation), and rollback capability (can revert in <10 minutes). Decision: a readiness review with representatives from ML, engineering, product, and security—not just ML team sign-off.

**Q: How do you load test an ML model serving endpoint before production launch?**
A: Use realistic traffic patterns: real request payload sizes from historical data, real traffic shape (not just sustained constant load—include spikes). Test: sustained load at 1.5x expected peak, burst test (2x peak for 60 seconds), ramp test (gradual increase to identify breaking point), soak test (sustained load for 4 hours to detect memory leaks or resource exhaustion). Measure: P50/P95/P99 latency, error rate, GPU/CPU utilization, memory growth. Automate load tests as part of the staging environment.

**Q: What are the most common production readiness failures that cause outages after launch?**
A: Inadequate load testing: system performs fine at 10 RPS but fails at 100 RPS. Missing error handling: model serves 500s on edge-case inputs that weren't in test data. Insufficient monitoring: failure isn't detected for hours because no alert was configured. No runbook: on-call engineer doesn't know how to respond to the model's specific failure modes. Underestimated startup time: deployment causes outage because new pods take 3 minutes to become ready but the deployment strategy assumed 30 seconds. Each of these is preventable with a comprehensive readiness checklist.

**Q: How do you document a model's known limitations for production users?**
A: Create a model card (standardized by Google/Hugging Face) that includes: intended use cases and out-of-scope uses, performance metrics by demographic group and data slice, known failure modes with examples, data sources and training methodology, recommended minimum input requirements, and edge cases to handle explicitly. Store the model card in the model registry and require its completion as part of the promotion gate. Communicate limitations to consuming teams—they need to build appropriate safeguards in their applications.

**Q: What security review elements are specific to ML systems vs. standard software?**
A: ML-specific security concerns: model inversion attacks (extracting training data from model outputs), adversarial examples (inputs crafted to fool the model), data poisoning (attacking the training pipeline to embed backdoors), model stealing (using API queries to replicate the model), and prompt injection (for LLM-based systems). Review: access controls on training data and model artifacts, rate limiting on inference endpoints, input validation to detect adversarial patterns, monitoring for unusual query patterns that may indicate model extraction attacks.

""",

"24-bias-detection": """\
## Interview Q&A

**Q: What is the difference between disparate impact and disparate treatment in ML models?**
A: Disparate treatment: using a protected characteristic (race, gender) directly as a model feature—illegal in most jurisdictions for consequential decisions. Disparate impact: the model doesn't use protected characteristics directly but produces significantly different outcomes for protected groups—can also be illegal even without discriminatory intent. Test for both: check which features the model uses, and separately measure outcomes across demographic groups. A model can have disparate impact even with no disparate treatment if proxy variables correlate with protected characteristics.

**Q: How do you measure bias in a model when you don't have demographic labels?**
A: Proxy inference: use name-based ethnicity inference (Bayesian Improved Surname Geocoding), zip code as a proxy for race/income, gender inference from name. These proxies are imperfect but can detect gross disparities. Audit vendor: hire a third-party audit firm with specialized bias detection tools. Analyze proxy features: if the model heavily weights zip code, investigate whether that creates disparate impact. Test with synthetic data: create matched pairs that differ only in demographic-correlated attributes and measure prediction differences.

**Q: What are the fundamental trade-offs between different fairness metrics?**
A: It is mathematically proven that you cannot simultaneously satisfy demographic parity (equal positive prediction rates across groups), equalized odds (equal TPR and FPR across groups), and predictive parity (equal precision across groups) unless base rates are equal across groups (which they rarely are). Choose the fairness metric that aligns with your use case: criminal justice—prefer equalized odds (equal error rates). Hiring—prefer demographic parity (equal opportunity). Medical diagnosis—prefer predictive parity (equal reliability of positive prediction).

**Q: How do you implement a bias monitoring system for production models?**
A: Compute fairness metrics (demographic parity ratio, equalized odds difference, disparate impact) on a rolling window of recent predictions. Compare against: legal thresholds (4/5 rule: adverse impact ratio <0.8 triggers investigation), historical baselines, and peer models. Alert when: fairness metrics degrade significantly, prediction volume for specific groups changes (may indicate distribution shift), or outcomes for groups diverge. Store all bias metrics with the same rigor as accuracy metrics—they're equally important for responsible deployment.

**Q: What interventions can you use to mitigate bias in a deployed model?**
A: Pre-processing: rebalance training data, remove or transform biased features. In-processing: add fairness constraints to the loss function (adversarial debiasing, fairness regularization). Post-processing: adjust decision thresholds per demographic group to equalize outcomes. Monitoring: implement feedback loops to detect bias creep after retraining. Choosing between them: post-processing is fastest to implement but may reduce overall accuracy; pre-processing addresses root causes but requires new training. Start with post-processing to demonstrate the fix is feasible, then address root causes through data and training changes.

""",

"25-fairness-metrics": """\
## Interview Q&A

**Q: When is demographic parity the appropriate fairness metric to use?**
A: Demographic parity (equal positive prediction rates across groups) is appropriate when: the selection process should be equal-opportunity regardless of group-specific base rates (e.g., hiring from historically underrepresented groups), the harm of under-selection is severe, and when you're trying to correct historical imbalances. It's inappropriate when: there are genuine group differences in the underlying construct being measured (e.g., demographic parity in medical diagnosis would mean treating equal numbers of sick and healthy people from each group regardless of disease rates).

**Q: How do you choose between equalized odds and equal opportunity as fairness criteria?**
A: Equal opportunity: only requires TPR (sensitivity) to be equal across groups—acceptable false positive rate disparity. Use when: false positives are low-harm (showing an ad to someone not interested). Equalized odds: requires both TPR and FPR to be equal—stricter, ensures errors are equally distributed. Use when: both false positives and false negatives have significant consequences (credit decisions, criminal justice). In practice, perfect equalized odds is impossible without equal base rates—choose which error type is more important and enforce equality on that.

**Q: How do you handle intersectionality in fairness evaluation?**
A: Intersectionality: disparities may exist for combinations of protected characteristics (e.g., Black women may face bias that doesn't appear when analyzing race or gender separately). Test: measure fairness metrics for all intersectional subgroups in your data. Challenge: small sample sizes for intersection subgroups reduce statistical power. Mitigation: combine rare intersection groups thoughtfully (don't merge to the point of obscuring real disparities), use bootstrap confidence intervals to communicate uncertainty, and prioritize reporting subgroups with enough data for reliable measurement.

**Q: What is calibration fairness and why does it matter?**
A: Calibration fairness: the model's predicted probabilities match actual rates equally well across groups. A model is calibrated for group A if "70% confident predictions" turn out to be correct 70% of the time for group A—same requirement for all groups. Calibration parity is important for: any application where the model's confidence score is used for decision-making (risk scores, medical probabilities), because a model that is well-calibrated overall but poorly calibrated for specific groups systematically misleads decisions for those groups.

**Q: How do you communicate fairness metrics to business stakeholders who aren't familiar with the technical definitions?**
A: Use concrete language: "The model approves loan applications from Group A at 68% and Group B at 55%—a 19% difference" rather than "Demographic parity difference is 0.13." Show impact: "This gap means approximately 3,000 additional Group B applicants per year would be approved if approval rates were equal." Frame in terms of regulatory risk: "Disparate impact ratios below 0.8 (currently 0.81) trigger regulatory scrutiny." Use comparison to human baseline: "Before the model, the human underwriter gap was 25%—the model reduced it to 19%."

""",

"26-data-governance": """\
## Interview Q&A

**Q: What data governance processes are needed specifically for ML training data?**
A: ML-specific governance requirements beyond standard data governance: consent documentation for personal data used in training, purpose limitation documentation (data collected for X used to train model Y—is that in scope?), lineage tracking from raw data to model artifacts, training data retention policies that align with model lifecycle, and re-use policies when sharing training data across teams. Additionally: documentation of which demographic groups are represented in training data (affects bias and fairness documentation requirements).

**Q: How do you implement data access controls for ML training pipelines?**
A: Principle of least privilege: the ML training job should only access the specific data needed for that training run. Implement: IAM roles scoped to specific S3 buckets/prefixes, time-limited access tokens for training jobs, row-level security in databases (the training pipeline only sees the rows it's authorized for). Audit: log all data access during training and store with the model metadata. Prevent: training pipelines from having write access to source data, production models from having direct access to training data after deployment.

**Q: What is the right approach to managing PII in ML training data?**
A: Assess necessity: does the model actually need PII to work? Anonymize or pseudonymize if not. If PII is needed: document the legal basis, implement data minimization (only the PII fields needed), set retention limits (delete PII from training data after model deployment + x months), implement purpose binding (this dataset can only be used for model Y), and encrypt PII at rest. For model outputs: ensure the model can't reproduce training PII (test for memorization), implement output filtering for PII patterns.

**Q: How do you handle data governance for externally sourced datasets?**
A: Review the license terms carefully: can you use this data for commercial models? Is attribution required? Are there restrictions on model distribution? Common gotchas: web-scraped data may have conflicting copyright status, some open datasets prohibit commercial use, GDPR-covered data has special requirements. Track all external dataset licenses in your data catalog. Before using a new external dataset, require legal review. Avoid the "interesting dataset" trap: don't collect datasets without a clear use case and governance plan.

**Q: What does a data governance review process look like before training a new model?**
A: Checklist: (1) data provenance review—where does this data come from, is there consent/license? (2) PII assessment—what personal data is included, what's the legal basis for use? (3) bias assessment—which demographic groups are represented, is the sample representative? (4) retention plan—how long will training data be stored? (5) access control review—who can access this training data? (6) purpose documentation—what is this model for, is this data appropriate for that purpose? Require sign-off from data governance, legal, and privacy teams before training.

""",

"27-ml-governance": """\
## Interview Q&A

**Q: What is model risk management and when is it required?**
A: Model risk management (MRM) is the process of identifying, assessing, and mitigating risks from model errors. Required by: US banking regulators (SR 11-7 guidance for banks), insurance regulators, and increasingly for high-risk AI under EU AI Act. MRM includes: model documentation (model purpose, assumptions, limitations), validation (independent review of model performance), ongoing monitoring (performance degradation detection), and governance (approval process for model deployment and changes). Companies outside financial services often implement MRM voluntarily for high-stakes models.

**Q: How do you structure model documentation for an ML governance audit?**
A: Required documentation: model card (purpose, capabilities, limitations), training data description (source, size, preprocessing, known biases), model architecture description, performance metrics (on multiple slices, not just overall), validation methodology (how was the model tested?), change log (what changed from previous version), known issues and mitigations, and intended use vs. out-of-scope use. Store documentation with model artifacts in the registry, version alongside the model. An audit should be able to reproduce key performance claims from the documentation alone.

**Q: What approval workflows are needed for high-risk ML model deployments?**
A: Approval chain for high-risk models: ML team sign-off (technical validation), data governance review (data usage compliance), security review (vulnerability assessment), legal/compliance review (regulatory requirements), business owner sign-off (accepts business risk), and for regulated industries, independent model validator sign-off. Codify approvals as blocking gates in the CI/CD pipeline—a model shouldn't deploy without all required approvals recorded. Track approval status in the model registry.

**Q: How do you handle a model that has been in production but was never properly validated?**
A: Don't immediately shut down the model—that would disrupt business operations. Instead: (1) document current model version and performance as the baseline; (2) run retrospective validation against historical data and compare to the undocumented baseline; (3) if validation reveals significant issues, implement monitoring and mitigations while working toward a proper replacement; (4) establish a governance timeline for bringing the model into compliance. Prioritize: high-risk models (consequential decisions) over low-risk (internal analytics).

**Q: What is the incident response process when an ML model causes a governance violation?**
A: Immediate: assess whether to take the model offline or implement emergency mitigations (e.g., filter outputs, escalate to human review). Document: what happened, when it was detected, who was affected, what the model did. Root cause analysis: was this a known limitation, a data issue, a model failure, or a deployment error? Remediation: implement technical fix, update governance documentation, improve monitoring to prevent recurrence. Reporting: depending on severity and regulation, may require notifying affected individuals, regulators, or senior management. Establish an ML-specific incident response playbook.

""",

"28-privacy-preserving-ml": """\
## Interview Q&A

**Q: What are the main privacy-preserving ML techniques and when do you use each?**
A: Differential privacy (DP): adds calibrated noise during training to prevent individual data points from being identifiable in model outputs—use for models where model weights or outputs could reveal training data. Federated learning: train without centralizing data—use when data can't leave devices or organizational silos. Secure multi-party computation: compute on encrypted data—use for collaborative ML between competitors. Homomorphic encryption: compute on fully encrypted data—use for inference on sensitive data without decrypting. Match the technique to the threat model—DP doesn't help if an attacker has access to the training logs.

**Q: How does differential privacy affect model accuracy and how do you tune the privacy budget?**
A: DP adds Gaussian noise scaled to the sensitivity of the computation. Higher privacy (lower epsilon) requires more noise, reducing model accuracy. In practice: for epsilon=8 (strong privacy), image classifiers lose 5-15% accuracy; for epsilon=1, loss can be 20-30%. Tune epsilon by: determining the minimum acceptable accuracy on your task, testing DP training at different epsilon values, and choosing the smallest epsilon where accuracy meets the threshold. Document the privacy budget in the model card and treat it as a model property that informs deployment decisions.

**Q: What is model memorization and how do you test for it?**
A: Model memorization: the model stores specific training examples and can reproduce them when queried. Test with: membership inference attacks (can you tell if a specific example was in the training set?), training data extraction (can you prompt the model to reproduce verbatim training data?), and shadow model attacks (train a shadow model on similar data and compare behavior). LLMs are particularly susceptible to memorization of frequently-repeated text (addresses, emails, code). Use differential privacy training, deduplication of training data, and output filtering to mitigate.

**Q: How do you implement federated learning for an ML application?**
A: Federated learning architecture: (1) server sends current model to all clients; (2) each client trains on local data for N steps; (3) clients send gradient updates (not data) to server; (4) server aggregates updates (FedAvg: weighted average of gradients); (5) server updates global model and repeats. Key challenges: communication overhead (sending model updates each round), data heterogeneity (each client has a different data distribution), and stragglers (slow clients delay aggregation). Use frameworks: TensorFlow Federated, PySyft, or Flower for implementation.

**Q: What are the privacy trade-offs of model inversion vs. membership inference attacks?**
A: Model inversion: attacker reconstructs input data from model outputs—more dangerous for models with rich outputs (face recognition, generative models). Defends against: output rounding, output perturbation, limiting access to model internals. Membership inference: attacker determines if a specific data point was in training—enables stalking (was this person's medical record used?). More practical attack on most ML systems. Defends against: differential privacy, early stopping, temperature tuning, and monitoring for unusual query patterns. Test your model's vulnerability to both attacks before deploying with sensitive training data.

""",

"29-differential-privacy": """\
## Interview Q&A

**Q: What does epsilon in differential privacy mean intuitively and what values are considered strong?**
A: Epsilon bounds the privacy loss: an attacker's ability to distinguish whether a specific individual was in the training set improves by at most e^epsilon. Small epsilon means strong privacy. Interpretations: epsilon=0: perfect privacy (no information leakage); epsilon=1: 2.7x advantage for attacker with one individual's data; epsilon=8: 3000x advantage (barely better than no privacy). In practice: epsilon<1 is strong but often impractical (too much noise); epsilon=1-10 is commonly used; epsilon>10 provides weak protection. The appropriate epsilon depends on your threat model and how sensitive the data is.

**Q: How do you compose privacy budgets across multiple DP computations?**
A: Basic composition: running k DP mechanisms with epsilon values epsilon_1 through epsilon_k gives total privacy budget epsilon_total = sum of all epsilon values—budget accumulates with each computation on the same data. This means you can't run unlimited DP queries—the privacy budget depletes. Advanced composition (Renyi DP, moments accountant): achieves tighter bounds, allowing more queries within the same total budget. In practice: use Opacus's privacy engine which automatically tracks cumulative privacy loss using moments accountant. Set a total budget and stop training when it's exhausted.

**Q: When is DP-SGD preferable to aggregation-level DP (adding noise to final statistics)?**
A: Aggregation-level DP: add noise to aggregate statistics (counts, means) after computation—simpler and allows tighter privacy guarantees for simple statistics. Sufficient for: releasing demographic statistics, publishing aggregate model metrics. DP-SGD: clips and noisifies gradients during training—provides protection for each training step but accumulates privacy budget. Necessary for: training neural networks where you want the model itself to be DP. If you just need to release model performance statistics publicly, aggregation-level DP is cheaper. If you need the model weights to be DP (e.g., to prevent model inversion), use DP-SGD.

**Q: How does clipping gradient norms interact with differential privacy in DP-SGD?**
A: DP-SGD clips each per-sample gradient to a maximum L2 norm C before adding Gaussian noise. Clipping bounds the sensitivity—the noise level is calibrated to C. Too small C: clips most gradients, destroying signal and degrading accuracy significantly. Too large C: gradients barely clipped, noise is small relative to gradient magnitude but privacy loss per step is lower. Tune C to the typical per-sample gradient norm: use the median gradient norm from the first few non-DP training steps as a starting point, then adjust based on accuracy vs. privacy budget.

**Q: What are the implementation pitfalls when applying DP to ML training?**
A: Common mistakes: applying DP after batch normalization (batch norm leaks per-sample info through batch statistics—use group norm or layer norm instead), forgetting to clip per-sample gradients (not batch-averaged gradients), reusing the privacy budget for hyperparameter tuning (each evaluation depletes budget—use public validation data for tuning), and applying DP only to training but not to the evaluation data (evaluation can also leak information). Use Opacus (PyTorch) or TF Privacy (TensorFlow) rather than implementing DP from scratch.

""",

"30-federated-learning": """\
## Interview Q&A

**Q: What are the communication efficiency challenges in federated learning and how do you address them?**
A: Challenge: sending full model updates (1GB+ for large models) from millions of clients per round is infeasible. Solutions: gradient compression (top-k sparsification: only send the top-k% largest gradients, 99% compression with <5% quality loss), quantization of updates (8-bit gradients), local steps (clients take N gradient steps before sending updates, reducing communication frequency), and model architecture choices (smaller models with fewer parameters to communicate). In cross-device FL (mobile), communication cost often exceeds compute cost.

**Q: How do you handle data heterogeneity (non-IID data) in federated learning?**
A: Non-IID (non-independent identically distributed) data: each client has a different data distribution (one user only uses the app for cooking, another for travel). Standard FedAvg assumes IID data and can diverge with extreme heterogeneity. Solutions: FedProx (adds proximal term to prevent local model from diverging too far from global), SCAFFOLD (variance reduction technique), and personalized FL (train a global model + per-client adaptation layer). In practice: understand your data heterogeneity before choosing an aggregation algorithm, as different methods have different heterogeneity tolerances.

**Q: How do you prevent adversarial clients from poisoning a federated learning model?**
A: Byzantine attacks: malicious clients send adversarial gradients to steer the global model. Defenses: robust aggregation (coordinate-wise median instead of mean, Krum: select the update closest to the majority), anomaly detection (reject updates that deviate significantly from the mean), differential privacy (DP aggregation masks individual updates making poisoning harder), and reputation systems (clients with a history of good updates weighted more). No single defense is sufficient—combine multiple approaches for production FL systems.

**Q: When does federated learning provide meaningful privacy protection vs. just data locality?**
A: Data locality: data doesn't leave the device (good for compliance, data sovereignty). Privacy: the server and other clients can't infer the client's data from gradient updates. Gradient updates can leak significant private information—gradient inversion attacks can reconstruct training images from gradients. For strong privacy, combine FL with differential privacy (add noise to client gradients before sending). Without DP, FL provides data locality but not strong privacy protection—be precise about which property you're claiming when communicating about FL.

**Q: What evaluation methodology do you use for federated learning models when you can't centralize test data?**
A: Three approaches: (1) holdout clients—reserve 10-20% of clients who don't participate in training, evaluate on their local data after each round; (2) federated evaluation—send the global model to all clients, each evaluates locally and reports metrics (aggregated without sharing data); (3) public proxy dataset—use a publicly available dataset that approximates the distribution for centralized evaluation. Each has trade-offs: holdout clients reduces training data, federated evaluation adds communication overhead, and proxy datasets may not represent actual distribution.

""",

"31-disaster-recovery": """\
## Interview Q&A

**Q: What is the RTO and RPO for ML systems and how do they differ from standard software?**
A: RTO (Recovery Time Objective): how long the system can be unavailable. For real-time ML serving: minutes. For batch inference: hours. RPO (Recovery Point Objective): how much data/state can be lost. For ML systems: typically measured in terms of model version (can we recover to the last good model) and training data (can we recover the last training dataset). ML-specific RPO consideration: if the model was retrained 3 times since the last backup, recovering the last good model version is more important than recovering the last training run.

**Q: How do you design an ML system to automatically failover when the primary model fails?**
A: Implement a fallback hierarchy: (1) primary model (latest deployed); (2) previous model version (always keep the N-1 version warm); (3) rule-based fallback (simple heuristic that handles the most common cases); (4) default response (safe fallback: "unable to process at this time"). Implement health checks that trigger automatic failover: if the primary model returns error rate >5% for 60 seconds, automatically switch traffic to the previous model version. Test failover regularly in staging—circuit breakers that aren't tested often fail silently.

**Q: What backup strategy is appropriate for ML model artifacts?**
A: Treat model artifacts like critical infrastructure: daily snapshots, retained for 30 days. Keep: model weights (or compressed serialized format), model architecture definition, preprocessing pipeline, evaluation results, and model card documentation. Cross-region replication for disaster recovery (primary failure should not cause data loss). Test recovery: monthly drill where you restore a model from backup and verify it produces correct predictions. The backup is worthless if you can't restore from it in a timely manner.

**Q: How do you handle ML system recovery when the underlying data pipeline is corrupted?**
A: A corrupted data pipeline can silently produce wrong features, causing model predictions to degrade without obvious errors. Recovery steps: (1) identify the last known good data state (use data versioning); (2) halt all model retraining that uses the corrupted pipeline; (3) assess impact: which models were trained on corrupted data? (4) roll back affected models to the last version trained on good data; (5) fix the pipeline; (6) validate fixed pipeline output matches pre-corruption baselines; (7) retrain affected models. This recovery can take days if data versioning is inadequate.

**Q: What is the difference between a model rollback and a model recovery and when do you use each?**
A: Rollback: intentional reversion to a previous known-good model version—used when a newly deployed model has degraded performance. Fast (minutes), no data loss, preserves operational continuity. Recovery: restoring a model after catastrophic failure (data corruption, lost model artifacts, infrastructure failure)—used when the current model can't serve at all. May take hours, requires backup restoration. Design for rollback first (it's more common) by keeping N-1 model versions always warm and ready to serve. Recovery is the fallback when rollback isn't possible.

""",

}  # end QA_CONTENT


def process_file(filepath: str, slug: str) -> str:
    """Read file, insert Q&A section before ## Interview Quick-Reference, write back."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    marker = "## Interview Quick-Reference"
    if marker not in content:
        return f"WARNING: marker not found in {os.path.basename(filepath)}"

    if "## Interview Q&A" in content:
        return f"SKIP: Q&A already present in {os.path.basename(filepath)}"

    qa_block = QA_CONTENT.get(slug)
    if qa_block is None:
        return f"WARNING: No Q&A content defined for slug '{slug}'"

    # Insert Q&A block immediately before the marker
    new_content = content.replace(marker, qa_block + marker, 1)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    return f"OK: {os.path.basename(filepath)}"


def main():
    files = sorted(f for f in os.listdir(PATTERNS_DIR) if f.endswith(".md"))

    updated = skipped = warned = 0
    for filename in files:
        slug = filename[:-3]  # strip .md
        filepath = os.path.join(PATTERNS_DIR, filename)
        result = process_file(filepath, slug)
        print(result)
        if result.startswith("OK"):
            updated += 1
        elif result.startswith("SKIP"):
            skipped += 1
        else:
            warned += 1

    print(f"\nSummary — Updated: {updated}, Skipped (already had Q&A): {skipped}, Warnings: {warned}")


if __name__ == "__main__":
    main()
