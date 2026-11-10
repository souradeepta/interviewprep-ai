# Inference Systems Interview Practice

Use these prompts to rehearse concise system explanations. State assumptions, quantify the relevant constraint, and identify an evaluation that could disprove your design.

## Mixture of Experts

### Prompt

A top-2 MoE layer has 64 experts. In a batch, three experts receive most assignments and several tokens are dropped. Explain how you would diagnose and improve the layer without assuming uniform expert demand is the product goal.

### Role/Domain

LLM inference and model architecture.

### Difficulty

Advanced.

### Expected Answer

Inspect per-layer pre-cap gate probabilities, selected routes, accepted counts, overflow rate, gate entropy, and expert-parallel communication. Determine whether the issue is router collapse, finite capacity, skewed data or dispatch overhead. Test balancing objectives, initialization, capacity and overflow policy against task quality and utilization; measure on realistic batches.

### Evaluation

Score diagnosis quality, correct distinction between preference and accepted routing, quality-aware trade-offs, and a concrete measurement plan.

### Follow-up

How would you debug a layer where accepted counts appear balanced but validation quality drops after increasing the capacity factor?

### Common Mistake

Claiming uniform routing is always optimal or claiming sparse activation guarantees lower serving latency.

## Long Context Management

### Prompt

A question's answer is present in a long document, but the model misses it when the relevant passage is near the middle. Propose a context construction and evaluation plan under a strict token budget.

### Role/Domain

Retrieval-augmented generation and LLM application engineering.

### Difficulty

Intermediate.

### Expected Answer

Count tokens for instructions, history, evidence and reserved output. Compare source-order truncation with query retrieval and query-focused compression. Preserve source identifiers, measure supporting-span recall and answer faithfulness, and position-controlled probes to identify lost-in-middle behavior. Do not summarize away exact qualifiers without checking claims against sources.

### Evaluation

Score budget correctness, evidence recall, provenance preservation and a plan to measure answer quality and latency.

### Follow-up

How would you debug a retrieval result that is topically relevant but does not support the generated claim?

### Common Mistake

Assuming a larger context window or higher semantic similarity automatically resolves evidence-use errors.

## Speculative Decoding

### Prompt

A speculative decoder has a high average token acceptance rate but does not improve p95 latency. What do you inspect and what experiment would decide whether to keep it?

### Role/Domain

LLM serving performance.

### Difficulty

Advanced.

### Expected Answer

Break down accepted-prefix distribution, draft time, target verification time, cache/memory traffic and cycle variance by request class. Sweep draft length and compare end-to-end latency and quality parity against target-only decoding. Include zero-accept cycles, tail latency, batch shape and fallback behavior; do not infer speedup from acceptance alone.

### Evaluation

Score algorithm understanding, cost accounting, correct distinction between formula and measurement, and tail-aware experiment design.

### Follow-up

How would you debug output-distribution drift when speculation is enabled only for sampling requests?

### Common Mistake

Multiplying acceptance rate by draft length and calling the result speedup.

## KV Cache Optimization

### Prompt

Estimate the KV cache for a serving configuration, then propose a safe prefix-sharing policy for common system prompts.

### Role/Domain

LLM inference platform engineering.

### Difficulty

Advanced.

### Expected Answer

Use approximately 2 × bytes/value × layers × KV heads × head dimension × cached tokens, then account for blocks, parallel replication, temporary memory and concurrency. Prefix keys must include model and adapter version, tokenized content, positions and mask semantics. Pin live blocks, reference-count shared blocks, measure hit rate and parity-test cached against recomputed outputs.

### Evaluation

Score arithmetic, completeness of memory accounting, correctness of cache identity and eviction safety.

### Follow-up

How would you debug an output mismatch that occurs only after a model adapter rollout?

### Common Mistake

Using parameter precision as the KV precision or sharing cache entries by text alone.

## Sparse Attention

### Prompt

You are asked to replace dense causal attention with a sliding window to support longer inputs. Describe the correctness and performance gates you require.

### Role/Domain

Transformer architecture and inference optimization.

### Difficulty

Advanced.

### Expected Answer

Define mask semantics and test causal boundaries, row degree and reachability across layers. Probe dependencies farther apart than the window and compare task quality. Confirm a kernel exploits the structure, then benchmark end-to-end latency and memory against a dense optimized kernel at target lengths and batch shapes.

### Evaluation

Score mask correctness, task-fit reasoning, realistic complexity analysis and hardware-aware measurement.

### Follow-up

How would you debug lower memory use but worse latency after introducing sparsity?

### Common Mistake

Assuming a sparse mask causes the underlying dense matrix multiplication to skip work.

## Quantization and Distillation

### Prompt

A four-bit distilled model is smaller but has a regression on multilingual prompts. Design an ablation to locate the cause and decide whether the deployment trade-off is acceptable.

### Role/Domain

Model compression and deployment.

### Difficulty

Advanced.

### Expected Answer

Compare teacher FP, teacher quantized, student FP and student quantized under the same prompt and decoding suite. Inspect language slices, outlier ranges, calibration coverage and actual target-kernel performance. Separate quality loss from storage and measured latency. Set acceptance thresholds and preserve a rollback artifact.

### Evaluation

Score attribution strategy, slice-aware quality measurement, hardware-aware benchmarking and release safeguards.

### Follow-up

How would you debug a regression that appears only at long context lengths?

### Common Mistake

Treating nominal bit width as guaranteed memory reduction or latency improvement.

## Multimodal Fusion

### Prompt

A production model combines text, image and audio scores. Image capture is sometimes absent and sometimes corrupted. Explain a robust fusion and evaluation design.

### Role/Domain

Multimodal machine learning.

### Difficulty

Intermediate.

### Expected Answer

Represent absence explicitly, calibrate modality scores on paired held-out data, and define whether remaining weights renormalize, abstain or use a trained missing-input path. Evaluate every likely availability subset, corrupted channels, disagreement, coverage and subgroup outcomes. Audit pairing and preprocessing for leakage.

### Evaluation

Score missingness semantics, calibration, modality-slice testing and treatment of conflicting evidence.

### Follow-up

How would you debug confident predictions when image and text evidence disagree?

### Common Mistake

Replacing a missing score with zero and interpreting it as negative evidence.

## Prefill and Decode Scheduling

### Prompt

Long prompts are delaying first tokens for short interactive requests. Would you disaggregate prefill and decode? Explain how you would decide.

### Role/Domain

LLM serving systems design.

### Difficulty

Advanced.

### Expected Answer

Separate queue wait, prefill service, cache transfer, decode service and total latency. Compare co-located, chunked-prefill and disaggregated configurations on mixed bursty traffic; report TTFT and completion p95/p99, throughput, fairness and transferred bytes. Disaggregate only if reduced contention exceeds handoff and coordination cost.

### Evaluation

Score latency decomposition, queueing reasoning, transfer-cost accounting and workload design.

### Follow-up

How would you debug improved median TTFT with a worse p99 completion latency?

### Common Mistake

Assuming more worker pools improve latency without considering utilization and state movement.

## Post-Training Specialization

### Prompt

A support assistant gives stale policy answers and inconsistently follows a required response format. Choose an adaptation plan and explain what belongs in evaluation.

### Role/Domain

Applied LLM engineering.

### Difficulty

Intermediate.

### Expected Answer

Use retrieval for changing, citable policy facts and first test whether prompting can enforce the stable format. If a measured behavior gap persists, compare adapter tuning against prompting with representative examples. Evaluate freshness, citation support, format adherence, general regressions, latency and fallback when retrieval is empty.

### Evaluation

Score separation of knowledge and behavior gaps, minimal intervention choice and measurable release gates.

### Follow-up

How would you debug a system that retrieves the correct policy but still gives the old answer?

### Common Mistake

Fine-tuning mutable policy text into weights without a freshness strategy.

## Decode-Length Prediction

### Prompt

An output-length predictor has low mean absolute error but requests still hit cache pressure and p99 latency breaches. Propose a revised policy and validation plan.

### Role/Domain

Serving capacity planning.

### Difficulty

Advanced.

### Expected Answer

Measure underprediction and exceedance by task class, calibrate conditional quantiles, and account for the asymmetric cost of under- versus over-reservation. Replay representative arrivals through the scheduler; track unused blocks, preemption, truncation, admission and p99. Use an output cap and conservative fallback for uncertain inputs.

### Evaluation

Score tail-risk reasoning, calibration, feature-timestamp correctness and operational outcome metrics.

### Follow-up

How would you debug a predictor that is calibrated overall but underestimates code-generation requests?

### Common Mistake

Optimizing average error without measuring tail exceedance or the scheduler consequences.
