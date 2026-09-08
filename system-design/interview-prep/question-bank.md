# ML/AI System-Design Question Bank

Run one prompt for 45–60 minutes using the [framework](system-design-framework.md)
and scorecard. Clarify scale, latency, labels, privacy, cost, and success
before proposing models. Every prompt requires a baseline, data/split plan,
serving path, offline/online evaluation, monitoring, failure handling, and
trade-offs. The linked reference is remediation, not a prepared answer.

## Product ranking and prediction

### Q1 — Recommendation feed with cold-start users and items — [inference caching](../patterns/08-inference-caching.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Recommendation feed with cold-start users and items — [inference caching](../patterns/08-inference-caching.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q2 — Search retrieval and ranking for rare identifiers — [model serving](../patterns/05-model-serving.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Search retrieval and ranking for rare identifiers — [model serving](../patterns/05-model-serving.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q3 — Ads click prediction with delayed conversions — [A/B testing](../patterns/14-ab-testing.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Ads click prediction with delayed conversions — [A/B testing](../patterns/14-ab-testing.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q4 — Transaction fraud detection with asymmetric costs — [model debugging](../patterns/17-model-debugging.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Transaction fraud detection with asymmetric costs — [model debugging](../patterns/17-model-debugging.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q5 — Demand forecasting with promotions and missing labels — [data pipelines](../patterns/02-data-pipelines.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Demand forecasting with promotions and missing labels — [data pipelines](../patterns/02-data-pipelines.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).


## Data, training, and serving platforms

### Q6 — Point-in-time feature platform for batch and online consumers — [feature store](../patterns/03-feature-store.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Point-in-time feature platform for batch and online consumers — [feature store](../patterns/03-feature-store.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q7 — Training platform with reproducible datasets and rollback — [reproducibility](../patterns/21-reproducibility.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Training platform with reproducible datasets and rollback — [reproducibility](../patterns/21-reproducibility.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q8 — Model registry and promotion workflow across environments — [model registry](../patterns/04-model-registry.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Model registry and promotion workflow across environments — [model registry](../patterns/04-model-registry.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q9 — Low-latency online inference with burst traffic — [online vs batch](../patterns/07-online-vs-batch-inference.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Low-latency online inference with burst traffic — [online vs batch](../patterns/07-online-vs-batch-inference.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q10 — Batch scoring platform with freshness and backfill guarantees — [data pipelines](../patterns/02-data-pipelines.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Batch scoring platform with freshness and backfill guarantees — [data pipelines](../patterns/02-data-pipelines.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).


## Evaluation, quality, and governance

### Q11 — Experiment platform with sample-ratio mismatch detection — [A/B testing](../patterns/14-ab-testing.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Experiment platform with sample-ratio mismatch detection — [A/B testing](../patterns/14-ab-testing.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q12 — Delayed-label monitoring for a risk model — [monitoring](../patterns/16-monitoring-and-observability.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Delayed-label monitoring for a risk model — [monitoring](../patterns/16-monitoring-and-observability.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q13 — Data-quality gates for a feature pipeline — [data pipelines](../patterns/02-data-pipelines.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Data-quality gates for a feature pipeline — [data pipelines](../patterns/02-data-pipelines.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q14 — Drift detection and retraining policy — [drift detection](../patterns/15-drift-detection.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Drift detection and retraining policy — [drift detection](../patterns/15-drift-detection.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q15 — Fairness and privacy controls for a personalization model — [fairness metrics](../patterns/25-fairness-metrics.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Fairness and privacy controls for a personalization model — [fairness metrics](../patterns/25-fairness-metrics.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).


## LLM and multimodal systems

### Q16 — RAG assistant with citations and tenant isolation — [production readiness](../patterns/23-production-readiness.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** RAG assistant with citations and tenant isolation — [production readiness](../patterns/23-production-readiness.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q17 — LLM serving with streaming, batching, and cost routing — [model serving](../patterns/05-model-serving.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** LLM serving with streaming, batching, and cost routing — [model serving](../patterns/05-model-serving.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q18 — Fine-tuning pipeline with data review and regression gates — [model versioning](../patterns/06-model-versioning.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Fine-tuning pipeline with data review and regression gates — [model versioning](../patterns/06-model-versioning.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q19 — Evaluation platform for quality, safety, cost, and latency — [monitoring](../patterns/16-monitoring-and-observability.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Evaluation platform for quality, safety, cost, and latency — [monitoring](../patterns/16-monitoring-and-observability.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q20 — Multimodal search with image/text retrieval and reranking — [inference caching](../patterns/08-inference-caching.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Multimodal search with image/text retrieval and reranking — [inference caching](../patterns/08-inference-caching.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).


## Agents and human workflows

### Q21 — Tool-using customer-support agent with permissioned actions — [production readiness](../patterns/23-production-readiness.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Tool-using customer-support agent with permissioned actions — [production readiness](../patterns/23-production-readiness.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q22 — Human-in-the-loop claims workflow with approval and audit — [model governance](../patterns/27-ml-governance.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Human-in-the-loop claims workflow with approval and audit — [model governance](../patterns/27-ml-governance.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q23 — Multi-agent research workflow with bounded fan-out — [cost optimization](../patterns/22-cost-optimization.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Multi-agent research workflow with bounded fan-out — [cost optimization](../patterns/22-cost-optimization.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q24 — Safety architecture for an agent that can modify records — [privacy-preserving ML](../patterns/28-privacy-preserving-ml.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Safety architecture for an agent that can modify records — [privacy-preserving ML](../patterns/28-privacy-preserving-ml.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).

### Q25 — Cost-aware routing across models and deterministic fallbacks — [online vs batch](../patterns/07-online-vs-batch-inference.md).
**Round/time:** 45–60 minutes.
**Scenario and constraints:** Cost-aware routing across models and deterministic fallbacks — [online vs batch](../patterns/07-online-vs-batch-inference.md). Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [system-design patterns](../patterns/) and [scorecard](scorecard.md).


## Follow-ups to ask yourself

For each prompt, explain the baseline, label maturity, leakage prevention,
candidate/model boundary, rollback path, load shedding, quality slices, and the
trade-off you would revisit first if the budget were cut in half.
