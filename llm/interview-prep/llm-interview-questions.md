# LLM Interview Questions

Each prompt is a 5–10 minute LLM engineering scenario. State assumptions,
constraints, implementation outline, evaluation plan, failure modes, and a
follow-up before reading the linked remediation page.

## Architecture and training

### Q1 — Design a next-token training data pipeline with deduplication and a held-out set.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design a next-token training data pipeline with deduplication and a held-out set. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q2 — Explain how tokenizer choice changes cost, context capacity, and multilingual quality.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Explain how tokenizer choice changes cost, context capacity, and multilingual quality. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q3 — Diagnose training loss falling while held-out loss rises.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Diagnose training loss falling while held-out loss rises. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q4 — Choose a model size and data mixture under a fixed compute budget.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Choose a model size and data mixture under a fixed compute budget. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q5 — Compare pretraining from scratch, continued pretraining, and instruction tuning.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Compare pretraining from scratch, continued pretraining, and instruction tuning. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q6 — Design a distributed training recovery plan after a worker failure.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design a distributed training recovery plan after a worker failure. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q7 — Explain data contamination and construct a contamination-resistant evaluation split.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Explain data contamination and construct a contamination-resistant evaluation split. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q8 — Diagnose catastrophic forgetting after domain adaptation.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Diagnose catastrophic forgetting after domain adaptation. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q9 — Choose LoRA, full fine-tuning, or adapters for a small domain dataset.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Choose LoRA, full fine-tuning, or adapters for a small domain dataset. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q10 — Design a safe instruction-tuning dataset and reviewer agreement process.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design a safe instruction-tuning dataset and reviewer agreement process. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).


## Retrieval and RAG

### Q11 — A RAG system retrieves irrelevant chunks: separate query, chunking, index, and ranker causes.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** A RAG system retrieves irrelevant chunks: separate query, chunking, index, and ranker causes. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [retrieval failure remediation](retrieval-failure-remediation.md), [LLM concepts](../concepts/), and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q12 — Choose chunk size, overlap, metadata filters, and parent-document reconstruction.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Choose chunk size, overlap, metadata filters, and parent-document reconstruction. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q13 — Evaluate embeddings and a reranker when recall@K improves but answer quality does not.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Evaluate embeddings and a reranker when recall@K improves but answer quality does not. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q14 — Design a hybrid lexical/vector retrieval system for rare identifiers.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design a hybrid lexical/vector retrieval system for rare identifiers. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q15 — Decide between fine-tuning and RAG for frequently changing policy documents.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Decide between fine-tuning and RAG for frequently changing policy documents. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q16 — Make citations faithful when the model combines multiple retrieved sources.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Make citations faithful when the model combines multiple retrieved sources. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q17 — Set an offline-to-online RAG evaluation plan with attribution, latency, and cost.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Set an offline-to-online RAG evaluation plan with attribution, latency, and cost. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).


## Alignment, safety, and structured output

### Q18 — Compare SFT, preference optimization, RLHF, and DPO for a support assistant.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Compare SFT, preference optimization, RLHF, and DPO for a support assistant. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q19 — Explain LLM-as-judge bias and design human calibration and judge-agreement checks.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Explain LLM-as-judge bias and design human calibration and judge-agreement checks. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q20 — Make JSON output reliable when schemas evolve and malformed output is costly.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Make JSON output reliable when schemas evolve and malformed output is costly. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q21 — Defend against prompt injection in retrieved documents and tool results.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Defend against prompt injection in retrieved documents and tool results. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q22 — Design refusal and safe-completion evaluation without rewarding over-refusal.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design refusal and safe-completion evaluation without rewarding over-refusal. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q23 — Turn production failures into a versioned regression set and release gate.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Turn production failures into a versioned regression set and release gate. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).


## Inference and operations

### Q24 — Route requests across models under quality, latency, and token-cost budgets.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Route requests across models under quality, latency, and token-cost budgets. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q25 — Diagnose a latency regression by separating queueing, prefill, decode, and network time.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Diagnose a latency regression by separating queueing, prefill, decode, and network time. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q26 — Explain KV caching, continuous batching, and when speculative decoding helps.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Explain KV caching, continuous batching, and when speculative decoding helps. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q27 — Choose quantization levels while protecting rare-token and long-context quality.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Choose quantization levels while protecting rare-token and long-context quality. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q28 — Design rate limits, retries, timeouts, and fallbacks for a model-serving tier.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design rate limits, retries, timeouts, and fallbacks for a model-serving tier. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q29 — Respond to a quality incident with no code deploy: scope, rollback, and root cause.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Respond to a quality incident with no code deploy: scope, rollback, and root cause. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

### Q30 — Design observability for prompt versions, retrieval traces, token use, safety, and user outcomes.
**Round/time:** 5–10 minutes.
**Scenario and constraints:** Design observability for prompt versions, retrieval traces, token use, safety, and user outcomes. Treat the stated topic as the interview scenario; quantify scale, latency, quality, privacy, and cost assumptions before choosing an approach.
**A strong candidate clarifies:** users, data/label boundaries, success metric, operational budget, and what is explicitly out of scope.
**Success criteria:** propose a baseline, an end-to-end design or diagnosis, measurable offline/online checks, and a safe rollback or fallback.
**Failure modes/edge cases:** leakage or contamination, distribution/slice regressions, stale or missing data, malformed inputs, budget overruns, and unsafe actions.
**Interviewer follow-up:** ask for the highest-risk trade-off, one adversarial example, and how the answer changes when quality, latency, or cost is constrained.
**Rubric (0–3):** 0 = missing or unsafe; 1 = names components without evidence; 2 = coherent plan with metrics and boundaries; 3 = quantified trade-offs, failure handling, and validation.
**Remediation:** [LLM concepts](../concepts/) and [evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).


Evaluation follow-up: connect Q13, Q17, Q19, Q23, and Q30 to the
[cross-domain evaluation drills](../../ml/interview-prep/evaluation-experiment-questions.md).

## Interviewer follow-ups

For any prompt, ask what the candidate would measure, what would invalidate the
plan, how quality changes by slice, and how they would bound cost and latency.
Required scenarios include retrieval failure, chunking, reranking, judge
limitations, fine-tuning versus RAG, structured output, injection, routing, and
incident response.
