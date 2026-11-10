# Curriculum Map and Role Paths

The manifest is the canonical lesson prerequisite graph. This map explains recommended sequences, role paths, and self-checks; its week-to-week ordering is guidance rather than a hard prerequisite. Start at the earliest stage you cannot yet demonstrate, and use the checks below to test out without assuming that similarly named tracks are equivalent.

## Prerequisite sequence

```mermaid
flowchart LR
  A[Programming and math] --> B[Statistics]
  B --> C[Core ML]
  C --> D[Deep learning]
  D --> E[LLM / NLP / CV / RL]
  E --> F[MLOps and system design]
  F --> G[Agents and modern AI engineering]
```

The arrows show a useful default sequence, not a requirement to finish every domain. A specialist can branch after core ML and return to MLOps/system design when deploying or evaluating a system.

## How `ai/` and `ml/` fit together

| Track | Purpose | If topics overlap, study this lesson | Relationship |
|---|---|---|---|
| [`ai/`](../ai/README.md) | Broad foundations: optimization, classical algorithms, neural networks, and general AI ideas. Use it to build shared vocabulary and mathematical intuition. | [Gradient descent](../ai/concepts/01-gradient-descent.md) for the general optimization mechanism. | Broad conceptual foundation; emphasizes why the method works across AI settings. |
| [`ml/`](../ml/README.md) | Model selection and practice: supervised learning, evaluation, features, generalization, algorithms, and ML engineering decisions. | [Optimization](../ml/concepts/optimization.md), [evaluation metrics](../ml/concepts/evaluation-metrics.md), and [supervised learning](../ml/concepts/supervised-learning.md). | Applied ML path; choose models, validate them, and reason about their failure boundaries. |
| Shared topic: gradient descent | Both tracks may explain gradients and iterative updates. | Begin with AI for the general mechanism; use [ML optimization](../ml/concepts/optimization.md) to connect optimizer choices to model fitting and validation. | Read both only when you need both the derivation and the applied decision context. |
| Shared topic: neural networks | AI introduces general neural-network components; ML focuses on training, regularization, and selection. | [AI neural networks](../ai/concepts/13-neural-networks.md), then [ML neural networks](../ml/concepts/neural-networks.md). | The ML lesson assumes the broad representation idea and emphasizes training behavior and model use. |

## Role paths

Hours are focused study hours per week, not calendar availability. Every week includes a lesson, a practice activity, and a check that can be explained or demonstrated. Pick the linked capstone as the path’s final project.

### Entry-level ML engineer — 12 weeks

**Pace:** 6–8 hours/week. **Capstone:** [Tabular ML in production](../projects/01-tabular-ml-production.md).

| Week | Hours | Lessons | Practice activity | Exit check |
|---:|---:|---|---|---|
| 1 | 6 | [Coding foundations](../coding/README.md), [probability](../stats/concepts/01-probability-fundamentals.md) | Trace a data transform by hand; answer two probability questions. | Explain expectation, variance, and a Python function’s input/output contract. |
| 2 | 6 | [Distributions](../stats/concepts/02-distributions-reference.md), [estimation](../stats/concepts/04-maximum-likelihood-estimation.md) | Fit a simple rate estimate and state its assumptions. | Distinguish a sample estimate from the population parameter. |
| 3 | 7 | [Confidence intervals](../stats/concepts/06-confidence-intervals.md), [hypothesis testing](../stats/concepts/05-hypothesis-testing.md) | Interpret a confidence interval and p-value aloud. | Explain why non-significance does not prove no effect. |
| 4 | 7 | [Supervised learning](../ml/concepts/supervised-learning.md), [linear regression](../ai/concepts/06-linear-regression.md) | Compare a mean baseline with a linear model on a held-out split. | State the target, features, baseline, and split strategy. |
| 5 | 7 | [Logistic regression](../ai/concepts/07-logistic-regression.md), [evaluation metrics](../ml/concepts/evaluation-metrics.md) | Choose a threshold under asymmetric error costs. | Defend a metric and threshold for an imbalanced binary task. |
| 6 | 7 | [Overfitting and underfitting](../ml/concepts/overfitting-underfitting.md), [regularization](../ml/concepts/regularization.md) | Diagnose a train/validation gap and propose a controlled change. | Separate high bias from high variance using observed evidence. |
| 7 | 8 | [Decision trees](../ai/concepts/08-decision-trees.md), [random forests](../ai/concepts/09-random-forests.md) | Explain a tree split and compare bagging behavior. | Name one case where a tree ensemble is a poor fit. |
| 8 | 8 | [Gradient boosting](../ai/concepts/10-gradient-boosting.md), [feature engineering](../ml/concepts/feature-engineering.md) | Design leakage-safe features for a timestamped prediction. | Explain point-in-time availability for every feature. |
| 9 | 7 | [Cross-validation](../ml/concepts/cross-validation-strategies.md), [data leakage](../ml/concepts/data-leakage.md) | Review a split plan for grouped and temporal leakage. | Select a split that matches the deployment unit. |
| 10 | 7 | [A/B testing](../stats/concepts/08-ab-testing-statistics.md), [statistical power](../stats/concepts/07-statistical-power-sample-size.md) | Outline a powered experiment for a model change. | Identify randomization unit, metric, MDE, and stopping rule. |
| 11 | 8 | [Data pipelines](../mlops/concepts/01-data-pipelines.md), [model testing](../mlops/concepts/09-model-testing.md) | Define schema checks and a model regression test. | Show how bad input is detected before promotion. |
| 12 | 8 | [ML roadmap](ml-roadmap.md), [ML interview practice](../ml/interview-prep/ml-theory-questions.md) | Complete the capstone and a mock model review. | Capstone runs locally; explain metric, failure case, and rollback trigger. |

### Production MLE — 14 weeks

**Pace:** 7–9 hours/week. **Capstone:** [ML platform reliability](../projects/03-ml-platform-reliability.md).

| Week | Hours | Lessons | Practice activity | Exit check |
|---:|---:|---|---|---|
| 1 | 7 | [Statistics probability](../stats/concepts/01-probability-fundamentals.md), [confidence intervals](../stats/concepts/06-confidence-intervals.md) | Estimate uncertainty for a service metric. | Explain sampling unit and interval assumptions. |
| 2 | 7 | [A/B testing](../stats/concepts/08-ab-testing-statistics.md), [causal inference](../stats/concepts/09-causal-inference.md) | Diagnose a sample-ratio mismatch scenario. | Name checks before interpreting a treatment estimate. |
| 3 | 7 | [Supervised learning](../ml/concepts/supervised-learning.md), [evaluation](../ml/concepts/evaluation-metrics.md) | Build a metric-to-decision table for a model. | Defend offline metrics and their blind spots. |
| 4 | 8 | [Data leakage](../ml/concepts/data-leakage.md), [cross-validation](../ml/concepts/cross-validation-strategies.md) | Design point-in-time joins and temporal validation. | Identify future information in a feature pipeline. |
| 5 | 8 | [Data pipelines](../mlops/concepts/01-data-pipelines.md), [data validation](../mlops/concepts/03-data-validation.md) | Write batch contract checks and failure routing. | Specify reject/quarantine behavior for invalid data. |
| 6 | 8 | [Feature stores](../mlops/concepts/02-feature-stores.md), [feature engineering](../ml/concepts/feature-engineering.md) | Trace one entity-time pair through offline and online features. | Explain parity and freshness tests. |
| 7 | 8 | [Experiment tracking](../mlops/concepts/05-experiment-tracking.md), [reproducibility](../mlops/concepts/07-reproducibility.md) | Produce a lineage record for a training run. | Reconstruct code, data, configuration, and artifact version. |
| 8 | 8 | [Model testing](../mlops/concepts/09-model-testing.md), [evaluation](../mlops/concepts/12-evaluation-metrics.md) | Define data, model, and slice regression gates. | Distinguish a passing job from a valid model. |
| 9 | 8 | [Model serving](../mlops/concepts/14-model-serving.md), [online vs batch inference](../system-design/patterns/07-online-vs-batch-inference.md) | Compare latency, freshness, and cost for two serving modes. | Match serving topology to the prediction deadline. |
| 10 | 8 | [Deployment strategies](../mlops/concepts/16-deployment-strategies.md), [canary deployment](../system-design/patterns/11-blue-green-deployment.md) | Write canary gates and a tested rollback procedure. | Identify compatibility risks in rollback. |
| 11 | 8 | [Drift detection](../system-design/patterns/15-drift-detection.md), [monitoring](../system-design/patterns/16-monitoring-and-observability.md) | Triage an input shift with delayed labels. | Separate service health, data quality, and model quality signals. |
| 12 | 8 | [Model registry](../mlops/concepts/15-model-registry.md), [governance](../system-design/patterns/26-data-governance.md) | Review an artifact promotion record. | Show immutable identity, approval, and provenance. |
| 13 | 8 | [Disaster recovery](../system-design/patterns/31-disaster-recovery.md), [MLOps interview practice](../mlops/interview-prep/mlops-interview-questions.md) | Run a timed incident response tabletop. | State containment, recovery evidence, and communication owner. |
| 14 | 9 | [MLOps roadmap](../mlops/MLOPS_ROADMAP.md), [system design framework](../system-design/interview-prep/system-design-framework.md) | Complete the platform reliability capstone and mock design interview. | Meet local success criterion and defend SLO, cost, and rollback choices. |

### LLM/RAG engineer — 10 weeks

**Pace:** 7–9 hours/week. **Capstone:** [RAG evaluation and retrieval](../projects/02-rag-evaluation-and-retrieval.md).

| Week | Hours | Lessons | Practice activity | Exit check |
|---:|---:|---|---|---|
| 1 | 7 | [Tokenization](../llm/concepts/01-tokenization.md), [embeddings](../llm/concepts/02-embeddings.md) | Compare lexical and vector similarity on a small example. | Explain token, embedding, and similarity limitations. |
| 2 | 7 | [RAG](../llm/concepts/18-rag.md), [retrieval-augmented generation](../llm/concepts/19-retrieval-augmented-generation.md) | Draw ingestion and query-time data flows. | Identify where source provenance is retained. |
| 3 | 8 | [Vector databases](../llm/concepts/20-vector-databases.md), [semantic search](../llm/concepts/21-semantic-search.md) | Select a retrieval strategy for a small corpus. | Explain recall/latency and filtering trade-offs. |
| 4 | 8 | [Retrieval systems](../llm/concepts/40-retrieval-systems.md), [advanced RAG](../modern-ai/concepts/05-advanced-rag-patterns.md) | Compare chunking and top-k using a fixed query set. | Show a failure caused by chunk boundary or retrieval miss. |
| 5 | 8 | [LLM evaluation](../llm/concepts/32-evaluation.md), [evaluation harness](../modern-ai/concepts/01-llm-evaluation-harness.md) | Build a small evidence and answer-quality rubric. | Separate retrieval quality from answer faithfulness. |
| 6 | 8 | [Prompting](../llm/concepts/12-prompting.md), [prompt injection security](../llm/concepts/41-prompt-injection-security.md) | Threat-model hostile retrieved content. | Describe an instruction/data boundary and a test case. |
| 7 | 8 | [Context window](../llm/concepts/25-context-window.md), [long-context handling](../llm/concepts/39-long-context-handling.md) | Budget tokens across system, evidence, and output. | Explain truncation and context-selection failure modes. |
| 8 | 8 | [KV cache](../llm/concepts/23-kv-cache.md), [continuous batching](../llm/concepts/26-continuous-batching.md) | Estimate serving memory and compare queueing choices. | Identify the latency/cost trade-off without claiming benchmark results. |
| 9 | 8 | [RAG memory store](../modern-ai/concepts/23-rag-memory-store.md), [AI gateway routing](../modern-ai/concepts/19-ai-gateway-routing.md) | Design an abstain/fallback rule for low evidence. | State behavior for empty, conflicting, and stale retrieval. |
| 10 | 9 | [Modern AI roadmap](../modern-ai/README.md), [LLM interview practice](../llm/interview-prep/llm-interview-questions.md) | Complete the RAG capstone and defend an evaluation report. | Reproduce local metrics and name the largest remaining evidence gap. |

### Applied AI / agent engineer — 12 weeks

**Pace:** 7–9 hours/week. **Capstone:** [Agent workflow safety](../projects/04-agent-workflow-safety.md).

| Week | Hours | Lessons | Practice activity | Exit check |
|---:|---:|---|---|---|
| 1 | 7 | [Agent loops](../agentic-ai/concepts/02-agent-loops.md), [tool use](../agentic-ai/concepts/03-tool-use.md) | Trace a bounded tool cycle and label each state transition. | Explain when a direct function call is preferable to an agent loop. |
| 2 | 7 | [Structured output](../agentic-ai/concepts/06-structured-output.md), [function calling](../agentic-ai/concepts/05-function-calling.md) | Specify a validated local tool schema. | Reject malformed arguments before side effects. |
| 3 | 8 | [Planning](../agentic-ai/concepts/07-planning-reasoning.md), [ReAct](../agentic-ai/concepts/08-react-reasoning-acting.md) | Compare plan-first and observe/act behavior on a fixed task. | Identify replanning condition and step bound. |
| 4 | 8 | [RAG agents](../agentic-ai/concepts/25-retrieval-augmented-generation.md), [LLM RAG](../llm/concepts/18-rag.md) | Make evidence-backed response and abstention cases. | Preserve source IDs and distinguish retrieval from proof. |
| 5 | 8 | [Agent security](../agentic-ai/concepts/60-agent-security-sandboxing.md), [prompt injection](../llm/concepts/41-prompt-injection-security.md) | Threat-model untrusted tool output and retrieved text. | Demonstrate a refusal or safe handling test. |
| 6 | 8 | [Error recovery](../agentic-ai/concepts/26-error-recovery.md), [agent debugging](../agentic-ai/concepts/27-agent-debugging.md) | Inject timeout and invalid output into a local mock. | Bound retries and preserve a useful failure trace. |
| 7 | 8 | [Agent testing](../agentic-ai/concepts/28-agent-testing.md), [agent evals](../agentic-ai/concepts/29-agent-evals.md) | Build deterministic success, safety, and regression cases. | Explain metric denominator and one blind spot. |
| 8 | 8 | [Tracing](../agentic-ai/concepts/32-tracing-agents.md), [observability](../agentic-ai/concepts/31-observability-for-agents.md) | Inspect a trace with tool inputs, outputs, latency, and outcome. | Locate a failure stage without logging sensitive payloads. |
| 9 | 8 | [Cost optimization](../agentic-ai/concepts/34-agent-cost-optimization.md), [agent cost analysis](../agentic-ai/concepts/62-agent-cost-analysis.md) | Estimate per-task cost from calls, tokens, and retries. | Identify a cost ceiling and graceful fallback. |
| 10 | 8 | [Human-agent collaboration](../agentic-ai/concepts/37-human-agent-collaboration.md), [safety/alignment](../agentic-ai/concepts/36-safety-alignment.md) | Define approval gates for an irreversible action. | Show an authorization boundary and escalation path. |
| 11 | 8 | [Agent deployment](../agentic-ai/concepts/56-agent-deployment-patterns.md), [state management](../agentic-ai/concepts/57-agent-state-management.md) | Design idempotent resumable execution using synthetic state. | Explain duplicate delivery and recovery behavior. |
| 12 | 9 | [Agent interview practice](../agentic-ai/interview-prep/agent-interview-questions.md), [Modern AI safety](../modern-ai/concepts/02-ai-red-teaming.md) | Complete the safety capstone and mock an agent incident review. | Meet local criterion and defend limits, audit trail, and human controls. |

## Fast track: test out of prerequisites

Use these small checks before skipping a stage. Passing one check is evidence for that narrow skill, not a claim that two courses or tracks are equivalent.

| Stage | Self-check to test out |
|---|---|
| Programming and math | Implement a function with a loop and dictionary; explain vector dot product and derivative of a squared error. If not, start with [coding foundations](../coding/README.md) and the math prerequisites in [Statistics](../stats/README.md). |
| Statistics | Explain a confidence interval, distinguish correlation from causation, and design a powered A/B test with a stopping rule. Use [statistics interview practice](../stats/interview-prep/statistics-interview-questions.md) for a timed self-check. |
| Core ML | Given a small imbalanced dataset, choose a split, baseline, metric, and threshold; identify one leakage risk. Use [ML theory practice](../ml/interview-prep/ml-theory-questions.md). |
| Deep learning | Explain backpropagation, regularization, and a train/validation gap; run a tiny model training example. Start at [ML neural networks](../ml/concepts/neural-networks.md) if any part is unclear. |
| Specialist domain | Explain one domain-specific failure and evaluation measure for LLM/NLP, CV, or RL. Check the relevant section README before skipping its foundations. |
| MLOps and system design | Draw data-to-serving flow, name an SLO, and describe rollback plus delayed-label monitoring. Use [MLOps interview practice](../mlops/interview-prep/mlops-interview-questions.md). |
