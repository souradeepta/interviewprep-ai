# Post-Training Specialization Decisions

**TL;DR:** Choose the lightest adaptation method that closes a measured gap: prompts for instruction following, retrieval for changing facts, adapters for bounded behavior, and full tuning only when evidence supports it.

## Prerequisites

[LLM evaluation](01-llm-evaluation-harness.md), retrieval and supervised learning.

## Learning Outcomes

- Classify a specialization gap as instruction, knowledge or model behavior.
- Compare prompting, RAG, LoRA and full fine-tuning against that gap.
- Specify evaluation evidence and regression checks before training.
- Explain why data freshness and parameter updates are different concerns.

## Core Intuition

If an employee needs clearer instructions, rewrite the checklist. If they need current customer records, give them a searchable database. If they need a durable new style or workflow, practice may help. Re-teaching all general knowledge is expensive and risks forgetting. Model adaptation has the same distinction between instructions, external knowledge and changed parameters.

## How It Works

Start with a held-out task evaluation and classify failures. Prompting changes the input policy without updating weights. Retrieval-augmented generation fetches external evidence and conditions the model on it; it suits private or frequently changing facts, but retrieval and citation become new failure points. LoRA freezes base weights and learns low-rank updates to selected matrices, reducing trainable parameter count. Full fine-tuning updates many or all parameters and may address broad behavior or capability gaps but needs more data, compute, regression testing and governance.

These categories are not mutually exclusive. A tuned model can still use retrieval, and prompts remain part of every method. “We need fine-tuning” is not a diagnosis. Build examples of failures, define measurable acceptance criteria, split data to avoid leakage and compare the least expensive candidate against the baseline. Test general capability retention, safety and distribution shifts in addition to target-task score.

Start with an error taxonomy. Separate missing or stale knowledge, instruction-following failures, formatting errors, reasoning limitations, tool-selection mistakes and safety-policy failures. Each category suggests a different experiment. For a knowledge error, check whether the fact exists in the model's trusted source and whether retrieval can return it. For a behavior error, test whether a clearer instruction and a few demonstrations solve it. For a broad capability gap, estimate whether curated supervised examples cover the desired behavior and whether the base architecture can plausibly learn it.

Retrieval systems add an evidence path that can be updated without changing model weights. Their quality depends on indexing freshness, access control, chunking, ranking and generation grounded in retrieved evidence. Evaluate retrieval recall and answer faithfulness separately. Fine-tuning may teach the model to use citations or follow a domain workflow, but it does not guarantee that remembered facts stay current. A useful hybrid is to tune stable behavior and retrieve changing facts, with explicit tests that prevent unsupported answers when retrieval fails.

LoRA changes a selected set of model transformations through low-rank matrices. This reduces trainable parameters and makes task-specific adapters easier to store than full copies. The rank and target layers control capacity and resource use; higher rank is not automatically better. Compare adapters against prompt-only and full fine-tuning baselines when feasible. Test for interference when multiple adapters or tasks are combined, and version each adapter with its training data, evaluation results and base-model checksum.

Full fine-tuning is an expensive intervention and should have a strong data and evaluation case. Deduplicate near-identical examples, remove evaluation-set contamination, inspect label quality and represent the production distribution. Split by time, customer, document family or other meaningful group so validation is not a memorized neighbor of training. Preserve general instruction and safety examples if those skills must remain. Use staged checkpoints and compare regressions across language, demographic and task slices before replacing a baseline.

Cost includes more than accelerator hours. Count data annotation, experiment iteration, model hosting, adapter or checkpoint storage, evaluation maintenance and rollback. A prompt change is cheap to reverse; a model update creates release and monitoring obligations. For every candidate, record quality delta, latency, token usage, serving memory, freshness and operational burden. A decision matrix is a triage aid: final selection follows measured outcomes and constraints, not fixed example-count thresholds.

## Trade-offs

| Method | Choose when | Main cost / risk |
|---|---|---|
| Prompting | Behavior is expressible as instructions/examples | Context limits and prompt sensitivity |
| RAG | Facts change, are private or need provenance | Retrieval misses and unsupported synthesis |
| LoRA | Stable, bounded style or task behavior | Dataset coverage, adapter management, forgetting |
| Full fine-tuning | Broad capability change and substantial curated data | Compute, regression surface and update governance |

If the behavior gap is not measurable, invest first in evaluation examples. If the knowledge changes daily, updating weights creates stale copies; use retrieval with freshness and access controls.

## Implementation with Python

Run `python3 modern-ai/implementations/34-post-training-specialization.py`. Four explicit requirement cases exercise a transparent rule-based decision matrix. Its sample-count thresholds are pedagogical flags, not universal data requirements.

```python
import runpy
choose = runpy.run_path("modern-ai/implementations/34-post-training-specialization.py")["recommend"]
assert choose(True, False, False, False, 0) == "RAG"
```

## Failure Modes

| Failure | Diagnostic | Mitigation |
|---|---|---|
| Fine-tuning used to memorize mutable facts | Update cadence and stale-answer rate | Move facts to retrieval and test freshness |
| RAG added for a behavior problem | Retrieval succeeds but output policy still fails | Improve instructions or train against behavior examples |
| Target score improves while general skill regresses | Broad regression suite by capability | Add replay/mix data, constrain updates or roll back |

- Mutable facts become stale after tuning: measure stale-answer rate after source updates and move frequently changing knowledge to a versioned retrieval index.
- Retrieval is selected for an instruction-following defect: compare evidence retrieval separately from response behavior and test prompt or adapter changes against the same cases.
- Target benchmark gains mask broad regressions: inspect capability and safety slices on a held-out suite and roll back if the release gate fails.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| RAG or fine-tuning for new company policies? | Usually RAG for changing, citable facts; ask what stable behavior may still need tuning. |
| What evidence justifies LoRA? | Repeatable behavior gap and representative examples; ask how to detect general regression. |
| How do you compare adaptation choices? | Same held-out tasks, cost/latency and safety metrics; follow up on data leakage. |

## Related Concepts

See [advanced RAG](05-advanced-rag-patterns.md), [quantization and distillation](31-quantization-aware-distillation.md), [notebook 34](../notebooks/34-post-training-specialization.ipynb), and [inference interview practice](../interview-prep/inference-systems.md#post-training-specialization).

When selecting an approach, account for the lifecycle after training. Prompt policies require prompt versioning and regression evaluation. Retrieval requires index refresh, access-control testing and evidence monitoring. Adapters require compatibility tracking and deployment routing. Full fine-tuning requires checkpoint governance, reproducibility and a rollback path. The cheapest experiment is often the one that produces the clearest evidence; keep the candidate artifact isolated until target-task and broad regression checks pass.

## References

[LoRA](https://arxiv.org/abs/2106.09685) and [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401).
