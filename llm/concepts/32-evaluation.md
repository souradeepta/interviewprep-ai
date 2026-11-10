# Evaluation (for LLMs)

TL;DR: LLM evaluation is a decision process that combines representative tasks, reliable measurements, uncertainty, and human or safety review to determine whether a model is fit for a defined use.

## Prerequisites

This lesson assumes basic classification metrics and prompt/model comparisons. It focuses on evaluation of generative language models and systems, including prompts, retrieval, and tool use; see [general evaluation metrics](../../ml/concepts/evaluation-metrics.md) for broader statistical foundations. The manifest places this note after [multimodal](31-multimodal.md), but the methods here also apply to text-only systems.

## Learning Outcomes

- Translate a deployment goal into evaluation dimensions, datasets, baselines, and decision thresholds.
- Distinguish reference-based metrics, task-specific checks, human judgments, and model-based judges.
- Compute deterministic classification-style metrics for a defined generative task and identify their limits.
- Diagnose leakage, sampling noise, judge bias, and aggregate-score regressions before release.

## Core Intuition

An evaluation is like a carefully designed instrument: it can answer only the questions its sample and measurement rules support. A summarizer's ROUGE score may measure overlap with references, but not whether it preserves the one critical safety instruction. A helpfulness rating may measure preference under a rubric, but not factual correctness. A benchmark score may compare systems on its fixed examples, but does not automatically predict behavior in a different language, product, or user population.

Start by stating the decision. Is the goal to select between two checkpoints, prevent a regression, certify a constrained behavior, or monitor a live product? Define the user task and unacceptable failures before choosing a metric. A system-level evaluation should often include the model plus its prompt, retrieval index, tools, guardrails, and serving configuration because those components jointly determine behavior.

## How It Works

Build an evaluation set from a target population and stratify it by meaningful conditions: task type, language, difficulty, prompt length, user group, and known risk. Keep development data separate from a locked test set. Prevent training contamination where possible, remove duplicate or near-duplicate examples, and record provenance and allowed use. For changing products, maintain a stable regression set and a rotating set that captures new behavior. A small curated suite is useful for diagnosis; it is not a probability sample of all future users unless designed as one.

Choose measures matched to each outcome. Exact match and accuracy suit tasks with unambiguous labels. Precision, recall, and F1 expose different error costs for imbalanced labels. BLEU and ROUGE compare token overlap and can penalize valid paraphrases. Semantic similarity metrics add learned representations but can miss factual contradictions. For generation, direct checks can measure format validity, citation support, tool-call correctness, refusal behavior, or task completion. Human assessment can judge usefulness and nuance, but requires clear rubrics, calibrated raters, and a way to resolve disagreement.

Model-based judges can scale comparisons, yet are measurements with their own error. Validate the judge against blinded human ratings on the target domain, inspect subgroup disagreement, and test for position, verbosity, and style preferences. Randomize candidate order and hide model identity. Use pairwise preference when the rubric supports it, and separate dimensions such as factuality and helpfulness rather than collapsing them prematurely. A deterministic judge prompt does not eliminate judge bias.

For a candidate system A and baseline B, use paired examples so both systems see the same cases. Report sample size, per-slice scores, uncertainty, and the comparison method. For a binary outcome with k successes from n independent examples, accuracy is k/n; a confidence interval helps show sampling uncertainty. For paired systems, differences on the same example carry more information than comparing two unrelated averages. Bootstrap or randomization procedures can estimate uncertainty, but the resampling unit must match the data-generating process (for example, user rather than turn when turns are correlated).

Before release, define a gate: required minimums, allowed regressions, critical failure review, and who can override the gate. During deployment, log privacy-safe outcome signals, monitor distribution shift and failure reports, and periodically rerun offline evaluations. Online A/B tests can measure product outcomes under randomized exposure, but require adequate sample size and guardrails; they do not replace targeted safety tests. Never use a single scalar score as a complete description of model quality.

## Trade-offs

| Method | Strength | Limitation | Best use |
|---|---|---|---|
| Exact match/task metric | Cheap, reproducible, easy to gate | Brittle for valid alternative wording; depends on label quality | Constrained outputs and clearly specified tasks |
| BLEU/ROUGE overlap | Fast and reference-based | Misses semantic equivalence, factuality, and usefulness | Narrow generation comparisons with suitable references |
| Human rubric | Can assess nuanced usefulness and harms | Cost, rater variability, and limited throughput | High-impact decisions and rubric calibration |
| LLM judge | Scales structured comparisons | Bias, prompt sensitivity, and correlated model errors | Triage after validation against humans |
| Online experiment | Measures real product outcomes | Exposure risk, confounding, and slow rare-event detection | Product-level changes with safe rollout controls |

No metric removes the need for sampling judgment. A higher average can hide a critical slice regression; confidence intervals can be narrow around a biased dataset; human agreement can be high when a rubric rewards the wrong behavior. Keep a metric-to-decision map so every reported number has a defined interpretation and owner.

## Implementation

This deterministic example computes accuracy, precision, recall, and F1 for a binary “acceptable response” label. It demonstrates metric definitions, not a complete LLM evaluation; the labels are supplied judgments, and the result says nothing about how representative the examples are or whether the rubric is valid.

```python
def binary_metrics(predictions, labels):
    if len(predictions) != len(labels) or not labels:
        raise ValueError("predictions and nonempty labels must have equal length")
    if any(value not in (0, 1) for value in predictions + labels):
        raise ValueError("values must be binary integers")
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(labels), "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


scores = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
assert scores["accuracy"] == 0.5
assert scores["precision"] == 0.5
assert scores["recall"] == 0.5
assert scores["f1"] == 0.5
assert (scores["tp"], scores["fp"], scores["fn"], scores["tn"]) == (1, 1, 1, 1)
```

For a real generation task, define labels with an explicit rubric and adjudicate a sample before scaling annotation. Store the dataset version, prompt/model revision, decoding parameters, evaluator version, and metric implementation with every run. Re-run a fixed seed or deterministic decoding for regression checks, while separately evaluating stochastic output distributions when production uses sampling.

## Failure Modes

- **Metric does not match the decision:** A lexical score is treated as factuality or helpfulness. Map every metric to a stated construct and add direct factuality, task-completion, or human checks where needed.
- **Test leakage or benchmark contamination:** Repeated public examples inflate apparent capability. Deduplicate against training/development sources where feasible, use held-out private or newly authored cases, and record provenance.
- **Small or unrepresentative sample:** A few examples produce unstable averages or omit important groups. Report sample counts and intervals, stratify by risk, and collect data from the target population.
- **Aggregate score masks critical regressions:** A model improves common tasks while failing a safety-critical slice. Set slice-level gates and inspect example-level errors before release.
- **Uncalibrated LLM judge:** The judge favors longer answers, a particular position, or its own style. Blind identities, randomize order, test counterbalanced pairs, and compare agreement with human raters by slice.
- **Rater disagreement is hidden:** Averaged scores conceal ambiguous rubric dimensions. Measure agreement, adjudicate disagreements, refine examples and instructions, then re-annotate affected cases.
- **Non-independent observations:** Multiple turns from one user are treated as independent, overstating certainty. Resample or cluster at the user/session level and state the unit of analysis.
- **Offline score mistaken for production outcome:** Synthetic benchmarks miss workflow and distribution effects. Use guarded online measurement for product impact and retain targeted offline tests for rare failures.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| How would you evaluate a new assistant model? | Define target decisions, representative tasks, baselines, metrics, risk slices, and release gates; combine automatic checks with calibrated human review. | Debug an average score increase with a safety-slice regression. |
| What does ROUGE tell you? | Reference-token overlap for generation; it does not establish factual correctness or usefulness. | Design a complementary factuality check. |
| When is an LLM judge useful? | For scalable rubric-based comparisons after validation against blinded human ratings. | Test whether position bias changes pairwise preferences. |
| How do you compare two systems? | Use the same cases, paired analysis, uncertainty, and per-slice results; preserve a locked test set. | Choose an uncertainty method when turns are grouped by user. |

## Related Concepts

- [Prompt Optimization](17-prompt-optimization.md) uses evaluation results to compare prompt variants.
- [Instruction Tuning](05-instruction-tuning.md) uses held-out evaluation to assess adaptation.
- [Multimodal](31-multimodal.md) extends evaluation to inputs and outputs beyond text.
- [Evaluation of Text Generation: A Survey](https://arxiv.org/abs/2006.14799) reviews reference-based and learned evaluation methods.

## Practice

- [Evaluation interview practice](../interview-prep/curriculum-practice.md#evaluation-for-llms) applies metric selection, uncertainty, and slice-level release gates.

## References

- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) defines a reference-based n-gram metric and its intended setting.
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) describes overlap metrics for summarization evaluation.
- [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110) presents broad scenario and metric coverage for language model evaluation.
