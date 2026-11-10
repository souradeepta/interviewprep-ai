# Few-Shot Learning
TL;DR: Few-shot learning supplies a small set of labeled examples at inference time so a model can apply a task pattern to a new input without updating its weights.

## Prerequisites

Know how a prompt pairs an input with an expected output; [prompting](12-prompting.md) and [zero-shot learning](14-zero-shot-learning.md) provide useful contrasts. Here “few-shot” refers to examples supplied in the prompt, not a small fine-tuning dataset.

## Learning Outcomes

- Construct a few-shot prompt with a clear task, consistent input/output schema, and representative demonstrations.
- Explain how label balance, example relevance, ordering, and context length can affect evaluation.
- Compare prompting with zero-shot prompting and parameter-updating fine-tuning under fixed constraints.
- Diagnose a prompt failure by stratifying errors and changing one demonstration variable at a time.

## Core Intuition

Suppose a colleague has never seen your ticket categories. Giving them three labeled examples can clarify what “billing,” “account access,” and “product defect” mean. The examples do not rewrite the colleague's long-term memory; they provide temporary evidence for the current decision. Few-shot prompting uses a similar interface: the model receives demonstrations followed by a new query and generates an output conditioned on the entire sequence.

The analogy has limits. A language model does not necessarily infer the intended rule as a person would. It may rely on lexical overlap, label wording, position, or familiar patterns from pretraining. A prompt that appears to show a rule can therefore produce a shortcut. Few-shot performance depends on the model, task, prompt format, examples, and decoding settings. There is no universal “three examples is enough” rule, and adding examples does not guarantee improvement.

In this lesson, the term describes an inference-time setting. It does not mean the model's parameters were trained on a few examples. The [In-Context Learning lesson](15-in-context-learning.md) discusses the broader phenomenon and its less settled mechanisms; few-shot is one common way to provide the context. This distinction matters when estimating cost and reproducibility: each query may carry demonstrations in its input, while fine-tuning incurs a separate training step and produces updated parameters.

## How It Works

A prompt commonly contains an instruction, demonstrations `(x_i, y_i)`, and a test input `x*`:

```text
Instruction: Assign one of {billing, access, defect}. Return only the label.
Ticket: I was charged twice.       Label: billing
Ticket: My password reset link expired. Label: access
Ticket: The export file is empty.  Label: defect
Ticket: The invoice includes a duplicate fee. Label:
```

The model predicts a continuation after the final label cue. The application can constrain or validate that continuation against the permitted label set. A useful abstraction is `y* ~ p(y | instruction, (x_1,y_1), ..., (x_k,y_k), x*)`; it describes conditioning, not a guarantee that the model learned an explicit symbolic rule.

Example selection is a data problem. Demonstrations should make the intended mapping identifiable and should reflect the evaluation population. For classification, cover labels and meaningful boundary cases. For extraction, include variations in wording and missing fields. For generation, examples should illustrate both the requested structure and the allowed content. Similarity-based retrieval can select examples near the current query, but it can also overrepresent a narrow region or retrieve a misleading neighbor. Measure selection quality on a held-out set instead of assuming that nearest examples are best.

Format consistency reduces ambiguity: use stable field names, delimiters, and output rules. Label names themselves can act as cues, so use meaningful labels where appropriate and test alternate verbalizers if results are unexpectedly sensitive. Order can affect outcomes. Some experiments find position effects, but “put hard examples last” is not a general law; compare candidate orderings on a fixed validation set. Likewise, include varied examples only when their variation teaches a feature relevant to the task. Irrelevant diversity adds tokens without useful evidence.

The operational cost scales with prompt length and repeated input. If each demonstration averages `t` tokens, `k` examples add roughly `k*t` tokens to every query, plus instruction and query tokens. This can increase billing and prefill latency, consume context needed for the answer, or reduce throughput. Prompt caching may reduce repeated processing cost in some systems, but must be measured under the actual API and workload. Fine-tuning can amortize repeated demonstrations across requests, while few-shot prompting is easier to update and inspect.

## Trade-offs

| Method | Update required | Per-query input | Useful when | Main limitation |
|---|---|---|---|---|
| Zero-shot prompt | No | Instruction and query | Task is clear from instructions and base-model capability | No task examples to disambiguate schema or edge cases |
| Few-shot prompt | No | Instruction, examples, query | Fast iteration or a small, changing task set | Repeated token cost and sensitivity to example choice |
| Fine-tuning | Yes | Query, often shorter task instruction | High-volume stable task with suitable training data | Training, versioning, and evaluation overhead |
| Retrieval-selected examples | No model update; example index may be built | Retrieved examples plus query | Large example library and query-dependent demonstrations | Retrieval mistakes and additional latency |

Choose by running a controlled comparison: hold the test split, base model, output parser, and decoding settings fixed; compare zero-shot and several few-shot prompts; report per-class or per-slice quality, invalid-output rate, tokens per query, and latency. If the use case is high volume, compare the accumulated prompt cost with fine-tuning and its maintenance cost. Keep development examples separate from final evaluation data to avoid prompt-selection leakage. If demonstrations are selected dynamically, evaluate retrieval on realistic queries and include cases where no good example exists.

## Implementation

This small example validates the allowed label schema and builds a prompt. It does not prove that every class is represented, call a model, estimate accuracy, or show that these examples are representative.

```python
LABELS = {"billing", "access", "defect"}
examples = [
    ("I was charged twice.", "billing"),
    ("My reset link expired.", "access"),
    ("The export file is empty.", "defect"),
]

def build_prompt(demonstrations, query):
    seen = {label for _, label in demonstrations}
    if not demonstrations or not seen <= LABELS:
        raise ValueError("provide examples with allowed labels")
    rows = ["Assign one of billing, access, or defect. Return only the label."]
    rows.extend(f"Ticket: {text}\nLabel: {label}" for text, label in demonstrations)
    rows.append(f"Ticket: {query}\nLabel:")
    return "\n\n".join(rows)

prompt = build_prompt(examples, "The invoice includes a duplicate fee.")
assert prompt.endswith("Label:")
assert "charged twice" in prompt and "duplicate fee" in prompt
try:
    build_prompt([("bad label", "other")], "query")
except ValueError:
    pass
else:
    raise AssertionError("unknown labels must be rejected")
```

For an experiment, save the exact prompt text, model identifier, decoding parameters, demonstration IDs, and evaluation-set version. Then compare variants by changing one factor at a time: examples, ordering, label wording, or instruction. A single successful output is not evidence of reliability. Include ambiguous and out-of-scope inputs, and decide whether the system should abstain, ask for clarification, or return a constrained error when the model does not produce a valid label.

## Failure Modes

- **Demonstrations do not represent the task distribution.** Aggregate accuracy may conceal a class or language with poor results. Report metrics by class, source, language, and relevant edge-case slice; replace or retrieve examples using only development data.
- **The prompt permits multiple interpretations.** Labels correlate with topic or sentiment in a way that suggests an unintended rule. Add an explicit instruction and counterexamples that distinguish competing interpretations, then evaluate on disambiguating cases.
- **One example is mislabeled or contradictory.** The model may copy the bad mapping, and a few examples provide little redundancy. Validate demonstration labels before prompt construction and add a test that checks the curated example set.
- **Order or wording changes cause a large shift.** Repeated prompt edits appear to “fix” individual cases while moving other results. Run a small controlled prompt sweep and report variability across candidate orders; pin the chosen template.
- **Prompt length consumes context or budget.** Truncation may remove late examples or the query. Measure token counts with the target tokenizer, reserve output capacity, and test the longest supported input.
- **Model output violates the expected schema.** A label explanation or invented category breaks downstream code. Constrain generation where supported and validate the result with an explicit parser and a safe fallback.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What is few-shot learning here? | Conditioning an inference on a small number of task examples in the prompt, without changing model parameters. |
| How do you select examples? | Cover relevant labels and boundary cases, avoid leakage, and test relevance-based selection against a stable development set. |
| Few-shot or fine-tuning? | Few-shot suits fast-changing or lower-volume tasks; fine-tuning may amortize repeated examples for stable high-volume tasks when data and evaluation support it. |
| How do you measure prompt quality? | Compare against zero-shot on held-out examples and report slice quality, invalid outputs, tokens, and latency. |
| Debugging follow-up: performance drops after adding examples—what do you inspect? | Check label balance, order, contradictory cues, truncation, and per-slice metrics; revert to a controlled baseline before changing multiple variables. |

## Related Concepts

- [In-Context Learning](15-in-context-learning.md) — broader inference-time conditioning phenomenon.
- [Zero-Shot Learning](14-zero-shot-learning.md) — task instruction without demonstrations.
- [Prompting](12-prompting.md) — prompt structure and instruction design.
- [Chain-of-Thought](16-chain-of-thought.md) — demonstrations may include intermediate solution steps for selected reasoning tasks.
- [Few-Shot Learning practice](../interview-prep/curriculum-practice.md#few-shot-learning) — matching manifest-owned assessment.
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 evaluation across zero-, one-, and few-shot settings.
- [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837) — analysis of which demonstration properties influence in-context performance.

## Real-World Examples

A support team can prototype routing categories by adding a small, reviewed set of ticket-label pairs to a prompt. This makes category definitions easy to edit while policy changes are frequent. Before release, the team should compare with a zero-shot baseline, inspect errors for each category and product, test ambiguous tickets, and measure the repeated prompt cost at expected traffic. If a high-volume category remains weak despite representative examples, the next step may be better labeled data and fine-tuning, or a rule-based route for cases with clear deterministic criteria. The choice should follow evidence from the task, not a universal shot count.
