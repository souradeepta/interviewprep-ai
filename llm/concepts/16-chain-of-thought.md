# Chain-of-Thought (CoT)
TL;DR: Chain-of-thought prompting asks a language model to generate intermediate steps before an answer, which can help on some multi-step tasks but does not guarantee correct or faithful reasoning.

## Prerequisites

Understand [prompting](12-prompting.md) and [in-context learning](15-in-context-learning.md). This lesson covers elicited intermediate text at inference; it does not treat generated explanations as verified records of a model's internal computation.

## Learning Outcomes

- Describe zero-shot and example-based chain-of-thought prompting and the types of tasks they target.
- Explain why intermediate text can help while still containing unsupported or invalid steps.
- Compare direct answers, CoT, and self-consistency using task quality, latency, and token cost.
- Design a task-appropriate validation method that checks final answers independently of their explanations.

## Core Intuition

For a multi-step arithmetic problem, writing intermediate quantities can expose where a calculation goes wrong: start with 12, subtract 3, then subtract 2. CoT prompting asks a model to produce a sequence of intermediate statements before its final answer. Those extra tokens give the model a structured continuation and can improve performance on selected reasoning benchmarks, especially for sufficiently capable models and tasks where decomposition is useful.

The important limit is that a plausible sequence is not proof of correctness. The model may generate a wrong step, rationalize an answer after the fact, or produce text that does not faithfully represent the computation that led to the output. The chain is generated language and must be checked against the task. For an exact math problem, execute or independently verify the arithmetic. For a factual question, check authoritative evidence. For a classification task, score the final label against labeled examples. A longer explanation is not a confidence estimate.

CoT is also distinct from a formal planner or proof system. It usually does not enforce valid transitions between steps. If correctness depends on invariants, arithmetic precision, or external facts, a calculator, program, database, or verifier may be more reliable than asking for more prose. The design goal is therefore not “always make the model think longer”; it is to test whether an intermediate representation improves the final task metric enough to justify its cost and risks.

## How It Works

In direct prompting, the model produces an answer from the question. A CoT prompt changes the requested output sequence, for example by asking for a concise derivation or by supplying worked demonstrations. Zero-shot CoT provides an instruction to break the problem into steps. Few-shot CoT provides examples whose intermediate steps and final answers model the expected structure. Neither method changes the model weights during inference.

Let `q` be a problem, `r` an intermediate text sequence, and `a` the final answer. The model generates from `p(r, a | q, prompt)`. The application should evaluate `a` against a task-specific criterion, and may separately evaluate whether `r` satisfies useful constraints. This notation does not imply that `r` is a causal transcript of hidden reasoning. CoT may be inappropriate where internal-looking explanations are sensitive or where users could mistake them for audited evidence; a concise answer with citations or a structured, externally verifiable trace can be a better interface.

Self-consistency is a related inference strategy: sample multiple reasoning paths, extract each final answer, and aggregate them, often by majority vote. It can improve some reasoning benchmarks when errors are diverse and answers can be compared. It also multiplies generation work, and voting does not help when samples share the same systematic error. It requires robust answer extraction: naive parsing of the last token can turn formatting variation into a false result. For numerical answers, normalize representations and reject malformed outputs before aggregation.

CoT should be evaluated at the level of the application outcome. Use a fixed test set and compare direct-answer and CoT variants with the same model and decoding conditions where feasible. For sampling strategies, report the number of generations and total tokens. Break results down by difficulty and problem type. Include cases that tempt a model to confidently invent intermediate facts, and verify that the final answer is not merely consistent with a flawed explanation. On safety-sensitive tasks, keep expert review and established decision procedures in the loop.

## Trade-offs

| Method | Potential benefit | Cost | Main limitation |
|---|---|---|---|
| Direct answer | Low output length and latency | Minimal | May fail to organize a problem requiring multiple steps |
| Zero-shot CoT | Adds a decomposition cue without authoring worked examples | More output tokens and latency | Effect varies by model and task; instruction can elicit irrelevant prose |
| Few-shot CoT | Demonstrates an expected solution format and task-specific steps | Prompt plus output tokens; example curation | Demonstrations may encode errors or bias the solution path |
| Self-consistency | Aggregates multiple sampled answers when errors vary | Multiple generations and answer parsing | Shared errors remain; majority vote is not verification |
| Tool or verifier | Can enforce arithmetic or task-specific constraints | Integration and tool execution cost | Requires a correct tool, schema, and failure handling |

The choice depends on whether the task has objectively checkable outputs and on the service's latency budget. For a short arithmetic expression, direct execution can be both cheaper and more exact. For a word problem, a concise structured solution may help identify interpretation errors, but the arithmetic should still be checked. For open-ended analysis, evaluate factual support and usefulness rather than rewarding length. A benchmark gain in one model or dataset does not guarantee a gain in another.

## Implementation

This dependency-light example compares a direct arithmetic baseline with a hand-written step sequence and verifies the result with a separate function. It illustrates independent checking; it does not call an LLM or measure CoT effectiveness.

```python
def direct_answer(start, bought, used):
    return start + bought - used

def check_steps(start, bought, used, stated_answer):
    after_purchase = start + bought
    after_use = after_purchase - used
    return after_purchase, after_use, after_use == stated_answer

answer = direct_answer(12, 3, 2)
intermediate, final_value, valid = check_steps(12, 3, 2, answer)
assert (intermediate, final_value, valid) == (15, 13, True)
assert check_steps(12, 3, 2, 7)[2] is False
```

In a model-backed application, keep the output contract separate from the prompt for reasoning. Parse the final answer into a typed field, reject missing or malformed values, and validate it with an independent task check where possible. If you compare self-consistency, cap the sample count and record total cost; never silently return a majority from malformed or incompatible answers. Retain only the explanation detail needed by the user and the task, and avoid claiming that generated text is a faithful internal trace.

## Failure Modes

- **A plausible but invalid step leads to the answer.** The explanation reads smoothly but violates arithmetic or a domain rule. Recompute with code, a trusted tool, or an explicit verifier and measure final-answer correctness separately from explanation quality.
- **The task does not benefit from decomposition.** CoT adds latency and verbosity without improving outcomes. Run an ablation against direct answers and remove the extra generation if the task metric does not justify it.
- **The prompt elicits unsupported facts.** The model fills missing premises with invented details. Require the system to identify insufficient information, verify claims against sources, and test under-specified examples.
- **Self-consistency shares correlated errors.** Many samples repeat the same wrong assumption. Add adversarial or counterexample tests, use an independent verifier, and compare the method with a deterministic tool.
- **Answer extraction is brittle.** A correct value followed by punctuation or units is parsed incorrectly. Use a constrained structured output when available and validate typed results with explicit test cases.
- **Explanation is mistaken for a faithful trace.** Reviewers infer that every generated step reveals the model's true internal process. Describe it as generated rationale, and base audits on observable task behavior and independent evidence.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What is CoT prompting? | Asking the model to generate intermediate text before its final answer, sometimes using worked examples to show a solution format. |
| When can it help? | On tasks where decomposition is useful and the model can use the extra context; validate the gain on the target task rather than assuming it. |
| Is a correct-looking chain proof of a correct answer? | No. Steps can be wrong, unsupported, or unfaithful to the computation. Check the final result independently. |
| What is self-consistency? | Generate multiple sampled solutions and aggregate normalized final answers; it may help when errors differ but increases cost and is not a verifier. |
| Design follow-up: how do you decide whether to deploy CoT? | Compare against direct prompting on held-out cases, independently score outputs, and report quality, token usage, latency, and failure slices. |

## Related Concepts

- [In-Context Learning](15-in-context-learning.md) — inference-time context can include reasoning demonstrations.
- [Prompting](12-prompting.md) — instructions and output structure.
- [Planning and Reasoning](../../agentic-ai/concepts/07-planning-reasoning.md) — explicit decomposition in multi-step agent workflows.
- [Chain-of-Thought practice](../interview-prep/curriculum-practice.md#chain-of-thought-cot) — matching manifest-owned assessment.
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) — primary study of the prompting method.
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — primary study of sampled reasoning and answer aggregation.

## Real-World Examples

A grading assistant might produce a short solution outline for a word problem and compare the final value with an independently computed answer. The outline can help a teacher locate a student's misconception, but it should not determine the grade by itself: an incorrect explanation can end at the right number, and a valid alternative method may differ from the generated steps. A sensible evaluation tests answer accuracy and explanation usefulness separately, measures the additional response time, and routes ambiguous cases for human review. This is a design example, not a claim about measured educational outcomes.
