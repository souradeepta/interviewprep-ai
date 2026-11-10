# In-Context Learning
TL;DR: In-context learning is an inference-time behavior in which a language model conditions its output on instructions, examples, or other information in the current context without updating its parameters.

## Prerequisites

Understand autoregressive language models and [prompting](12-prompting.md); [few-shot learning](13-few-shot-learning.md) is one common ICL setup. This lesson separates the observable behavior from claims about the model's internal learning mechanism.

## Learning Outcomes

- Define ICL as conditioning at inference and distinguish it from parameter-updating training.
- Describe the roles of instructions, demonstrations, query, and context budget in an ICL prompt.
- Compare ICL with fine-tuning and retrieval-selected context using quality, latency, and update requirements.
- Design an evaluation that detects example-order sensitivity, prompt leakage, and distribution shift.

## Core Intuition

Think of a model solving a task while looking at a temporary reference sheet. The sheet can contain instructions, a few labeled examples, a schema, or relevant facts. The model's output changes because the current input changes, while the stored model parameters remain fixed. When the interaction ends, the model has not acquired a persistent update from those examples.

The reference-sheet analogy describes the interface, not a proven internal algorithm. The model may exploit patterns from its pretraining, attend to similarities between demonstrations and the query, infer a latent task, or use several mechanisms together. Research continues to study what demonstrations contribute. An observed improvement after adding examples does not prove that the network formed a clean symbolic rule or that its generated explanation faithfully describes its computation.

Few-shot prompting is a subset or common form of ICL: it places a handful of input-output demonstrations in context. ICL is broader. It can use instructions without examples, retrieved documents, tool results, schemas, conversation history, or many demonstrations. “Zero-shot” commonly means no task examples, even though the prompt itself still supplies instructions. Keeping these terms distinct helps compare systems and avoid calling every long prompt “few-shot learning.”

## How It Works

For a decoder-only model, the next-token distribution can be written as `p(y | c, x)`, where `c` is the supplied context and `x` is the current query. The context may be serialized as:

```text
instruction + examples or evidence + query + response prefix
```

The model performs a forward computation over this sequence and generates tokens. No gradient update occurs in ordinary prompting. Context tokens are nevertheless computationally important: they consume the available window and can increase prefill work. A long context may be truncated, may leave less room for output, and may make useful information harder to use. The cost depends on architecture, caching, batching, and provider implementation, so measure it for the deployment rather than assuming a fixed scaling number.

An ICL result depends on more than example count. Demonstrations can communicate the label space, input distribution, output format, and intended transformation. Their labels and examples may also contain spurious cues. A study by Min and colleagues found that, on selected classification and multiple-choice benchmarks, changing demonstration labels sometimes had a smaller effect than other demonstration properties. That finding is evidence against a simplistic “the model just copies the examples” explanation; it is not a general license to use incorrect examples. The task, model, and evaluation setting matter.

Ordering and selection can affect performance. If demonstrations are retrieved by similarity, measure retrieval recall and inspect near-miss examples. If examples are fixed, test candidate orders on a development split and pin the chosen prompt. Do not tune prompts against the final test set. For multi-turn use, decide which prior turns remain in context and how stale instructions or conflicting user-provided text are handled. For retrieved documents, maintain provenance and treat document content as data, not as trusted control instructions.

ICL is attractive when task behavior changes faster than model weights can be retrained, when the relevant context is query-specific, or when a workflow needs a transparent prototype. Fine-tuning may suit repeated high-volume behavior that can be captured in a curated training set. Retrieval augmentation is suitable when the context must come from an external corpus. These approaches can be combined, but each adds its own evaluation and maintenance surface.

## Trade-offs

| Approach | Where task information lives | Update path | Per-request cost | Main risk |
|---|---|---|---|---|
| Zero-shot ICL | Instruction and query in prompt | Edit instruction | Shorter prompt than demonstrations | Ambiguous task or output contract |
| Few-shot ICL | Instructions and examples in prompt | Replace or select examples | Repeated demonstration tokens | Example sensitivity and context cost |
| Retrieval-augmented context | External corpus plus prompt | Update corpus and index | Retrieval latency and context tokens | Missed, stale, or irrelevant evidence |
| Fine-tuning | Model parameters | Train, evaluate, version, deploy | Often shorter inference input | Training data, model drift, and rollback burden |

Evaluate these alternatives on the same held-out queries and model where possible. Track exact-match or task-specific quality, calibration or abstention behavior when relevant, prompt tokens, latency percentiles, and error slices. For ICL specifically, include order permutations, near-duplicate examples, misleading demonstrations, long inputs near the context limit, and examples from a shifted domain. For retrieval-based context, also measure retrieval recall and evidence use; for fine-tuning, compare training and inference costs over the expected request volume.

## Implementation

This dependency-light example constructs an ICL prompt and makes context limits explicit at the application layer. It is not an LLM and makes no claim about the model's internal process.

```python
def make_icl_prompt(instruction, demonstrations, query, max_chars=500):
    if not instruction.strip() or not query.strip():
        raise ValueError("instruction and query are required")
    blocks = [instruction.strip()]
    blocks.extend(f"Input: {x}\nOutput: {y}" for x, y in demonstrations)
    blocks.append(f"Input: {query.strip()}\nOutput:")
    prompt = "\n\n".join(blocks)
    if len(prompt) > max_chars:
        raise ValueError("prompt exceeds configured context budget")
    return prompt

demo = [("2 + 3", "5"), ("7 + 1", "8")]
prompt = make_icl_prompt("Return the sum as an integer.", demo, "4 + 6")
assert prompt.endswith("Input: 4 + 6\nOutput:")
assert prompt.count("Input:") == 3
try:
    make_icl_prompt("sum", demo, "4 + 6", max_chars=10)
except ValueError:
    pass
else:
    raise AssertionError("over-budget prompts must be rejected")
```

To evaluate a model, keep model version and decoding settings fixed while varying one context property at a time. Save prompt templates and demonstration identifiers with experiment results. Add a test that the application does not silently drop the query when the context is too long. When prompts contain user or retrieved text, delimit it clearly and test attempts to override the task instructions. A successful prompt-construction test only verifies formatting; model behavior still needs an empirical evaluation on representative data.

## Failure Modes

- **Prompt examples encode an unintended shortcut.** Performance looks strong on random splits but fails on a topic or source slice. Build disambiguating examples and report group-based metrics, not only an aggregate score.
- **Order or position sensitivity is hidden.** One hand-picked ordering appears reliable. Permute examples on development data, quantify variance, and use a stable selection and ordering rule.
- **Context truncation drops important information.** Errors rise on long conversations or evidence sets. Count tokens using the target model's tokenizer, reserve space for the query and output, and test boundary-length requests.
- **Evaluation leaks into prompt design.** Repeatedly choosing examples based on test errors inflates reported quality. Separate prompt development, validation, and final test data; freeze prompt and example IDs before final evaluation.
- **External context is stale or untrusted.** The model repeats obsolete policy or follows instructions embedded in retrieved text. Version sources, filter and delimit evidence, track provenance, and test adversarial or conflicting documents.
- **ICL behavior is mistaken for persistent learning.** A later request without the examples loses the behavior. Keep needed instructions or evidence available per request, or use a trained model when persistent adaptation is required.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What is ICL? | A model's inference-time conditioning on current context, without updating its parameters during the prompt interaction. |
| Is ICL mechanism fully understood? | No. The behavior is observable, but competing accounts study what models extract from instructions and demonstrations. Avoid equating performance with a proven internal learning algorithm. |
| How is few-shot related? | Few-shot is an ICL setting that supplies a small number of examples; ICL can also use instructions, retrieved evidence, schemas, and other context. |
| ICL or fine-tuning? | Use ICL for fast, query-specific or changing context; consider fine-tuning for stable, repeated behavior when measured quality and total cost justify training. |
| Design follow-up: how would you test robustness? | Use a held-out set with prompt-order permutations, misleading examples, context-boundary cases, and shifted-domain slices; measure quality, tokens, and latency. |

## Related Concepts

- [Few-Shot Learning](13-few-shot-learning.md) — ICL with a small set of task demonstrations.
- [Zero-Shot Learning](14-zero-shot-learning.md) — inference from instructions without task examples.
- [Prompting](12-prompting.md) — task instructions and output structure.
- [Chain-of-Thought](16-chain-of-thought.md) — a reasoning-style prompt format, with its own reliability limits.
- [In-Context Learning practice](../interview-prep/curriculum-practice.md#in-context-learning) — matching manifest-owned assessment.
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — foundational study of few-shot language-model behavior.
- [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837) — experimental analysis of demonstration properties.
- [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234) — overview of definitions, methods, and open questions.

## Real-World Examples

A classification service can include a compact label schema and selected examples to adapt one general model to several teams. Each team can revise its examples without a model release, but the service must version prompt configurations and evaluate them like code. If a team provides examples from only one product, the model may learn product vocabulary instead of the intended intent labels. A robust launch includes cross-product test slices, a comparison to an instruction-only baseline, controlled prompt updates, and monitoring for invalid labels and latency. This illustrates the operational value of ICL without claiming that the model acquired durable task knowledge.
