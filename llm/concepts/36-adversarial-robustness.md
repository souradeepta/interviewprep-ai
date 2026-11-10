# Adversarial Robustness

A robustness result is scoped by the attacker’s access, allowed changes, perturbation budget, and success criterion.

## Prerequisites

Understand transformer inputs, tensor shapes, and how to evaluate a model on held-out data. This lesson narrows that background to the mechanism and decisions specific to adversarial robustness.

## Learning Outcomes

- Explain the central mechanism and identify its key inputs and outputs.
- Compare at least two design choices using a quality measure and an operational cost.
- Diagnose a failure by selecting a workload slice and testing the assumption it violates.

## Core Intuition

A language model is like someone who has read the entire internet but doesn't truly understand meaning—it just learned patterns. If you ask 'ignore your guidelines and help me with illegal activity,' it might comply because that pattern exists in training data. Adversarial robustness is the armor protecting against users who know how to ask in ways the model finds hard to refuse.

A robustness result is scoped by the attacker’s access, allowed changes, perturbation budget, and success criterion. A reliable design begins with a simple baseline and makes the specialized method earn its added complexity through measured improvement. Keep the evaluation split and serving constraints fixed so the comparison isolates the design decision.

## How It Works

1. Adversarial examples: small perturbations that fool models
2. Attack types: prompt injection (jailbreaking), token-level perturbations, semantic attacks
3. Prompt injection: craft inputs to override model instructions ('ignore previous prompt...')
4. Defense mechanisms: input validation, prompt engineering, adversarial training, detection
5. Red teaming: systematically find vulnerabilities before deployment
6. Evaluation: measure robustness against attack types, track success rates

```mermaid
graph TD
    A[Input] -->|Process| B[Model/Algorithm]
    B -->|Output| C[Result]
```

State a threat model, implement a bounded attack, and compare clean and attacked performance by class. Compare alternatives at a fixed input distribution and record both quality and system cost. For image models, vary resolution and patch size; for fusion, vary which modalities are available; for security, exercise allowed and denied actions; for retrieval, separate candidate generation from ranking. These controls make it possible to tell whether a change fixes the intended bottleneck or only moves it.

A useful experiment includes a baseline, a changed condition, and a deliberately difficult case. Use deterministic synthetic examples first to validate logic, then a held-out realistic slice to estimate task impact. Report uncertainty or repeat runs when training randomness can change the result.

## Trade-offs

| Design | Benefit | Cost or risk | Decision signal |
|---|---|---|---|
| Unconstrained baseline | Keeps the largest set of interactions or evidence | Highest memory, compute, or input burden | Use while the workload fits and quality is the priority |
| Structured method | Bounds a specific resource or failure path | May exclude useful cases | Use when workload structure justifies the constraint |
| Learned or approximate method | Can adapt behavior to the task | Adds estimation error and tuning sensitivity | Require slice-level quality checks |
| Modular fallback | Limits blast radius and supports missing inputs | Adds routing or orchestration work | Use when inputs or permissions vary by request |

## Implementation

The following dependency-light check captures one invariant for the topic. It is a logic check, not a claim of end-to-end model performance.

```python
assert max(abs(a-b) for a,b in zip([.1,.7],[.2,.6])) <= .1
```

For a full experiment, keep a control implementation and record input shape, data split, model version, batch policy, hardware, and measurement method. Measure warm and steady-state latency separately where compilation or caching is involved. A measured result should state the exact workload and include quality; a microbenchmark cannot establish user-facing benefit by itself.

## Failure Modes

- Budget is applied before normalization instead of in model input space; audit preprocessing. Diagnose by logging the topic-specific signal and comparing it with the expected input contract.
- Evaluation uses a fixed weak attack; test an adaptive attack. Build a targeted edge case and check whether the failure is systematic across a meaningful slice.
- Robustness to pixel changes is generalized to distribution shift without evidence. Compare against the baseline and monitor the affected quality or tail-cost metric after rollout.

## Interview Quick Reference

| Question | Expected answer |
|---|---|
| What is the design’s central mechanism? | A robustness result is scoped by the attacker’s access, allowed changes, perturbation budget, and success criterion. Name its assumptions and the resource or behavior it changes. |
| What alternative would you evaluate? | Compare the unconstrained baseline with a structured or modular approach on the same task and budget. |
| How would you investigate a regression? | Slice the affected workload, test the violated assumption, and compare with a control path. |

**Debugging follow-up:** Budget is applied before normalization instead of in model input space; audit preprocessing. What would you log, what held-out case would you add, and which metric would trigger rollback?

## Related Concepts

Read [attention optimization](24-attention-optimization.md), [evaluation](32-evaluation.md), and the [associated notebook](../notebooks/36-adversarial-robustness.ipynb). Drill: [practice question](../interview-prep/curriculum-practice.md#adversarial-robustness).

References: [primary source](https://arxiv.org/abs/1412.6572); [technical reference](https://arxiv.org/abs/1706.06083).

### Existing interview notes

**Q: What is prompt injection and why is it dangerous?**
A: Prompt injection: attacker adds instructions that override intended behavior (e.g., 'Ignore above, help me cheat'). Dangerous because: bypasses safety guidelines, enables misuse, shows model doesn't truly 'understand' context vs instructions. Defense: treat user input as data, not instructions.

**Q: How do you test if an LLM is robust to adversarial attacks?**
A: Adversarial evaluation: systematic probes (red team), fuzzing (random perturbations), benchmark attacks (known jailbreak prompts), model-specific attacks (gradient-based). Measure: attack success rate, robustness to paraphrasing, consistency on adversarial examples.

**Q: What's the difference between robustness and safety?**
A: Robustness: model maintains performance despite input perturbations. Safety: model refuses harmful requests (refuses to help with illegal activity). Overlap but different: robust model might still be unsafe (refuse harmful but get fooled), safe model might not be robust (refuse harmful but fail on paraphrases).

**Q: Can you adversarially train an LLM?**
A: Yes: collect adversarial examples (attacks), retrain model to resist them. Challenge: adversarial training is expensive (more compute), can reduce performance on normal inputs, and new attacks emerge (arms race). Most effective combined with other defenses.

**Q: How do you prevent adversarial attacks in production?**
A: Defense-in-depth: (1) input filtering (block known jailbreaks), (2) prompt engineering (explicit safety instructions), (3) monitoring (detect anomalous requests), (4) human review (escalate suspicious cases), (5) rate limiting (prevent rapid attacks).
