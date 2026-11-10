# Meta-Learning

**TL;DR:** Meta-learning optimizes how a learner adapts across related tasks, aiming to make a new task learnable from limited examples or updates.

## Prerequisites

Prerequisites: supervised learning, gradient descent, and train/validation separation. After this lesson, you can explain the distinction between task-level training and ordinary example-level training, describe episodic evaluation, compare initialization-based and metric-based methods, and state what assumptions permit adaptation to transfer. This ML lesson addresses how to learn an adaptation procedure; the broader [AI roadmap](../../ai/README.md) uses the term learning systems more generally and does not replace these statistical details.

## Learning Outcomes

- Explain the central assumptions and mechanism of meta learning.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Ordinary supervised learning asks, “What parameters work for the training distribution?” Meta-learning asks, “What starting point or update rule makes a family of related tasks easy to learn?” Imagine a field technician who services many similar machines: practice on varied machines may teach the technician where to look first on an unfamiliar one. The benefit exists only if the new machine shares useful structure with those practice machines. Memorizing a list of training machines is not adaptation.

A meta-learner therefore needs a task distribution, not just a shuffled pile of examples. Each task has support data used for adaptation and query data used to measure whether that adaptation worked. This split guards against reporting performance on examples that shaped the adapted parameters. The choice of task family defines the claim: adapting between product categories is different from adapting between hospitals or languages.

The support/query distinction is also useful in product design. A deployment may permit a learner to see a few labeled examples for a new user group, but it may not permit gradient updates in the live request path. One system can adapt offline after a support collection phase and publish a task-specific checkpoint; another can adapt per request using a learned distance rule. Their latency, privacy, rollback, and audit requirements differ. When labels arrive gradually, the method must specify whether support examples are retained, whether the model adapts continuously, and how earlier predictions are corrected if labels later change.

## How It Works

Let tasks τ be sampled from p(τ), with a support set Sτ and query set Qτ. A generic objective is minimize over meta-parameters φ the expected query loss after an adaptation procedure Aφ: Eτ[L(Qτ, Aφ(Sτ))]. In model-agnostic meta-learning (MAML), φ is an initialization θ; one or more inner gradient steps produce θ′τ=θ−α∇θL(Sτ,θ), and the outer update changes θ to reduce L(Qτ,θ′τ). Differentiating through the inner step yields higher-order derivatives unless a first-order approximation is used. Reptile also seeks an initialization that moves toward task-adapted parameters without full second-order differentiation.

Metric-based methods such as prototypical networks learn an embedding space. During each episode, they form a prototype (usually the mean embedding) for each support class and classify query examples by distance. This is attractive when tasks share labels only within episodes. Learned optimizers instead parameterize update behavior, but can be expensive and harder to inspect. Complexity depends on task count, inner steps, model size, and whether gradients through adaptation are retained; MAML can require substantial activation memory.

Evaluation must mimic deployment: sample held-out tasks, reveal only the support data allowed in practice, adapt with the declared budget, and score on untouched query examples. Split by task identity or domain, not merely by row, when the claim concerns new tasks. If the target task distribution differs, meta-training's adaptation advantage may disappear.

Task granularity is a consequential modeling choice. If an episode represents one class subset from a shared image collection, success demonstrates few-shot class recognition under that image domain. It does not establish rapid adaptation to a new camera or population. Conversely, if each episode is an institution, random rows from each institution in both training and test can make adaptation look easy by leaking site-specific patterns. Write down exactly which task attributes vary between episodes and which remain shared. Then build held-out tasks around the dimensions expected to vary at deployment.

The baseline must also receive equivalent information. A meta-learned initialization trained across many tasks has seen more total examples than a model trained on one target task. Compare it with pooled multitask pretraining, frozen features, target-only training, and a standard fine-tuning recipe using the same support set. Report performance as a function of support-set size and adaptation steps, rather than choosing one favorable point. If meta-learning only wins with an unrealistic number of gradient steps or a hand-curated task distribution, its practical advantage may be small.

## Trade-offs and Selection Boundary

| Family | What it learns | Best fit and boundary |
|---|---|---|
| Initialization based (MAML/Reptile) | Parameters that adapt quickly | Related tasks with small support sets; inner-loop tuning and compute are required |
| Metric based | Embedding geometry and class prototypes | Few-shot classification with episode-local labels; less suited to arbitrary output structures |
| Learned optimizer | An update rule | Repeated structured tasks; can overfit to task generator and be difficult to debug |
| Transfer learning | Reuses pretrained representations | Often simpler when one large source dataset exists; does not necessarily optimize fast adaptation |

Choose meta-learning when deployment repeatedly encounters new but related tasks and can provide a small support set before prediction. Prefer ordinary pretraining plus fine-tuning when there is one stable target task or a strong general pretrained model. Establish a simple baseline that receives the same support examples and adaptation compute; otherwise an apparent gain may be due only to extra training.

## Implementation

```python
def adapt_constant(ys, theta, rate):
    # One gradient step for a task-specific constant predictor.
    grad = 2 * sum(theta - y for y in ys) / len(ys)
    return theta - rate * grad

def task_loss(ys, theta):
    return sum((theta-y)**2 for y in ys)/len(ys)

# Each task has a different level; query labels remain held out during adaptation.
support_y, query_y = [6., 7.], [8., 9.]
before = task_loss(query_y, 0.0)
adapted = adapt_constant(support_y, 0.0, rate=.5)
after = task_loss(query_y, adapted)
assert after < before
assert adapted == 6.5 and after == 4.25
```

## Failure Modes and Diagnostics

1. **Task leakage:** examples from one task appear in both meta-training and held-out evaluation, so the learner recognizes a task rather than adapts. Split by task/domain identity and verify disjoint identifiers.
2. **Unrepresentative task generator:** episodes are artificially balanced or homogeneous while deployment has skewed classes or larger shifts. Compare task-level metadata and metrics against deployment, and vary the episode sampler deliberately.
3. **Support set does not match usage:** adaptation consumes labels or compute unavailable in production. Enforce the actual support budget and include adaptation time in reported latency.
4. **Inner-loop instability:** large adaptation steps reduce support loss but damage query performance. Track support and query curves separately; tune step size and number of steps on validation tasks.
5. **Negative transfer:** the new task is unrelated, so meta-initialization underperforms a generic or target-only baseline. Measure per-task gains and provide a fallback that skips adaptation when evidence is weak.

## Interview Quick Reference

| Question | Answer |
|---|---|
| Meta-learning versus transfer learning? | Meta-learning explicitly optimizes adaptation across a task distribution; transfer learning broadly reuses source knowledge. |
| How evaluate few-shot learning fairly? | Hold out tasks/domains, adapt only on support examples, and score on separate query examples under the deployment budget. |
| Design follow-up: new customers provide five labeled examples. | Define customer-level tasks, prevent customer overlap, compare meta-learned initialization with pretrained fine-tuning, and measure query quality plus adaptation cost. |

## Related Concepts

- [Transfer learning](transfer-learning.md), [domain adaptation](domain-adaptation.md), and [optimization](optimization.md)
- [Matching notebook](../notebooks/21-meta-learning.ipynb)
- [Interview practice: model selection](../interview-prep/ml-theory-questions.md#q-how-does-cross-validation-work-and-when-do-you-use-it)

## References

- Finn, Abbeel, and Levine, [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400).
- Snell, Swersky, and Zemel, [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175).
