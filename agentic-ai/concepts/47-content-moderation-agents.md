# Content Moderation Agents
**TL;DR:** Moderation agents combine policy-specific classification, context gathering, and calibrated action routing with appeal and audit mechanisms.

## Prerequisites
Classification metrics, safety policies, and human review workflows.

## Learning Outcomes
- Distinguish content labels from enforcement actions.
- Choose thresholds using harm severity and review capacity.
- Evaluate error rates and appeal outcomes across language and user groups.

## Core Intuition
A smoke alarm should be sensitive enough to catch danger, but a false alarm still has a cost. Moderation has multiple alarm levels: hide immediately, queue for review, add friction, or allow. The policy defines which evidence justifies each action. A model score is evidence for routing; it is not itself a policy decision.

## How It Works
First define a policy taxonomy with examples, counterexamples, severity, and allowed actions. Classifiers produce calibrated scores for a particular policy label; contextual tools may retrieve surrounding conversation, account signals permitted by policy, and language identification. A decision layer maps evidence to actions using thresholds that reflect both harm and review capacity. High-impact enforcement should preserve the content snapshot, policy version, model version, evidence, and action reason. Appeals go to an independent review path and can reverse or restore content.
Context windows matter: quoting abusive language to report it differs from directing abuse at a person, and a single message can be harmless in one conversation but threatening in another. Include only context necessary to interpret the policy, and apply retention limits because moderation data can be sensitive. Avoid inferring intent from protected traits or unreliable account proxies. When language identification or model confidence is weak, the system should route to a capable reviewer or apply a reversible limited action rather than pretending the score is certain.
A moderation action is a policy operation with consequences for speakers and readers. Store the relevant policy clause and the evidence class behind an action so an appeal reviewer can understand the decision without relying on hidden model reasoning. Keep thresholds versioned by category because threats, language coverage, and review capacity differ. When reviewers disagree, preserve disagreement as an uncertainty signal and update examples through governance rather than allowing a noisy label to silently alter policy.
Select thresholds on policy-labeled data and report precision, recall, calibration, and action rates separately by policy class, language, and relevant content context. An aggregate F1 score obscures the relative cost of allowing severe harm versus removing benign speech. Sample allowed items to estimate false negatives, review removed items for false positives, and treat appeal reversals as a delayed quality signal. Measure reviewer queue time and exposure to harmful content as operational constraints.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Automatic removal | Fast response to clear severe harm | False positives can suppress speech |
| Temporary friction | Limits spread while uncertain | Can burden benign users |
| Human review | Handles nuanced context | Capacity and latency limits |
| Allow with sampling | Preserves expression | Misses harm until audit |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def moderation_action(score, threshold, high_impact):
    if high_impact or score < threshold:
        return 'human_review'
    return 'apply_policy'

assert moderation_action(0.99, 0.8, True) == 'human_review'
```

## Failure Modes
- Quoted or satirical content is misclassified: include bounded conversation context and test policy counterexamples.
- Minority-language error rates are hidden by aggregate scores: publish per-language calibration and route weak coverage to review.
- Appeal reversals rise: audit the action threshold and policy examples, then replay affected cases.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| How do you choose an enforcement threshold? | Set it per policy severity using labeled error costs and reviewer capacity; measure both missed harmful content and wrongful actions. Design follow-up: what happens when the queue fills? |
| How do you handle ambiguous context? | Use the minimum relevant context, choose reversible actions, and route uncertainty to trained reviewers. Debug follow-up: inspect quoted speech and language coverage. |
| How do you detect disparate quality? | Audit calibrated error and action rates across languages and policy slices, including sampled allowed content and appeal outcomes. |

## Related Concepts
See [Safety alignment](36-safety-alignment.md), [human-agent collaboration](37-human-agent-collaboration.md), [agent evaluation](29-agent-evals.md).

## References
- [transparency.meta.com](https://transparency.meta.com/policies/community-standards/)
- [www.nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)

## Further Topic Detail
## Core Intuition
A smoke alarm should be sensitive enough to catch danger, but a false alarm still has a cost. Moderation has multiple alarm levels: hide immediately, queue for review, add friction, or allow. The policy defines which evidence justifies each action. A model score is evidence for routing; it is not itself a policy decision.


## How It Works
First define a policy taxonomy with examples, counterexamples, severity, and allowed actions. Classifiers produce calibrated scores for a particular policy label; contextual tools may retrieve surrounding conversation, account signals permitted by policy, and language identification. A decision layer maps evidence to actions using thresholds that reflect both harm and review capacity. High-impact enforcement should preserve the content snapshot, policy version, model version, evidence, and action reason. Appeals go to an independent review path and can reverse or restore content.
Context windows matter: quoting abusive language to report it differs from directing abuse at a person, and a single message can be harmless in one conversation but threatening in another. Include only context necessary to interpret the policy, and apply retention limits because moderation data can be sensitive. Avoid inferring intent from protected traits or unreliable account proxies. When language identification or model confidence is weak, the system should route to a capable reviewer or apply a reversible limited action rather than pretending the score is certain.
A moderation action is a policy operation with consequences for speakers and readers. Store the relevant policy clause and the evidence class behind an action so an appeal reviewer can understand the decision without relying on hidden model reasoning. Keep thresholds versioned by category because threats, language coverage, and review capacity differ. When reviewers disagree, preserve disagreement as an uncertainty signal and update examples through governance rather than allowing a noisy label to silently alter policy.
Select thresholds on policy-labeled data and report precision, recall, calibration, and action rates separately by policy class, language, and relevant content context. An aggregate F1 score obscures the relative cost of allowing severe harm versus removing benign speech. Sample allowed items to estimate false negatives, review removed items for false positives, and treat appeal reversals as a delayed quality signal. Measure reviewer queue time and exposure to harmful content as operational constraints.


## Detailed Explanation

Moderation agents flag harmful content: toxicity, hate speech, misinformation, copyright, spam. Mechanisms: rule-based (regex), ML-based (classification), semantic (understanding meaning). Challenges: false positives (legitimate posts flagged), false negatives (miss violations), cultural context (same phrase offensive in one context, fine in another). Best for: social platforms, comment sections, user-generated content, platform safety.



## Interview Q&A

**Q: How do you balance recall (catching all harmful content) vs. precision (avoiding false positives) in content moderation?**
A: The balance depends on the harm severity and platform context. For high-severity content (CSAM, terrorism incitement): maximize recall even at cost of false positives—human reviewers verify flagged content. For lower-severity content (mild rudeness, borderline spam): optimize for precision to avoid over-removal. Operationalize: set threshold by measuring false positive and false negative rates on labeled data at different thresholds. Track both rates over time and retune quarterly as content patterns evolve.

**Q: How do you handle edge cases and context-dependent content in moderation?**
A: Context determines meaning: satire looks like harmful content without context, medical discussions involve sensitive topics that should be permitted. Design a multi-stage pipeline: first classify the most obvious cases confidently, then pass ambiguous cases to context-aware analysis (does the account have a history? is this a recognized satire format? is this a medical platform?). For genuinely ambiguous cases, escalate to human review with the full context. Maintain a case library of borderline decisions with explanations for consistency.

**Q: What are the ethical implications of content moderation agents and how do you address them?**
A: Key issues: disparate impact (moderation may be less accurate for non-English or non-Western content, leading to over-moderation of minority communities), lack of transparency (users don't understand why content was removed), inconsistent enforcement (same content treated differently), and chilling effects (users self-censor fearing moderation). Address by: measuring accuracy across demographic groups and languages, providing clear moderation policies and specific reasons for removals, appeal mechanisms, and regular audits of moderation patterns.

**Q: How do you keep moderation models up-to-date with evolving harmful content tactics?**
A: Bad actors adapt to detection: new slang, coded language, adversarial perturbations. Maintain: active adversarial red-teaming (attempt to evade the current system and use failures as training data), regular model retraining with recent data (monthly), canary testing (known-bad content tested daily to detect degradation), and human review sampling (regularly review a random sample of allowed content to catch false negatives). Treat moderation as an ongoing arms race, not a one-time deployment.

**Q: What appeals process should accompany automated content moderation?**
A: Every moderation action should be appealable. The appeal process must include: human review (not just the same model), access to the specific reason for moderation, reasonable SLA (24-48 hours for most content), and actual reversal capability (not just acknowledgment). For high-stakes removals (accounts, verified creators), provide expedited review. Track appeal outcomes: high reversal rates indicate over-moderation in that category. Appeals are your most valuable signal for improving precision.

**Q: How do you moderate content across multiple languages with unequal model quality?**
A: Measure accuracy separately for each language using labeled test sets. For languages with low model accuracy, increase the threshold for automated action (require higher confidence before removing) and increase the routing rate to multilingual human reviewers. Build language-specific training data actively—work with native speakers to label edge cases. Prioritize languages by user volume: languages with many users justify dedicated model development. Acknowledge and communicate limitations: don't claim uniform coverage if accuracy varies significantly.
