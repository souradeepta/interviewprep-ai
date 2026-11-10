# Multiple Testing and False-Discovery Control

TL;DR: Multiple-testing procedures adjust a family of claims so repeated hypothesis tests do not create more false discoveries than the analysis can tolerate.

## Prerequisites

Review [hypothesis testing](05-hypothesis-testing.md), [confidence intervals](06-confidence-intervals.md), and [A/B testing](08-ab-testing-statistics.md).

## Learning Outcomes

- Define the family of hypotheses for a scientific or product decision.
- Distinguish family-wise error rate (FWER) from false discovery rate (FDR).
- Apply and interpret Bonferroni, Holm, and Benjamini–Hochberg procedures.
- Explain the role of dependence, exploratory selection, and effect-size reporting.

## Core Intuition

If a fair coin is flipped once, a surprising result is uncommon. If it is flipped a thousand times and every result is inspected, some streak will look surprising by chance. Testing many hypotheses creates the same selection opportunity. The correction depends on which mistake matters: one false claim among many may be intolerable in a confirmatory clinical decision, while a discovery screen may accept a controlled fraction of false leads for follow-up.

The testing family is a scientific definition, not just a software list. It might be all metrics in a launch decision, all genes in a screen, or all model slices evaluated for a claim. If an analyst explores many outcomes then reports only those that pass a threshold, the effective family is larger than the final table suggests. Corrections do not undo outcome-driven analysis that was hidden or unrecorded; preregistration and transparent exploratory labeling are complementary safeguards.

## How It Works

FWER is the probability of at least one false rejection in the family. Bonferroni rejects a hypothesis when (p_i \le \alpha/m), where m is the number of tests, guaranteeing FWER no greater than α without requiring independence. Holm sorts p-values, then compares the smallest to α/m, the next to α/(m−1), and stops at the first failure; it controls FWER and is usually at least as powerful as plain Bonferroni.

FDR is the expected proportion of false rejections among all rejections, with the proportion treated as zero when none are rejected. Benjamini–Hochberg (BH) sorts p-values (p_{(1)} \le \dots \le p_{(m)}), finds the largest rank k such that (p_{(k)} \le (k/m)q), and rejects ranks 1 through k. Under independence and certain positive dependence structures, this controls FDR at q. Under arbitrary dependence, a more conservative procedure such as Benjamini–Yekutieli or dependence-aware analysis may be needed.

The guarantee is about the procedure over repeated datasets, not a per-result probability that a reported discovery is false. For example, FDR of 5% does not mean every selected hypothesis has exactly a 5% chance of being false, nor that exactly 5 of 100 results will be false. The realized false fraction can be much larger in a particular dataset, especially when few discoveries are made. Communicate the controlled criterion and validate promising findings on independent data when decisions are consequential.

Dependence deserves particular attention in ML and product work. Metrics often share users, labels, and instrumentation; gene-expression features may be correlated; subgroup estimates overlap. BH is often used under positive dependence, but not every correlation structure satisfies its formal conditions. When no defensible dependence condition is available, Benjamini–Yekutieli controls FDR under arbitrary dependence at the cost of a harmonic-factor penalty, or a design-specific resampling method may be appropriate. The choice should reflect the data structure and cost of missed discoveries rather than defaulting to the least conservative option.

The family also interacts with estimation. Selecting estimates because they pass a threshold creates a winner’s curse: selected effect sizes tend to be exaggerated even if the testing procedure controls an error rate. Adjusted p-values do not automatically produce valid post-selection confidence intervals. Separate discovery and confirmation datasets, shrink estimates using a justified model, or report selection-aware uncertainty if effect magnitudes will drive downstream decisions.

Adjusted p-values summarize the smallest target level at which a result would be selected under a chosen procedure. They are not posterior probabilities that the hypothesis is true. A rejected null also says nothing about practical magnitude. Report raw effect sizes and confidence intervals, make the testing family explicit, and distinguish prespecified primary outcomes from exploratory scans.

## Trade-offs

| Method | Error criterion | Useful setting | Main cost |
|---|---|---|---|
| Bonferroni | FWER | Few high-stakes claims; any false positive is costly | Conservative, especially for many tests |
| Holm step-down | FWER | Confirmatory family with varied p-values | Still controls any false positive, though stronger than single-step Bonferroni |
| Benjamini–Hochberg | FDR | Discovery screens where follow-up validates leads | Dependence conditions and expected proportion interpretation matter |
| Benjamini–Yekutieli | FDR under arbitrary dependence | Highly dependent test families | Often substantially more conservative |
| No adjustment | None beyond per-test level | One prespecified primary hypothesis | Invalid if used to select from many outcomes |

## Implementation

The function below returns indices rejected by BH. It operates on a fixed family supplied by the caller; deciding what belongs in that family precedes the computation. The example uses synthetic p-values and is not a replacement for validating assumptions in a real design.

```python
def benjamini_hochberg(p_values, q=0.05):
    if not 0 <= q <= 1 or any(not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values and q must be between zero and one")
    ordered = sorted(enumerate(p_values), key=lambda pair: (pair[1], pair[0]))
    cutoff = 0
    m = len(ordered)
    for rank, (_, p_value) in enumerate(ordered, start=1):
        if p_value <= q * rank / m:
            cutoff = rank
    return {index for index, _ in ordered[:cutoff]}


p = [0.001, 0.012, 0.031, 0.20, 0.73]
assert benjamini_hochberg(p, q=0.05) == {0, 1, 2}
```

If `p_values` is empty, handle that explicitly before division; in production, the function should return an empty set for an empty family. Ties do not invalidate BH: the sorted order is only used to find a qualifying rank, after which all p-values at or below the cutoff are rejected. Preserve original test identifiers when sorting so rejected hypotheses can be matched to estimates and intervals.

## Failure Modes

- **Family defined after seeing results:** Reporting only favorable metrics understates the number of opportunities. Predefine the family or label analysis exploratory and disclose the selection process.
- **Wrong error criterion:** FDR does not cap the probability of any false claim. Use FWER when even one false positive is unacceptable; choose based on downstream consequences.
- **Assuming independence:** Correlated outcomes can change guarantees for a chosen procedure. Examine design/dependence and consider a method with a valid guarantee or resampling strategy.
- **Ignoring power loss:** Strict correction can make real effects hard to detect. Increase sample size, reduce redundant outcomes, prioritize a primary endpoint, or separate discovery from confirmation rather than silently dropping correction.
- **Treating adjusted significance as effect magnitude:** A tiny, unimportant change can be significant. Report estimates, intervals, domain thresholds, and decision costs.
- **Applying correction to a biased test:** Multiplicity adjustment cannot repair confounding, invalid standard errors, or p-hacking. Validate each test’s assumptions before correcting the family.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| What does BH control? | Expected false-discovery proportion among rejected hypotheses under its dependence conditions. | Contrast with probability of any false positive. |
| When prefer Holm to BH? | When the decision requires strong FWER control across a confirmatory family. | Debug a product dashboard with one primary and many exploratory metrics. |
| Does an adjusted p-value measure effect importance? | No; report effect size, interval, and practical threshold separately. | Design a validation study for selected discoveries. |

## Related Concepts

Continue with [hypothesis testing](05-hypothesis-testing.md), [A/B testing](08-ab-testing-statistics.md), and [permutation tests](17-permutation-tests-randomization-inference.md). The paired notebook is [18-multiple-testing-fdr.ipynb](../notebooks/18-multiple-testing-fdr.ipynb). Practice the metrics-dashboard scenario in [Statistics interview practice](../interview-prep/statistics-interview-questions.md#13-handle-multiple-metric-tests).

References: [Benjamini and Hochberg, “Controlling the False Discovery Rate” (1995)](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x) and [Holm, “A Simple Sequentially Rejective Multiple Test Procedure” (1979)](https://doi.org/10.1007/BF01907414).
