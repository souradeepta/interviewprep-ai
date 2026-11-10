# Sequential Experimentation and Peeking Control

TL;DR: Sequential experiments allow evidence to accumulate over time, but valid stopping requires decision boundaries designed for repeated looks rather than repeated use of a fixed-horizon test.

## Prerequisites

Review [A/B testing](08-ab-testing-statistics.md), [hypothesis testing](05-hypothesis-testing.md), and [multiple-testing control](18-multiple-testing-fdr.md).

## Learning Outcomes

- Explain why repeated unadjusted fixed-horizon tests inflate false-positive probability.
- Distinguish fixed-horizon analysis from planned interim and continuous sequential designs.
- Describe alpha spending, sequential likelihood-ratio tests, and always-valid inference at a conceptual level.
- Specify monitoring, stopping, and reporting rules before observing experimental outcomes.

## Core Intuition

A fixed-horizon test is like agreeing to open one envelope after collecting a planned number of observations. If the experimenter opens an envelope every hour and stops as soon as one looks favorable, the procedure has many chances to find a lucky fluctuation. The nominal five-percent error rate described the one planned look, not the new rule “keep looking until significant.”

This does not mean teams must hide dashboards. Operational monitoring and inferential decisions serve different purposes. It is reasonable to check instrumentation, sample-ratio mismatch, safety signals, and service health continuously. The statistical stopping rule must account for outcome looks. A well-designed sequential test specifies boundaries or an error-spending schedule in advance, preserving a target error rate while often stopping early when evidence is strong. A prespecified fixed horizon remains the simplest approach when early stopping has little value.

## How It Works

Let (H_0) represent no effect and (H_1) a specified alternative. In a sequential probability ratio test, each observation updates a likelihood ratio (L_n = \prod_{i=1}^n f_1(X_i)/f_0(X_i)), or its log sum. Stop for H1 when the ratio crosses an upper boundary, stop for H0 below a lower boundary, and otherwise continue. Wald’s approximations relate the boundaries to Type I error α and Type II error β. The design can be sample-efficient when the alternative and distribution assumptions are suitable, but model misspecification and a poorly chosen alternative matter.

For experiments with planned interim analyses, alpha-spending functions allocate a cumulative error budget over looks. O’Brien–Fleming-style boundaries spend little alpha early and more near the end, discouraging premature stops; Pocock-style boundaries are more even. Group-sequential designs may also include futility boundaries. Always-valid p-values or confidence sequences provide guarantees under optional stopping when their assumptions hold. These methods do not authorize unplanned metric searches or changes to eligibility after outcomes are seen.

Sequential inference also depends on the experimental data process. Repeated peeks should use new information without double-counting outcomes, and delayed labels require a clear definition of which cohorts have matured. A safety stop may be triggered by a separate operational rule rather than a claim of efficacy. Predefine primary metrics, analysis population, look schedule, missing-data handling, stopping boundaries, and whether the test can restart. Document deviations and report the method used, not merely the final p-value.

A simple Bonferroni correction over K planned looks tests each look at α/K, controlling family-wise error even when looks are dependent, although it can be conservative. It is useful as an educational baseline and sometimes a practical fallback, but not equivalent in efficiency to a tailored sequential design. Simulation under the null can evaluate the complete planned decision procedure, including sample sizes and stopping rules. Simulations must preserve the assignment and outcome process; simulated calibration is evidence about the model, not a proof for a different deployment distribution.

## Trade-offs

| Method | Stopping flexibility | Error control | Main limitation |
|---|---|---|---|
| Fixed horizon | One planned final analysis | Simple at chosen α under test assumptions | No statistical early efficacy stop |
| Bonferroni across planned looks | Prespecified interim peeks | Conservative family-wise control | Can waste power, especially with many looks |
| Group-sequential alpha spending | Planned interim analyses | Allocates α across looks | Requires a prespecified design and boundary calculation |
| SPRT / likelihood-ratio design | Sequential decisions | Controls errors for specified hypotheses under assumptions | Alternative and sampling model must be specified |
| Always-valid inference | Potentially continuous looks | Valid under method-specific conditions | More specialized interpretation and implementation |

## Implementation

The notebook simulates a global null and compares naive repeated checks with a conservative Bonferroni threshold across a fixed set of looks. The simulation treats sequential means as cumulative observations. It demonstrates the direction of error inflation; it does not replace a production experiment’s design-specific power and calibration analysis.

```python
import math
import random

random.seed(21)

def two_sided_normal_p(z):
    return math.erfc(abs(z) / math.sqrt(2))

trials, looks, alpha = 3000, 15, 0.05
naive_reject = adjusted_reject = 0
for _ in range(trials):
    values = []
    naive = adjusted = False
    for _look in range(looks):
        values.append(random.gauss(0, 1))
        z = (sum(values) / len(values)) * math.sqrt(len(values))
        p = two_sided_normal_p(z)
        naive |= p < alpha
        adjusted |= p < alpha / looks
    naive_reject += naive
    adjusted_reject += adjusted

naive_rate = naive_reject / trials
adjusted_rate = adjusted_reject / trials
assert naive_rate > 0.08
assert adjusted_rate < 0.08
print("null rejection rates:", naive_rate, adjusted_rate)
```

For a real design, choose the sequential procedure before launch, then use an implementation validated for the outcome distribution and allocation scheme. Report both the effect estimate and uncertainty at stopping. Do not describe a continuously monitored fixed-horizon p-value as sequentially valid merely because the dashboard was convenient.

## Failure Modes

- **Stop at first p < 0.05:** Repeated unadjusted tests raise the chance of a false positive. Predefine a fixed horizon or use calibrated boundaries; test the complete stopping policy under a null simulation.
- **Change the number of looks after seeing data:** The original error allocation no longer matches the procedure. Lock look timing and document deviations; do not apply the planned correction to an unplanned search.
- **Confuse health alerts with efficacy decisions:** Service or safety monitoring can stop traffic, but it does not prove an efficacy effect. Keep operational triggers and inferential boundaries distinct.
- **Ignore delayed outcomes:** Immature cohorts can bias interim estimates or reduce apparent effects. Define maturation windows and only analyze data with comparable outcome opportunity.
- **Choose the favorable metric after peeking:** Sequential control for one primary outcome does not cover many unplanned metrics. Prespecify the family or adjust for the broader search and label exploration.
- **Assume a sequential method is assumption-free:** SPRT, alpha spending, and always-valid procedures rely on design and data-process conditions. Validate those conditions and report the method and boundaries.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| Why does peeking inflate error? | The stopping rule selects among repeated noisy looks, increasing the chance of crossing a fixed threshold by chance. | Debug a team stopping at day 10. |
| What does alpha spending do? | It allocates a cumulative Type I error budget across planned interim analyses. | Compare early and late boundaries. |
| Can operational monitoring continue during a fixed-horizon test? | Yes, for safety and system health; inferential efficacy decisions must obey the valid analysis plan. | Design separate safety and efficacy stops. |

## Related Concepts

Continue with [A/B testing](08-ab-testing-statistics.md), [power and sample size](07-statistical-power-sample-size.md), and [multiple-testing control](18-multiple-testing-fdr.md). The paired notebook is [21-sequential-experimentation-peeking-control.ipynb](../notebooks/21-sequential-experimentation-peeking-control.ipynb). Practice the peeking scenario in [Statistics interview practice](../interview-prep/statistics-interview-questions.md#12-review-a-peeking-experiment).

References: [Wald, “Sequential Tests of Statistical Hypotheses” (1945)](https://doi.org/10.1214/aoms/1177731118) and [Howard et al., “Time-uniform, nonparametric, nonasymptotic confidence sequences” (2021)](https://doi.org/10.1214/20-AOS1991).
