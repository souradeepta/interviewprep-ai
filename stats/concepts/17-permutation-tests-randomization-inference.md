# Permutation Tests and Randomization Inference

TL;DR: Randomization inference builds a null distribution by reassigning treatment according to the experiment’s actual assignment design, making exchangeability the key validity condition.

## Prerequisites

Review [hypothesis testing](05-hypothesis-testing.md), [A/B testing](08-ab-testing-statistics.md), and [causal inference](09-causal-inference.md).

## Learning Outcomes

- State the null hypothesis and statistic for a randomization test.
- Generate only assignments allowed by the experiment’s randomization procedure.
- Compute an exact or Monte Carlo permutation p-value with appropriate tail handling.
- Explain why unrestricted row shuffling is invalid for paired, blocked, clustered, or networked designs.

## Core Intuition

In a randomized experiment, chance assignment creates a known mechanism for who receives treatment. Under a sharp null that treatment changes no unit’s outcome, the observed outcomes would remain fixed while allowed treatment labels vary. Replaying those possible assignments produces the reference distribution of the test statistic. The question is not “what data could we have seen under any shuffle?” but “what assignments could this design have produced under the null?”

Consider measuring a new onboarding flow on the same users before and after a change. Each user’s two measurements form a pair. A valid sharp-null randomization may flip which observation is called treated within each pair, preserving the pairing. Shuffling all before and after values across users discards that design and inflates or distorts the reference distribution. The arithmetic can be flawless while the inferential question is wrong.

Permutation tests can also be used outside randomized trials when a null makes observations exchangeable, but this assumption must be justified. A test based on exchangeability under equal distributions is not automatically a test of equal means when distribution shapes differ. A randomization test under an assignment mechanism often has a clean finite-sample interpretation for a sharp null; it does not by itself estimate heterogeneous treatment effects or prove a particular average-effect null.

## How It Works

Choose a statistic (T(Y,Z)), such as a mean difference between treated and control outcomes. Under the null, generate a set of allowed assignments (Z^*) according to the design and calculate (T(Y,Z^*)). The two-sided exact p-value is the fraction of allowed assignments whose statistic is at least as extreme as observed, commonly (|T^*| \ge |T_{obs}|). When assignments are sampled Monte Carlo rather than exhaustively enumerated, a standard plus-one estimate is ((1 + \#\{|T_b^*| \ge |T_{obs}|\})/(B+1)); this avoids a zero p-value.

For complete randomization with fixed treatment count, shuffle labels while preserving the number treated. For blocked assignment, permute within each block. For paired designs, flip treatment within pairs or sign-flip paired differences under the relevant sharp null. For cluster randomization, reassign clusters rather than individuals. Restricted randomization must preserve the original allocation probabilities and constraints.

Enumerating all assignments can be expensive: with n units and k treated, there are ({n \choose k}) possible assignments. Monte Carlo permutation costs roughly O(Bn) statistic work, where B is the number of sampled assignments. Efficient code can update statistics incrementally, but should first establish a transparent correct baseline and verify it against exact enumeration on a tiny example.

## Trade-offs

| Procedure | Assignment structure | Strength | Limitation |
|---|---|---|---|
| Exact randomization test | Known finite set of allowed assignments | Finite-sample calibration for a sharp null | Number of assignments may be enormous |
| Monte Carlo randomization | Sampled assignments from the design | Scales to larger experiments | Adds simulation error; resolution depends on B |
| Unrestricted permutation | Exchangeable observations under a suitable null | Simple for genuinely exchangeable units | Invalid under pairing, clustering, blocking, or unequal distributions for some nulls |
| Sign-flip paired test | Matched pairs with exchangeable treatment assignment | Preserves within-pair structure | Null and assignment symmetry must be appropriate |
| Parametric test | Model-based sampling assumptions | Efficient and familiar | Calibration depends on distribution/model assumptions |

## Implementation

This example exactly enumerates paired sign assignments for the mean change. It contrasts the design-respecting test with a deliberately naive unrestricted label shuffle. The synthetic numbers are for illustrating the reference-distribution difference, not for making a substantive treatment claim.

```python
import itertools

before = [10, 12, 9, 15, 11, 14]
after = [11, 14, 8, 15, 14, 15]
deltas = [a - b for a, b in zip(after, before)]
observed = sum(deltas) / len(deltas)
null_stats = [
    sum(sign * delta for sign, delta in zip(signs, deltas)) / len(deltas)
    for signs in itertools.product((-1, 1), repeat=len(deltas))
]
p_value = sum(abs(value) >= abs(observed) for value in null_stats) / len(null_stats)
assert len(null_stats) == 2 ** len(deltas)
assert 0 < p_value <= 1
print("observed mean change:", observed, "paired randomization p:", p_value)
```

In a randomized between-user test, preserve the observed treatment count in every shuffle. In a blocked test, retain the block membership and the treatment count within each block. If the randomization used unequal assignment probabilities, simulate from those probabilities rather than drawing uniform permutations. Always report the test statistic and its units alongside the p-value so the result remains interpretable.

## Failure Modes

- **Shuffling the wrong units:** Individual shuffles break clusters or pairs. Inspect assignment logs and reproduce the actual unit and constraints; a randomization test should mirror the implementation.
- **Testing the wrong null:** A sharp null of no effect for every unit differs from a null of zero average treatment effect. State which null supports the reference distribution and use methods appropriate to the estimand.
- **Ignoring blocks or stratification:** Unrestricted assignment changes the design’s balance properties. Reassign within blocks and verify invariant counts in every simulated draw.
- **Using a one-sided statistic after seeing direction:** Choosing the tail from the observed result inflates error. Prespecify direction or use a two-sided statistic.
- **Zero Monte Carlo p-value:** No sampled permutation being more extreme does not imply probability zero. Use the plus-one estimator and report B and its attainable resolution.
- **Confusing statistical significance with effect size:** The test evaluates compatibility under the null, not whether the effect is important. Report the observed difference and uncertainty or a separate interval procedure.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| What makes a randomization test valid? | The reassignment distribution must match the experiment design under a stated null. | Debug a blocked trial analyzed with unrestricted shuffling. |
| When would you use exact enumeration? | When the allowed assignment set is small enough to enumerate all outcomes. | How does Monte Carlo change the p-value calculation? |
| What is the paired randomization unit? | The pair; treatment labels or signs vary within each matched pair. | What if there are repeated observations per subject? |

## Related Concepts

Continue with [A/B testing](08-ab-testing-statistics.md), [causal inference](09-causal-inference.md), and [bootstrap resampling](16-bootstrap-resampling.md). The paired notebook is [17-permutation-tests-randomization-inference.ipynb](../notebooks/17-permutation-tests-randomization-inference.ipynb). See [Statistics interview practice](../interview-prep/statistics-interview-questions.md#10-diagnose-confounding-in-an-observational-launch) for design assumptions.

References: [Fisher, *The Design of Experiments* (1935)](https://archive.org/details/designofexperime00fish) and [Imbens and Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/04EEA5D0A7C1D3B2E70D320D6BEA11FA).
