import json
from pathlib import Path


LESSONS = {
    16: {
        "slug": "bootstrap-resampling", "title": "Bootstrap and Resampling",
        "prereq": "Probability fundamentals and sampling distributions", "minutes": 35,
        "outcomes": "Construct a nonparametric bootstrap interval; choose the resampling unit; identify settings where bootstrap coverage is unreliable.",
        "recap": "The bootstrap approximates an estimator's sampling distribution by sampling observed independent units with replacement. The sample must represent the target population, and dependence must be respected by resampling clusters or blocks.",
        "code": """import random\nrandom.seed(16022)\n\n# Positive, right-skewed observations: a normal interval can be implausible for a median.\nsample = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.4, 2.0, 5.8, 9.1]\ndef median(values):\n    values = sorted(values); n = len(values)\n    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2\ncenter = median(sample)\n# Naive shortcut: use a normal-theory SE formula for a median.\nnaive_se = (max(sample) - min(sample)) / (2 * len(sample) ** 0.5)\nnaive_interval = (center - 1.96 * naive_se, center + 1.96 * naive_se)\n# Corrected approach for this lesson: resample independent observations and recompute median.\nboot = [median([random.choice(sample) for _ in sample]) for _ in range(3000)]\nboot.sort()\nbootstrap_interval = (boot[74], boot[2924])\nprint('median', center, 'naive', naive_interval, 'bootstrap percentile', bootstrap_interval)\nassert bootstrap_interval[0] >= min(sample) and bootstrap_interval[1] <= max(sample)\nassert len(boot) == 3000""",
        "metric": "Compare interval endpoints and width; inspect the bootstrap estimator distribution rather than assuming it is symmetric. A percentile interval is pedagogical here, not universally optimal.",
        "edge": "Resampling one observation produces a degenerate interval. Repeated identical values also make empirical resamples uninformative; dependent observations require cluster/block resampling.",
        "exercise": "Change the estimator to a 90th percentile and use the same resamples. Compare percentile and a normal interval; test singleton and tied samples. State the independent-unit assumption.",
        "summary": "Resampling approximates repeated sampling under an empirical distribution. Choosing the right unit and diagnosing small-sample/tail behavior matter more than increasing replicate count.",
    },
    17: {
        "slug": "permutation-tests-randomization-inference", "title": "Permutation Tests and Randomization Inference",
        "prereq": "Hypothesis testing and exchangeability", "minutes": 35,
        "outcomes": "Construct a randomization test under a sharp null; preserve the experimental assignment design; distinguish exact enumeration from Monte Carlo approximation.",
        "recap": "A randomization test compares the observed statistic with assignments allowed by the design under a null hypothesis. Permutations are valid only when units are exchangeable under that null; matched and clustered designs require restricted shuffles.",
        "code": """import random\nrandom.seed(17022)\n\n# Matched before/after measurements; pair identity is part of the design.\nbefore = [10, 12, 9, 15, 11, 14]\ndelta = [1, 2, -1, 0, 3, 1]\nafter = [b + d for b, d in zip(before, delta)]\nobserved = sum(delta) / len(delta)\n# Naive: shuffle labels across all 12 rows, destroying pair membership.\npooled = before + after\nnaive_stats = []\nfor _ in range(4000):\n    random.shuffle(pooled)\n    naive_stats.append(sum(pooled[6:]) / 6 - sum(pooled[:6]) / 6)\n# Design-respecting null: flip the sign of each paired difference.\npaired_stats = []\nfor mask in range(1 << len(delta)):\n    signed = [d if mask & (1 << i) else -d for i, d in enumerate(delta)]\n    paired_stats.append(sum(signed) / len(signed))\np_paired = sum(abs(x) >= abs(observed) for x in paired_stats) / len(paired_stats)\np_naive = (1 + sum(abs(x) >= abs(observed) for x in naive_stats)) / (len(naive_stats) + 1)\nprint('observed paired mean change', observed, 'naive p', p_naive, 'paired randomization p', p_paired)\nassert len(paired_stats) == 64\nassert p_paired >= 1 / 64""",
        "metric": "Report the observed effect and its randomization p-value; Monte Carlo p-values use a plus-one correction. Randomization inference tests a sharp no-effect null under the actual assignment mechanism.",
        "edge": "A treatment assigned by cluster cannot be shuffled by individual. Highly unbalanced groups and constrained/block randomization also require honoring assignment restrictions.",
        "exercise": "Enumerate sign flips for 8 pairs and compare with the invalid pooled-label shuffle. Add a block label and shuffle only within blocks; assert every simulated assignment preserves block counts.",
        "summary": "The assignment mechanism determines the valid reference distribution. An unrestricted shuffle can answer the wrong question even when the arithmetic is correct.",
    },
    18: {
        "slug": "multiple-testing-fdr", "title": "Multiple Testing and False-Discovery Control",
        "prereq": "Hypothesis testing and p-value interpretation", "minutes": 35,
        "outcomes": "Define a testing family; distinguish FWER from FDR; apply Benjamini–Hochberg to ordered p-values and interpret its assumptions.",
        "recap": "Testing many null hypotheses creates opportunities for chance discoveries. Family-wise error controls the probability of any false rejection; false-discovery rate controls the expected false fraction among rejections under stated dependence conditions.",
        "code": """import random\nrandom.seed(18022)\n\ndef benjamini_hochberg(p_values, q=0.05):\n    ordered = sorted(enumerate(p_values), key=lambda pair: pair[1])\n    m = len(ordered); cutoff = -1\n    for rank, (_, p) in enumerate(ordered, 1):\n        if p <= q * rank / m: cutoff = rank\n    rejected = {idx for idx, _ in ordered[:cutoff]} if cutoff > 0 else set()\n    return rejected\n# Synthetic null p-values plus three deliberately small alternatives.\np = [random.random() for _ in range(40)] + [0.0004, 0.001, 0.003]\nraw = {i for i, value in enumerate(p) if value < 0.05}\nadjusted = benjamini_hochberg(p, q=0.05)\nprint('raw discoveries', len(raw), 'BH discoveries', sorted(adjusted))\nassert len(raw) >= len(adjusted)\nassert {40, 41, 42} <= adjusted""",
        "metric": "Report the number tested, prespecified family, raw and adjusted discoveries, effect sizes, and uncertainty. BH is a step-up procedure that controls FDR under independence and certain positive dependence structures.",
        "edge": "If many hypotheses are correlated in arbitrary ways, ordinary BH guarantees may not apply. Exploratory selection after seeing results also changes the effective family.",
        "exercise": "Implement Holm's step-down family-wise procedure and compare it with BH on the same p-values. Test empty input, all-large p-values, and tied values; do not infer effect size from adjusted significance.",
        "summary": "Multiplicity correction aligns the error criterion with the decision. It does not repair biased data, p-hacking, or an ill-defined family of claims.",
    },
    19: {
        "slug": "missing-data-imputation-assumptions", "title": "Missing Data and Imputation Assumptions",
        "prereq": "Estimation, conditional probability, and regression", "minutes": 40,
        "outcomes": "Distinguish MCAR, MAR, and MNAR; diagnose observed missingness patterns; select an estimator under explicit assumptions and perform sensitivity analysis.",
        "recap": "Missingness mechanisms describe what the probability of being missing depends on. MCAR is independent of data, MAR is explainable by observed variables conditional on them, and MNAR depends on unobserved values even after conditioning.",
        "code": """import random\nrandom.seed(19022)\n\n# MAR example: response probability depends on observed group, whose outcome means differ.\nrows = []\nfor group, mean, response_rate in [('A', 10.0, 0.9), ('B', 20.0, 0.3)]:\n    for _ in range(1000):\n        y = mean + random.gauss(0, 2)\n        observed = random.random() < response_rate\n        rows.append((group, y if observed else None, response_rate))\nfull_mean = sum((10.0 if g == 'A' else 20.0) for g, _, _ in rows) / len(rows)\ncomplete = [y for _, y, _ in rows if y is not None]\ncomplete_mean = sum(complete) / len(complete)\n# Under the known MAR response model, inverse-probability weighting recovers the target mean.\nweighted_sum = sum(y / rate for _, y, rate in rows if y is not None)\nweighted_n = sum(1 / rate for _, y, rate in rows if y is not None)\nipw_mean = weighted_sum / weighted_n\nprint('true target', full_mean, 'complete case', complete_mean, 'IPW', ipw_mean)\nassert abs(ipw_mean - full_mean) < abs(complete_mean - full_mean)\nassert len(complete) < len(rows)""",
        "metric": "Compare complete-case and weighted estimates to the known synthetic target. In real data the target is unknown; report missingness by observed covariates and uncertainty from the imputation/weighting procedure.",
        "edge": "If a response probability is near zero, inverse weights explode and positivity fails. If missingness depends on the missing outcome, this MAR correction is not identified without extra assumptions.",
        "exercise": "Repeat with response depending on the unobserved outcome sign (MNAR). Compare complete-case, mean-imputed, and IPW behavior; explain why no observed-data method identifies the truth without assumptions.",
        "summary": "Imputation is an inferential choice tied to a missingness mechanism. Preserve uncertainty and test sensitivity to assumptions instead of filling blanks and proceeding as if observed.",
    },
    20: {
        "slug": "regression-diagnostics-robust-standard-errors", "title": "Regression Diagnostics and Robust Standard Errors",
        "prereq": "Linear regression, residuals, and sampling variance", "minutes": 40,
        "outcomes": "Use residual diagnostics to identify model problems; compute a heteroskedasticity-consistent slope standard error; distinguish robust inference from model repair.",
        "recap": "OLS estimates a conditional mean using a linear projection. Homoskedastic standard errors assume constant residual variance; heteroskedasticity-consistent covariance estimators relax that variance assumption asymptotically, but do not fix a wrong mean function or dependence.",
        "code": """import random, math\nrandom.seed(20022)\n\nx = [i / 10 for i in range(1, 101)]\ny = [2 + 3 * value + random.gauss(0, 0.15 + 0.12 * value) for value in x]\nn = len(x); xbar = sum(x) / n; ybar = sum(y) / n\nsxx = sum((v - xbar) ** 2 for v in x)\nslope = sum((a - xbar) * (b - ybar) for a, b in zip(x, y)) / sxx\nintercept = ybar - slope * xbar\nresid = [b - (intercept + slope * a) for a, b in zip(x, y)]\nsse = sum(e * e for e in resid); sigma2 = sse / (n - 2)\n# Homoskedastic textbook SE versus HC0 sandwich slope SE.\nnaive_se = math.sqrt(sigma2 / sxx)\nhc0_se = math.sqrt(sum(((a - xbar) * e) ** 2 for a, e in zip(x, resid)) / (sxx ** 2))\nlow_var = sum(e*e for a,e in zip(x,resid) if a < 5) / sum(a < 5 for a in x)\nhigh_var = sum(e*e for a,e in zip(x,resid) if a >= 5) / sum(a >= 5 for a in x)\nprint('slope', slope, 'naive SE', naive_se, 'HC0 SE', hc0_se, 'residual variance low/high', low_var, high_var)\nassert high_var > low_var\nassert naive_se > 0 and hc0_se > 0""",
        "metric": "Inspect residual-versus-fitted and scale-location plots, leverage/influence, and uncertainty by the dependence unit. Robust standard errors change estimated uncertainty, not OLS coefficients or predictions.",
        "edge": "Small samples, high leverage, serial dependence, or clustered errors can invalidate simple HC0 inference. Use finite-sample corrections, HAC, or cluster-robust methods appropriate to the design.",
        "exercise": "Add a quadratic mean term and compare residual structure before/after; then create serially correlated noise. Explain why HC0 does not fix either omitted nonlinearity or autocorrelation.",
        "summary": "Diagnostics first identify which assumption failed. A robust covariance estimator addresses heteroskedastic uncertainty under conditions; it is not a universal repair for regression misspecification.",
    },
    21: {
        "slug": "sequential-experimentation-peeking-control", "title": "Sequential Experimentation and Peeking Control",
        "prereq": "A/B testing and multiple testing", "minutes": 40,
        "outcomes": "Explain how repeated fixed-horizon tests inflate false positives; implement a valid simple multiple-look correction; distinguish planned sequential designs from ad hoc stopping.",
        "recap": "A fixed-horizon test calibrates its error rate at one planned analysis. Rechecking and stopping at the first significant result changes the procedure. Valid sequential methods use alpha spending, likelihood-ratio boundaries, or always-valid inference designed for optional stopping.",
        "code": """import random, math\nrandom.seed(21022)\n\ndef two_sided_normal_p(z):\n    return math.erfc(abs(z) / math.sqrt(2))\nnaive_wins = adjusted_wins = 0\ntrials, looks, alpha = 3000, 15, 0.05\nfor _ in range(trials):\n    observations = []\n    naive = adjusted = False\n    for look in range(1, looks + 1):\n        observations.append(random.gauss(0, 1))  # true null: zero mean\n        mean = sum(observations) / len(observations)\n        se = 1 / math.sqrt(len(observations))\n        p = two_sided_normal_p(mean / se)\n        naive |= p < alpha\n        # Bonferroni across the 15 planned looks: conservative but valid.\n        adjusted |= p < alpha / looks\n    naive_wins += naive; adjusted_wins += adjusted\nnaive_rate = naive_wins / trials; adjusted_rate = adjusted_wins / trials\nprint('false positive rate, naive looks', naive_rate, 'Bonferroni looks', adjusted_rate)\nassert naive_rate > 0.08\nassert adjusted_rate < 0.08""",
        "metric": "Estimate empirical false-positive rate across repeated null experiments and compare it with the nominal alpha. Bonferroni protects a fixed family of looks but can be conservative; designed sequential procedures can use information more efficiently.",
        "edge": "Changing the number or timing of looks after observing outcomes invalidates the planned correction. Outcome delays and repeated metric selection also create unplanned looks.",
        "exercise": "Implement an alpha-spending schedule with cumulative spend never exceeding 0.05. Compare stopping time and type-I error in simulation; explain why the same fixed-horizon p-value cannot be reused after arbitrary peeking.",
        "summary": "Monitoring may be continuous, but decision boundaries must match the monitoring plan. A valid sequential design predefines how evidence accumulates and when to stop.",
    },
}


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


for number, lesson in LESSONS.items():
    cells = [
        md(f"# {number:02d}. {lesson['title']}\n\n**Prerequisites:** {lesson['prereq']}. **Estimated time:** {lesson['minutes']} minutes.\n\n**Objectives:** {lesson['outcomes']}"),
        md(f"## Concept recap and assumptions\n\n{lesson['recap']} This notebook uses only the Python standard library and synthetic data."),
        md("## Naive baseline and corrected method\n\nRun the deterministic comparison below. It prints both approaches and asserts key invariants."),
        code(lesson['code']),
        md("## Comparison and interpretation\n\n" + lesson['metric']),
        md("## Edge and failure cases\n\n" + lesson['edge']),
        md("## Exercises\n\n" + lesson['exercise']),
        md("## Summary\n\n" + lesson['summary']),
        md("## Next lessons and linked practice\n\nContinue with the adjacent Statistics lessons and use [Statistics interview practice](../interview-prep/statistics-interview-questions.md) to explain assumptions, diagnostics, and design choices."),
    ]
    doc = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.x"}}, "nbformat": 4, "nbformat_minor": 5}
    path = Path("stats/notebooks") / f"{number:02d}-{lesson['slug']}.ipynb"
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
