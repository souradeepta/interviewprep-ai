# ML/AI Evaluation and Experimentation Drills

Use 15–25 minutes per drill. Every answer must calculate or specify a metric,
make a decision, report uncertainty, and name a quality/cost/latency guardrail.
Pure reference functions are in `solutions/evaluation_exercises.py`.

## Classical prediction

1. Choose a threshold from a precision-recall and expected-cost curve.
2. Compare calibration and discrimination for two imbalanced classifiers.
3. Build a slice table with uncertainty and decide where more labels have value.
4. Construct a delayed-label evaluation cohort without treating immature rows as negatives.

## Retrieval and ranking

5. Calculate recall@K before reranking and NDCG after reranking.
6. Diagnose offline ranking gains that disappear online under position bias.
7. Compare embedding, lexical, and reranker quality with latency budgets.

## Experimentation

8. Detect sample-ratio mismatch and decide whether an A/B result is usable.
9. Explain interference, network effects, and cluster randomization.
10. Define a primary metric, guardrails, stopping rule, and confidence interval.

## LLM evaluation

11. Separate retrieval attribution, answer faithfulness, and helpfulness.
12. Calibrate an LLM judge against human labels and report agreement/bias.
13. Turn failures into a versioned regression set and release gate.

## Agent evaluation

14. Measure task success separately from tool success and language quality.
15. Estimate pass@K for sampled solutions and report cost and latency.
16. Compare agent variants when quality rises but cost or latency violates budget.

For every exercise, explain offline-to-online disagreement and which failure
reports become future regression cases. Do not claim a fixed production target
without evidence from the relevant workload.
