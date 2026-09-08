# LLM Coding Exercises

All tasks are API-free and deterministic. Use 20 minutes per exercise, write
clarifying questions, state complexity, test edge cases, and link the result
to the named implementation concept.

1. **Chunk-window selection:** select a bounded window of adjacent chunks around
   the highest-scoring evidence while preserving document boundaries and stable ties.
2. **Filtered retrieval:** filter candidate IDs by tenant, freshness, and ACL,
   then return deterministic top-K results and a recall report.
3. **Schema validation:** validate nested JSON-like dictionaries, report all
   missing/wrong-type fields, and reject unknown fields in strict mode.
4. **Citation coverage:** map answer claims to retrieved chunk IDs and compute
   unsupported-claim rate without calling a model.
5. **Token-budget routing:** choose a model from quality, latency, and cost
   budgets, with an explicit unavailable outcome when no route is safe.
6. **Regression-set comparison:** compare baseline and candidate scores by slice,
   reject quality regressions, and report cost/latency guardrails.

Reference primitives for retrieval and evaluation are in
`../../ml/interview-prep/solutions/evaluation_exercises.py`; live model calls
are out of scope.
