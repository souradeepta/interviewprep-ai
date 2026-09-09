# ML/AI Interview Practice Expansion Specification

**Date:** 2026-09-07
**Status:** Proposed implementation plan
**Scope:** ML and AI interview preparation only: ML coding and theory, LLM and
agent interviews, ML system design, data/SQL, and evaluation/experimentation.

## 1. Decision and intended outcome

Build a coherent, simulation-ready interview-preparation layer on top of the
repository's existing concept, notebook, MLOps, and system-design material.
The work must make a candidate able to practice timed ML/AI interview rounds,
not merely read a larger set of reference articles.

This is deliberately not a general LeetCode expansion, a rewrite of the
existing concept library, or a framework/API tutorial project. Existing
algorithm guides remain useful only where they support ML/AI interview
scenarios such as feature aggregation, event ordering, retrieval, ranking, or
agent state handling.

## 2. Review evidence

### Existing strengths to preserve

| Area | Current evidence | Planning implication |
|---|---|---|
| ML theory | `ml/interview-prep/ml-theory-questions.md` has 50 structured questions. | Extend with missing decision-oriented topics; do not replace it. |
| ML coding | `ml/interview-prep/ml-coding-questions.md` has 7 from-scratch exercises: KNN, attention, softmax/cross-entropy, PCA, rejection sampling, topic modeling, and ranking metrics. | Add core training/data exercises and executable reference solutions. |
| ML system design | 15 case studies, 31 patterns, and a concise framework already exist. | Convert the material into explicit timed prompts and scorecards. |
| LLM and agents | 44 LLM and 64 agent concepts include local interview Q&A; LLM has 45 implementations and agents have 58. | Add centralized interview banks and exercise paths rather than duplicating every concept's Q&A. |
| MLOps/evaluation | Three JSON question banks contain 62 prompts; MLOps concepts cover data testing, model testing, metrics, and A/B tests. | Reuse this knowledge for integrated, executable evaluation drills. |
| Recent-gap pass | The current-gap spec added prefix sums, intervals, grids, simulation, rejection sampling, topic modeling, and ranking metrics. | Treat these as complete input coverage; do not reopen or duplicate them. |

### Material gaps

1. **ML coding depth and verifiability.** Seven prompts do not cover the
   core from-scratch implementation and debugging tasks commonly used to
   assess ML engineers: linear/logistic training, split criteria, clustering,
   data splits, calibration, and gradient checking. Existing examples embed
   code in Markdown, so regressions in answer code are not independently
   tested.
2. **ML theory decision gaps.** The existing bank covers leakage, time-series
   leakage, basic precision/recall, and production degradation. It lacks
   focused simulations on calibration/thresholds and business cost, delayed or
   censored labels, selection bias and counterfactual evaluation, and feature
   freshness/point-in-time correctness.
3. **LLM and agent interview navigation.** Roadmaps promise 30+ LLM and 20+
   agentic simulation questions, but neither section contains a dedicated
   interview-practice entry point. Roadmaps instead link to general ML theory
   questions. Local concept Q&As are useful references but cannot form a
   balanced mock or a trackable plan.
4. **ML system-design mock gap.**
   `system-design/interview-prep/system-design-framework.md` offers a
   45-minute framework and two worked examples, while the roadmap refers to
   questions 1--25 that do not exist. The case studies are rich but lack a
   prompt-first practice mode, timing, scoring rubric, and deliberate
   follow-up coverage.
5. **Data/SQL exercise gap.** There are no `.sql` files or SQL interview
   exercises. This leaves MLE screening preparation incomplete despite strong
   existing material on pipelines, feature stores, validation, leakage, and
   ranking.
6. **Evaluation integration gap.** Evaluation knowledge is distributed across
   statistics, MLOps, LLM, and agent concepts. No exercise set forces a
   candidate to compute metrics, choose a time-safe holdout, diagnose
   offline/online disagreement, or define regression gates across classical
   ML, RAG, and agents.
7. **Agent implementation coverage is uneven.** Six agent concepts lack a
   paired implementation (`10`, `18`, `23`, `26`, `61`, `64`). Not all need
   implementation merely for parity; error recovery, multi-turn state, and
   real-time behavior are the ones that enable high-value, deterministic
   interview exercises.

## 3. Content contract used by every new exercise

Every new question or mock must state:

1. role and round type (MLE coding, applied ML, LLM/agent, system design, or
   data/SQL);
2. expected duration and allowed dependencies;
3. a prompt with explicit inputs, constraints, and success criteria;
4. clarifying questions expected from a strong candidate;
5. a solution outline or interviewer rubric, including complexity and
   trade-offs where relevant;
6. edge cases, failure modes, and one or more follow-up questions; and
7. links to the existing concepts/case studies for remediation.

Answer text must distinguish facts, assumptions, and example targets. Do not
present invented company-specific questions or fixed performance targets as
verified interview reports.

## 4. Prioritized implementation plan

### P0 — Establish truthful, navigable practice tracks

#### P0.1 Add a single ML interview-prep map

Modify:

- `ml/interview-prep/README.md`
- `roadmaps/ml-roadmap.md`
- `coding/README.md`

Add links and a progression table for theory, ML coding, data/SQL,
evaluation/experimentation, case studies, and behavioral practice. Correct
the roadmap's current references to coding questions 11--20 until those
questions actually exist. Link the already-added ML-relevant pattern guides
(`prefix-sums.md`, `intervals-sweep-line.md`, `matrix-grid.md`, and
`simulation-state-machines.md`) only as optional OA support; do not position
them as general-SWE requirements.

Acceptance criteria:

- A candidate can choose MLE, applied scientist, LLM/agent, or ML system
  design preparation from one page.
- All stated question ranges correspond to real numbered questions.
- No new generic DSA topic is added.

#### P0.2 Create dedicated LLM and agent interview entry points

Add:

- `llm/interview-prep/README.md`
- `llm/interview-prep/llm-interview-questions.md`
- `llm/interview-prep/llm-coding-exercises.md`
- `agentic-ai/interview-prep/README.md`
- `agentic-ai/interview-prep/agent-interview-questions.md`
- `agentic-ai/interview-prep/agent-coding-exercises.md`

Modify:

- `llm/README.md`
- `agentic-ai/README.md`
- `roadmaps/llm-roadmap.md`
- `roadmaps/agentic-roadmap.md`

`llm-interview-questions.md` must contain at least 30 uniquely numbered,
simulation-ready questions. Group them as architecture/training, retrieval and
RAG, fine-tuning/alignment, evaluation and safety, and inference/operations.
The mandatory scenarios include retrieval failure analysis, corpus/chunking
choice, embedding and reranker evaluation, LLM-as-judge limitations,
fine-tuning-versus-RAG, structured output reliability, prompt injection,
latency/cost routing, and incident response.

`agent-interview-questions.md` must contain at least 20 uniquely numbered
questions. Group them as tool contracts and state, planning/orchestration,
evaluation and observability, reliability and human oversight, and security.
The mandatory scenarios include retries and idempotency, memory boundaries,
tool permissioning, state recovery, trace-level debugging, task-success
measurement, cost/latency budgets, unsafe tool output, and multi-agent
handoffs.

The two coding-exercise documents must each provide 6--8 API-independent,
deterministic tasks. Use mock tools/responses rather than live model calls:
for example, RAG chunk-window selection, retrieval filtering, JSON/schema
validation, trace aggregation, retry/backoff policy, state checkpointing, and
tool-result validation. Each task links to an existing implementation or a
new reference module described in P2.

Acceptance criteria:

- Roadmap readiness claims (30+ LLM and 20+ agent questions) become true.
- Every new question has the content contract in section 3 and no duplicate
  prompt title within its bank.
- All exercise paths resolve and are discoverable from their section README.

#### P0.3 Make ML system-design practice real

Add:

- `system-design/interview-prep/question-bank.md`
- `system-design/interview-prep/scorecard.md`

Modify:

- `system-design/interview-prep/system-design-framework.md`
- `system-design/README.md`
- `roadmaps/system-design-roadmap.md`

`question-bank.md` must have 25 prompts, split into five tracks of five:

1. recommendation, search, ads, fraud, and forecasting;
2. feature/data platforms, training, registry, batch/online serving, and
   monitoring;
3. experimentation, delayed labels, data quality, drift, and privacy/fairness;
4. RAG, LLM serving, fine-tuning, evaluation platform, and multimodal search;
5. tool-using agent, human-in-the-loop workflow, multi-agent orchestration,
   agent safety, and cost-aware routing.

Each prompt must use the existing framework, specify time budget and scale,
link to one existing case study or pattern, and include interviewer follow-ups
that test requirements, data/labels, model choice, serving, evaluation,
failure handling, and trade-offs. `scorecard.md` defines a 0--3 rubric for
each dimension, with explicit red flags such as leakage, missing baseline,
unbounded costs, ignoring label delay, and unsafe agent autonomy.

Acceptance criteria:

- Every Q1--Q25 reference in the system-design roadmap resolves to this bank.
- Each prompt can be run as a 45--60 minute mock without reading a solution
  first.
- Existing 15 case studies remain reference material and are not rewritten.

### P1 — Fill ML coding, theory, data, and evaluation gaps

#### P1.1 Expand ML coding from seven to twenty exercises

Modify:

- `ml/interview-prep/ml-coding-questions.md`
- `roadmaps/ml-roadmap.md`

Add:

- `ml/interview-prep/solutions/linear_models.py`
- `ml/interview-prep/solutions/clustering_and_trees.py`
- `ml/interview-prep/solutions/training_and_evaluation.py`

Add Q8--Q20 in this order:

| Questions | Exercise focus | Required interview skill |
|---|---|---|
| Q8--Q9 | Linear regression and binary logistic regression with stable loss, regularization, and gradient checks | vectorized numerical implementation and optimization debugging |
| Q10 | K-means with initialization, empty-cluster handling, convergence, and reproducibility | unsupervised learning trade-offs |
| Q11 | Decision-stump/tree split selection with weighted Gini or entropy | feature thresholds, impurity, and complexity |
| Q12 | Leakage-safe temporal split and point-in-time feature construction | data correctness before modeling |
| Q13 | Probability calibration, threshold selection, and expected-cost decisioning | metrics tied to business cost |
| Q14 | Mini-batch training loop with gradient accumulation, clipping, early stopping, and seed control | training systems reasoning |
| Q15 | Approximate nearest-neighbor retrieval evaluation (recall@K, latency budget, filtered results) | retrieval implementation and measurement |
| Q16 | Stratified/bootstrap confidence interval for a metric difference | uncertainty and reproducibility |
| Q17 | Ranking candidate generation plus reranking contract | retrieval/ranking pipeline boundaries |
| Q18 | Online feature freshness and training-serving parity checker | production feature correctness |
| Q19 | Debug a deliberately faulty metric implementation | test design and failure isolation |
| Q20 | End-to-end mini take-home: baseline, evaluation, error analysis, and ship/no-ship memo | applied ML judgment |

The three modules contain importable, dependency-light reference solutions.
Use NumPy only where numerical arrays are essential; keep deterministic unit
test fixtures small. The Markdown questions explain the approach and link to
the reference module, while the modules—not copied Markdown snippets—are the
tested source of truth.

Acceptance criteria:

- Twenty unique numbered ML coding questions exist, each with inputs,
  constraints, tests, complexity, and follow-ups.
- Solutions produce deterministic results with an explicit seed.
- Numerical code uses stable softmax/sigmoid/log-loss calculations and tests
  invalid shapes, empty inputs, class imbalance, and ties where applicable.

#### P1.2 Add the missing applied-ML theory decisions

Modify:

- `ml/interview-prep/ml-theory-questions.md`
- `ml/interview-prep/README.md`

Add six questions after the existing 50, preserving all current question
numbers and adding a topical index rather than renumbering old content:

1. calibration versus discrimination and threshold choice under asymmetric
   costs;
2. delayed/censored labels and how they change training and evaluation;
3. point-in-time correctness, feature freshness, and entity joins;
4. selection bias, feedback loops, and counterfactual evaluation for ranking;
5. error slicing, uncertainty, and deciding whether more labels are worth it;
6. reproducible experiment design: data, code, seed, environment, and
   comparison baseline.

Each answer must cite and link to existing `stats/`, `mlops/`, and
`system-design/` material rather than re-explaining it in full.

Acceptance criteria:

- The theory index labels the six additions as applied/evaluation practice.
- The questions require a decision procedure, not a definition-only answer.
- The existing 50 questions retain stable anchors and wording unless a factual
  correction is independently required.

#### P1.3 Add ML-specific SQL and data-manipulation practice

Add:

- `ml/interview-prep/data-sql-questions.md`
- `ml/interview-prep/sql/01-feature-aggregations.sql`
- `ml/interview-prep/sql/02-point-in-time-joins.sql`
- `ml/interview-prep/sql/03-label-windows.sql`
- `ml/interview-prep/sql/04-ranking-and-experiment-analysis.sql`
- `ml/interview-prep/sql/fixtures.sql`

Modify:

- `ml/interview-prep/README.md`
- `roadmaps/ml-roadmap.md`

Use SQLite-compatible SQL only. The question document has at least 12
problems across feature aggregation/window functions, sessionization,
point-in-time/as-of joins, label look-forward windows, train/validation/test
splits by time/entity, duplicate and late-event handling, top-K ranking,
exposure/CTR aggregation, experiment guardrails, and metric debugging.

`fixtures.sql` provides a compact, synthetic event/user/item/label/experiment
schema with edge cases: duplicate events, missing IDs, late arrivals, changing
features, and delayed labels. Each query file has expected result tables or
assertions in comments. Do not use production-like personal data or make
unverified company claims.

Acceptance criteria:

- Every query executes against a fresh in-memory SQLite database.
- The accepted query explicitly prevents future-data leakage where the prompt
  requires it.
- The question bank explains why a superficially plausible query is wrong.

#### P1.4 Create one cross-domain evaluation drill

Add:

- `ml/interview-prep/evaluation-experiment-questions.md`
- `ml/interview-prep/solutions/evaluation_exercises.py`

Modify:

- `ml/interview-prep/README.md`
- `llm/interview-prep/llm-interview-questions.md`
- `agentic-ai/interview-prep/agent-interview-questions.md`

The document provides at least 12 exercises, grouped by classical prediction,
ranking/retrieval, experimentation, LLM evaluation, and agent evaluation.
It must require candidates to calculate/interpret calibration, PR/ROC and
cost curves, ranking metrics, confidence intervals, sample-ratio mismatch,
interference and guardrails, retrieval attribution, judge agreement,
pass@k/task success, and quality-cost-latency trade-offs. It must explicitly
cover offline-to-online disagreement and how failure reports become regression
sets.

`evaluation_exercises.py` contains small pure functions for metric
calculation, confidence intervals, slice aggregation, and regression-gate
comparison. It must not call a model API or depend on credentials.

Acceptance criteria:

- Each evaluation type has a concrete expected decision, not only a formula.
- LLM and agent exercises differentiate model quality from retrieval/tool/task
  success, and report cost and latency alongside quality.
- Existing MLOps, stats, LLM, and agent pages are linked as deeper references
  without duplicating their full prose.

### P2 — Add deterministic agent reliability exercises and evidence checks

#### P2.1 Implement only the missing agent components needed for practice

Add:

- `agentic-ai/implementations/26-error-recovery.py`
- `agentic-ai/implementations/61-multi-turn-conversation.py`
- `agentic-ai/implementations/64-real-time-agent-systems.py`

Modify:

- `agentic-ai/interview-prep/agent-coding-exercises.md`

The implementations must be local simulations with injectable clock, mock
tool, and fake model response. Cover bounded retry/backoff, idempotency key
handling, checkpoint/restore, timeout/cancellation, state-machine invariants,
and cost/latency accounting. They are exercises and reference solutions, not
production framework abstractions.

Do not create implementations for concepts `10`, `18`, and `23` solely to
make counts match. Reassess them after the practice paths ship based on an
explicit interview-value case.

Acceptance criteria:

- No network, API key, sleep, or wall-clock dependence in tests.
- A failing tool, duplicate event, stale checkpoint, and timeout each have a
  deterministic expected outcome.
- Agent exercise links point to a runnable reference implementation.

#### P2.2 Add durable coverage validation

Add:

- `tests/test_interview_prep_coverage.py`
- `tests/test_ml_interview_solutions.py`
- `tests/test_ml_sql_exercises.py`
- `tests/test_agent_interview_exercises.py`

`test_interview_prep_coverage.py` checks document presence, question-number
uniqueness, promised minimum counts, content-contract headings, roadmap links,
and valid local links. It must avoid brittle word-count assertions.

`test_ml_interview_solutions.py` runs algorithmic fixtures: finite-difference
gradient checks, deterministic K-means behavior, split/feature leakage
rejection, calibration/expected-cost output, ranking metrics, and error cases.

`test_ml_sql_exercises.py` loads `fixtures.sql` into `sqlite3`, executes all
four query files, and compares named result sets to expected rows. It must
start from a new database per test so one query cannot affect another.

`test_agent_interview_exercises.py` checks retry bounds, idempotency,
checkpoint recovery, timeout handling, and budget enforcement using fake
dependencies.

Also modify `tests/test_repository_integrity.py` only if its active link audit
needs the new directory added to a currently explicit scope; retain its
existing smoke-test behavior.

Acceptance criteria:

- New tests are deterministic, offline, and complete in under 10 seconds on a
  typical developer machine, excluding existing opt-in notebook execution.
- A deliberately changed expected result, duplicate question number, broken
  roadmap link, future-data join, or retry regression makes the relevant test
  fail.

## 5. Implementation sequencing

1. Land P0.1--P0.3 as documentation/navigation only, then run link auditing.
   This makes the practice contract visible before adding content at scale.
2. Build P1.1 and its tests first. It is the shared substrate for ML coding,
   data correctness, retrieval, and evaluation.
3. Add P1.2--P1.4, using the existing MLOps/statistics/LLM/agent references
   instead of copying their explanatory material.
4. Add P2.1 only after the agent exercise prompts identify exactly which
   behavior needs reference code. Land P2.2 with each content group rather
   than as a late bulk test pass.
5. After each phase: run the targeted tests, the full pytest suite, and
   `python3 scripts/audit_markdown_links.py --root .`. Review the explicit file
   list before committing. Preserve unrelated dirty-worktree files and never
   use broad staging or cleanup commands.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Duplicate content creates a larger but less usable repository. | New banks are prompt-first and link to the existing concept answer as remediation. Maintain a topic-to-source index in each README. |
| Numbers promised by roadmaps drift again. | Make document-count/link assertions part of `test_interview_prep_coverage.py`. |
| Answer code teaches unstable or incorrect numerical behavior. | Test finite differences, numerical stability, seeded outputs, shape validation, and adversarial input fixtures. |
| SQL answers leak future information or depend on a vendor dialect. | Use SQLite fixtures, state the cutoff timestamp, and assert point-in-time behavior in tests. |
| Live model examples make the suite flaky or require credentials. | Use injected fakes and static fixtures only; live API evaluation is out of scope. |
| The scope expands back into generic software-interview content. | Accept a new DSA item only if it is explicitly tied to an ML/AI data, retrieval, ranking, scheduling, or agent workflow scenario. |
| Existing uncommitted work is overwritten or accidentally staged. | Modify only the files named in this plan, inspect status before each phase, and stage explicit paths through the documented writable-clone workflow. |

## 7. Definition of done

- ML interview prep has a complete, navigable theory/coding/data/evaluation/
  case-study/behavioral path.
- ML coding has 20 tested prompts and deterministic reference solutions for
  newly added algorithmic exercises.
- LLM has at least 30 and agents at least 20 simulation-ready questions, each
  with a discoverable entry point and no live API dependency.
- ML system design has 25 timed prompts and a reusable scoring rubric; roadmap
  references are accurate.
- SQL exercises execute under SQLite and explicitly test leakage-safe logic.
- Evaluation exercises span classical ML, retrieval/ranking, experimentation,
  LLMs, and agents with quality, cost, and latency considerations.
- Targeted tests, full pytest, and the active Markdown link audit are clean.
- No unrelated source, historical document, worktree, or prior-run artifact is
  deleted or silently rewritten.

## Implementation status — 2026-09-07

The expansion was implemented and then reviewed for correctness. The follow-up
repair is specified in
[2026-09-07-interview-practice-correctness-repair.md](2026-09-07-interview-practice-correctness-repair.md).
That repair preserves the original questions while adding behavioral tests,
safe NDCG/agent/SQL contracts, and independently runnable simulation cards.
