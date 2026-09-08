# ML/AI Interview Practice Correctness Repair Specification

**Date:** 2026-09-07
**Status:** Proposed repair plan
**Scope:** Correctness, safety, and simulation readiness of the ML/AI interview
practice expansion reviewed at commit `682f8c8`. This is restricted to ML,
LLM, agentic AI, ML system design, data/SQL, evaluation, and their tests.

## 1. Decision and non-goals

Repair the current practice material rather than replace it. The expansion
introduced useful coverage and navigation; this work makes the exercises
technically correct, testable, and usable as interview simulations.

Preserve all existing questions, solution modules, historical plans, and
specifications unless a statement is factually inaccurate. Do not broaden the
repository into generic software-engineering or general LeetCode practice. Do
not add live-model calls, production credentials, external services, or
non-deterministic tests.

## 2. Review findings and repair principles

| Finding | Why it matters for interview prep | Repair principle |
|---|---|---|
| `ndcg_at_k` ranks the already truncated list when computing ideal DCG. | It can certify an obviously bad ranking as perfect. | Compute DCG from the displayed top `k`, but compute ideal DCG from all supplied candidate relevances, limited to `k`. |
| Agent examples promise budgets, idempotency, recovery, and cancellation that their code does not guarantee. | Candidates can learn unsafe patterns that are especially harmful for tool-using agents. | State the delivery semantics explicitly, reject invalid state/accounting inputs, and model durable progress, cancellation, and recovery deterministically. |
| SQL labels and experiment fixtures blur observation time, attribution, and randomization. | These are core MLE leakage and causal-inference interview concepts. | Model prediction identity, label availability, attribution windows, and one randomized arm per experiment assignment explicitly. |
| Regression tests mostly count headings or execute SQL. | Tests pass while the promised learning contract is false. | Test observable semantics, malformed inputs, counterexamples, expected SQL rows, and exercise-document contracts. |
| ML coding questions 1–7 are unnumbered and later questions overpromise their implementations. | Roadmap references become ambiguous and learners cannot tell what is executable. | Give every exercise a stable number; make stated capabilities match tested reference functions. |
| LLM notebook count and K-means status are stale. | Navigation and scope claims lose trust. | Derive/document current counts accurately and remove only the stale claim, not history. |
| Interview banks are topic indexes with global instructions. | They are not independently runnable mocks. | Add a compact, repeatable per-prompt simulation card with constraints, rubric, follow-up, and failure modes. |

### Content quality contract

Every modified or newly structured practice question must distinguish:

1. the candidate prompt and assumptions;
2. success criteria and evaluation measures;
3. expected clarifying questions or constraints;
4. edge cases/failure modes;
5. interviewer follow-up and scoring guidance; and
6. remediation links, which must remain reference material rather than a
   hidden solution.

## 3. Prioritized delivery plan

### P0 — Correct executable reference behavior and lock it with tests

#### P0.1 Fix ranking evaluation semantics

Modify:

- `ml/interview-prep/solutions/evaluation_exercises.py`
- `tests/test_interview_prep_expansion.py`
- `ml/interview-prep/ml-coding-questions.md`
- `ml/interview-prep/evaluation-experiment-questions.md` (only if it states
  the old, incorrect NDCG contract)

Implementation requirements:

- Define `ndcg_at_k(relevances, k)` as graded relevance in ranked order.
  Reject non-finite relevance values and non-integral/non-positive `k` as
  appropriate for the module’s public API; retain `0.0` for an empty list or
  a valid `k` with no gain.
- Calculate displayed DCG from `relevances[:k]`; calculate ideal DCG from
  `sorted(relevances, reverse=True)[:k]`. Do not sort a truncated input when
  calculating the denominator.
- Document that duplicate candidate IDs must be removed before constructing
  the relevance vector; NDCG itself does not infer candidate identity.
- Make the Q17/Q19 wording explicitly tell candidates to test the
  "high-relevance result outside the top-k" counterexample.

Acceptance criteria:

- `ndcg_at_k([1, 0, 3], 2)` is approximately `0.131` (not `1.0`).
- A perfect ordering returns `1.0`; all-zero relevance returns `0.0`; a
  ranking with relevant items outside `k` is bounded below `1.0` when
  appropriate.
- Invalid numeric inputs fail loudly rather than returning NaN/inf.

Required tests:

- Unit cases for graded relevance, binary relevance, `k > n`, `k <= 0`, empty
  input, all-zero relevance, ties, and the regression counterexample above.
- A property-style assertion that finite valid inputs yield a finite result in
  `[0, 1]`.

#### P0.2 Make agent safety claims true

Modify:

- `agentic-ai/implementations/26-error-recovery.py`
- `agentic-ai/implementations/61-multi-turn-conversation.py`
- `agentic-ai/implementations/64-real-time-agent-systems.py`
- `agentic-ai/interview-prep/agent-coding-exercises.md`
- `agentic-ai/interview-prep/agent-interview-questions.md`
- `tests/test_interview_prep_expansion.py`

Add:

- `tests/test_agent_interview_exercises.py`

`26-error-recovery.py` requirements:

- Keep retry scheduling side-effect free and validate `transient` is a tuple
  of exception classes.
- Replace the false generic "at most once" claim with an explicit durable
  idempotency state machine: `pending`, `completed`, and `unknown`/`requires
  reconciliation` for an operation that may have performed its side effect
  before raising.
- Accept a caller-supplied reconciliation function or result lookup for an
  uncertain key. Never automatically rerun an operation in the uncertain
  state.
- Explain exactly-once is not guaranteed across an arbitrary external side
  effect without downstream idempotency or a transactional outbox.

`61-multi-turn-conversation.py` requirements:

- Reject boolean, zero, negative, and non-integer `max_messages`; protect the
  invariant in construction and restore.
- Validate each checkpoint message has an allowed role and string content;
  require `task_state` to be a dictionary; deep-copy all restored values.
- Include a schema/version field, completed action IDs, and checkpoint time
  supplied by an injected clock. Reject stale checkpoints using an explicit
  `max_checkpoint_age` argument instead of silently restoring them.
- Keep bounded conversation messages separate from durable task progress so
  truncating message history cannot erase completed-side-effect records.

`64-real-time-agent-systems.py` requirements:

- Validate all limits and charges are finite, non-negative numeric values;
  reject booleans and NaNs/infinities.
- Make a rejected charge atomic: no counter may change if any dimension
  exceeds its budget.
- Add a cancellation/deadline path with deterministic injected time or
  pre-measured step data. Return a structured partial result containing
  completed actions, skipped/cancelled action, reason, and cleanup actions.
- Validate each `run_steps` entry before charging, including required name,
  elapsed time, tokens, and cost.

Documentation requirements:

- Map each of the six agent coding exercises to a concrete reference function
  or class. Where a task spans modules, name each primitive.
- State delivery semantics and recovery limitations in the exercise text; do
  not promise transparent recovery from unknown external side effects.
- Add per-prompt success criteria, boundary conditions, and a recovery or
  safety follow-up as described in P2.

Acceptance criteria:

- Invalid message limits and invalid budget charges raise `ValueError`.
- A negative token/time/cost charge cannot increase remaining budget.
- A post-side-effect exception produces `requires_reconciliation`; a second
  call with the same key cannot duplicate the effect until reconciliation
  resolves it.
- Restoring a stale/malformed checkpoint is rejected. A valid checkpoint
  preserves durable completed action IDs even after message trimming.
- Cancellation returns a bounded partial result and leaves counters
  internally consistent.

Required tests:

- Retry success, retry exhaustion, unsupported exception propagation, and
  uncertain-side-effect reconciliation.
- Zero/negative/non-integer message limits; malformed role/content/task state;
  deep-copy isolation; stale checkpoint; trimmed history with retained durable
  progress.
- Negative, NaN, infinite, and over-limit budget values; atomic failure;
  deadline cancellation; valid exact-limit charge; malformed steps.

#### P0.3 Repair SQL attribution and experimental-unit semantics

Modify:

- `ml/interview-prep/sql/fixtures.sql`
- `ml/interview-prep/sql/03-label-windows.sql`
- `ml/interview-prep/sql/04-ranking-and-experiment-analysis.sql`
- `ml/interview-prep/data-sql-questions.md`
- `tests/test_interview_prep_expansion.py`

Add:

- `tests/test_ml_sql_exercises.py`

Fixture model requirements:

- Give `predictions` a prediction/decision identifier and define one explicit
  outcome-attribution key (for example prediction ID or exposure ID) in
  `labels`; do not attribute every user-level conversion to every prediction.
- Include `observed_at`/`available_at` separately from event/label occurrence
  time so each query can enforce a named as-of cutoff.
- Add a clearly documented prediction-time, label-maturity window, and
  evaluation cutoff parameter/CTE. A label that happened in-window but was
  not available at the cutoff must remain unknown.
- Separate experiment assignment from exposures. Each user must have exactly
  one `variant` for an experiment ID; multiple exposure rows are allowed only
  beneath that assignment. Remove the contradictory cross-arm `u1` fixture by
  replacing it with valid repeated exposures or a separate experiment.
- Retain intentionally difficult synthetic rows (duplicate delivery, missing
  entity, future feature, late arrival), but annotate what each tests.

Query requirements:

- `03-label-windows.sql` returns one row per prediction with mature positive,
  mature negative, or unknown/not-mature label status. It must join by the
  attribution key and use both horizon and availability cutoff.
- `04-ranking-and-experiment-analysis.sql` calculates exposure-level CTR from
  valid assignments and includes a deterministic sample-ratio-mismatch result
  against an expected allocation. It must not treat unexposed users or future
  outcomes as negatives.
- Extend the guide so its wording matches reality: the four `.sql` files are
  executable reference answers; the remaining eight are intentionally prompt
  drills until their reference files and expected outputs are added. Do not
  claim all twelve currently have answers/tests.

Acceptance criteria:

- One conversion cannot label more than its explicitly attributed prediction
  or exposure.
- A future/unavailable label is returned as unknown, not zero.
- Every user has one assignment per experiment and all exposures agree with
  that assignment.
- All four reference SQL files execute on a new in-memory SQLite database.

Required tests:

- Assert fixture invariants with SQL: unique `(experiment_id, user_id)`
  assignment; no exposure/assignment arm mismatch; no duplicate prediction
  identifier; labels refer to valid attribution IDs.
- Compare expected result rows, including a known mature positive, mature
  negative, and unknown label.
- Insert a deliberately future `available_at` label and prove it is excluded
  at the cutoff.
- Assert expected CTR/arm counts and a known sample-ratio-mismatch outcome.
- Execute every query in a fresh connection; never rely on state left by a
  previous query.

### P1 — Align ML coding promises, numbering, and executable solutions

Modify:

- `ml/interview-prep/ml-coding-questions.md`
- `ml/interview-prep/solutions/training_and_evaluation.py`
- `ml/interview-prep/solutions/evaluation_exercises.py`
- `ml/interview-prep/README.md` (only if its question counts or solution links
  become inaccurate)
- `roadmaps/ml-roadmap.md` (only if it names an affected exercise range)
- `tests/test_interview_prep_expansion.py`

Add:

- `tests/test_ml_interview_solutions.py`

#### P1.1 Stabilize question identity and document scope honestly

- Renumber existing unnumbered exercises as Q1 through Q7 in their current
  order: KNN, attention, softmax/cross-entropy, PCA, rejection sampling, topic
  modeling, and ranking metrics. Preserve their prose and links.
- Retain Q8–Q20 numbers as-is; confirm there are exactly twenty unique,
  monotonically ordered question headings.
- Delete only the stale sentence saying K-means remains a follow-up; retain
  other useful follow-up topics and label them as optional after Q20.
- Amend Q14 to match the final reference implementation. Either implement a
  genuine dependency-light training-loop helper with gradient accumulation,
  clipping, early stopping, callback-based gradients, and deterministic seed,
  or narrow the prompt to seeded batch construction. Preferred repair: provide
  the real small training-loop helper because the question explicitly tests
  those behaviors.
- Ensure Q13 says `choose_threshold` requires binary labels and that ties use
  an explicit deterministic policy (for example lowest threshold among
  minimum-cost candidates).
- Redesign `point_in_time_features` to receive `events` and `feature_rows` as
  separate inputs, use strict availability before each event, preserve event
  order, and reject malformed timestamps/rows. Do not derive feature history
  from events themselves.

Acceptance criteria:

- Heading extraction returns Q1 through Q20 exactly once, in order.
- Every `See` reference resolves to an existing named public function/class.
- Q14’s documented behavior is present in source and covered with direct
  tests, rather than being a conceptual aside.
- Q12’s solution cannot select a feature row at or after the event time.
- Q13 rejects labels other than 0/1 and has deterministic tie behavior.

Required tests:

- Heading-number uniqueness/order and question-to-reference contract checks.
- Linear/logistic input validation and stability regressions retained from the
  current suite.
- K-means determinism, duplicated points, `k=1`, and empty-cluster repair.
- Point-in-time test with a separate feature table containing a past, same-time,
  future, and another-entity feature.
- Training-loop gradient accumulation equivalence under a fixed toy gradient,
  clipping bound, early-stop behavior, and reproducible batch order.
- Threshold non-binary labels, costs, and equal-cost ties.
- Retrieval duplicates/filtering contract and NDCG regression from P0.

### P1.2 — Make regression coverage test the published learning contract

Modify:

- `tests/test_interview_prep_expansion.py`
- `tests/test_repository_integrity.py` (if this is the repository’s shared
  place for documentation and local-link contracts)

Add:

- `tests/test_interview_bank_contracts.py`

Test design:

- Replace lower-bound-only heading counts with unique sequential-number tests:
  LLM Q1–Q30, agent Q1–Q20, system-design Q1–Q25, and ML coding Q1–Q20.
- Require every bank prompt to contain a compact simulation card. A regular
  expression or parser should require stable markers such as `Scenario`,
  `Success criteria`, `Clarifying questions`, `Failure modes`, `Follow-up`,
  and `Rubric`; avoid brittle full-text snapshots.
- Verify every coding-exercise task names at least one concrete local reference
  module/function. For LLM tasks with no implementation yet, mark the task as
  design-only rather than falsely naming an unrelated evaluation primitive.
- Test SQL result sets and fixture invariants as P0.3 describes.
- Keep tests dependency-light: standard library, NumPy, and SQLite only. Use
  dynamic module loading for numbered filenames rather than renaming existing
  files.

Acceptance criteria:

- Removing a required simulation-card section, duplicating a number, or
  changing an executable reference to a non-existent symbol fails a focused
  test.
- Tests prove behavior, not just that files parse or SQL executes.
- Full suite remains deterministic and does not modify Markdown or SQL files.

### P2 — Convert topic banks into independently runnable interview mocks

Modify:

- `llm/interview-prep/llm-interview-questions.md`
- `llm/interview-prep/llm-coding-exercises.md`
- `agentic-ai/interview-prep/agent-interview-questions.md`
- `agentic-ai/interview-prep/agent-coding-exercises.md`
- `system-design/interview-prep/question-bank.md`
- `system-design/interview-prep/scorecard.md`
- `llm/README.md`

Optional additions, only if six LLM coding tasks are to be claimed as
executable reference exercises:

- `llm/interview-prep/solutions/retrieval_exercises.py`
- `llm/interview-prep/solutions/structured_output_exercises.py`
- `llm/interview-prep/solutions/routing_exercises.py`

#### P2.1 Bank-card format

Convert each bank item without changing its core topic or its ordinal number.
Use a concise repeated format so 75 prompts remain scannable:

```markdown
### Q12 — [short scenario title]
**Round/time:** ...
**Scenario and constraints:** ...
**A strong candidate clarifies:** ...
**Success criteria:** ...
**Failure modes/edge cases:** ...
**Interviewer follow-up:** ...
**Rubric (0–3):** ...
**Remediation:** ...
```

For LLM Q1–Q30, add concrete data/traffic/quality-cost-latency constraints
and task-specific follow-ups. For agent Q1–Q20, require a typed side-effect,
trust, approval, recovery, or budget boundary appropriate to each prompt. For
system-design Q1–Q25, state starting scale, required output, success metrics,
and two role-specific follow-ups; preserve links to the existing patterns.

The system-design scorecard remains the cross-mock summary. Add a small
per-prompt scoring mapping only where a prompt needs special criteria (for
example attribution correctness for ads or isolation for RAG). Do not duplicate
the full scorecard 25 times.

#### P2.2 Exercise-reference honesty

- In LLM coding exercises, either add tested reference modules for schema
  validation, citation coverage, chunk-window selection, filtered retrieval,
  routing, and regression gating, or explicitly label each as an interview
  design task with expected test cases but no bundled reference answer.
- Preferred repair: add the three small API-free modules above and map every
  task to named functions. Keep their responsibility narrow; generic ML
  metrics remain in `ml/interview-prep/solutions/evaluation_exercises.py`.
- In agent coding exercises, map tool validation and cancellation to concrete
  new or expanded functions in the P0 agent modules. Ensure the documentation
  names the input/output contract and failure result for each.

Acceptance criteria:

- Each individual LLM, agent, and system-design question can be run by an
  interviewer without relying on the document preamble.
- Each prompt supplies a scenario, constraints, clarifications, success
  criteria, failure modes, a tailored follow-up, and a 0–3 scoring signal.
- All six LLM and six agent coding exercises are either truly reference-backed
  or honestly labeled design-only; none uses an unrelated function as a
  placeholder reference.

Required tests:

- Bank-contract parser checks every numbered entry independently.
- Reference-map tests import every claimed function and exercise one normal
  and one invalid/edge input.
- Markdown link audit remains clean.

### P3 — Reconcile live navigation facts and preserve history

Modify:

- `llm/README.md`
- `ml/interview-prep/ml-coding-questions.md`
- `docs/superpowers/specs/2026-09-07-ml-ai-interview-practice-expansion.md`
  (append a dated implementation-status note only; do not rewrite the
  historical proposal)

Requirements:

- Correct the LLM notebook count from 46 to the current 45 files, or replace
  the hard-coded count with wording that remains accurate under intended
  maintenance conventions.
- Remove the stale K-means-is-missing statement after Q10; retain the original
  expansion spec as historical context and append a concise note explaining
  that this repair addresses issues discovered in review.
- Add a link from the original expansion spec to this repair specification.

Acceptance criteria:

- README count matches `llm/notebooks/*.ipynb` excluding README/non-notebook
  files.
- No active ML practice page describes K-means as both implemented and
  outstanding.
- Historical documents retain their original decision context and date.

## 4. Delivery order and validation gates

1. Implement P0.1–P0.3 and their focused tests first. Run:
   `pytest -q tests/test_interview_prep_expansion.py tests/test_agent_interview_exercises.py tests/test_ml_sql_exercises.py`.
2. Implement the ML exercise/source alignment in P1 and run:
   `pytest -q tests/test_ml_interview_solutions.py tests/test_interview_prep_expansion.py`.
3. Convert the mock banks and add contract tests in P2. Run:
   `pytest -q tests/test_interview_bank_contracts.py`.
4. Reconcile documentation in P3, then run:
   `python3 scripts/audit_markdown_links.py --root .`.
5. Run the complete suite with `pytest -q`, plus a fresh SQLite execution for
   each reference query. Report exact pass/skip counts; do not claim a full
   run from targeted checks.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Durable exactly-once semantics are impossible to demonstrate with a local function alone. | Teach explicit at-least-once/unknown-result boundaries, require reconciliation, and document downstream idempotency assumptions. |
| SQL fixes become over-engineered or database-specific. | Keep the schema small, SQLite-compatible, and focused on attribution, availability, and assignment semantics. |
| Per-question cards make banks too repetitive. | Use compact required labels and a shared scorecard; tailor only the scenario and special rubric elements. |
| Tests become brittle prose snapshots. | Parse stable headings/markers and semantic links; test source behavior with small fixtures. |
| Renumbering breaks historical references. | Preserve current order, add numbers only to Q1–Q7, and update only active references; do not rewrite historical plans. |
| New LLM reference code drifts into model API wrappers. | Enforce API-free, deterministic pure functions with synthetic inputs. |
| Existing uncommitted work is accidentally overwritten. | Patch only named files, review diffs per phase, and do not delete/reset/reformat unrelated artifacts. |

## 6. Definition of done

This repair is complete only when:

- the NDCG regression and all identified agent/SQL safety counterexamples are
  covered by behavioral tests;
- every public claim in the repaired exercise documents matches executable
  code or is explicitly marked design-only;
- all four banks have unique stable numbering and per-question simulation
  contracts;
- active documentation has accurate notebook/K-means statements;
- targeted tests, full `pytest -q`, SQLite result tests, and the Markdown link
  audit pass; and
- no historical plan, unrelated working artifact, or existing practice
  material was deleted merely to satisfy coverage counts.
