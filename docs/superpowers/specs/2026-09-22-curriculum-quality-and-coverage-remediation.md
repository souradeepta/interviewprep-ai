# Curriculum Quality and Coverage Remediation Specification

**Date:** 2026-09-22  
**Status:** Proposed implementation specification  
**Audience:** GPT-Luna implementing sequentially in this checkout

## Objective

Turn the repository from a large collection of nominally paired artifacts into
a coherent, trustworthy interview-preparation curriculum. The implementation
must first repair learner-facing material that is incomplete, generic, or
malformed. It must then add the missing learning paths, practice modes, and
production-ML coverage. A concept, notebook, or question counts only when it
meets a content contract and is covered by automated checks.

This is curriculum work. Do not change unrelated historical reports or the
uncommitted documentation/status work already present in the checkout.

## Audit Evidence and Gaps

The findings below were measured from the 2026-09-22 checkout. File counts
alone are not evidence of usable curriculum quality.

| Area | Evidence | Learner impact | Required response |
|---|---|---|---|
| ML concept notes | All 40 `ml/concepts/*.md` files are 98-101 words and retain literal template text such as `[To be filled with intuitive explanation]`, `Important property 1`, and `[Key insight]`. | The ML roadmap presents empty notes as an interview-ready foundation. | Replace all 40 notes with complete, topic-specific lessons before adding more ML topics. |
| Modern AI, concepts 26-35 | Concepts 27-35 are 532-794 words. Several have generic diagrams, generic challenge text, or answers that direct the learner to a notebook instead of answering the question. | The advanced serving track appears deep but cannot support an interview explanation or an implementation decision. | Rewrite concepts 26-35 and their matching notebooks as one connected inference-systems module. |
| Modern AI, notebooks 26-35 | Ten notebooks contain the identical generic objective list (`Core concept`, `Implementation`, `Production`, `Comparison`) and toy linear-layer examples unrelated to their titles. | A learner cannot learn MoE, KV cache, speculative decoding, or serving from the advertised code. | Replace all ten notebooks and Python implementations with topic-specific deterministic exercises. |
| Agentic notebooks | `10-mcts-for-agents.ipynb` has trailing JSON data and `26-error-recovery.ipynb` has invalid JSON. Twelve valid notebooks have fewer than nine cells. | Two advertised notebooks cannot open; the short notebooks lack the required learning arc. | Repair JSON, then rebuild the twelve listed notebooks to the common notebook contract. |
| LLM and Agentic concepts | Nine LLM notes (33-35, 38-40, 42-44) and 23 Agentic notes are under 800 words. Representative files use generic architecture tables, generic best practices, and generic related-topic bullets. | The explanations do not provide enough specific trade-offs or interview guidance. | Replace the named LLM notes and the Agentic notes that fail the new content audit. |
| Practice coverage | Dedicated interview tracks exist for ML, LLM, Agentic AI, and system design. Statistics, MLOps, RL, CV, and NLP have no domain interview banks; no track has a scored project/capstone brief. The four recently added DSA guides are only 201-245 words compared with 1,000+ words in established guides. | Learners cannot convert study into interview performance or demonstrate portfolio-level work in most domains. | Add focused interview practice, repair the four short DSA guides, and add rubric-scored capstones. |
| Learning progression | Only four role roadmaps exist. The root navigation exposes ten instructional domains, but it has no prerequisite graph, role paths for MLE/applied/LLM-platform roles, or guidance that resolves overlap between `ai/`, `ml/`, `stats/`, `nlp/`, and `rl/`. | A learner sees breadth but cannot choose a minimal, ordered path. | Add one canonical curriculum map and role-based paths that link existing material without duplicating it. |
| Production ML breadth | The numbered MLOps curriculum ends at 16 despite its README naming monitoring, governance, scaling, and incident response as important. Statistics lacks bootstrap, multiple-comparison control, missing-data mechanisms, sequential testing, and regression diagnostics. | The curriculum is weak on the operational and experimental decisions expected in senior MLE interviews. | Add bounded MLOps and statistics modules with concepts, notebooks, implementations where appropriate, and practice questions. |
| Validation | Current tests validate inventory parity and ten representative notebook executions. They do not parse every notebook, reject placeholders, enforce curriculum metadata, or test that roadmaps give valid prerequisite paths. | Invalid or generic content can be added while all current checks pass. | Add deterministic curriculum quality tests and a manifest-driven audit before content additions are considered complete. |

### Baseline inventory to preserve

- Paired tracks currently include AI 40, ML 40, LLM 44/45 notebooks,
  Agentic AI 64, Modern AI 55, RL 20, Statistics 15, CV 8, NLP 8, and MLOps
  16 concepts plus a concept-map notebook.
- The current repository-status count may still report nominal artifacts that
  fail the quality criteria above. Do not lower counts to hide defects.
- The existing root and section README count edits are uncommitted user work.
  Do not overwrite them. Reconcile counts only after the new artifacts exist.

## Delivery Order

Implement in the order below. Finish, test, and commit-quality-check each
phase before beginning the next. Do not mass-generate content from one generic
template: every required lesson needs topic-specific examples and decisions.

| Phase | Outcome | Why it comes first |
|---|---|---|
| 0 | A manifest and validators that define “done” | Prevents another inflated inventory. |
| 1 | ML foundations and broken/generic notebook repair | Removes the most severe learner-facing failures. |
| 2 | Role paths, assessment model, and practice expansion | Makes the repaired material navigable and usable in interviews. |
| 3 | MLOps and statistics coverage expansion | Adds the highest-value missing production and experiment content. |
| 4 | CV/NLP/RL practice and capstones | Extends the assessment model to specialist paths. |
| 5 | Documentation reconciliation and full verification | Publishes only truthful curriculum claims. |

## Phase 0 — Curriculum Contract and Automated Audit

### New files

1. `data/curriculum_manifest.json`
2. `scripts/audit_curriculum.py`
3. `tests/test_curriculum_quality.py`
4. `docs/CURRICULUM_CONTENT_STANDARD.md`

### Manifest schema

Use JSON and validate it with the standard library. Each lesson entry must
contain:

```json
{
  "id": "ml/activation-functions",
  "track": "ml",
  "level": "foundation",
  "concept": "ml/concepts/activation-functions.md",
  "notebook": "ml/notebooks/01-activation-functions.ipynb",
  "implementation": null,
  "prerequisites": ["ai/gradient-descent"],
  "outcomes": [
    "Explain why nonlinear activations are necessary.",
    "Choose an activation for a hidden layer and justify it."
  ],
  "practice": ["ml/interview-prep/ml-theory-questions.md#activation-functions"],
  "estimated_minutes": 60
}
```

Use stable `track/slug` IDs. `prerequisites` must refer to other manifest IDs;
an empty list is allowed only for explicitly designated entry lessons. Do not
infer order from filenames. Include all learner-facing concepts in the ten
instructional tracks after Phases 1-4, plus every new practice module and
capstone.

### Content standards

`docs/CURRICULUM_CONTENT_STANDARD.md` is the source of truth for authors.
It must include the following contracts.

#### Concept note contract

Each concept note must contain, in this order:

1. `#` title and a one-sentence TL;DR.
2. Prerequisites and 3-5 measurable learning outcomes.
3. Core intuition with a concrete analogy or counterexample.
4. How it works, including relevant notation, assumptions, and complexity.
5. A topic-specific decision/trade-off table with at least three rows.
6. One runnable, dependency-light Python example or a clearly labeled
   executable pseudocode exception for a genuinely infrastructure-only topic.
7. At least three non-generic failure modes and a diagnostic or mitigation for
   each.
8. An interview quick-reference with at least three question/answer rows,
   including one design or debugging follow-up.
9. Related concepts with real relative links and at least two primary or
   authoritative references.

The substantive prose target is 900-1,800 words. A narrow reference page may
be 650-900 words only when the manifest sets `lesson_type` to `reference` and
the note still meets every section requirement. Do not count headings, code,
or reference URLs toward the target.

#### Notebook contract

Every notebook must be valid `nbformat` v4 JSON and contain at least nine
non-empty cells:

1. Title, prerequisites, estimated time, and measurable objectives.
2. A compact concept recap with assumptions.
3. A from-scratch or transparent baseline implementation.
4. Assertions for the baseline’s expected behavior.
5. A realistic extension or library implementation, when the dependency is
   included in the notebook requirements.
6. A comparison, metric, or visualization tied to the named topic.
7. At least two deliberately chosen edge or failure cases.
8. Exercises with expected properties or test cases.
9. Summary, next lessons, and linked practice.

Use a fixed random seed. Do not write files outside the notebook directory or
`/tmp`. Do not require an API key for a default smoke notebook. A notebook may
mock an external service only when it says so, describes what behavior the
mock cannot prove, and tests the surrounding deterministic logic.

#### Interview and capstone contract

Each question must give the prompt, role/domain, difficulty, a short expected
answer, evaluation dimensions, at least one follow-up, and a common mistake.
Coding questions also need runnable test cases and time/space complexity.
Every capstone must include a problem brief, supplied/synthetic data contract,
deliverables, evaluation rubric, timebox, extension prompt, and links to the
prerequisite lessons.

### Audit behavior

`scripts/audit_curriculum.py` must support:

```bash
python3 scripts/audit_curriculum.py
python3 scripts/audit_curriculum.py --format json
python3 scripts/audit_curriculum.py --track ml
```

It must return nonzero for:

- malformed notebook JSON or a notebook without a v4 `cells` list;
- missing manifest path, unresolved manifest prerequisite, duplicate ID, or
  prerequisite cycle;
- a broken local link in a manifest-referenced concept, roadmap, practice
  page, or capstone;
- a concept missing a required section, containing known scaffold phrases, or
  below the appropriate prose threshold;
- a notebook with too few non-empty cells, no executable code, no assertion,
  no exercise, or a known generic scaffold phrase;
- a practice page or capstone missing required headings.

The scaffold denylist must include the literal phrases found in the audit:
`[To be filled`, `Important property 1`, `Mistake 1`, `[Key insight]`,
`Core idea: understand the fundamental principle`, `See notebook`,
`Challenge 1`, `1. Core concept`, and `Modify parameters`.

Avoid false positives inside a historical document or a test fixture: restrict
the audit to manifest-owned curriculum paths. Keep exact detection rules and
thresholds in `tests/test_curriculum_quality.py`, with fixture-based tests for
both passing and failing examples.

## Phase 1 — Repair the Core Learning Material

### 1A. Replace the 40 ML concept stubs

Rewrite every file in `ml/concepts/`. Preserve filenames and links used by the
ML roadmap. Use these six modules to establish a deliberate progression:

| Module | Files |
|---|---|
| Math, data, and generalization | `supervised-learning`, `probability-statistics`, `evaluation-metrics`, `optimization`, `regularization`, `overfitting-underfitting`, `feature-engineering` |
| Linear and probabilistic models | `linear-regression`, `logistic-regression`, `naive-bayes`, `support-vector-machines`, `calibration` |
| Trees and ensembles | `decision-trees`, `random-forests`, `gradient-boosting`, `ensemble-methods` |
| Unsupervised and representation methods | `unsupervised-learning`, `clustering`, `dimensionality-reduction`, `anomaly-detection` |
| Deep-learning fundamentals | `neural-networks`, `activation-functions`, `weight-initialization`, `batch-normalization`, `layer-normalization`, `dropout`, `backpropagation`, `loss-functions`, `learning-rate-schedules`, `momentum-and-acceleration`, `gradient-accumulation`, `mixed-precision-training` |
| Transfer, sequence, and scale | `cnns`, `rnns-lstms`, `attention-mechanism`, `transformers`, `transfer-learning`, `domain-adaptation`, `knowledge-distillation`, `model-compression`, `quantization`, `pruning`, `meta-learning`, `distributed-training`, `early-stopping` |

Before writing, compare filenames with `find ml/concepts -name '*.md'` and
place each actual file exactly once. The table is an organizing guide, not
permission to rename files or omit any of the 40.

For every rewritten file:

- explain its relationship to the similarly named `ai/` topic where one
  exists, and state the role of this ML-track lesson so learners do not study
  duplicated material by accident;
- use a tiny deterministic dataset in code where the algorithm permits it;
- state a selection boundary, such as when to use calibration after a
  classifier, when to prefer layer norm over batch norm, or when a tree is a
  poor choice;
- link to its matching notebook and a relevant question anchor.

### 1B. Rebuild the Modern AI inference systems module

Replace concepts, notebooks, and Python implementations numbered 26-35:

| ID | Topic-specific minimum demonstration |
|---:|---|
| 26 | MoE top-k routing, capacity factor, load-balance loss, and dropped-token rate |
| 27 | Context-budget allocator comparing truncation, retrieval, and summarization |
| 28 | Speculative decoding acceptance/rejection accounting and expected speedup limits |
| 29 | KV-cache memory calculator, block allocator, prefix sharing, and eviction behavior |
| 30 | Sparse-attention masks with complexity and task-fit comparison |
| 31 | Quantization/distillation error versus memory/latency trade-off |
| 32 | A multimodal fusion baseline with modality-missing failure cases |
| 33 | Prefill/decode queue simulation with latency percentiles and scheduling trade-offs |
| 34 | Post-training specialization decision matrix: prompting, RAG, LoRA, or full fine-tuning |
| 35 | Decode-length prediction with calibration, underprediction, and tail-latency analysis |

Do not claim measured production speedups unless the notebook measures them on
the learner’s environment. Use formulas and clearly labeled simulations where
hardware-specific results are unavailable. Update the notebook captions in
the matching concepts to their real filenames; do not leave unsuffixed names
such as `kv-cache-optimization.ipynb`.

### 1C. Repair Agentic notebook validity and lesson arc

First repair these files so `nbformat.read` succeeds:

- `agentic-ai/notebooks/10-mcts-for-agents.ipynb`
- `agentic-ai/notebooks/26-error-recovery.ipynb`

Then rebuild these short notebooks to the Phase 0 notebook contract:

- `07-planning-reasoning`, `08-react-reasoning-acting`,
  `39-code-analysis-agents`, `40-customer-service-agents`,
  `41-medical-agents`, `42-legal-document-agents`, `43-finance-agents`,
  `44-logistics-agents`, `45-data-analysis-agents`, `46-research-agents`,
  `47-content-moderation-agents`, `48-recommendation-agents`.

Domain notebooks must use safe local synthetic data and teach the engineering
constraint that distinguishes the domain (for example, auditability in legal,
abstention in medical, idempotency in finance, or privacy in customer support).
They must not simulate real orders, diagnoses, or legal conclusions.

### 1D. Repair thin LLM and Agentic notes

Rewrite these LLM notes: 33, 34, 35, 38, 39, 40, 42, 43, and 44. Replace their
generic diagrams and tables with specific architecture, cost, quality, or
evaluation trade-offs. Every interview answer must stand on its own rather
than pointing to another artifact.

Run the Phase 0 audit across all Agentic concepts. Rewrite every note it flags;
the initial expected set includes 39, 44, 47, 48, 53, 54, and 58-64. Do not
use word count as the only repair rule: a longer generic note still fails.

## Phase 2 — Make the Curriculum Navigable and Assessable

### 2A. Canonical curriculum map

Add `roadmaps/curriculum-map.md` and link it from the root README and
`roadmaps/README.md`. It must contain:

- a prerequisite diagram with these stages: programming/math -> statistics ->
  core ML -> deep learning -> LLM/NLP/CV/RL -> MLOps/system design -> agents
  and modern AI engineering;
- a table that states the purpose of `ai/` versus `ml/`, and links to the
  recommended lesson when topics overlap;
- four role paths: entry-level ML engineer (12 weeks), production MLE (14
  weeks), LLM/RAG engineer (10 weeks), and applied AI/agent engineer (12
  weeks);
- weekly study plans with realistic hours, existing lesson links, one practice
  activity, and a phase exit check each week;
- a fast-track section that identifies prerequisites learners may test out of,
  with a small self-check instead of an unsupported claim of equivalence.

Update the four existing roadmaps so they identify this map as their common
prerequisite source and contain only path-specific sequencing. Do not add a
second contradictory order in section READMEs.

### 2B. Practice banks

Create five files and their README links:

| File | Questions | Required coverage |
|---|---:|---|
| `stats/interview-prep/statistics-interview-questions.md` | 20 | distributions, estimation, confidence intervals, power, A/B tests, causal assumptions, bootstrap, multiple tests, missing data, sequential testing |
| `mlops/interview-prep/mlops-interview-questions.md` | 20 | data contracts, leakage, feature parity, deployment, monitoring, rollback, drift, incident response, governance, cost |
| `rl/interview-prep/rl-interview-questions.md` | 15 | MDPs, Bellman updates, off-policy stability, exploration, reward hacking, offline RL, RLHF |
| `cv/interview-prep/cv-interview-questions.md` | 15 | labels and augmentation, detection/segmentation metrics, class imbalance, distribution shift, serving/edge constraints, multimodal evaluation |
| `nlp/interview-prep/nlp-interview-questions.md` | 15 | tokenization, embeddings, retrieval, ranking, sequence labeling, multilingual evaluation, LLM boundary decisions |

At least 40% of each bank must be scenario, debugging, or system-design
questions. Do not pad with alternate phrasings of definition questions.

Expand these DSA guides to 900-1,300 substantive words each, with recognition
criteria, one worked trace, two complete problems, common failure cases,
complexity, and a practice list: `prefix-sums.md`, `intervals-sweep-line.md`,
`matrix-grid.md`, `simulation-state-machines.md`.

### 2C. Capstones and scoring

Add `projects/README.md`, four project briefs, and a shared scorecard:

- `projects/01-tabular-ml-production.md`
- `projects/02-rag-evaluation-and-retrieval.md`
- `projects/03-ml-platform-reliability.md`
- `projects/04-agent-workflow-safety.md`
- `projects/scorecard.md`

Each is a 6-10 hour project based on deterministic synthetic or openly
available small data. A capstone must specify an executable local success
criterion and a scoring rubric totaling 100 points. Include engineering,
measurement, failure handling, communication, and interview debrief sections.
The four roles in the curriculum map must each point to one required capstone.

## Phase 3 — Add High-Value Coverage

### 3A. Statistics: six lessons, notebooks, and practice links

Add IDs 16-21 in `stats/concepts/` and `stats/notebooks/`:

1. Bootstrap and resampling
2. Permutation tests and randomization inference
3. Multiple testing and false-discovery control
4. Missing data and imputation assumptions
5. Regression diagnostics and robust standard errors
6. Sequential experimentation and peeking control

Every notebook must use a seeded synthetic dataset and show a failure of a
naive method next to the correct method. Update `stats/README.md`, its learning
paths, the manifest, and the statistics interview bank. Do not add standalone
implementations in this track unless a notebook’s logic cannot be expressed
clearly inside the notebook.

### 3B. MLOps: eight lessons, notebooks, implementations, and practice links

Add IDs 17-24 in `mlops/concepts/`, `mlops/notebooks/`, and
`mlops/implementations/`:

1. Model monitoring and alert design
2. Data and concept drift detection
3. ML pipeline orchestration and backfills
4. Online/offline feature parity
5. ML incident response and rollback
6. ML governance, lineage, and auditability
7. ML security and privacy controls
8. Cost, capacity, and reliability planning

Each Python implementation must be dependency-light and support direct unit
tests. Use simulated metrics/logs rather than cloud credentials. Each lesson
must include one operational decision with a measurable threshold, such as an
alert policy, rollback trigger, freshness SLO, or error budget. Update
`mlops/README.md`, `mlops/MLOPS_ROADMAP.md`, the root inventory, manifest,
and MLOps interview bank.

## Phase 4 — Specialist Assessment Integration

For CV, NLP, and RL, add only the practice banks from Phase 2 and connect their
existing concept paths to the role paths. Do not add arbitrary concepts simply
to equalize counts. Add one capstone extension prompt per domain to the
relevant project brief:

- CV: data shift and thresholding in visual inspection.
- NLP: hybrid retrieval and ranking evaluation.
- RL: offline-policy evaluation and reward-hacking analysis.

## Phase 5 — Documentation and Verification

### Required documentation changes

- Update root `README.md` inventory only after exact files are present:
  Statistics becomes 21 concepts/notebooks; MLOps becomes 24 concepts and 25
  notebooks including the map, with 24 implementations.
- Update `docs/REPOSITORY_STATUS.md` with the actual test count and separate
  nominal artifact counts from validated curriculum counts if any legacy item
  is intentionally exempted.
- Update all affected section READMEs, `roadmaps/README.md`, and
  `CONTRIBUTING.md` to link the content standard and manifest contract.
- Mark superseded historical plans as historical only when their own content
  claims would otherwise mislead a contributor. Do not rewrite their original
  chronology.

### Test requirements

Add tests that prove all of the following:

1. Every manifest item resolves to existing artifacts and an acyclic
   prerequisite graph.
2. Every manifest concept meets its content contract and has valid local
   links.
3. Every notebook in the instructional tracks parses through `nbformat`.
4. Every notebook passes the structural contract; the ten repaired Modern AI
   notebooks also pass a topic-token test that rejects unrelated generic
   examples.
5. All new MLOps Python implementations have focused unit tests.
6. Every role path has links, a weekly practice action, exit checks, and one
   capstone.
7. Every new interview bank meets its minimum question count and structure.
8. Repository counts in active README/status pages match the filesystem.

Run these commands before handoff:

```bash
python3 scripts/audit_curriculum.py
python3 scripts/audit_markdown_links.py
python3 -m pytest -q
git diff --check
```

Run the default notebook smoke suite. Add deterministic representatives for
the repaired Agentic MCTS notebook, a Modern AI inference notebook, a new
statistics notebook, and a new MLOps notebook. If a full notebook requires
optional heavyweight dependencies, retain a dependency-light representative
that tests the instructional core; do not skip JSON or structural validation.

## Acceptance Criteria

The work is complete only when all items are true:

- No manifest-owned concept retains a scaffold phrase or fails the content
  contract; the 40 ML concept files are substantive topic-specific lessons.
- Every shipped notebook parses, and every instructional notebook meets the
  notebook contract.
- Modern AI 26-35 have real, different demonstrations that match their titles;
  none retain generic objectives, toy `nn.Linear` substitutions, or placeholder
  exercises.
- The two malformed Agentic notebooks open in Jupyter and the twelve short
  Agentic notebooks have a complete learning arc.
- The curriculum map provides a valid, linked, time-bounded plan for all four
  roles and resolves `ai/`/`ml/` overlap.
- The five new practice banks meet their question counts and format; all four
  short DSA guides meet their revised quality bar.
- Four scored capstones exist and are linked from role paths.
- Statistics has 21 paired concept/notebook lessons; MLOps has 24 paired
  concept/notebook/implementation lessons plus its concept map.
- The manifest audit, link audit, full test suite, and diff check pass.
- Active counts and status claims match the resulting filesystem and test
  output.

## Explicit Non-Goals

- Do not execute API-backed notebooks against paid or private services.
- Do not claim benchmark, latency, accuracy, or production adoption results
  without a cited source or a reproducible local measurement.
- Do not replace all existing notebooks with one uniform generator output.
- Do not add more paper summaries in this pass; the problem is curriculum
  progression, practice, and executable quality.
