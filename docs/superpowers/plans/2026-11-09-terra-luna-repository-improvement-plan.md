# Terra/Luna Repository Improvement Plan

**Date:** 2026-11-09  
**Status:** Ready for implementation  
**Implementation owners:** Terra and Luna  
**Review model:** independent author/reviewer cycles for every bounded batch

## 1. Purpose

Turn the repository into a trustworthy, navigable, and verifiable interview-
preparation curriculum without expanding the inventory faster than its quality
can be demonstrated.

This plan follows the independent Terra and Luna reviews. The immediate
priority is auditability: the repository already contains a manifest, content
standard, audit script, curriculum map, new practice material, and expanded
MLOps/statistics content, but the automated quality gate covers only part of the
advertised repository and structural checks cannot establish instructional
correctness by themselves.

The plan therefore treats quality as four separate dimensions:

1. **Inventory:** every advertised artifact is accounted for.
2. **Structure:** files, links, metadata, notebooks, and prerequisites are valid.
3. **Instructional quality:** explanations, examples, exercises, and answers are
   specific, correct, and useful.
4. **Execution:** dependency-light code actually runs and heavier material has
   an explicit alternative validation method.

## 2. Baseline and constraints

Baseline commit: `26aadef` on `origin/master`.

Observed baseline evidence:

- `472 passed, 10 skipped` from the local test suite.
- `git diff --check` clean before the baseline was committed.
- The curriculum audit reports 172 required lessons and no automated errors.
- The manifest currently covers six tracks: ML, Modern AI, MLOps, Statistics,
  LLM, and Agentic AI.
- The README advertises additional tracks and artifact families that are not
  yet represented by the same audit denominator.
- Notebook execution is representative rather than repository-wide.
- Manually maintained status counts and contributor instructions can drift from
  the newer content standard.

Constraints:

- Preserve existing learner-facing paths and filenames unless a migration is
  explicitly reviewed.
- Do not delete or hide incomplete material to make counts pass.
- Do not mass-generate generic prose or notebooks from a shared template.
- Keep each implementation batch small enough for a human to inspect fully.
- Do not mix unrelated historical-report edits into curriculum batches.
- Every change must leave the repository in a testable state.

## 3. Ownership model

Terra owns architecture, inventory, schema, validation, CI, dependency policy,
and release readiness. Luna owns learner-facing content, examples, exercises,
question banks, capstones, and pedagogical coherence.

Ownership does not grant approval authority. The author of a batch cannot be its
sole reviewer.

| Workstream | Primary | Independent reviewer | Required handoff |
|---|---|---|---|
| Manifest, graph, and audit | Terra | Luna | Human-readable scope and learner-impact notes |
| Content and notebook repair | Luna | Terra | Changed artifact ledger and technical evidence |
| Practice and capstones | Luna | Terra | Rubric, answer key, and link/execution evidence |
| CI and notebook tiers | Terra | Luna | False-positive/false-negative fixture review |
| Navigation and status | Terra | Luna | Count/inventory reconciliation report |
| Final release | Terra | Luna plus project owner | Signed validation and unresolved-risk summary |

## 4. Review-cycle protocol

Every batch follows the same cycle.

### Cycle A — Plan

The author creates a batch record containing:

- batch ID and one-sentence learner outcome;
- exact lesson IDs and paths in scope;
- files expected to change and files explicitly out of scope;
- prerequisites and downstream links;
- references to authoritative sources;
- acceptance criteria and validation commands;
- known risks and proposed exceptions.

Terra checks scope, dependency boundaries, and machine-verifiable criteria.
Luna checks learner progression, topic coverage, and reviewability. The batch
does not start until both approve the scope.

### Cycle B — Implement

The author works only on the approved batch. Each changed lesson keeps its
concept, notebook, implementation, practice, manifest, and navigation links in
sync where those artifacts are applicable.

The author maintains a changed-artifact ledger with one row per lesson:

| Lesson ID | Concept | Notebook | Implementation | Practice | Manifest | Review status |
|---|---|---|---|---|---|---|
| `track/slug` | path/status | path/status | path/status | path/status | path/status | pending |

### Cycle C — Independent review

The reviewer checks the complete batch, not only the diff summary. Findings are
recorded with severity, path, section/cell, evidence, and requested disposition:

- **Blocking:** incorrect claim, misleading prerequisite, broken learner path,
  invalid code, missing required artifact, or failed gate.
- **Major:** materially weak explanation, generic example, unsupported source,
  or missing assessment signal.
- **Minor:** wording, formatting, or non-blocking navigation issue.

Technical correctness and pedagogical usefulness are reviewed separately.

### Cycle D — Repair and re-review

The author resolves blocking and major findings or supplies an evidence-based
exception. The reviewer rechecks every changed finding plus neighboring
prerequisites, links, references, and tests affected by the repair.

No author may close their own blocking finding. Disagreement goes to the project
owner with both positions and evidence.

### Cycle E — Integration gate

The batch closes only when:

- no blocking findings remain;
- targeted tests pass;
- the full suite passes;
- the manifest audit passes for the affected tracks;
- link and notebook checks pass or documented exceptions exist;
- the changed-artifact ledger is complete;
- the batch record names residual follow-up work and an owner.

## 5. Delivery phases

### Phase 0 — Preserve and inventory

**Owner:** Terra  
**Reviewer:** Luna  
**Dependency:** none

Create a complete repository ledger before broad edits continue. Reconcile the
manifest against actual concepts, notebooks, implementations, interview pages,
roadmaps, and capstones. Classify every artifact as `reviewed`, `required`,
`baseline`, `excluded` with reason, or `missing`.

Deliverables:

- inventory report by top-level track and artifact type;
- changed-file ledger for the baseline expansion;
- explicit manifest denominator and exclusions;
- generated counts or checked-in count validation;
- list of all skipped notebook executions and reasons;
- status-page update that distinguishes audited scope from total inventory.

Acceptance criteria:

- Every README-advertised track has an inventory row.
- Every manifest entry resolves to an existing path.
- Every learner-facing concept is either indexed or visibly classified as
  baseline/excluded.
- No claim of completeness exceeds the reviewed denominator.

### Phase 1 — Establish the curriculum architecture

**Owner:** Terra  
**Reviewer:** Luna  
**Dependency:** Phase 0

Make the manifest the canonical machine-readable source for lesson identity,
track, level, outcomes, prerequisites, artifact paths, practice, and estimated
time. Define how it relates to `roadmaps/curriculum-map.md`; the map is the
learner-facing explanation, while the manifest is the consistency contract.

Add checks for:

- stable IDs and duplicate paths;
- unresolved prerequisites and prerequisite cycles;
- manifest/map disagreement;
- missing paired artifacts;
- broken local links and anchors;
- stale or manually inconsistent counts;
- changed lessons whose paired artifacts were not reviewed together.

Acceptance criteria:

- A fresh contributor can determine the canonical prerequisite from one source.
- The activation-functions example and similar overlaps have an explicit,
  reviewed relationship between AI and ML tracks.
- Audit output reports indexed, baseline, excluded, and missing counts by track.
- CI runs the audit as an explicit named job.

### Phase 2 — Prove the quality gate with adversarial fixtures

**Owner:** Terra  
**Reviewer:** Luna  
**Dependency:** Phase 1

Strengthen `scripts/audit_curriculum.py` and its tests. Structural checks remain
valuable, but the audit must clearly state which requirements require human
review.

Add fixtures for:

- malformed notebook JSON and non-v4 notebooks;
- missing cells, assertions, exercises, or deterministic seeds;
- misleading section headings with generic content;
- scaffold phrases and placeholder prose;
- broken relative links and anchors;
- duplicate IDs, missing paths, cycles, and invalid prerequisites;
- practice pages and capstones missing rubric fields;
- notebooks that parse but fail at runtime;
- a valid narrow reference lesson below the general prose threshold.

Acceptance criteria:

- Every fixture fails for the intended reason.
- No historical document or test fixture is accidentally audited.
- Audit output identifies exact path, rule, and remediation.
- CI failure messages are actionable without reading the validator source.
- Human-review-only criteria are documented rather than disguised as proxies.

### Phase 3 — Pilot a complete learner journey

**Owner:** Luna  
**Reviewer:** Terra  
**Dependency:** Phases 1-2

Select a small, dependency-light pilot spanning statistics, core ML, and
evaluation. The pilot must be fully coherent before repository-wide rewrites.

For each pilot lesson:

- align concept, notebook, practice, manifest, and roadmap entry;
- state assumptions, limitations, and decision boundaries;
- use topic-specific examples and edge cases;
- provide an answer key with follow-up questions and common mistakes;
- execute the notebook from a fresh kernel where feasible.

Acceptance criteria:

- A new learner can start at the stated entry point and follow the path without
  unexplained prerequisites or broken links.
- Every pilot notebook tests a learning objective, not merely JSON validity.
- A reviewer can distinguish measured output from simulation or pseudocode.
- Pilot review findings produce updated content rules before scaling.

### Phase 4 — Review the current expansion in bounded batches

**Owner:** Luna  
**Reviewer:** Terra  
**Dependency:** Phase 3

Do not review the 485-file expansion as one unit. Use these batches:

1. **MLOps operations:** monitoring, drift, orchestration, feature parity,
   incident response, governance, security, cost, and reliability.
2. **Statistics:** bootstrap, permutation tests, multiple testing, missing data,
   regression diagnostics, and sequential experimentation.
3. **Agentic AI:** concepts, notebooks, and practice requiring topic-specific
   agent loops, tools, memory, safety, and evaluation.
4. **LLM:** retrieval, inference, optimization, safety, and interview practice.
5. **Modern AI:** inference-system concepts, implementations, notebooks, and
   trade-off explanations.
6. **Practice and projects:** domain interview tracks, role paths, and scored
   capstones.

Each batch must include its own changed-artifact ledger and review record. A
batch may be split further if the reviewer cannot inspect every lesson fully.

Acceptance criteria:

- No generic template language is accepted as topic-specific instruction.
- References support the claims they are attached to.
- Examples include failure modes and explain what the result demonstrates.
- Every added practice page has prompts, expected answers, dimensions,
  follow-ups, and common mistakes.
- Every capstone has a data contract, deliverables, rubric, timebox, and
  extension prompt.

### Phase 5 — Expand execution assurance

**Owner:** Terra  
**Reviewer:** Luna  
**Dependency:** Phases 2-4

Introduce notebook validation tiers:

- **Tier 1:** parse every notebook, validate nbformat, policy, links, cells,
  seed, and output rules.
- **Tier 2:** execute every dependency-light notebook in a clean kernel.
- **Tier 3:** run a maintained representative set for notebooks requiring large
  dependencies, downloads, accelerators, credentials, or external services.

Every Tier 3 exclusion records its reason and an alternative structural or unit
check. CI reports checked, skipped, and failed notebooks explicitly.

Acceptance criteria:

- No notebook silently leaves the execution matrix.
- Default smoke execution needs no API key or model download.
- Assertions test the notebook’s stated objective and edge cases.
- Repeated runs are deterministic within documented numeric tolerances.

### Phase 6 — Reconcile documentation and release

**Owner:** Terra  
**Reviewer:** Luna and project owner  
**Dependency:** Phases 0-5

Reconcile root README, section READMEs, `CONTRIBUTING.md`, repository status,
roadmaps, and historical handoffs. Make the content standard the single source
of truth for contributor requirements. Generate or verify counts from the
inventory wherever practical.

Release acceptance criteria:

- Root and section counts agree with the verified inventory.
- README distinguishes audited content from baseline content.
- Contributor instructions identify files, metadata, validation commands, and
  review expectations.
- Full suite, curriculum audit, link check, and applicable notebook tiers pass.
- The release record includes exact commands, results, skipped checks, and
  accepted residual risks.

## 6. Definition of done

The improvement program is complete when:

1. All advertised tracks have an explicit inventory status.
2. The manifest, curriculum map, and navigation have one documented contract.
3. Automated checks reject structural regressions and known scaffold patterns.
4. A risk-based human review has promoted the required content; unfinished work
   remains visibly baseline rather than silently passing.
5. At least one complete learner journey passes end to end.
6. Notebook execution tiers are explicit and reproducible.
7. Practice and capstones are scored, linked, and reviewable.
8. Terra and Luna have independently reviewed every closed batch.
9. Documentation claims are generated or verified from the same inventory.
10. The final release record contains no unowned blocking issue.

## 7. Operating cadence

For each batch, Terra and Luna should produce:

- a short plan record before implementation;
- an implementation commit or bounded commit series;
- an independent review memo;
- a repair/re-review note;
- a validation transcript and residual-risk entry.

Recommended batch size is 5-12 concept lessons or one coherent artifact family,
whichever is smaller. A batch larger than that requires an explicit reviewability
justification.

At the end of every phase, pause for a checkpoint. The next phase may begin
only after the previous phase's acceptance criteria and review record are
complete. This keeps the repository's quality claims synchronized with the work
actually inspected.
