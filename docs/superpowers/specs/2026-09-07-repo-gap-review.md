# Repository Gap Review and Remediation Specification

**Date:** 2026-09-07
**Scope:** Repository-wide review after the A–K enhancement work and the
statistics/CV notebook repairs.
**Status:** Specification only; implementation is a subsequent task.

## 1. Executive Summary

The enhancement content is substantially present, but the repository is not
yet internally consistent or fully verifiable. The highest-value gaps are:

1. Root navigation and coverage counts do not reflect the current repository.
2. The link graph contains 69 missing local targets under the current Markdown
   audit, concentrated in roadmaps and legacy system-design paths.
3. Automated tests validate file shape and Python syntax for the newer sections,
   but do not execute notebook cells or validate dependencies, outputs, or
   cross-section coverage.
4. Handoff/session documents contain stale “empty”, “missing”, and “TBD” claims
   that conflict with the completed artifacts.
5. Untracked files mix managed worktrees, generated outputs, helper scripts, and
   potentially intentional user work. Cleanup needs classification, not bulk
   deletion.
6. The checkout’s `.git` directory is a read-only mount, so local branch/index
   state cannot be synchronized even though remote delivery is possible through
   a verified writable staging clone.

## 2. Evidence Snapshot

The audit was run from the repository root on 2026-09-07.

| Area | Observed state | Expected state |
|---|---:|---:|
| Full test suite | 408 passing | Keep green |
| AI concepts / notebooks | 40 / 40 | README currently matches |
| ML concepts / notebooks | 40 / 40 | README says 37 / 40 |
| LLM concepts / notebooks | 44 / 45 | README says 44 / 46 |
| Agentic AI concepts / notebooks | 64 / 64 | README omits this from the table’s implementation detail |
| Modern AI concepts / notebooks | 56 / 55 | README says 55 / 20 |
| MLOps concepts / notebooks | 16 / 17 | README matches |
| RL concepts / notebooks / implementations | 20 / 20 / 20 | README omits RL |
| Statistics concepts / notebooks | 15 / 15 | README omits stats |
| CV concepts / notebooks | 8 / 8 | README omits CV |
| NLP concepts / notebooks | 8 / 8 | README omits NLP |
| Papers concepts / notebooks | 27 / 27 | README coverage is present |
| Broken local Markdown targets | 69 | Zero unresolved intentional links |
| New-section structural tests | 10 passing | Retain and extend |

The broken-link count must be re-run with a repository-aware auditor before
fixing because some targets may be intentional historical references, while
others are genuine path drift.

## 3. Findings and Required Remediation

### P0 — Make the repository navigable and truthful

#### P0.1 Root README coverage drift

Update `README.md` to include the completed RL, Statistics, Computer Vision,
NLP, and ML Interview Prep sections. Correct the stale counts for ML, LLM,
Modern AI, and any other section whose count differs from the filesystem.
Add direct “Start Here” links for statistics, CV/NLP foundations, RL, and
interview preparation.

Acceptance criteria:

- Every top-level learning section with a README appears in the root coverage
  table or is explicitly classified as infrastructure/documentation.
- Every numeric claim in the table is generated or checked against a scripted
  count; no hand-edited count can silently drift.
- Root README links resolve.

#### P0.2 Broken local link graph

Create a link-audit script that:

- parses Markdown links while ignoring URLs, anchors, code blocks, and image
  URLs;
- resolves paths relative to the source file;
- reports missing files separately from missing anchors;
- supports an allowlist for historical documents that are intentionally
  snapshots;
- exits non-zero for an unallowlisted missing target.

Fix genuine path drift in `roadmaps/`, `coding/README.md`, and system-design
references. Do not rewrite historical session notes solely to make the link
checker green; classify them explicitly instead.

Acceptance criteria: zero unallowlisted missing file targets in active READMEs,
roadmaps, and contribution docs.

### P1 — Add execution-level validation

#### P1.1 Notebook smoke execution

Extend `tests/test_new_sections.py` or add a dedicated test module that runs a
small, deterministic smoke test for each notebook family. The test must:

- parse notebooks with `nbformat`;
- compile every code cell;
- execute representative notebooks with `nbclient` or an equivalent isolated
  runner;
- use a non-interactive Matplotlib backend;
- set bounded timeouts and deterministic seeds;
- report missing optional dependencies clearly;
- reject uncaught exceptions and unexpected empty result sections.

Full execution may be a separate opt-in CI job if runtime is too high, but the
default test suite must at least execute one notebook per section and all
special notebooks (stats 08/10 and CV 06).

#### P1.2 Cross-file coverage validation

Validate that every concept has exactly one matching notebook where the section
contract requires it, and that README tables point to existing files. Extend
the validator to papers, RL, MLOps, Modern AI, and Agentic AI instead of only
stats/CV/NLP.

Acceptance criteria:

- No concept-without-notebook or notebook-without-concept mismatches in covered
  sections.
- Every required notebook has the expected cell count and minimum code volume.
- Runtime failures identify section, notebook, cell index, and dependency.

### P1 — Reconcile documentation state

#### P1.3 Historical handoffs and session summaries

Mark `docs/superpowers/plans/HANDOFF-2026-05-30.md` and related session
documents as historical snapshots. Replace contradictory present-tense claims
such as “empty”, “missing”, and “TBD” with a dated note or link to the current
status. Do not erase historical context.

#### P1.4 Plan checklist integrity

Either check off completed tasks in active plans or add a machine-readable
completion block that maps each sub-project to its evidence files and commit.
The active plan must not simultaneously say “all A–K implemented” and leave
the same projects presented as pending without an explicit historical label.

Acceptance criteria: a cold-start agent can identify current pending work from
one status section without inferring it from Git history.

### P2 — Classify workspace artifacts safely

Create an artifact inventory policy, not a deletion script. Classify each
untracked path into:

1. managed worktree state (`.claude/worktrees/`) — preserve;
2. reproducible generated output — preserve until provenance and regeneration
   are documented;
3. source/helper script — preserve unless explicitly retired;
4. disposable cache (`__pycache__`, `.pytest_cache`) — ignore or remove only
   with explicit scope;
5. unknown user work — preserve and ask before changing.

Add narrowly scoped ignore rules only after confirming they do not hide source
files. Never use broad `rm`, `git clean`, or recursive deletion to “clean” the
workspace.

Acceptance criteria: the cleanup report lists every changed/deleted path and
its classification; no untracked source or image is deleted without a
recoverable copy or explicit authorization.

### P2 — Git metadata and delivery workflow

Document and test the read-only `.git` fallback:

- diagnose with `findmnt -T .git`, `stat`, and a direct write probe;
- distinguish mount immutability from a stale `index.lock`;
- if host remount is unavailable, create a temporary clone;
- attach the current worktree only after verifying the clone’s index base;
- reset the temporary clone’s origin from local path to the configured GitHub
  URL;
- stage only an explicit file list;
- review `git show --stat` and `git diff --cached --check`;
- push only after explicit authorization;
- verify the remote ref with `git ls-remote`.

The workflow must never push a local-path origin inherited from `git clone
/path`.

## 4. Non-Goals

- Do not rewrite or delete historical session documents merely because their
  original counts are outdated.
- Do not execute every heavyweight notebook in the default unit-test path if
  that makes local development impractical; use a bounded smoke matrix plus a
  longer CI job.
- Do not clean `.claude/worktrees` without independent confirmation that each
  worktree is stale and no process owns it.
- Do not regenerate missing untracked plots or scripts without knowing their
  intended provenance and output contract.

## 5. Proposed Implementation Order

1. Add the repository-wide inventory and link-audit scripts.
2. Correct root README counts/navigation and classify historical docs.
3. Extend cross-section structural tests and add the bounded notebook smoke
   matrix.
4. Review dependency/runtime failures and repair only reproducible defects.
5. Produce the artifact classification report; make only explicitly approved
   ignore/cleanup changes.
6. Validate, review the exact diff, commit, push, and verify the remote ref.

## 6. Definition of Done

- Root README coverage and counts match the filesystem.
- Active local Markdown links have no unresolved targets.
- Full tests pass, including structural and bounded runtime notebook checks.
- Active plan status is consistent and historical notes are labeled.
- All untracked artifacts have a documented classification.
- No source or prior-run artifact was deleted without evidence and recovery.
- Remote commit and ref are verified after delivery.

## 7. Implementation Status — 2026-09-07

Completed in the current worktree:

- Root README and index counts/navigation were reconciled with the filesystem.
- Active Markdown link auditing and deterministic repair utilities were added.
- The active Markdown audit now reports zero missing files or anchors.
- Structural coverage tests remain green for statistics, computer vision, and
  NLP; the repository-wide suite passes (`409 passed, 3 skipped`).
- Three representative notebook smoke tests were added. They execute when a
  `python3` Jupyter kernel is installed and skip with an explicit reason when
  the environment has no kernels, as in this worktree.
- An artifact inventory documents preserved worktrees, source helpers, plans,
  tests, and disposable caches.

Remaining delivery steps are operational rather than unimplemented fixes:

1. Review the explicit staged file list and exclude `.claude` and caches.
2. Create the next checkpoint through the writable staging clone because the
   workspace `.git` mount is read-only.
3. Obtain explicit push authorization, push the reviewed commit, and verify
   the remote ref with `git ls-remote`.
