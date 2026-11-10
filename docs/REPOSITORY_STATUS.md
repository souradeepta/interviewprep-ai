# Repository Status

**Last reviewed:** 2026-11-10
**Inventory command:** `python3 scripts/report_repository_inventory.py`
**Curriculum audit:** `python3 scripts/audit_curriculum.py --format json`

This page distinguishes the on-disk instructional inventory from the manifest
audit. Counts below are recursive instructional files by directory, excluding
folder index READMEs. Counts do not imply that each artifact has passed the
content standard. Manifest lessons are the audited denominator; tracks with
some manifest entries may still contain unindexed artifacts.

For baseline records with empty prerequisites, the relationship is provisional.
This includes AI, RL, CV, NLP, and lessons newly inventoried from Statistics,
MLOps, and Modern AI. Empty prerequisites mean they have not yet been reviewed,
not that a lesson is suitable for a learner with no background. Resolve those
links before promoting records to `required`.

## Current instructional inventory

| Track | Concepts | Notebooks | Python implementations | Manifest lessons |
|---|---:|---:|---:|---:|
| AI fundamentals (`ai`) | 40 | 40 | 40 | 40 baseline |
| Machine learning (`ml`) | 40 | 40 | 0 | 40 |
| LLM (`llm`) | 44 | 45 | 45 | 44 |
| Agentic AI (`agentic-ai`) | 64 | 64 | 64 | 64 |
| Modern AI (`modern-ai`) | 55 | 55 | 55 | 55 (10 required, 45 baseline) |
| MLOps (`mlops`) | 24 | 25 | 24 | 24 (8 required, 16 baseline) |
| Reinforcement learning (`rl`) | 20 | 20 | 20 | 20 baseline |
| Statistics (`stats`) | 21 | 21 | 0 | 21 (6 required, 15 baseline) |
| Computer vision (`cv`) | 8 | 8 | 0 | 8 baseline |
| NLP (`nlp`) | 8 | 8 | 0 | 8 baseline |

Additional advertised families include papers, system design patterns,
architecture reviews, coding guides, cheat sheets, and projects. Their file
counts are reported separately by the inventory command; they are not
represented as lesson tracks in the current manifest. These counts establish
presence only. They do not establish quality or that the README descriptions
match the contents.

## Validation status

- Latest completed full suite: `481 passed, 20 skipped` after the LLM batch
  committed as `6a2f1cc`.
- Latest LLM curriculum audit: 44 required lessons, zero errors or warnings.
- Latest active Markdown link audit: zero problems.
- The LLM batch's five changed notebooks executed successfully in cell order.
- The general notebook execution tier and skip ledger are not yet complete.
- Current generated tier ledger: 361 tracked notebooks; Tier 1 structural-only
  2, Tier 2 clean-kernel candidates 275, Tier 3 resource/dependency candidates
  84. The Tier 3 reasons come from executable imports/calls and require
  human confirmation before these are accepted as exclusions.
- All 361 tracked notebooks now pass nbformat v4.5 structural validation.
- The 275 Tier 2 notebooks skip locally when no `python3` Jupyter kernel is
  installed; CI installs the pinned kernel and executes them.
- Four maintained Tier 3 representative paths have offline stub/fixture tests
  in `tests/test_tier3_representatives.py`; these test provider setup, local
  retrieval, model wrapper shape handling, and mocked HTTP retries. Other Tier
  3 paths still rely on structural checks and proposed per-notebook alternatives.
- Manifest audit coverage is ten instructional tracks with 324 lessons: 172
  required and 152 baseline. The four newly inventoried tracks (AI
  fundamentals, RL, CV, NLP) and the remaining Statistics, MLOps, and Modern
  AI concepts remain baseline pending content review.
- Terra reviewed the curriculum-map agreement batch. Unsupported practice
  references were removed from 45 baseline Modern AI records; focused map and
  manifest checks pass.
- Full local suite after this batch: `490 passed, 275 skipped`; skips are
  Tier 2 notebook executions without a locally registered Python kernel.

## Current program work

The active repository improvement plan is
[`2026-11-09-terra-luna-repository-improvement-plan.md`](superpowers/plans/2026-11-09-terra-luna-repository-improvement-plan.md).
Work proceeds through repository inventory, curriculum architecture, learner
journey pilots, bounded content reviews, notebook execution tiers, and
documentation reconciliation. This report is a baseline inventory, not a
claim that those phases are complete.
