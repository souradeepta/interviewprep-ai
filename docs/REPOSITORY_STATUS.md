# Repository Status

**Last reviewed:** 2026-11-09
**Inventory command:** `python3 scripts/report_repository_inventory.py`
**Curriculum audit:** `python3 scripts/audit_curriculum.py --format json`

This page distinguishes the on-disk instructional inventory from the manifest
audit. Counts below are recursive instructional files by directory, excluding
folder index READMEs. Counts do not imply that each artifact has passed the
content standard. Manifest lessons are the audited denominator; tracks with
some manifest entries may still contain unindexed artifacts.

For the 76 baseline AI, RL, CV, and NLP records, empty prerequisites are
provisional inventory roots. They mean prerequisite relationships have not yet
been reviewed, not that the lesson is suitable for a learner with no
background. Resolve those links before promoting the records to `required`.

## Current instructional inventory

| Track | Concepts | Notebooks | Python implementations | Manifest lessons |
|---|---:|---:|---:|---:|
| AI fundamentals (`ai`) | 40 | 40 | 40 | 40 baseline |
| Machine learning (`ml`) | 40 | 40 | 0 | 40 |
| LLM (`llm`) | 44 | 45 | 45 | 44 |
| Agentic AI (`agentic-ai`) | 64 | 64 | 64 | 64 |
| Modern AI (`modern-ai`) | 55 | 55 | 55 | 10 |
| MLOps (`mlops`) | 24 | 25 | 24 | 8 |
| Reinforcement learning (`rl`) | 20 | 20 | 20 | 20 baseline |
| Statistics (`stats`) | 21 | 21 | 0 | 6 required |
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
- Manifest audit coverage is ten instructional tracks with 248 lessons: 172
  required lessons across ML, Statistics, Modern AI, MLOps, LLM, and Agentic
  AI; plus 76 baseline lessons across AI fundamentals, RL, CV, and NLP.

## Current program work

The active repository improvement plan is
[`2026-11-09-terra-luna-repository-improvement-plan.md`](superpowers/plans/2026-11-09-terra-luna-repository-improvement-plan.md).
Work proceeds through repository inventory, curriculum architecture, learner
journey pilots, bounded content reviews, notebook execution tiers, and
documentation reconciliation. This report is a baseline inventory, not a
claim that those phases are complete.
