# Workspace Artifact Inventory — 2026-09-07

This inventory records the untracked workspace state observed during the gap
review. It is intentionally a preservation record, not a cleanup instruction.

| Location | Classification | Decision |
|---|---|---|
| `.claude/worktrees/**` | Managed agent worktree state | Preserve; do not clean without ownership/staleness confirmation. |
| `.claude/settings.local.json` | Local tool configuration | Preserve; do not publish as repository content. |
| `docs/superpowers/plans/*.md` | Prior-run planning/source documentation | Preserve and include only the explicitly reviewed plans. |
| `docs/superpowers/specs/*.md` | Current review specifications | Preserve and track the gap review and this inventory. |
| `scripts/gen_nlp_notebooks.py` | Recovered source/helper script | Preserve; provenance was confirmed from a retained agent worktree. |
| `scripts/audit_markdown_links.py` | New repository validation utility | Track; standard-library-only and safe to run read-only. |
| `scripts/repair_*.py` | One-time link repair utilities | Preserve for review/reproducibility; do not run automatically in CI. |
| `tests/test_*.py` | Repository regression tests | Track the new tests; preserve existing tests. |
| `scripts/__pycache__/**`, `tests/__pycache__/**` | Disposable generated caches | Do not stage or publish; removal is optional and out of scope. |

No untracked source or image was deleted during this implementation pass. The
previously investigated root-level RL images and two unrecoverable helper
scripts remain absent because no copy or provenance was found; they were not
regenerated speculatively.

Before delivery, stage an explicit file list. Do not use `git add -A` while
the managed `.claude` worktrees and generated caches are present.
