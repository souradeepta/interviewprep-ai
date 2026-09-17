# Repository Gap Remediation Specification

**Date:** 2026-09-17  
**Status:** Implemented in the current checkpoint

## Objective

Make the repository truthful, reproducible, and continuously verifiable after
the 2026-09-17 gap audit. Preserve historical session context while ensuring
active documentation and CI describe the artifacts that actually exist.

## Scope

1. Reconcile Modern AI concept, notebook, and implementation numbering.
2. Correct stale section and root coverage claims.
3. Execute the complete repository test suite and expanded representative
   notebook matrix in CI, including LLM and previously uncovered tracks.
4. Run notebook smoke validation on pushes and pull requests as well as
   manually.
5. Publish a minimal pinned dependency set for the default test suite.
6. Mark stale session reports as historical snapshots.
7. Add regression coverage for numbered concept/notebook/implementation
   parity in sections that promise those pairings.
8. Publish one current status page that distinguishes active state from
   historical plans.
9. Fill the three deterministic Agentic AI implementation gaps for MCTS,
   bounded autonomy, and knowledge-graph traversal.

## Decisions

- The duplicate `modern-ai/concepts/26-speculative-decoding.md` is removed;
  the complete surviving implementation is the numbered 28 concept and its
  matching notebook/Python module. Modern AI therefore has 55 paired items.
- Heavy model-serving dependencies remain excluded from the default test
  requirements and notebook smoke workflow.
- Historical documents are labeled, not rewritten or deleted.
- The ten representative notebooks remain bounded smoke coverage; full
  execution of every notebook is explicitly out of scope for default CI.
- The LLM smoke representative is the dependency-free concept map; model-based
  LLM notebooks remain opt-in because they require packages and weights.

## Acceptance criteria

- Modern AI has identical numeric IDs for concepts, notebooks, and
  implementations, with no duplicate concept IDs.
- Root, Modern AI, and MLOps README counts match the filesystem.
- CI executes the full repository suite, including all ten representative
  notebooks, and runs on push, pull request, and manual dispatch.
- `requirements-dev.txt` is sufficient for the default test suite.
- Stale MLOps and notebook validation reports are visibly historical.
- `docs/REPOSITORY_STATUS.md` is the single current inventory/status reference.
- Agentic AI has deterministic implementations for concepts 10, 18, and 23,
  bringing its implementation inventory to 64.
- Full tests and diff checks pass.
