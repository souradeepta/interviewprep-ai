# Repository Status

**Last reviewed:** 2026-09-17  
**Authoritative source:** the current `master` checkout and this status page

This is the current status page. Dated plans and session handoffs are
historical records unless they explicitly identify themselves as active.

## Current inventory

| Section | Concepts | Notebooks | Implementations |
|---|---:|---:|---:|
| AI fundamentals | 40 | 40 | 40 |
| Machine learning | 40 | 40 | 8 |
| LLM | 44 | 45 | 45 |
| Agentic AI | 64 | 64 | 64 |
| Modern AI | 55 | 55 | 55 |
| MLOps | 16 | 17 (including map) | 15 |
| Reinforcement learning | 20 | 20 | 20 |
| Statistics | 15 | 15 | — |
| Computer vision | 8 | 8 | — |
| NLP | 8 | 8 | — |

## Validation

- Full suite: `446 passed, 10 skipped`
- Active Markdown links: `0 problem(s)`
- Ten deterministic notebook representatives are defined; CI executes all ten.
- The LLM representative is the local concept map because generated LLM
  implementation notebooks may require remote model packages or weights.
- The ten notebook runtime checks skip locally when no `python3` Jupyter
  kernel is installed. The opt-in/CI notebook requirements register one.

## Known intentional limitations

- Runtime execution covers a bounded representative matrix, not every
  heavyweight notebook.
- Some sections have fewer Python implementations than concepts because their
  implementation directory is supplemental rather than a one-to-one
  contract. Modern AI and RL do have one-to-one implementation contracts.
- Historical plans may retain unchecked task lists; they are not current work
  queues unless linked from a newer handoff.

## Next work

1. Periodically refresh interview-topic coverage.
2. Expand the notebook smoke matrix only when dependencies remain deterministic.
3. Add focused implementations where a section's interview track benefits
   from executable practice.
