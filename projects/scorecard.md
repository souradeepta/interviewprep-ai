# Shared Capstone Scorecard

Score each project against its own rubric. This common review sheet keeps feedback focused on evidence and records where the project rubric applies different point weights.

| Dimension | Review question | Typical evidence |
|---|---|---|
| Engineering | Is the implementation bounded, readable, and reproducible? | Clear interfaces, deterministic data, validation, runnable command. |
| Measurement | Do metrics answer the stated question with valid denominators and baselines? | Baseline comparison, slice analysis, uncertainty, metric limitations. |
| Failure handling | Are edge cases detected and outcomes safe? | Tests for malformed, missing, repeated, stale, or ambiguous inputs; explicit fallback. |
| Communication | Can another engineer understand choices and limitations? | README/report with assumptions, trade-offs, result, and next step. |
| Interview debrief | Can the learner explain design choices under follow-up? | 5-minute summary, one alternative, one failure mode, one improvement. |

## Scoring guidance

- Award points for observed evidence, not intent or tool choice.
- A failed success command earns no credit for the behavior it is meant to verify until repaired.
- A limitation stated clearly is better than a broad unsupported claim.
- A project is complete at 70/100 only when its executable success criterion passes; use the remaining points to distinguish quality and judgment.
- Record the highest-impact defect and one next experiment in the debrief.
