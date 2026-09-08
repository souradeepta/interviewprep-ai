# ML/AI System-Design Scorecard

Score each dimension 0–3 after a 45–60 minute mock. A strong answer is
specific about assumptions and trade-offs; a high score is not earned by naming
many technologies.

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Requirements and scale | Missing | Vague | Mostly quantified | Prioritized SLAs, users, and constraints |
| Data and labels | Missing | Model-first | Covers collection/splits | Handles delay, leakage, quality, and privacy |
| Baseline and model choice | Unjustified | Names a model | Gives trade-off | Baseline, fallback, and complexity are explicit |
| Architecture and serving | Incoherent | Components only | End-to-end path | Capacity, freshness, batching, and failure paths |
| Evaluation | Accuracy only | Offline metric | Offline + online | Slices, uncertainty, guardrails, and regression gates |
| Reliability and operations | Ignored | Generic monitoring | Alerts and rollback | Incident response, drift, ownership, and recovery |
| Safety/fairness/cost | Ignored | Afterthought | Some controls | Threat model, human boundary, cost and latency budget |
| Communication | No structure | Overconfident | Clear answer | Assumptions, decisions, alternatives, and follow-ups |

**Red flags:** future-data leakage; no baseline; accuracy for an imbalanced
problem; unbounded retrieval, model, or agent costs; ignoring delayed labels;
no rollback; treating offline gains as causal; unsafe autonomous side effects;
or claiming fixed performance targets without evidence.

Suggested interpretation: 0–8 needs fundamentals, 9–16 is developing,
17–21 is interview-ready for many rounds, and 22–24 is a strong senior answer.
Use the score to choose the next remediation topic, not as a hiring claim.
