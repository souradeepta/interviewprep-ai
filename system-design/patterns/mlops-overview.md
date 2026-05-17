# MLOps Overview

## TL;DR
ML engineering: not just training. Covers data pipelines, model serving, monitoring, retraining, versioning. Enables reliable, scalable production ML.

## Core Intuition
ML systems are fragile. Code, data, models all evolve. MLOps manages this: CI/CD for ML, automated retraining, monitoring.

## How It Works
```
Data → Pipeline → Training → Evaluation → Serving → Monitoring
                                              ↓
                                        Drift detected?
                                              ↓
                                          Retrain
```

**Key components:**
- Data pipeline (ETL)
- Feature store
- Model registry
- Experiment tracking
- Automated retraining
- Monitoring + alerting
- Deployment orchestration

## Common Mistakes / Gotchas
- **Manual processes:** retraining by hand is error-prone. Automate.
- **No versioning:** can't rollback. Version code, data, models.
- **Missing monitoring:** find issues in production too late.

## Interview Quick-Reference
**MLOps?** Entire ML system: pipelines, training, serving, monitoring, retraining. Automation is key.

## Related Topics
- [Data Pipelines](data-pipelines.md) — ETL part
- [Monitoring & Observability](monitoring-and-observability.md) — detection

## Resources
- [Hidden Technical Debt in Machine Learning](https://arxiv.org/abs/1503.04811)
