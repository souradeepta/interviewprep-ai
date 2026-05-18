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

## Interview Q&A

**Q: What is the most common reason ML models fail in production?**
A: Training-serving skew: the model was trained on data preprocessed differently than production data—different scaling, encoding, or feature order. Other top causes: data drift (distribution of production inputs changes), inadequate monitoring (failures go undetected), and insufficient testing of edge cases. The infrastructure to catch these issues (feature stores, monitoring, CI/CD) is what MLOps provides.

**Q: When should a company invest in MLOps infrastructure vs. keeping ML simple?**
A: Invest when: you have 3+ models in production, models need to be retrained regularly, multiple teams contribute to ML pipelines, or model failures have significant business impact. Keep simple when: you have 1-2 models that rarely change, the models are low-stakes, or you're still validating the ML use case. Over-engineering MLOps for one model is wasteful; under-engineering for 10 models creates technical debt that slows the entire team.

**Q: How does MLOps differ from DevOps and what does that mean for tooling?**
A: DevOps handles code and infrastructure; MLOps handles code + data + models + experiments. Unique MLOps challenges: data versioning (code versioning tools don't handle large datasets), experiment tracking (many hyperparameter configurations to compare), model validation (both technical metrics and business metrics), and data drift (models degrade without code changes). Tooling must address all four: MLflow/W&B for experiments, DVC for data versioning, Evidently for drift, and custom model cards for documentation.

**Q: What does a minimal viable MLOps setup look like?**
A: Minimum: (1) version control for code and model artifacts, (2) automated retraining pipeline with data validation, (3) staging environment for model validation before production, (4) basic monitoring of prediction distribution and key business metrics. This can be built with MLflow + GitHub Actions + simple dashboards in a few weeks. Don't let perfect MLOps be the enemy of shipping—start minimal and add components as team pain points emerge.

**Q: What is the ROI calculation for investing in MLOps infrastructure?**
A: Quantify: engineer hours spent debugging production issues (monitoring reduces this), time to retrain and deploy models (CI/CD reduces this), models that failed silently before monitoring caught them (value of prevented failures). Typical improvements: 3-5x faster model deployment, 50-70% reduction in production incidents, 2-3x more models maintained per engineer. The investment pays off when the cost of infrastructure is less than the cost of the manual work it replaces, usually at 3-5 production models.

## Interview Quick-Reference
**MLOps?** Entire ML system: pipelines, training, serving, monitoring, retraining. Automation is key.

## Related Topics
- [Data Pipelines](02-data-pipelines.md) — ETL part
- [Monitoring & Observability](16-monitoring-and-observability.md) — detection

## Resources
- [Hidden Technical Debt in Machine Learning](https://arxiv.org/abs/1503.04811)
