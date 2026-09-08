# ML System Design Roadmap

## Who This Is For
Engineers preparing for ML/AI system design interview rounds. Assumes you can already discuss
ML models and have some engineering background. System design rounds typically last 45–60 minutes
and expect you to design a complete ML-powered product feature.

---

## Phase 1 — Foundations (Beginner)
**Goal:** Learn the ML system design framework. Understand core infrastructure patterns.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [System Design Framework](../system-design/interview-prep/system-design-framework.md)
- [ ] [Online vs. Batch Inference](../system-design/patterns/07-online-vs-batch-inference.md)
- [ ] [Feature Store](../system-design/patterns/03-feature-store.md)
- [ ] [Data Pipelines](../system-design/patterns/02-data-pipelines.md)
- [ ] Practice: [System Design Framework](../system-design/interview-prep/system-design-framework.md) — Q1–Q5 (use framework, don't worry about depth yet)

**Phase 1 exit check:**
- Can you describe the ML system design framework in under 2 minutes?
- Can you explain the difference between online and batch feature computation?

---

## Phase 2 — Case Studies (Intermediate)
**Goal:** Practice designing complete ML systems end-to-end.
**Estimated time:** 2–3 weeks at 10 hrs/week

- [ ] [Model Registry](../system-design/patterns/04-model-registry.md)
- [ ] [A/B Testing](../system-design/patterns/14-ab-testing.md)
- [ ] [MLOps Overview](../system-design/patterns/01-mlops-overview.md)
- [ ] [ML Case Studies — Recommendation, Search, and Fraud](../ml/interview-prep/case-studies.md)
- [ ] Practice: [System Design Framework](../system-design/interview-prep/system-design-framework.md) — Q6–Q15

**Phase 2 exit check:**
- Can you design a recommendation system end-to-end in 45 minutes?
- Can you explain how to run an A/B test for an ML model and detect interference effects?

---

## Phase 3 — Advanced + LLM/Agentic Design (Advanced)
**Goal:** Design LLM-powered systems and agentic platforms under interview conditions.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [ML Case Studies — Moderation and Ads](../ml/interview-prep/case-studies.md)
- [ ] [LLM System Design Patterns](../system-design/README.md)
- [ ] [LLM Serving Pattern](../system-design/patterns/07-online-vs-batch-inference.md)
- [ ] [Agentic Deployment Patterns](../agentic-ai/concepts/56-agent-deployment-patterns.md)
- [ ] Practice: [System Design Framework](../system-design/interview-prep/system-design-framework.md) — Q16–Q25

**Phase 3 exit check:**
- Can you design a production RAG system with latency SLAs, observability, and fallbacks?
- Can you design an ad click prediction system at 1M QPS?

---

## Interview Readiness Checklist
- [ ] Memorized the ML system design framework (can recite structure from memory)
- [ ] Completed at least 3 full case study mocks (recommendation, fraud, ranking)
- [ ] Can design a RAG system with production concerns addressed
- [ ] Completed 15+ system design questions in simulation format

---

## Suggested Weekly Schedule

| Week | Focus | Files |
|------|-------|-------|
| 1 | Framework + core patterns | system-design-framework.md, online-vs-batch-inference.md, feature-store.md |
| 2 | Case studies: recommendation + ranking | recommendation-system.md, search-ranking.md |
| 3 | Case studies: fraud + moderation + ads | fraud-detection.md, content-moderation.md, ads-click-prediction.md |
| 4 | LLM + agentic system design | rag-system-design.md, llm-serving-design.md, agentic-system-design.md |
