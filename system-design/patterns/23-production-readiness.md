# Production Readiness

## TL;DR
Checklist before deploying ML to production: versioning, monitoring, fallbacks, SLA, disaster recovery, documentation, testing.

## Core Intuition
ML experiments work, but production is hard. Checklist prevents fires.

## How It Works
**Pre-deployment checklist:**
- Model versioning: can identify, rollback
- Monitoring: latency, accuracy, drift alerts
- Fallback: if model fails, have backup
- SLA defined: latency, availability targets
- Testing: load test, edge cases
- Documentation: team knows what it does
- Disaster recovery: plan if system down

**Post-deployment:**
- Monitor continuously
- Respond to alerts
- Retraining schedule
- Regular audits

## Common Mistakes / Gotchas
- **Ship and forget:** no monitoring → issues go undetected
- **No fallback:** service fails → bad user experience
- **Poor documentation:** knowledge leaves with engineer

## Interview Q&A

**Q: What constitutes an ML system being "production ready" and who decides?**
A: Production readiness is multidimensional: functional (correct predictions on held-out test set meeting accuracy SLA), operational (latency/throughput SLAs met under load), reliability (99th percentile availability, graceful degradation under failure), observability (monitoring, alerting, and runbooks in place), compliance (security review, data governance, explainability documentation), and rollback capability (can revert in <10 minutes). Decision: a readiness review with representatives from ML, engineering, product, and security—not just ML team sign-off.

**Q: How do you load test an ML model serving endpoint before production launch?**
A: Use realistic traffic patterns: real request payload sizes from historical data, real traffic shape (not just sustained constant load—include spikes). Test: sustained load at 1.5x expected peak, burst test (2x peak for 60 seconds), ramp test (gradual increase to identify breaking point), soak test (sustained load for 4 hours to detect memory leaks or resource exhaustion). Measure: P50/P95/P99 latency, error rate, GPU/CPU utilization, memory growth. Automate load tests as part of the staging environment.

**Q: What are the most common production readiness failures that cause outages after launch?**
A: Inadequate load testing: system performs fine at 10 RPS but fails at 100 RPS. Missing error handling: model serves 500s on edge-case inputs that weren't in test data. Insufficient monitoring: failure isn't detected for hours because no alert was configured. No runbook: on-call engineer doesn't know how to respond to the model's specific failure modes. Underestimated startup time: deployment causes outage because new pods take 3 minutes to become ready but the deployment strategy assumed 30 seconds. Each of these is preventable with a comprehensive readiness checklist.

**Q: How do you document a model's known limitations for production users?**
A: Create a model card (standardized by Google/Hugging Face) that includes: intended use cases and out-of-scope uses, performance metrics by demographic group and data slice, known failure modes with examples, data sources and training methodology, recommended minimum input requirements, and edge cases to handle explicitly. Store the model card in the model registry and require its completion as part of the promotion gate. Communicate limitations to consuming teams—they need to build appropriate safeguards in their applications.

**Q: What security review elements are specific to ML systems vs. standard software?**
A: ML-specific security concerns: model inversion attacks (extracting training data from model outputs), adversarial examples (inputs crafted to fool the model), data poisoning (attacking the training pipeline to embed backdoors), model stealing (using API queries to replicate the model), and prompt injection (for LLM-based systems). Review: access controls on training data and model artifacts, rate limiting on inference endpoints, input validation to detect adversarial patterns, monitoring for unusual query patterns that may indicate model extraction attacks.

## Interview Quick-Reference
**Production ready?** Versioning, monitoring, fallbacks, SLA, testing, docs, disaster recovery.

## Related Topics
- [Monitoring & Observability](16-monitoring-and-observability.md)
- [Disaster Recovery](31-disaster-recovery.md)

## Resources
- [ML Systems in Production](https://www.oreilly.com/library/view/machine-learning-systems-design/9781492072929/)
