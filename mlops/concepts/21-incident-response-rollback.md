# ML Incident Response and Rollback

**TL;DR:** Respond to model incidents by validating impact, containing the blast radius, preserving evidence and using a tested rollback gate tied to measurable service or quality thresholds.

## Prerequisites

[Deployment strategies](16-deployment-strategies.md), [model monitoring](17-monitoring-alert-design.md), and [model versioning](06-model-versioning.md).

## Learning Outcomes

- Distinguish service incidents from delayed quality incidents and data failures.
- Triage severity using evidence, affected population and safety impact.
- Define a rollback trigger that includes an observable threshold and a tested recovery path.
- Preserve model, feature, data and configuration lineage during incident response.

## Core Intuition

Incident response is a controlled decision process under uncertainty. A monitor may show elevated errors, a delayed quality metric may show a regression, or a user report may reveal a problem that telemetry misses. The response should protect users first, keep the affected scope small, then identify cause from preserved evidence. “Roll back” is not a complete plan if the old model cannot read the current feature schema or if the previous artifact is unavailable. Recovery needs the right compatible model, data contract and routing configuration.

## How It Works

Prepare an incident playbook before deployment. It should name severity levels, on-call owners, dashboards, safe fallbacks, rollback commands, escalation paths and customer communication duties. Define service triggers such as request error rate above 2% or p95 latency above 500 ms for a minimum sample; define model-quality triggers separately, since labels may mature later. A meaningful threshold specifies the measurement window, sample floor, affected route and action. Tune it against baseline variation and user impact rather than copying a generic number.

Triage begins by validating the signal. Check telemetry freshness, request volume, feature schema and recent releases. Compare canary and control cohorts, slice outcomes by route or population, and inspect whether the change is an instrumentation bug, upstream data issue, serving fault or model behavior regression. A failing service objective can warrant immediate rollback before quality labels mature. A feature shift without quality impact usually warrants investigation, not automatic model replacement. The implementation uses a simple policy: error rate above 2% or matured quality drop at least 3% recommends rollback. These example limits must be calibrated for a real system.

Containment should be reversible and proportionate. Options include pausing a rollout, routing traffic to the last known-good model, disabling a feature, using a safe non-model fallback, limiting traffic to an affected segment, or rejecting requests that cannot be served safely. Keep a tested path to restore the compatible previous release. If a feature schema changed with the model, a model-only rollback may fail; release model, schema, preprocessing and serving configuration as a compatibility unit. Preserve enough capacity for both versions during a canary or blue-green switch.

Evidence preservation is part of response. Record alert time, model and data versions, code revision, feature contract, affected cohorts, threshold values, request traces or privacy-safe samples, action owner and each decision timestamp. Do not retain sensitive raw payloads beyond policy. Capture before-and-after metrics and annotate dashboards. Without this record, post-incident review cannot distinguish whether rollback helped or whether the metric naturally recovered.

Use a clear incident timeline: detect, acknowledge, assess impact, contain, diagnose, recover, validate, communicate and review. During a high-severity incident, assign an incident lead who coordinates while technical responders investigate. Avoid concurrent uncoordinated changes that make causal attribution impossible. State what is known, what is uncertain and the next update time. Recovery means verified behavior under a bounded canary, not merely a successful deployment command.

After recovery, conduct a blameless review focused on system conditions. Identify detection gaps, missing parity checks, unsafe defaults, rollout guardrails and runbook ambiguity. Convert findings into owners and deadlines. Test rollback drills and failure injection periodically; an unexercised rollback path tends to fail exactly when time is short. For label-delayed systems, keep the incident open or schedule a follow-up quality review after outcomes mature.

Rollback can itself introduce risk. A previous model may have known defects, depend on an older feature schema, or be incompatible with data that has already migrated. Keep the last compatible artifact and a tested fallback, and rehearse the transition while traffic is still on the current release. If direct rollback is unsafe, pause the rollout and route to a validated simpler policy or a human review queue. Document this choice in the runbook rather than improvising a fallback during a live event.

## Trade-offs

| Response | Benefit | Cost or boundary |
|---|---|---|
| Immediate rollback | Fast containment for clear service regression | Can revert a harmless change or hide an upstream fault |
| Pause rollout | Limits further exposure while retaining evidence | Existing canary traffic remains affected |
| Safe fallback | Preserves service when model is unhealthy | May reduce quality or business functionality |
| Segment-specific mitigation | Smaller blast radius | Requires reliable routing and cohort diagnosis |
| Wait for matured labels | Stronger quality evidence | Unsafe for severe immediate harm or service failure |

## Implementation with Python

Run `python3 mlops/implementations/21-incident-response-rollback.py`. The local policy raises severity based on service metrics and affected fraction, and recommends rollback when error rate exceeds 2% or matured quality drops by at least 3%. These thresholds are explicit demo policy inputs, not production defaults.

```python
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("incident", "mlops/implementations/21-incident-response-rollback.py")
incident = module_from_spec(spec); spec.loader.exec_module(incident)
decision = incident.should_rollback(current_error_rate=.03, baseline_error_rate=.005,
                                    quality_drop=0, labels_matured=False)
assert decision["rollback"]
```

## Failure Modes

- Rollback is incompatible with current features: deploy and test model-feature compatibility as a release unit.
- Delayed quality is treated as healthy: mark immature labels unknown and schedule a follow-up review.
- A noisy alert triggers an unsafe action: validate telemetry and sample floor while containing rollout growth.
- Evidence is lost during recovery: snapshot versions, thresholds, cohort and timestamps before changing state.

## Interview Quick Reference

| Question | Expected answer | Follow-up |
|---|---|---|
| What triggers an immediate model rollback? | A verified hard service/safety threshold or matured quality guardrail, with a compatible prior release ready. | Debug an error spike that began before the model rollout. |
| How do you respond when labels are delayed? | Contain on service/data signals and schedule quality assessment when cohorts mature. | Design a two-stage incident policy. |
| What is required for rollback to be safe? | Tested routing, compatible schema/preprocessing, artifact and configuration lineage, and capacity. | Design a rollback drill for a stateful feature migration. |

## Related Concepts

See [deployment strategies](16-deployment-strategies.md), [model monitoring](17-monitoring-alert-design.md), [feature parity](20-feature-parity.md), and [MLOps interview practice](../interview-prep/mlops-interview-questions.md#reliability-governance-and-cost).

## References

[Google SRE: Incident Management](https://sre.google/workbook/incident-response/) and [NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf).
