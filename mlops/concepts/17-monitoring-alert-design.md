# Model Monitoring and Alert Design

**TL;DR:** Monitor service health, data validity and delayed model quality with alerts tied to an owner and a reversible action; thresholds need minimum evidence and an explicit measurement window.

## Prerequisites

[Model evaluation](12-evaluation-metrics.md), [data testing](10-data-testing.md), and basic service-level objectives.

## Learning Outcomes

- Separate immediate service and input-quality signals from delayed outcome quality.
- Calculate windowed error rate, latency percentile and labeled quality summary.
- Define an alert threshold, minimum sample count, owner and first response.
- Explain how false positives, alert fatigue and missing labels affect alert policy.

## Core Intuition

A smoke alarm is useful because it signals a specific hazard and prompts an action. A dashboard line without an owner or response is closer to a thermometer left in a cupboard. Model monitoring is successful when the system can distinguish a model-quality concern from a serving or data failure, page the right person with enough context, and keep the incident bounded while evidence is gathered. More metrics do not automatically make better monitoring; each signal should answer a decision question.

## How It Works

Start with an operational contract. Define the model's service-level indicators (SLIs), the objective for each (SLO), the observation window and the action when the objective is missed. Typical immediate SLIs include request error rate, p95/p99 latency, timeout rate, prediction volume, feature freshness, schema violations and missing/default rates. Quality metrics such as loss, calibration, false-negative rate or business conversion usually arrive later because labels are delayed. They should be monitored on matured cohorts and not mixed with leading service metrics as though they were available at request time.

An alert should have at least four parts: a threshold, a minimum evidence requirement, a duration or repeated-window rule, and a response. For example, alert when a 5-minute request window contains at least 200 requests and the error rate exceeds 2% for two consecutive windows. The implementation in this lesson encodes a per-window sample floor and explicit error, latency and optional quality limits. It deliberately does not add persistence across windows; a production evaluator should do so where one noisy window would cause a page. The threshold is policy input, not a universal default.

Use separate alert classes. A page means a user-impacting condition needs immediate response. A ticket means an owner should investigate during working hours. A dashboard annotation or report may be sufficient for a shift that has no known user impact. Prediction-distribution movement alone is often a diagnostic signal, not a page: legitimate seasonality, campaign launches and upstream product changes can move inputs without causing harmful quality loss. Page when a metric is tied to a user-facing objective or a hard safety boundary.

Windows and aggregation matter. A mean latency can hide a small number of severe stalls; percentiles describe a chosen tail but need enough samples to be stable. Error rate can be dominated by one tenant or route, so aggregate and slice by meaningful dimensions. Avoid unbounded high-cardinality labels such as raw user IDs. For delayed labels, publish cohort maturity and coverage; do not declare model health from the subset whose outcomes arrived first if label delay correlates with outcomes.

Alert evaluation must handle missingness explicitly. If quality labels are absent, quality should be `unknown`, not zero and not healthy. If the traffic sample is below the minimum count, report insufficient evidence rather than silently passing. If telemetry collection itself fails, a monitoring-system alert should fire independently from the model alert. This prevents a broken observer from creating false confidence.

Alert messages should include the evaluated window, sample count, current and allowed values, affected model/version, dashboard or trace link, and the first safe action. A page that says only “drift detected” forces responders to rediscover context under pressure. Route alerts to an owning team and document escalation when the primary owner does not respond. Review noisy and missed alerts after incidents; changing a threshold is a policy change and should be versioned just like a model release.

## Trade-offs

| Choice | Benefit | Cost or boundary |
|---|---|---|
| Short window | Detects incidents quickly | Noisy rates and unstable percentiles at low volume |
| Long window | More stable aggregation | Slow response and can hide brief severe failures |
| Fixed threshold | Easy to explain and test | Seasonal baselines may create false alarms |
| Dynamic baseline | Adapts to expected cycles | Can normalize a slow harmful regression |
| Quality alert | Directly measures model outcomes | Requires mature, representative labels |

Prefer an SLO-bound threshold for latency and errors. Use an adaptive baseline only as a diagnostic until a measured user-impact threshold is established. Never auto-retrain or rollback solely because an unlabeled feature distribution moved.

## Implementation with Python

Run `python3 mlops/implementations/17-monitoring-alert-design.py`. The standard-library module summarizes a deterministic request window and evaluates a configurable minimum sample count and thresholds. The policy example pages when error rate exceeds 2% with at least 20 observations. Change the window or limits and observe why the result changes.

```python
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("monitor", "mlops/implementations/17-monitoring-alert-design.py")
monitor = module_from_spec(spec); spec.loader.exec_module(monitor)
rows = [{"success": True, "latency_ms": 40} for _ in range(20)]
metrics = monitor.summarize_window(rows)
assert monitor.evaluate_alert(metrics, min_requests=20)["state"] == "healthy"
```

## Failure Modes

- Too few samples create noisy pages: require a minimum count and inspect confidence or repeated windows before paging.
- Delayed labels are treated as current truth: track cohort maturity and label coverage; publish quality only on comparable matured cohorts.
- Aggregate monitoring hides one broken route: slice by stable, meaningful dimensions and cap metric cardinality.
- Telemetry outage looks like model health: monitor event ingestion and evaluator freshness as separate SLIs.

## Interview Quick Reference

| Question | Expected answer | Follow-up |
|---|---|---|
| What should page on-call for a ranking model? | User-impact SLIs such as errors, tail latency and a validated quality guardrail with sufficient sample. | How would you design the minimum sample and persistence rule? |
| Labels arrive a week late. How do you monitor quality now? | Use service/data leading indicators now and quality on matured cohorts with coverage. | Debug an apparent quality jump caused by partial labels. |
| Feature PSI crossed a threshold. Do you roll back? | Treat it as a diagnostic; inspect data and quality impact before a release action. | Design an alert that distinguishes a campaign shift from a pipeline break. |

## Related Concepts

See [model testing](09-model-testing.md), [evaluation metrics](12-evaluation-metrics.md), [deployment strategies](16-deployment-strategies.md), and [MLOps interview practice](../interview-prep/mlops-interview-questions.md#evaluation-release-and-serving).

## References

[Google SRE: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) and [NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf).
