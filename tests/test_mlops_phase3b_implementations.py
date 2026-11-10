"""Focused deterministic tests for the new MLOps operational examples."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    path = ROOT / f"mlops/implementations/{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module

def test_monitoring_suppresses_low_sample_alerts_and_trips_threshold():
    m = load("17-monitoring-alert-design")
    records = [{"success": i != 0, "latency_ms": 10 + i, "quality": .9} for i in range(20)]
    stats = m.summarize_window(records)
    assert m.evaluate_alert(stats, min_requests=21)["state"] == "insufficient_traffic"
    assert m.evaluate_alert(stats, error_rate_limit=.04)["reasons"] == ["error_rate"]

def test_drift_distinguishes_feature_distribution_and_labeled_rate_shift():
    m = load("18-drift-detection")
    base = [.1, .2, .3, .4]
    assert m.psi(base, base, [0, .25, .5, 1])["psi"] == 0
    moved = m.psi(base, [.7, .8, .9, .95], [0, .25, .5, 1])
    assert moved["psi"] > 1
    assert m.label_rate_shift([0, 0, 0, 1], [1, 1, 1, 0])["absolute_shift"] == .5

def test_pipeline_order_backfill_bound_and_idempotent_resume():
    m = load("19-pipeline-orchestration")
    tasks = ["extract", "validate", "train", "publish"]
    deps = {"validate": ["extract"], "train": ["validate"], "publish": ["train"]}
    first = m.run_partition(7, tasks, deps)
    assert [task for task, _ in first] == tasks
    assert m.run_partition(7, tasks, deps, completed=first) == []
    assert m.backfill_partitions(4, 6, 3) == [4, 5, 6]

def test_feature_parity_separates_value_mismatch_from_staleness():
    m = load("20-feature-parity")
    online = {"value": 3.0, "event_time": "2026-01-01T00:00:00", "observed_at": "2026-01-01T00:01:00"}
    same = {"offline": {"value": 3.0}, "online": online}
    changed = {"offline": {"value": 4.0}, "online": online}
    assert m.audit_rows([same], max_age_seconds=120)["decision"] == "pass"
    assert m.audit_rows([changed], max_age_seconds=120)["mismatch_rate"] == 1
    assert m.audit_rows([same], max_age_seconds=30)["stale_rate"] == 1

def test_incident_rollback_uses_service_and_matured_quality_thresholds():
    m = load("21-incident-response-rollback")
    assert m.should_rollback(current_error_rate=.03, baseline_error_rate=.005,
                             quality_drop=0, labels_matured=False)["rollback"]
    assert not m.should_rollback(current_error_rate=.01, baseline_error_rate=.005,
                                 quality_drop=.2, labels_matured=False)["rollback"]
    assert m.should_rollback(current_error_rate=.01, baseline_error_rate=.005,
                             quality_drop=.03, labels_matured=True)["rollback"]

def test_lineage_validation_and_fingerprint_are_stable():
    m = load("22-governance-lineage")
    record={field:field+"-value" for field in m.REQUIRED_FIELDS}
    assert m.validate_lineage(record)["valid"]
    assert m.fingerprint(record)==m.fingerprint(dict(reversed(list(record.items()))))
    assert not m.validate_lineage({})["valid"]

def test_security_defaults_to_deny_and_redacts_sensitive_fields():
    m = load("23-security-privacy-controls")
    grants={("oncall","read_metrics","aggregate")}
    assert m.authorize(role="oncall",action="read_metrics",resource_class="aggregate",grants=grants)["allowed"]
    assert not m.authorize(role="oncall",action="read_raw",resource_class="row",grants=grants)["allowed"]
    assert m.redact({"email":"x@example.invalid","score":.4})["email"]=="[REDACTED]"
    assert m.retention_action(20,30)=="retain"
    assert m.retention_action(20,10,legal_hold=True)=="retain_under_hold"

def test_cost_capacity_planning_respects_headroom_and_budget_reserve():
    m = load("24-cost-capacity-reliability")
    assert m.capacity_plan(100,25)["workers"]==8
    estimate=m.monthly_cost(100000,.8,fixed_cost=500)
    assert estimate==2900
    assert m.budget_decision(estimate,3000)["decision"]=="review_capacity_or_cost"
