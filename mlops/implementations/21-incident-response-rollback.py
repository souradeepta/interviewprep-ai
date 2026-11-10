"""Incident severity triage and measurable rollback decision policy."""
def assess_incident(*, error_rate, latency_p95_ms, quality_drop, affected_fraction, labels_matured):
    if min(error_rate,latency_p95_ms,quality_drop,affected_fraction)<0 or affected_fraction>1:
        raise ValueError("invalid incident metric")
    evidence={"service":error_rate>.02 or latency_p95_ms>500,
              "quality":bool(labels_matured and quality_drop>=.03),
              "blast_radius":affected_fraction}
    severity="SEV1" if evidence["service"] and affected_fraction>=.25 else (
        "SEV2" if evidence["service"] or evidence["quality"] else "SEV3")
    return {"severity":severity,"evidence":evidence}

def should_rollback(*, current_error_rate, baseline_error_rate, quality_drop, labels_matured,
                    error_rate_limit=.02, quality_drop_limit=.03):
    error_breach=current_error_rate>error_rate_limit
    quality_breach=bool(labels_matured and quality_drop>=quality_drop_limit)
    return {"rollback":error_breach or quality_breach,
            "reasons":[name for name,hit in (("error_rate",error_breach),("quality",quality_breach)) if hit],
            "baseline_error_rate":baseline_error_rate}

if __name__ == "__main__":
    print(assess_incident(error_rate=.08,latency_p95_ms=700,quality_drop=0,affected_fraction=.4,labels_matured=False))
    print(should_rollback(current_error_rate=.08,baseline_error_rate=.005,quality_drop=0,labels_matured=False))
