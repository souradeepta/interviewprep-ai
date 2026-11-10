"""Compare offline and online feature values with freshness and tolerance checks."""
from datetime import datetime

def compare_feature(offline,online,*,abs_tolerance=1e-9,max_age_seconds=300):
    """Inputs carry value and ISO-8601 event_time. NaN/null mismatch is never silently equal."""
    off=offline.get("value"); on=online.get("value")
    if off is None or on is None: equal=off is on
    else: equal=abs(float(off)-float(on))<=abs_tolerance
    event=datetime.fromisoformat(online["event_time"]); now=datetime.fromisoformat(online["observed_at"])
    age=(now-event).total_seconds()
    return {"values_equal":equal,"age_seconds":age,"fresh":0<=age<=max_age_seconds,
            "parity":equal and 0<=age<=max_age_seconds}

def audit_rows(rows,*,max_mismatch_rate=.01,max_age_seconds=300):
    if not rows: raise ValueError("at least one row is required")
    results=[compare_feature(r["offline"],r["online"],max_age_seconds=max_age_seconds) for r in rows]
    mismatch=sum(not r["values_equal"] for r in results)/len(results)
    stale=sum(not r["fresh"] for r in results)/len(results)
    return {"rows":len(rows),"mismatch_rate":mismatch,"stale_rate":stale,
            "decision":"block" if mismatch>max_mismatch_rate or stale>max_mismatch_rate else "pass"}

if __name__ == "__main__":
    row={"offline":{"value":3.0},"online":{"value":3.0,"event_time":"2026-01-01T00:00:00","observed_at":"2026-01-01T00:01:00"}}
    print(audit_rows([row],max_age_seconds=120))
