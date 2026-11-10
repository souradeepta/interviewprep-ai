"""Windowed model-service metrics and deterministic threshold alerting."""
from statistics import mean

def summarize_window(records):
    """Records have success, latency_ms and optional quality; missing quality is ignored."""
    if not records: raise ValueError("window must contain records")
    errors=sum(not bool(r["success"]) for r in records)/len(records)
    latencies=sorted(float(r["latency_ms"]) for r in records)
    p95=latencies[min(len(latencies)-1, int(.95*len(latencies)))]
    quality=[float(r["quality"]) for r in records if r.get("quality") is not None]
    return {"count":len(records),"error_rate":errors,"p95_latency_ms":p95,
            "mean_quality":mean(quality) if quality else None}

def evaluate_alert(metrics, *, min_requests=20, error_rate_limit=.02, p95_latency_limit_ms=500,
                   min_quality=None):
    """Alert only with enough traffic; limits are explicit SLO-style policy inputs."""
    if metrics["count"] < min_requests: return {"state":"insufficient_traffic","reasons":[]}
    reasons=[]
    if metrics["error_rate"] > error_rate_limit: reasons.append("error_rate")
    if metrics["p95_latency_ms"] > p95_latency_limit_ms: reasons.append("p95_latency")
    if min_quality is not None and metrics["mean_quality"] is not None and metrics["mean_quality"] < min_quality:
        reasons.append("quality")
    return {"state":"alert" if reasons else "healthy","reasons":reasons}

if __name__ == "__main__":
    sample=[{"success":i!=1,"latency_ms":100+i*8,"quality":.91} for i in range(25)]
    metrics=summarize_window(sample); print(metrics,evaluate_alert(metrics,error_rate_limit=.02))
