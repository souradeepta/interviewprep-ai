"""Pure model-routing and regression-gate exercises."""


def route_request(routes, *, min_quality, max_latency_ms, max_cost):
    eligible = [route for route in routes
                if route["quality"] >= min_quality
                and route["latency_ms"] <= max_latency_ms
                and route["cost"] <= max_cost]
    if not eligible:
        return {"state": "unavailable", "route": None}
    chosen = min(eligible, key=lambda route: (route["cost"], route["latency_ms"], route["name"]))
    return {"state": "available", "route": chosen["name"]}


def compare_regression(baseline, candidate, *, max_cost_increase=0.0,
                       max_latency_increase=0.0, min_quality_delta=0.0):
    """Gate candidate quality while enforcing operational guardrails."""
    quality_delta = candidate["quality"] - baseline["quality"]
    cost_delta = candidate["cost"] - baseline["cost"]
    latency_delta = candidate["latency_ms"] - baseline["latency_ms"]
    passed = (quality_delta >= min_quality_delta
              and cost_delta <= max_cost_increase
              and latency_delta <= max_latency_increase)
    return {"passed": passed, "quality_delta": quality_delta,
            "cost_delta": cost_delta, "latency_delta": latency_delta}
