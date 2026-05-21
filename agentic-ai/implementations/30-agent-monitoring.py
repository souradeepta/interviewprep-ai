"""
Auto-generated from 30-agent-monitoring.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Monitoring
# Learning objectives:
# - Understand SLO-based monitoring and alerting
# - Implement metrics collection and aggregation
# ======================================================================

import os
import json
import time
from anthropic import Anthropic
from dotenv import load_dotenv
from collections import defaultdict, deque
from datetime import datetime, timedelta
from statistics import mean, stdev

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready for agent monitoring!")


# ======================================================================
# ## Level 1: Basic Metrics Collection
# Simple logging of success/fail, latency, and cost.
# ======================================================================

class BasicMetricsCollector:
    """Collect basic metrics: success, latency, cost."""
    def __init__(self):
        self.client = Anthropic()
        self.metrics = []
    
    def execute_task(self, task: str, check_success: callable) -> dict:
        """Execute task and collect metrics."""
        start = time.time()
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": task}]
        )
        
        result_text = response.content[0].text
        latency = time.time() - start
        success = check_success(result_text)
        
        # Calculate cost (approx)
        cost = (response.usage.input_tokens * 0.003 + 
               response.usage.output_tokens * 0.006) / 1000
        
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "latency_seconds": round(latency, 2),
            "cost_usd": round(cost, 4),
            "tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        
        self.metrics.append(metric)
        return metric
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.metrics:
            return {}
        
        successes = sum(1 for m in self.metrics if m["success"])
        latencies = [m["latency_seconds"] for m in self.metrics]
        costs = [m["cost_usd"] for m in self.metrics]
        
        return {
            "total_tasks": len(self.metrics),
            "success_rate": f"{100*successes/len(self.metrics):.1f}%",
            "avg_latency": f"{mean(latencies):.2f}s",
            "max_latency": f"{max(latencies):.2f}s",
            "total_cost": f"${sum(costs):.3f}",
            "avg_cost_per_task": f"${mean(costs):.4f}"
        }

# Test
collector = BasicMetricsCollector()
for i in range(3):
    metric = collector.execute_task(
        f"Question {i+1}: What is {i*2}+{i*2}?",
        lambda r: "+" in r or "equal" in r.lower()
    )
    print(f"Task {i+1}: success={metric['success']}, latency={metric['latency_seconds']}s")

print(f"\nSummary: {json.dumps(collector.get_summary(), indent=2)}")


# ======================================================================
# ## Level 2: SLO-Based Monitoring with Alerts
# Define SLOs and alert when breached.
# ======================================================================

class SLOMonitor:
    """Monitor agent metrics against SLOs."""
    def __init__(self, slos: dict):
        self.client = Anthropic()
        self.slos = slos  # {"success_rate": 0.85, "latency": 5.0, "cost": 0.50}
        self.window = deque(maxlen=100)  # Last 100 requests
        self.alerts = []
    
    def execute_task(self, task: str, check_success: callable):
        """Execute and monitor SLOs."""
        start = time.time()
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": task}]
        )
        
        result_text = response.content[0].text
        latency = time.time() - start
        success = check_success(result_text)
        cost = (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.006) / 1000
        
        metric = {"success": success, "latency": latency, "cost": cost}
        self.window.append(metric)
        
        # Check SLOs
        self._check_slos()
        
        return metric
    
    def _check_slos(self):
        """Check if current metrics violate SLOs."""
        if len(self.window) < 10:
            return  # Need minimum data
        
        success_rate = sum(1 for m in self.window if m["success"]) / len(self.window)
        avg_latency = sum(m["latency"] for m in self.window) / len(self.window)
        avg_cost = sum(m["cost"] for m in self.window) / len(self.window)
        
        # Check thresholds
        if success_rate < self.slos["success_rate"]:
            alert = f"⚠️  SUCCESS ALERT: {success_rate:.1%} < SLO {self.slos['success_rate']:.0%}"
            self.alerts.append(alert)
            print(alert)
        
        if avg_latency > self.slos["latency"]:
            alert = f"⏱️  LATENCY ALERT: {avg_latency:.2f}s > SLO {self.slos['latency']:.1f}s"
            self.alerts.append(alert)
            print(alert)
        
        if avg_cost > self.slos["cost"]:
            alert = f"💰 COST ALERT: ${avg_cost:.4f} > SLO ${self.slos['cost']:.3f}"
            self.alerts.append(alert)
            print(alert)
    
    def get_status(self) -> dict:
        """Get current monitoring status."""
        if not self.window:
            return {"status": "No data"}
        
        success_rate = sum(1 for m in self.window if m["success"]) / len(self.window)
        avg_latency = sum(m["latency"] for m in self.window) / len(self.window)
        avg_cost = sum(m["cost"] for m in self.window) / len(self.window)
        
        return {
            "success_rate": f"{success_rate:.1%}",
            "avg_latency": f"{avg_latency:.2f}s",
            "avg_cost": f"${avg_cost:.4f}",
            "slos": self.slos,
            "alerts_fired": len(self.alerts)
        }

# Test
monitor = SLOMonitor(slos={"success_rate": 0.80, "latency": 2.0, "cost": 0.10})
for i in range(15):
    monitor.execute_task(
        f"Task {i}: compute {i}+1",
        lambda r: "compute" in r.lower() or str(i+1) in r
    )

print(f"\nMonitor status: {json.dumps(monitor.get_status(), indent=2)}")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Hourly Rollups with Trend Detection
# ======================================================================

class HourlyRollupMonitor:
    """Compute hourly metrics and detect trends."""
    def __init__(self):
        self.client = Anthropic()
        self.events = []
        self.hourly_snapshots = []
    
    def log_event(self, success: bool, latency: float, cost: float):
        """Log execution event."""
        self.events.append({
            "timestamp": time.time(),
            "success": success,
            "latency": latency,
            "cost": cost
        })
    
    def compute_hourly_rollup(self):
        """Compute metrics for past hour."""
        now = time.time()
        one_hour_ago = now - 3600
        
        recent = [e for e in self.events if e["timestamp"] > one_hour_ago]
        
        if not recent:
            return None
        
        success_rate = sum(1 for e in recent if e["success"]) / len(recent)
        avg_latency = mean([e["latency"] for e in recent])
        total_cost = sum(e["cost"] for e in recent)
        
        rollup = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_count": len(recent),
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "total_cost": total_cost
        }
        
        self.hourly_snapshots.append(rollup)
        return rollup
    
    def detect_trend(self) -> str:
        """Detect if metrics trending up or down."""
        if len(self.hourly_snapshots) < 2:
            return "Insufficient data"
        
        recent_rate = self.hourly_snapshots[-1]["success_rate"]
        prior_rate = self.hourly_snapshots[-2]["success_rate"] if len(self.hourly_snapshots) > 1 else recent_rate
        
        delta = recent_rate - prior_rate
        
        if delta > 0.05:
            return f"📈 Trend improving: +{delta:.1%}"
        elif delta < -0.05:
            return f"📉 Trend degrading: {delta:.1%}"
        else:
            return f"→ Stable: {delta:+.1%}"

# Test
monitor = HourlyRollupMonitor()
for i in range(10):
    success = i % 3 != 0  # Fail every 3rd
    monitor.log_event(success, 1.5, 0.05)

rollup = monitor.compute_hourly_rollup()
trend = monitor.detect_trend()

print(f"Rollup: {json.dumps({k: v if not isinstance(v, float) else round(v, 3) for k, v in rollup.items()}, indent=2)}")
print(f"Trend: {trend}")


# ======================================================================
# ### Example 2: Anomaly Detection with Statistical Thresholds
# ======================================================================

class AnomalyDetector:
    """Detect anomalies using z-score."""
    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold
        self.latency_window = deque(maxlen=50)
        self.success_window = deque(maxlen=50)
        self.anomalies = []
    
    def record(self, success: bool, latency: float):
        """Record metric and check for anomalies."""
        self.success_window.append(1 if success else 0)
        self.latency_window.append(latency)
        
        # Check for anomalies
        self._detect_anomalies()
    
    def _detect_anomalies(self):
        """Check if recent values are statistical outliers."""
        # Latency anomaly
        if len(self.latency_window) >= 10:
            lat_mean = mean(self.latency_window)
            lat_std = stdev(self.latency_window)
            current = self.latency_window[-1]
            
            if lat_std > 0:
                z_score = abs((current - lat_mean) / lat_std)
                if z_score > self.z_threshold:
                    anomaly = f"Latency anomaly: {current:.2f}s (mean {lat_mean:.2f}±{lat_std:.2f}, z={z_score:.1f})"
                    self.anomalies.append(anomaly)
                    print(f"⚠️  {anomaly}")
        
        # Success rate anomaly
        if len(self.success_window) >= 10:
            success_rate = mean(self.success_window)
            # Simple: if success suddenly drops to 0, alert
            if success_rate < 0.5 and self.success_window[-1] == 0:
                anomaly = f"Success rate dropped to {success_rate:.0%}"
                self.anomalies.append(anomaly)
                print(f"⚠️  {anomaly}")
    
    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "samples": len(self.latency_window),
            "avg_latency": f"{mean(self.latency_window):.2f}s" if self.latency_window else "N/A",
            "success_rate": f"{mean(self.success_window):.0%}" if self.success_window else "N/A",
            "anomalies_detected": len(self.anomalies)
        }

# Test
detector = AnomalyDetector(z_threshold=1.5)
for i in range(20):
    success = True
    latency = 1.0 if i < 10 else 5.0  # Jump at step 10
    detector.record(success, latency)

print(f"\nDetector stats: {json.dumps(detector.get_stats(), indent=2)}")


# ======================================================================
# ### Example 3: Production Dashboard with Regression Detection
# ======================================================================

class ProductionDashboard:
    """Full production monitoring dashboard."""
    def __init__(self, baseline_success_rate: float = 0.85, regression_threshold: float = 0.05):
        self.client = Anthropic()
        self.baseline = baseline_success_rate
        self.regression_threshold = regression_threshold
        self.events = []
        self.snapshots = {}
    
    def execute_task(self, task: str, check_success: callable):
        """Execute task and update dashboard."""
        start = time.time()
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": task}]
        )
        
        latency = time.time() - start
        success = check_success(response.content[0].text)
        cost = (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.006) / 1000
        
        self.events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "latency": latency,
            "cost": cost
        })
        
        return {"success": success, "latency": latency, "cost": cost}
    
    def compute_snapshot(self, name: str = "current"):
        """Create monitoring snapshot."""
        if not self.events:
            return None
        
        successes = sum(1 for e in self.events if e["success"])
        success_rate = successes / len(self.events)
        latencies = [e["latency"] for e in self.events]
        costs = [e["cost"] for e in self.events]
        
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_count": len(self.events),
            "success_rate": success_rate,
            "baseline": self.baseline,
            "avg_latency": mean(latencies),
            "total_cost": sum(costs),
            "regression_detected": success_rate < self.baseline - self.regression_threshold
        }
        
        self.snapshots[name] = snapshot
        return snapshot
    
    def generate_dashboard(self) -> dict:
        """Generate final dashboard report."""
        snapshot = self.compute_snapshot()
        
        if not snapshot:
            return {"status": "No data"}
        
        status = "🚨 CRITICAL" if snapshot["regression_detected"] else "✓ Healthy"
        
        return {
            "status": status,
            "timestamp": snapshot["timestamp"],
            "metrics": {
                "success_rate": f"{snapshot['success_rate']:.1%}",
                "baseline": f"{snapshot['baseline']:.0%}",
                "avg_latency": f"{snapshot['avg_latency']:.2f}s",
                "cost_per_task": f"${snapshot['total_cost']/max(1, snapshot['request_count']):.4f}"
            },
            "regression_detected": snapshot["regression_detected"]
        }

# Test
dashboard = ProductionDashboard(baseline_success_rate=0.70)
for i in range(10):
    success = i % 2 == 0  # 50% success rate
    result = dashboard.execute_task(
        f"Task {i}",
        lambda r: success
    )

report = dashboard.generate_dashboard()
print(json.dumps(report, indent=2))


# ======================================================================
# ## Key Takeaways
# 1. **Set SLOs before production.** Define acceptable thresholds (success rate, latency, cost) before deploying. Use these to gate launches.
# 2. **Monitor continuously, not once.** Evals are one-time; monitoring is permanent. Catch degradation within hours, not days.
# ======================================================================
