import nbformat as nbf
from dataclasses import dataclass

nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell("# Autonomous Agents\n\nObjectives: Bounded autonomy, safety constraints, monitoring, rollback, escalation"))

nb.cells.append(nbf.v4.new_code_cell("""# Level 1: Bounded Autonomous Agent

class SimpleAutonomousAgent:
    def __init__(self, budget_limit: float = 1000):
        self.budget_limit = budget_limit
        self.spent = 0
        self.decisions = []

    def approve_transaction(self, amount: float, confidence: float) -> str:
        '''Decide on transaction within boundaries.'''
        # Boundary 1: Budget
        if self.spent + amount > self.budget_limit:
            return "REJECTED_BUDGET"

        # Boundary 2: Confidence
        if confidence < 0.7:
            return "ESCALATED_LOW_CONFIDENCE"

        # Within boundaries: approve
        self.spent += amount
        self.decisions.append({"amount": amount, "status": "approved"})
        return "APPROVED"

print('Level 1 - Bounded Autonomy:\\n')
agent = SimpleAutonomousAgent(budget_limit=500)

test_cases = [
    {"amount": 100, "confidence": 0.9},
    {"amount": 300, "confidence": 0.8},
    {"amount": 200, "confidence": 0.6},
]

for tc in test_cases:
    status = agent.approve_transaction(tc["amount"], tc["confidence"])
    print(f"Amount: {tc['amount']}, Confidence: {tc['confidence']:.0%} → {status}")

print(f"\\nTotal spent: {agent.spent}/{agent.budget_limit}\\n"""))

nb.cells.append(nbf.v4.new_markdown_cell("**Key Points:** Define hard boundaries. Check before executing. Reject if violated. Maintain state."))

nb.cells.append(nbf.v4.new_code_cell("""# Level 2: Agent with Monitoring and Rollback

class MonitoredAutonomousAgent(SimpleAutonomousAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_history = []

    def approve_with_logging(self, amount: float, confidence: float, reason: str) -> str:
        '''Approve and log decision.'''
        status = self.approve_transaction(amount, confidence)

        # Log decision
        self.action_history.append({
            "amount": amount,
            "confidence": confidence,
            "reason": reason,
            "status": status
        })

        return status

    def rollback_last(self) -> bool:
        '''Undo last transaction.'''
        if self.decisions and self.action_history:
            self.decisions.pop()
            self.spent -= self.action_history[-1]["amount"]
            self.action_history.pop()
            return True
        return False

    def get_metrics(self) -> dict:
        '''Get agent metrics.'''
        total = len(self.action_history)
        approved = sum(1 for a in self.action_history if a["status"] == "approved")

        return {
            "total_decisions": total,
            "approved": approved,
            "approval_rate": approved / max(total, 1)
        }

print('Level 2 - Monitoring and Rollback:\\n')
agent = MonitoredAutonomousAgent(budget_limit=500)

agent.approve_with_logging(100, 0.9, "trusted_customer")
agent.approve_with_logging(200, 0.85, "verified_vendor")

print(f"Metrics: {agent.get_metrics()}")

agent.rollback_last()
print(f"After rollback: {agent.get_metrics()}\\n"""))

nb.cells.append(nbf.v4.new_markdown_cell("**Key Takeaways:** Log every decision with reasoning. Maintain history. Enable rollback. Monitor metrics."))

nb.cells.append(nbf.v4.new_code_cell("""# Example 1: Escalation to Human

class EscalatingAgent(MonitoredAutonomousAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.escalations = []

    def decide_with_escalation(self, amount: float, confidence: float, risk_level: str) -> str:
        '''Decide with escalation logic.'''
        # High-risk always escalate
        if risk_level == "high":
            self.escalations.append({"amount": amount, "reason": "high_risk"})
            return "ESCALATE_HIGH_RISK"

        # Low confidence escalate
        if confidence < 0.7:
            self.escalations.append({"amount": amount, "reason": "low_confidence"})
            return "ESCALATE_LOW_CONFIDENCE"

        # Otherwise auto-approve if within budget
        if amount > self.budget_limit - self.spent:
            self.escalations.append({"amount": amount, "reason": "budget_exceeded"})
            return "ESCALATE_BUDGET"

        return "AUTO_APPROVED"

print('Example 1 - Escalation to Human:\\n')
agent = EscalatingAgent(budget_limit=1000)

cases = [
    {"amount": 50, "confidence": 0.95, "risk": "low"},
    {"amount": 500, "confidence": 0.5, "risk": "low"},
    {"amount": 100, "confidence": 0.9, "risk": "high"},
]

for c in cases:
    result = agent.decide_with_escalation(c["amount"], c["confidence"], c["risk"])
    print(f"Amount: {c['amount']}, Risk: {c['risk']} → {result}")
print()"""))

nb.cells.append(nbf.v4.new_markdown_cell("**Example 1 Key Points:** Define escalation rules. Route to human for uncertain/risky. Log for audit."))

nb.cells.append(nbf.v4.new_code_cell("""# Example 2: Anomaly Detection

class AnomalyDetectingAgent(MonitoredAutonomousAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anomalies = []

    def check_anomalies(self) -> list:
        '''Detect unusual patterns.'''
        alerts = []
        metrics = self.get_metrics()

        # Too many approvals
        if metrics["approval_rate"] > 0.95:
            alerts.append("Approval rate too high (>95%)")

        # Too few approvals
        if metrics["approval_rate"] < 0.2 and metrics["total_decisions"] > 5:
            alerts.append("Approval rate too low (<20%)")

        # Check avg amount
        if self.decisions:
            avg = sum(d["amount"] for d in self.decisions) / len(self.decisions)
            if avg > 300:
                alerts.append(f"Average amount high ({avg:.0f})")

        return alerts

print('Example 2 - Anomaly Detection:\\n')
agent = AnomalyDetectingAgent(budget_limit=2000)

# Simulate approvals
for i in range(5):
    agent.approve_with_logging(100, 0.95, f"trans_{i}")

anomalies = agent.check_anomalies()
print(f"Metrics: {agent.get_metrics()}")
print(f"Anomalies detected: {anomalies if anomalies else 'None'}\\n"""))

nb.cells.append(nbf.v4.new_markdown_cell("**Example 2 Key Points:** Monitor key metrics. Alert on anomalies. Enables quick human intervention."))

nb.cells.append(nbf.v4.new_markdown_cell("""## Key Takeaways

**Autonomy Components:**
- Goals: what agent tries to achieve
- Boundaries: what agent cannot do
- Monitoring: watch for issues
- Escalation: hand off to human
- Rollback: undo bad decisions

**Safety Layers:**
1. Input validation (boundary checks)
2. Decision logging (audit trail)
3. Anomaly detection (catch drift)
4. Human escalation (expert judgment)
5. Easy rollback (undo capability)

**Design Pattern:**
- Initialize with narrow scope
- Gradually expand boundaries
- Continuous monitoring
- Regular audits
- Quick rollback readiness

**Related Concepts:** [[error-recovery]], [[monitoring]], [[safety-alignment]], [[human-agent-collaboration]]"""))

nbf.write(nb, '/home/sbisw/github/interviewprep-ml/agentic-ai/notebooks/autonomous-agents.ipynb')
