"""A bounded autonomous decision policy with auditability and escalation."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class Decision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class DecisionRecord:
    request_id: str
    decision: Decision
    reason: str
    amount: float


class BoundedAutonomousAgent:
    """Apply explicit limits before allowing an autonomous side effect."""

    def __init__(self, max_amount: float = 500.0, confidence_threshold: float = 0.8, daily_limit: float = 1000.0):
        if max_amount < 0 or daily_limit < 0 or not 0 <= confidence_threshold <= 1:
            raise ValueError("invalid policy bounds")
        self.max_amount = max_amount
        self.confidence_threshold = confidence_threshold
        self.daily_limit = daily_limit
        self.spent = 0.0
        self.audit_log: List[DecisionRecord] = []

    def decide(self, request: Dict) -> Decision:
        request_id = str(request.get("request_id", "unknown"))
        amount = float(request.get("amount", 0.0))
        confidence = float(request.get("confidence", 0.0))
        if amount < 0:
            decision, reason = Decision.REJECT, "negative amount"
        elif amount > self.max_amount:
            decision, reason = Decision.ESCALATE, "amount exceeds per-request limit"
        elif confidence < self.confidence_threshold:
            decision, reason = Decision.ESCALATE, "confidence below threshold"
        elif self.spent + amount > self.daily_limit:
            decision, reason = Decision.ESCALATE, "daily spending limit exceeded"
        else:
            decision, reason = Decision.APPROVE, "all policy bounds passed"
            self.spent += amount
        self.audit_log.append(DecisionRecord(request_id, decision, reason, amount))
        return decision

    def rollback_last_approval(self) -> None:
        if not self.audit_log or self.audit_log[-1].decision != Decision.APPROVE:
            raise ValueError("no approval available to roll back")
        record = self.audit_log.pop()
        self.spent -= record.amount
