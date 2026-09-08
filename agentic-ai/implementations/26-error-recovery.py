"""API-free bounded retry and recoverable side-effect exercise.

An arbitrary external side effect cannot be made exactly-once by local Python
state.  The executor therefore records an ``unknown`` result after an
exception and requires reconciliation before the key can be retried.
"""

from dataclasses import dataclass


@dataclass
class RetryResult:
    value: object | None
    attempts: int
    delays: list[float]
    error: Exception | None = None


def retry(operation, *, max_attempts=3, transient=(TimeoutError,), base_delay=1.0):
    """Retry without sleeping; return the schedule for a deterministic test."""
    if (isinstance(max_attempts, bool) or not isinstance(max_attempts, int)
            or max_attempts <= 0 or base_delay < 0):
        raise ValueError("invalid retry policy")
    if (not isinstance(transient, tuple)
            or not transient
            or not all(isinstance(error, type) and issubclass(error, BaseException)
                       for error in transient)):
        raise ValueError("transient must be a non-empty tuple of exception classes")
    delays = []
    for attempt in range(1, max_attempts + 1):
        try:
            return RetryResult(operation(), attempt, delays)
        except transient as error:
            if attempt == max_attempts:
                return RetryResult(None, attempt, delays, error)
            delays.append(base_delay * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


class ReconciliationRequired(RuntimeError):
    """The operation may have happened and must be looked up before retrying."""

    def __init__(self, key):
        super().__init__(f"reconcile side effect for key {key!r} before retrying")
        self.key = key


@dataclass
class OperationRecord:
    state: str
    value: object | None = None


class IdempotentExecutor:
    """Track completed and uncertain effects; never replay an uncertain key."""

    def __init__(self):
        self.records: dict[str, OperationRecord] = {}

    def execute(self, key, operation):
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        record = self.records.get(key)
        if record and record.state == "completed":
            return record.value
        if record and record.state == "unknown":
            raise ReconciliationRequired(key)
        self.records[key] = OperationRecord("pending")
        try:
            value = operation()
        except Exception as error:
            self.records[key] = OperationRecord("unknown")
            raise ReconciliationRequired(key) from error
        self.records[key] = OperationRecord("completed", value)
        return value

    def reconcile(self, key, lookup):
        """Resolve an unknown key using a durable result lookup.

        ``lookup`` returns ``None`` when the external system has no result yet.
        A non-None result marks the operation completed and makes future calls
        return that result without replaying the side effect.
        """
        record = self.records.get(key)
        if not record or record.state != "unknown":
            raise ValueError("key is not awaiting reconciliation")
        value = lookup(key)
        if value is None:
            raise ReconciliationRequired(key)
        self.records[key] = OperationRecord("completed", value)
        return value

    def state(self, key):
        record = self.records.get(key)
        return record.state if record else None
