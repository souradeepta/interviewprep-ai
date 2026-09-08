"""API-free bounded retry and idempotent side-effect exercise."""

from dataclasses import dataclass


@dataclass
class RetryResult:
    value: object | None
    attempts: int
    delays: list[float]
    error: Exception | None = None


def retry(operation, *, max_attempts=3, transient=(TimeoutError,), base_delay=1.0):
    """Retry without sleeping; return the schedule for a deterministic test."""
    if max_attempts <= 0 or base_delay < 0:
        raise ValueError("invalid retry policy")
    delays = []
    for attempt in range(1, max_attempts + 1):
        try:
            return RetryResult(operation(), attempt, delays)
        except transient as error:
            if attempt == max_attempts:
                return RetryResult(None, attempt, delays, error)
            delays.append(base_delay * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


class IdempotentExecutor:
    """Execute a side effect at most once per key, including after retries."""

    def __init__(self):
        self.completed: dict[str, object] = {}

    def execute(self, key, operation):
        if key in self.completed:
            return self.completed[key]
        value = operation()
        self.completed[key] = value
        return value
