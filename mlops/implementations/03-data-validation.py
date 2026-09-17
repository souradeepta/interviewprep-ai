"""Dependency-free layered data validation for interview practice."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ValidationResult:
    layer: str
    passed: bool
    failed_rows: Tuple[int, ...]
    details: str


class DataValidator:
    """Run schema, completeness, range, and business-rule checks."""

    def __init__(self, rows: Iterable[Mapping[str, object]]) -> None:
        self.rows = [dict(row) for row in rows]

    def _result(self, layer: str, failed: List[int], details: str) -> ValidationResult:
        return ValidationResult(layer, not failed, tuple(failed), details)

    def validate_schema(self, schema: Mapping[str, type]) -> ValidationResult:
        failed = [
            index
            for index, row in enumerate(self.rows)
            if any(column not in row or (row[column] is not None and not isinstance(row[column], expected))
                   for column, expected in schema.items())
        ]
        return self._result("schema", failed, f"{len(failed)} rows violate the declared schema")

    def validate_completeness(self, required: Iterable[str], max_null_fraction: float = 0.0) -> ValidationResult:
        if not 0 <= max_null_fraction <= 1:
            raise ValueError("max_null_fraction must be between 0 and 1")
        required = tuple(required)
        failed = []
        for index, row in enumerate(self.rows):
            missing = sum(row.get(column) is None for column in required)
            if required and missing / len(required) > max_null_fraction:
                failed.append(index)
        return self._result("completeness", failed, f"{len(failed)} rows exceed the null threshold")

    def validate_range(self, column: str, minimum: float, maximum: float) -> ValidationResult:
        if minimum > maximum:
            raise ValueError("minimum must not exceed maximum")
        failed = []
        for index, row in enumerate(self.rows):
            value = row.get(column)
            if not isinstance(value, (int, float)) or not minimum <= value <= maximum:
                failed.append(index)
        return self._result("range", failed, f"{len(failed)} rows fall outside [{minimum}, {maximum}]")

    def validate_business_rule(self, name: str, rule: Callable[[Mapping[str, object]], bool]) -> ValidationResult:
        failed = [index for index, row in enumerate(self.rows) if not rule(row)]
        return self._result(name, failed, f"{len(failed)} rows violate the business rule")

    def validate_all(self, schema: Mapping[str, type], required: Iterable[str], ranges: Optional[Mapping[str, Tuple[float, float]]] = None) -> List[ValidationResult]:
        results = [self.validate_schema(schema), self.validate_completeness(required)]
        for column, (minimum, maximum) in (ranges or {}).items():
            results.append(self.validate_range(column, minimum, maximum))
        return results


if __name__ == "__main__":
    sample = DataValidator([{"user_id": 1, "amount": 25.0}, {"user_id": 2, "amount": -1.0}])
    for result in sample.validate_all({"user_id": int, "amount": float}, ("user_id", "amount"), {"amount": (0, 10000)}):
        print(result.layer, "PASS" if result.passed else f"FAIL rows={result.failed_rows}")
