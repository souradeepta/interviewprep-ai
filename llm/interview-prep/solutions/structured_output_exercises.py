"""Deterministic structured-output and citation exercises."""


def validate_json_like(payload, schema, *, strict=True, path=""):
    """Return all missing, wrong-type, and (optionally) unknown fields."""
    errors = []
    if not isinstance(payload, dict) or not isinstance(schema, dict):
        return [f"{path or '$'}: expected object"]
    for key, expected in schema.items():
        location = f"{path}.{key}" if path else key
        if key not in payload:
            errors.append(f"{location}: missing")
        elif isinstance(expected, dict):
            errors.extend(validate_json_like(payload[key], expected, strict=strict, path=location))
        elif not isinstance(payload[key], expected) or (expected is int and isinstance(payload[key], bool)):
            errors.append(f"{location}: wrong_type")
    if strict:
        for key in payload:
            if key not in schema:
                location = f"{path}.{key}" if path else key
                errors.append(f"{location}: unknown")
    return errors


def citation_coverage(claims, retrieved_ids):
    """Measure claims whose citations all point to retrieved chunks."""
    retrieved = set(retrieved_ids)
    if not claims:
        return {"claims": 0, "supported": 0, "unsupported_rate": 0.0}
    supported = 0
    for claim in claims:
        citations = claim.get("citations", [])
        if citations and set(citations).issubset(retrieved):
            supported += 1
    return {"claims": len(claims), "supported": supported,
            "unsupported_rate": (len(claims) - supported) / len(claims)}
