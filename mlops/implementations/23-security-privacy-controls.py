"""Local access-policy checks and deterministic sensitive-field redaction."""
SENSITIVE_FIELDS={"email","phone","name","ssn","access_token"}
def authorize(*, role, action, resource_class, grants):
    allowed=(role,action,resource_class) in set(grants)
    return {"allowed":allowed,"reason":"granted" if allowed else "deny_by_default"}

def redact(record, fields=SENSITIVE_FIELDS):
    return {key:("[REDACTED]" if key in fields and value is not None else value)
            for key,value in record.items()}

def retention_action(age_days, retention_days, *, legal_hold=False):
    if age_days<0 or retention_days<0: raise ValueError("ages must be nonnegative")
    if legal_hold: return "retain_under_hold"
    return "delete_or_expire" if age_days>retention_days else "retain"

if __name__ == "__main__":
    grants={("oncall","read_metrics","aggregate")}
    print(authorize(role="oncall",action="read_metrics",resource_class="aggregate",grants=grants))
    print(redact({"model":"v2","email":"person@example.invalid","score":.8}))
