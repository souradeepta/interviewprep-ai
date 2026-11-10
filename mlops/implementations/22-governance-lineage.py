"""Validate minimum model lineage and produce a stable metadata fingerprint."""
import hashlib
import json

REQUIRED_FIELDS=("model_id","model_version","code_revision","data_snapshot","feature_schema",
                 "training_config","evaluation_report","owner","approved_by")
def validate_lineage(record):
    missing=[field for field in REQUIRED_FIELDS if not record.get(field)]
    return {"valid":not missing,"missing":missing}

def fingerprint(record):
    result=validate_lineage(record)
    if not result["valid"]: raise ValueError("incomplete lineage: "+", ".join(result["missing"]))
    payload=json.dumps(record,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()

if __name__ == "__main__":
    artifact={k:v for k,v in zip(REQUIRED_FIELDS,["risk","v7","abc123","snapshot-42","schema-3","cfg-9","eval-12","team-ml","reviewer-4"])}
    print(validate_lineage(artifact),fingerprint(artifact))
