# Model Versioning

## Detailed Description

Semantic versioning (MAJOR.MINOR.PATCH) for trained models. MAJOR: architecture change (XGBoost→GBDT). MINOR: feature set change (add feature). PATCH: hyperparameter tuning. Enables reproducibility, rollback, clear communication.

## Core Intuition

Versioning = clear semantics. v1.2.3 says: same architecture (1), new features (2), tweaked hyperparams (3). v1.3.0 says: new features added, probably need revalidation. Without versioning: confusing (which is better, 1.1 or 1.2?). With it: clear.

## How It Works

**Versioning Scheme:**
- MAJOR (e.g., 1.0.0 → 2.0.0): architecture change (XGBoost → neural net)
- MINOR (e.g., 2.0.0 → 2.1.0): feature set change (add/remove features)
- PATCH (e.g., 2.1.0 → 2.1.1): hyperparameter tuning, bugfix

**Metadata per version:**
- Code commit hash (git): enables exact code reproduction
- Data version (DVC): which training data version
- Hyperparameters: learning_rate, batch_size, epochs
- Metrics: accuracy, F1, AUC (comparison across versions)

## Key Properties / Trade-offs

| Aspect | Semantic | Calendar |
|--------|----------|----------|
| Example | 2.1.0 | 2024-01-15.3 |
| Readability | Clear meaning (major/minor/patch) | Timestamp-based |
| Rollback | "Go back to 1.9.0" | "Go back to yesterday" |
| When to use | Code-driven models | Daily retrained models |
| Clarity | Clear breaking changes | Less clear |

## Common Mistakes / Gotchas
- No versioning at all: models scattered, impossible to reproduce
- Calendar versioning without semantic: unclear if v2 is minor or major change
- Losing code: model v1.0.0 trained on deleted code branch → can't rebuild
- Data drift ignored: v1.0.0 trained on 2024-01 data, trying to use with 2024-03 data

## Best Practices

- **Git commit hash as anchor:** model_v1.0.0_commit_abc123def456. Reproducible.
- **Data version locked:** Pin with DVC or commit hash. Same data + code = same model.
- **Breaking changes → major:** architecture change increments major version.
- **Backward compatibility:** never tag the same version twice. Each model is immutable.
- **Deprecation policy:** versions < 6 months old marked deprecated, removed after 12 months.

## Detailed Trade-off Analysis

| Aspect | Semantic | Calendar | Git Hash |
|--------|----------|----------|----------|
| Readability | Clear (major/minor/patch) | Timestamp | Unique |
| Rollback clarity | "v1.9.0" obvious | Yesterday's | Specific |
| Metadata tracking | Manual | Auto | Auto |
| When to use | Code-driven | Daily retrains | CI/CD driven |

---

## Production Failure Scenarios

### Scenario 1: Can't rollback (model code deleted)
**Recover:** Git hash enables rebuild from commit. Without it: impossible.
**Prevent:** Always store git commit hash with model version.

### Scenario 2: Version incompatibility
**What breaks:** v2.0.0 uses new features not in v1.0.0 data.
**Prevent:** Tag breaking changes (major version). Never reuse version tag.

### Scenario 3: Data drift
**What breaks:** v1.0.0 trained on 2024-01 data, used with 2024-03 data (different distribution).
**Prevent:** Track data version. Monitor distribution shift.

---

## Implementation Guidance & Gotchas

**❌ Wrong: Unversioned or ambiguous versions**
```python
model.save("model.pkl")  # What version? No git? Impossible to rollback.
```

**✅ Right: Semantic + git commit**
```python
version = "v1.2.0"
git_hash = "abc123"
tag = f"{version}_{git_hash}"
registry.register(tag, model, metadata)
# Can rebuild from git if needed
```

---

## Sophisticated Interview Q&A

**Q1: Version 2.0.0 breaks backward compatibility. Handle?**
A: Major version indicates breaking change. Coordinate deployment: keep v1.9.0 available for gradual migration. Deprecate v1.x over 3 months. Update clients explicitly.

**Q2: Model 1.0.0 trained on data version A, now training on B. Same version?**
A: No. Different data = different model. Bump minor version (1.1.0). Monitor distribution shift.

---

## Cost & Resource Analysis

**Minimal cost.** Versioning overhead: 5 min per model registration.

**ROI:** Prevents rollback outages (time saved >> cost).

---

## Monitoring & Observability Patterns

**Metrics:**
- Versions deployed per environment
- Time since version deployed (age)
- Deprecation status

**Alerts:**
- if version >= 6 months old: mark deprecated
- if version >= 12 months old: archive

## Code Example

```python
import json, subprocess
from pathlib import Path

class ModelVersioning:
    def register_version(self, model, version, metrics, code_commit, data_version):
        metadata = {
            "version": version,
            "code_commit": code_commit,
            "data_version": data_version,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        version_dir = Path(f"models/v{version}")
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(f"{version_dir}/model.pkl")
        
        # Save metadata
        with open(f"{version_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Registered model v{version} with code commit {code_commit}")
```

## Interview Q&A

Q: When bump MAJOR vs MINOR vs PATCH?
A: A: MAJOR: architecture change (LSTM→CNN). MINOR: feature set change (add X). PATCH: hyperparameter only. Uncertain? Use MINOR (safe).

Q: v2.0.0 worse than v1.9.0?
A: A: Registry stores both. Deploy v1.9.0 (5min). Investigate v2.0.0 bug. Fix, retrain as v2.0.1.

Q: Have v1.2.0, v1.2.1, v1.2.2. Deploy which?
A: A: Compare validation: highest accuracy? Also check latency, size, cost. Deploy best.

Q: Version in code or assigned at deployment?
A: A: Assigned at deployment. Code doesn't mention version. System increments based on what changed (prevents mismatch).

Q: v1.0.0 on data_v1, v1.1.0 on data_v2, accuracy dropped?
A: A: Track data version with model. v1.1.0={code:abc, data:v2, accuracy:0.91}. Understand: accuracy dropped due to data, not code.

Q: Two teams release v1.2.0?
A: A: Prevent with central authority. One registry, one version number generator. Teams coordinate.

Q: Maintain how many old versions?
A: A: Production + 1 prior (rollback). Latest 3 for comparison. Archive older.

Q: v1.5.0 passes validation but fails in production?
A: A: Tag versions: v1.5.0{caveat:'works on segment_A, lower on segment_B'}. Use tags: 'stable', 'experimental', 'production_ready'.
## Interview Quick-Reference

| Semver | Example | When |
|--------|---------|------|
| MAJOR | 1.0.0 → 2.0.0 | Architecture change |
| MINOR | 2.0.0 → 2.1.0 | Feature set change |
| PATCH | 2.1.0 → 2.1.1 | Bugfix/hyperparameter |

## Related Topics
- [Model Registry](04-model-registry.md) - manages all versions
- [Data Pipelines](02-data-pipelines.md) - data versioning (DVC)

## Resources
- [Semantic Versioning](https://semver.org/)
- [DVC: Data Version Control](https://dvc.org/)

