# Model Versioning

## TL;DR
Semantic versioning (MAJOR.MINOR.PATCH) for trained models. Pin code commit, data version, hyperparameters. Enables reproduction: exact same inputs → exact same model.

## Core Intuition
Model v1.0.0 is a snapshot: specific code, specific data, specific hyperparameters. Re-run same code + data = reproducible model. No version → can't reproduce.

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

**Q: How do you ensure reproducibility across model versions?**
A: Pin three things: (1) code version via git commit hash, (2) data version via DVC or data hash, (3) hyperparameters explicitly. Given all three, re-run training produces identical model (within floating-point precision). Test: periodically rebuild old models from stored metadata, validate metrics match.

**Q: Model v2.0.0 is worse than v1.9.0. Rollback?**
A: Versioning enables instant rollback: switch production traffic to v1.9.0 (config change, no rebuild). Investigate v2.0. Find bug. Create v2.0.1 with fix. Redeploy. SLA: rollback <5 minutes.

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
