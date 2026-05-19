# Reproducibility

## TL;DR
Re-run training with same code + data + random seed → exact same model. Critical for: debugging (reproduce error), auditing (regulatory), collaboration (share model training).

## Core Intuition
Scientist reports: "trained model on dataset X, got 95% accuracy". Can you reproduce it? Yes? Reproducible. No? Not reproducible.

## How It Works

**Reproducibility requirements:**
1. **Code version:** git commit hash (exact code used)
2. **Data version:** data commit hash (exact training data)
3. **Random seed:** set seed for numpy, torch, etc
4. **Dependencies:** version all libraries (Python 3.10, torch 2.0, etc)
5. **Hardware:** note GPU type (different GPU might have floating-point differences)

| Component | How to Track |
|-----------|---|
| Code | git commit hash |
| Data | DVC commit, or data hash |
| Seed | np.random.seed(42) |
| Deps | requirements.txt, poetry.lock |
| Hardware | log GPU type in metadata |

## Key Properties / Trade-offs
- Effort: reproducibility requires discipline (version everything)
- Cost: maintain multiple versions of code/data
- Benefit: debugging, auditing, collaboration

## Common Mistakes / Gotchas
- Not seeding: same code, different seed → different model
- Changing dependencies: torch 1.12 vs 2.0 might differ
- Data versioning: forgot to version training data → can't rebuild
- Floating-point sensitivity: GPU vs CPU might differ slightly

## Best Practices
- **Seed early:** set random seed before any randomization
- **Lock dependencies:** use poetry.lock or requirements-exact.txt
- **Version data:** use DVC for datasets
- **Document hardware:** log GPU type, torch version, CUDA version
- **Validate reproduction:** weekly: re-run old training, verify metrics match

## Code Example
```python
import numpy as np, torch, random

# Seed everything
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

seed_everything(42)

# Train
model = train(X, y)  # Same code + seed → exact same model
```

## Interview Q&A
**Q: Model v1.0.0 trained 6 months ago. Re-train today with same code + data. Different accuracy. Why?**
A: (1) Random seed different. (2) Different torch version. (3) Different GPU. (4) Data changed (pipeline updated). Investigate each. To truly reproduce, must match all four.

**Q: Reproducibility overhead: worth it?**
A: Yes, for production models. Cost: 5% development time overhead. Benefit: debugging saved hours, regulatory audit easy. ROI positive at 2+ production models.

## Interview Quick-Reference
| Component | How to Version |
|-----------|---|
| Code | git commit |
| Data | DVC commit or hash |
| Random | seed=42 |
| Deps | poetry.lock |

## Related Topics
- [Model Versioning](06-model-versioning.md)
- [Model Registry](04-model-registry.md)

## Resources
- [MLflow Reproducibility](https://mlflow.org/docs/latest/tracking.html)
