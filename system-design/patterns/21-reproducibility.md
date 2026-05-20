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

## Detailed Trade-off Analysis

| Aspect | Manual Reproducibility | MLflow Tracking | Full Reproducibility Setup | Container-Based |
|--------|----------------------|-----------------|---------------------------|-----------------|
| Setup time | 1 day | 3 days | 1 week | 2 days |
| Reproducibility guarantee | 70% (errors possible) | 85% (some deps may vary) | 99%+ (complete control) | 95% (kernel variations) |
| Scalability | Manual per model | Tracks at scale | Requires discipline | Scales well |
| Cost | $0 (manual) | $500/month | $1000/month (storage) | $200/month (registry) |
| Speed to reproduce | 1-2 hours (debug) | 30 min (exact run) | 10 min (full snapshot) | 5 min (container pull) |

**Decision:** Simple projects → manual seeding. Production at scale → MLflow. Critical audit requirements → full reproducibility + containers.

---

## Production Failure Scenarios

**Scenario 1: Re-trained model, different accuracy, can't debug**
- Original trained 6 months ago. Retrain with same code. Accuracy drops 2%. Why?
- No seed recorded, dependencies updated, GPU changed. Can't isolate cause.
- Fix: Record seed, deps, GPU in model card. Re-train, isolate variable (re-seed, downgrade torch, use same GPU). Identifies the culprit.

**Scenario 2: Data pipeline quietly changed**
- Train on "training_data_v1.csv". 6 months later, use training_data_v1.csv. But pipeline updated, file changed.
- Can't reproduce original model because data different.
- Fix: Version data with hash or DVC. Store exact data version with model.

**Scenario 3: Floating-point differs between GPU/CPU**
- Train on GPU A (specific hardware). Reproduce on GPU B (newer hardware). Weights slightly different due to floating-point precision.
- Metrics differ by 0.01%. Technically not reproducible, but practically OK.
- Fix: Accept floating-point tolerance (±0.1%). Document hardware requirements. Or: use integer arithmetic if possible.

**Scenario 4: Upstream library breaks backward compatibility**
- Lock torch==1.12. Year later, PyPI drops torch 1.12 support. Can't install dependencies.
- Can't reproduce old model.
- Fix: Store docker image or freeze entire environment (conda pack). Or: pin PyPI mirror snapshot.

---

## Implementation Guidance

**Wrong:** Train model, don't record seed, GPU, dependencies. Hope it's reproducible.
**Right:** Record 5 components (code hash, data hash, seed, deps version, GPU type) in model card. Before production deployment, verify reproduction one more time.

**Wrong:** "Exact reproducibility is impossible, give up."
**Right:** 99% reproducibility is achievable (seed + frozen deps + docker). Accept floating-point tolerance. Document deviations.

---

## Sophisticated Interview Q&A

**Q1: Exact floating-point reproduction across GPU/CPU. Possible?**
A: No. GPUs use different precision (bfloat16, fp16) than CPU. Solution: (1) log GPU type with model. (2) Test both, document differences. (3) Accept ±0.1% tolerance. (4) If exact needed, use CPU (slower but deterministic). (5) Or: use quantized integer arithmetic.

**Q2: Model trained before torch updated. Torch API broke compatibility. Re-produce?**
A: (1) Pin old torch version in requirements.txt. (2) Or: containerize with exact torch version. (3) Or: use torch version compatibility layer (adapt code). (4) Best: maintain torch compatibility (update code, test, document). Avoid pinning forever.

**Q3: Reproducibility adds 20% overhead. Worth it for 1 model?**
A: For 1 model, overhead likely not justified. For 5+ models, yes (amortized cost). For regulated domain (finance, healthcare), yes (compliance). For startups, maybe not. Cost-benefit: model value × reproduction cost / total cost.

**Q4: How ensure team follows reproducibility practices?**
A: (1) Template + checklist (require in code review). (2) CI/CD gate: daily reproduction test (fails if model not reproducible). (3) Documentation: model cards include 5 components. (4) Automation: script to seed, lock deps, record metadata.

---

## Cost & Resource Analysis

**Manual reproducibility:** 5% development overhead (seed, comments, documentation). For 100 models: 2-3 engineer weeks/year.
**MLflow setup:** 1 engineer week initial setup. $500/month infrastructure. Handles 1000+ models.
**Full reproducibility:** 2 engineer weeks initial + 1 week/month maintenance. Data storage: $1K/month (versioned datasets).
**Container-based:** 1 engineer day per model + registry cost $200/month.

**ROI:** Debugging time saved: 1-2 hours per incident × 1-2 incidents/month = $2-4K saved/month. vs. $500-1000 infrastructure cost. Break-even at 1-2 incidents/month.

---

## Monitoring & Observability

**Key metrics:** Reproducibility score (% of models meet all 5 requirements), reproduction success rate (daily test pass rate), average time to reproduce a model, data staleness (how long since data version updated)

**Alerts:** Reproducibility test fails (model can't reproduce), dependency version unsupported (will break in future), data version older than 3 months (staleness), seed not set (reproducibility broken)

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
