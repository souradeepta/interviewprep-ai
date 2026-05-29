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

**Failure 1: CUDA Non-Determinism**
- **Symptom:** Same code plus same data gives different loss values on different runs (variance ~0.5% across runs with the same seed).
- **Root cause:** Non-deterministic CUDA kernels — operations like atomicAdd in parallel reductions produce results that depend on thread execution order, which varies run-to-run.
- **Detection:** Run the same training script three times with the same seed. Measure the standard deviation of the final validation loss. Anything above 0.1% signals non-determinism.
- **Fix:** Set `torch.use_deterministic_algorithms(True)` and `torch.backends.cudnn.deterministic = True`. Accept a 10-15% training slowdown as the cost of determinism. Document this overhead so teammates don't revert the setting for speed.

**Failure 2: Preprocessing Code Unversioned**
- **Symptom:** Cannot reproduce a six-month-old result despite having the model checkpoint and the same dataset file name — the reproduced model accuracy is 3% lower.
- **Root cause:** The dataset was versioned (DVC) but `preprocessing.py` was not. A normalization constant was updated six months ago without tracking. The "same" data now goes through different transformations.
- **Detection:** Hash the preprocessing pipeline code and log the hash as experiment metadata in MLflow. Alert when the hash changes between runs.
- **Fix:** Commit hash of all pipeline code in experiment metadata. Use DVC for both data and code versioning. The model card must include: git commit hash of training code, DVC hash of training data, DVC hash of preprocessing pipeline.

**Failure 3: Environment Drift Between Machines**
- **Symptom:** Training results differ between two nominally identical machines that both show the same seed, code commit, and data hash.
- **Root cause:** A library version mismatch — torch 1.12 vs torch 2.0 have different default behaviors for `nn.LayerNorm` and weight initialization. The requirements.txt said `torch>=1.10`, so pip resolved different versions on the two machines.
- **Detection:** Compare full `pip freeze` outputs between machines. Any version difference is a potential source of divergence.
- **Fix:** Pin ALL dependencies with exact versions (`torch==2.0.1`, not `torch>=1.10`). Better: use a Docker image per experiment and push the image hash to the model card. No Docker image = no reproducibility guarantee for long-lived models.

**Failure 4: Seed Not Set Globally**
- **Symptom:** Deterministic CUDA enabled, same torch seed, but results still vary between runs.
- **Root cause:** `torch.manual_seed` was set but `np.random.seed`, `random.seed`, and DataLoader worker seeds were not. Data augmentation (using NumPy's random state) produces different augmented samples each run.
- **Detection:** Audit every file that imports `random`, `numpy`, or `PIL` for random calls. Log the hash of the first 100 augmented samples as a run artifact.
- **Fix:** Set seeds for all: torch, numpy, Python's built-in random, and `DataLoader(worker_init_fn=seed_worker)`. The `seed_worker` function should call `np.random.seed` and `random.seed` inside each worker process.

---

## Implementation Guidance

**Wrong:** Train model, don't record seed, GPU, dependencies. Hope it's reproducible.
**Right:** Record 5 components (code hash, data hash, seed, deps version, GPU type) in model card. Before production deployment, verify reproduction one more time.

**Wrong:** "Exact reproducibility is impossible, give up."
**Right:** 99% reproducibility is achievable (seed + frozen deps + docker). Accept floating-point tolerance. Document deviations.

---

## Interview Q&A

**Q1: Is exact floating-point reproduction possible across different GPU hardware?**
A: No, and that is expected. GPUs use different precision modes (bfloat16, fp16, tf32) and different hardware generations produce different rounding. The practical solution is to (1) log the exact GPU model in experiment metadata, (2) define a tolerance (typically ±0.1% on validation metrics) rather than demanding bit-identical results, and (3) use CPU-based inference if you need exact determinism for regulatory audit — accept the 5-10× latency cost as the price of exactness.

**Q2: Model trained with torch 1.12 needs to be reproduced, but torch 1.12 is no longer supported on PyPI. What do you do?**
A: The clean solution is a Docker image with the exact environment frozen at training time. If the image was pushed to a registry, pull it and reproduce inside the container. If the image was never saved, you have a gap in the reproducibility chain — the best you can do is use an archived PyPI mirror snapshot or conda-pack the environment and archive it. Going forward: push Docker images to ECR or Artifact Registry for every production training run; images are cheap (10-20GB each) and eliminate this class of failure entirely.

**Q3: Reproducibility adds 20% development overhead. Is it worth it for a single experimental model?**
A: For a single exploratory experiment, the overhead is not justified. For any model that will be used in production, reviewed by regulators, or maintained for more than three months, the investment pays off at the first debugging incident. A good rule of thumb: apply full reproducibility (Docker + DVC + seed) to any model that is (a) customer-facing, (b) in a regulated domain (finance, healthcare), or (c) used to make decisions worth more than $10K/month. For pure research, a git commit + seed is sufficient.

**Q4: How do you ensure a team of 10 engineers follows reproducibility practices consistently?**
A: Checklist-based code review is necessary but not sufficient — people skip it under time pressure. The reliable solution is automation: a CI/CD gate that rejects model registration unless the five components (code hash, data hash, seed, dependency lock file, hardware type) are present in the MLflow run metadata. This converts a social contract into a technical constraint. Pair this with a model card template that engineers fill out at registration time. Review the model card in the deployment PR, not in a separate process.

**Q5: When would you NOT invest in full reproducibility?**
A: Three situations: (1) Pure research prototypes with a lifetime under two weeks — the overhead exceeds the benefit. (2) Online learning systems that update continuously — the model state is never the same twice by design; reproducibility is instead defined at the dataset and hyperparameter level, not the weight level. (3) Systems where the ground truth labels are inherently non-deterministic (e.g., human annotators who disagree) — exact reproduction of labels is impossible, so exact model reproduction is meaningless.

**Q6: You run the same training script with the same seed and get different loss values. How do you debug?**
A: Work through the layers in order. First, check whether `torch.use_deterministic_algorithms(True)` is set — if not, CUDA non-determinism is the likely cause. Second, verify that all random number generators are seeded (torch, numpy, random, DataLoader workers). Third, check whether any external data source has changed between runs (cloud storage file overwritten, database query returning different rows). Fourth, check whether a dependency was silently upgraded (compare `pip freeze` outputs between runs). Each of these explains a different class of non-reproducibility; the debugging order follows likelihood.

**Q7: What breaks at 10× scale — 1,000 models instead of 100?**
A: The storage model breaks first. 1TB of versioned datasets × 1,000 model lineages creates cross-dependencies that make it expensive to garbage-collect old data (you can't delete a dataset version if 50 models reference it). The fix is to version datasets at the experiment level and use reference counting with automated expiry policies. The second scaling problem is MLflow query performance — 500K runs in a single tracking server slows UI and API. The solution is to shard by team or model family and use a federated tracking setup.

**Q8: A regulatory auditor asks you to reproduce a model decision made 18 months ago. What do you need?**
A: You need the five components: (1) the exact model artifact (weights + architecture) from the model registry, (2) the exact input data (versioned in DVC or archived in S3 with an immutable key), (3) the serving code version (git commit hash), (4) the inference environment (Docker image), and (5) the feature computation code and its version. If any of these is missing, you cannot produce a legally defensible reproduction. In regulated industries (lending, healthcare), this checklist is not optional — it must be verified at every model deployment as part of the release process.

---

## Cost & Resource Analysis

**Manual reproducibility:** 5% development overhead (seed, comments, documentation). For 100 models: 2-3 engineer weeks/year.
**MLflow setup:** 1 engineer week initial setup. $500/month infrastructure. Handles 1000+ models.
**Full reproducibility:** 2 engineer weeks initial + 1 week/month maintenance. Data storage: $1K/month (versioned datasets).
**Container-based:** 1 engineer day per model + registry cost $200/month.

**ROI:** Debugging time saved: 1-2 hours per incident × 1-2 incidents/month = $2-4K saved/month. vs. $500-1000 infrastructure cost. Break-even at 1-2 incidents/month.

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| DVC data storage (S3) | $0.023/GB | 1TB versioned history | $23 |
| MLflow experiment tracking (managed) | $0.10/run | 500 runs/month | $50 |
| Docker image registry (ECR) | $0.10/GB | 10GB images | $1 |
| Reproducibility audit (engineer time) | $200/hr | 2 hr/month | $400 |
| **Total** | | | **~$474/month** |

The $474/month total is almost entirely engineer time ($400, ~84%). Infrastructure — S3 versioning, MLflow, Docker — costs under $75/month combined. This means reproducibility at scale is cheap to run but expensive to maintain if the discipline breaks down. The key investment is automation: a CI/CD step that verifies the five reproducibility components (code hash, data hash, seed, dependency lock, hardware log) are present before any model is registered. That one-time automation investment of ~$3K engineer time pays for itself by eliminating reproducibility incidents, each of which typically costs 4-8 hours of debugging ($800-1600).

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

## Interview Quick-Reference
| Component | How to Version |
|-----------|---|
| Code | git commit |
| Data | DVC commit or hash |
| Random | seed=42 |
| Deps | poetry.lock |
| Hardware | Docker image hash |

## Related Topics
- [Model Versioning](06-model-versioning.md)
- [Model Registry](04-model-registry.md)

## Resources
- [MLflow Reproducibility](https://mlflow.org/docs/latest/tracking.html)
