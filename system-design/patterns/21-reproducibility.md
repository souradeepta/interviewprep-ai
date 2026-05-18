# Reproducibility

## TL;DR
Core ML system design pattern for production.

## Core Intuition
[Intuitive explanation]

## How It Works
[Technical details]

## Key Properties / Trade-offs
- Property 1
- Property 2

## Common Mistakes / Gotchas
- Mistake 1
- Mistake 2

## Best Practices
- Pin all library versions in requirements.txt or conda environment file
- Set random seeds everywhere — Python, NumPy, PyTorch, TensorFlow
- Use deterministic algorithms when available (PyTorch: torch.use_deterministic_algorithms)
- Store training data snapshots as versioned artifacts, not live queries
- Record hardware specs — GPU type affects floating point results
- Containerize the training environment with Docker
- Automate full reproduction pipeline from data → trained model → metrics

## Interview Q&A

**Q: What are the minimum requirements for a reproducible ML experiment?**
A: Code: exact git commit hash. Data: exact dataset version (hash or immutable reference). Environment: exact library versions (requirements.txt or conda environment), hardware type. Random seeds: set in NumPy, PyTorch, and Python random for all operations that use randomness. Configuration: all hyperparameters stored in a config file or experiment tracker. Given these five things, you should be able to re-run the exact experiment. If you can't, identify which element is not reproducible and fix it.

**Q: How do you handle non-determinism in GPU training for reproducibility?**
A: GPU operations are non-deterministic by default (CUDA's parallel reduction algorithms are non-associative). Enable determinism: set CUDA_LAUNCH_BLOCKING=1, torch.use_deterministic_algorithms(True), and cuDNN benchmark off (torch.backends.cudnn.benchmark = False). Trade-off: deterministic mode can be 20-50% slower. Use it for: debugging experiments where you need to isolate changes, final training runs before deployment. Accept non-determinism in exploratory training but record the variance range to understand expected performance variation.

**Q: What is the difference between reproducibility and replicability in ML research?**
A: Reproducibility: re-running the same code with the same data produces the same result (computational reproducibility). Replicability: running a different implementation of the same method on different data produces similar results (scientific replicability). For production ML: focus on computational reproducibility—you need to rebuild models exactly in case of rollback or audit. Replicability matters for research claims but is a separate concern. Production model training should be reproducible within the same compute environment; replicability across environments is harder and often not required.

**Q: How do you maintain reproducibility when training data comes from a live production database?**
A: Never train directly against a live database—snapshot training data at a fixed point in time and store it immutably. Create a training data snapshot pipeline: extract data with explicit filters (date ranges, version flags), store in versioned storage (S3 with versioning, Delta Lake), log the query used and execution timestamp. If you must refresh training data, create a new dataset version rather than overwriting the old one. The training pipeline should accept a dataset version as input, not query the live database directly.

**Q: How do you audit which training data was used to produce a production model?**
A: Maintain a model registry entry that links: model artifact to training run to dataset version to raw data snapshots. Store the dataset hash alongside the model artifact. For regulated industries, this audit trail must be preserved for the lifetime of the model (often 5-7 years). Test your audit capability: given a model version in production, can you reproduce the exact training dataset? If not, your audit trail is incomplete. Tools: MLflow lineage tracking, SageMaker Experiments, or custom PostgreSQL lineage tables.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
