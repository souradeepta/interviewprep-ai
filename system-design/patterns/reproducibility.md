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

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
