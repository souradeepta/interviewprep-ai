# Differential Privacy

## TL;DR
Add noise to data or gradients during training. Model cannot memorize individuals (membership inference attack fails). Formally proven privacy: ε-differential privacy (smaller ε = more private).

## Core Intuition
Data = {Alice, Bob, Charlie}. Train model on data. Model memorizes Alice. Attack: "is Alice in training data?" → Yes (privacy leak). Solution: add noise so model can't memorize Alice.

## How It Works

**Differential privacy:**
- Add Gaussian noise to gradients during training
- Smaller ε = more noise = more privacy (but less accuracy)
- ε=1: good privacy. ε=10: less privacy.

| ε | Privacy | Accuracy |
|---|---------|----------|
| 0.5 | Very high | 90% |
| 1.0 | High | 92% |
| 5.0 | Medium | 94% |
| 10.0 | Low | 95% |

## Key Properties / Trade-offs
- Privacy vs accuracy: more privacy → more accuracy drop
- Complexity: requires clipping and noise addition
- Proof: differential privacy = formal proof (unlike heuristics)

## Common Mistakes / Gotchas
- DIY differential privacy (easy to get wrong)
- Insufficient noise (privacy bound too loose)
- No privacy audit (claims not validated)
- Confusing ε values (different libraries use different scales)

## Best Practices
- **Use library:** TensorFlow Privacy, Opacus (PyTorch)
- **Test privacy:** membership inference attack to validate
- **Report ε value:** always report privacy budget
- **Noise scheduling:** start high noise, reduce over training
- **Batch size:** smaller batch → more privacy needed

## Code Example
```python
import opacus
from opacus.utils.batch_memory_manager import BatchMemoryManager

# Attach DP to optimizer
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

privacy_engine = opacus.PrivacyEngine(
    model,
    batch_size=32,
    sample_size=len(dataset),
    alphas=[10, 100],
    noise_multiplier=1.0,
    max_grad_norm=1.0
)
privacy_engine.attach(optimizer)

# Train normally, DP is automatic
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        loss = model(batch_x, batch_y)
        loss.backward()
        optimizer.step()
```

## Interview Q&A
**Q: Model trained with DP (ε=1.0). Privacy guarantee?**
A: Formal guarantee: attacker with ~N/2 data samples can't determine if single person is in training set (at ε=1.0). More precisely: probability attacker guesses wrong is at least 50% + (1-e^(-1))/2 = 68%.

**Q: DP reduces accuracy 5%. Acceptable?**
A: Depends on use case. Recommendation (95% → 90%): maybe. Medical diagnosis (95% → 90%): unacceptable. Measure fairness-accuracy frontier, choose based on user needs.

## Interview Quick-Reference
| ε | Privacy Level |
|---|---|
| 0.1 | Excellent (academic research) |
| 1.0 | Good (industry standard) |
| 10.0 | Weak (privacy claim but not strong) |

## Related Topics
- [Privacy-Preserving ML](28-privacy-preserving-ml.md)
- [Data Governance](26-data-governance.md)

## Resources
- [Differential Privacy Library](https://github.com/tensorflow/privacy)
- [Opacus: PyTorch DP Library](https://opacus.ai/)
