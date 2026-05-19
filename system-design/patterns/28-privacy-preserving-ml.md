# Privacy-Preserving ML

## TL;DR
Train models without exposing private data. Techniques: differential privacy (add noise), federated learning (train on edge), homomorphic encryption (compute on encrypted data). Trade latency/accuracy for privacy.

## Core Intuition
Hospital has patient data. Can't share with cloud. Solution: send model to hospital, train locally, send results back (federated). Or: add noise to data, train, model doesn't memorize individuals (differential privacy).

## How It Works

**Privacy techniques:**

1. **Differential privacy:** add noise to data/gradients
   - Model can't memorize individuals
   - Cost: accuracy drops 1-5%

2. **Federated learning:** train on decentralized data
   - Phone/hospital trains locally, sends gradient updates
   - Server aggregates, sends back
   - Data never leaves device

3. **Homomorphic encryption:** compute on encrypted data
   - Encrypt data, train model on encrypted data
   - Cost: 1000x slower

| Technique | Privacy | Accuracy | Speed |
|-----------|---------|----------|-------|
| Diff privacy | High | 95% | Fast |
| Federated | High | 95% | Medium |
| Homomorphic | Highest | 95% | Very slow |

## Key Properties / Trade-offs
- Privacy vs accuracy: enforcing privacy reduces accuracy
- Privacy vs speed: more privacy usually means slower
- Complexity: privacy-preserving ML more complex to implement

## Common Mistakes / Gotchas
- Assuming anonymization = privacy (can be re-identified)
- Insufficient noise (privacy attack succeeds)
- No privacy audit (privacy claims unvalidated)
- Ignoring computational cost (homomorphic encryption very slow)

## Best Practices
- **Privacy audit:** hire external party to verify privacy
- **Differential privacy:** use published libraries (not DIY)
- **Federated: careful gradient aggregation (averagingalone leaks data)
- **Benchmark privacy vs accuracy:** measure trade-off

## Code Example
```python
from tensorflow_privacy import DPKerasOptimizer

# Differential privacy: add noise to gradients
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
dp_optimizer = DPKerasOptimizer(
    optimizer,
    l2_norm_clip=1.0,  # Clip gradients
    noise_multiplier=1.0  # Add Gaussian noise
)

# Train with DP optimizer
model.compile(optimizer=dp_optimizer, loss='mse')
model.fit(X, y, epochs=10)
```

## Interview Q&A
**Q: GDPR: user requests right-to-be-forgotten. Model trained on their data. What to do?**
A: (1) Remove their data from training. (2) Retrain model. Or: (3) Use differential privacy (model already doesn't memorize individuals). Or: (4) Use certified forgetting (retraining procedure that provably removes their influence).

**Q: Federated learning: train on hospital data. Hospital sues, says you stole data. How prevent?**
A: (1) Contract specifying data never leaves hospital. (2) Audit: verify no data exfiltrated. (3) Differential privacy: even if gradient intercepted, doesn't reveal individuals. (4) Encryption: data encrypted on hospital device.

## Interview Quick-Reference
| Technique | Best For |
|-----------|----------|
| Diff privacy | Quick privacy, accept slight accuracy drop |
| Federated | Decentralized data (hospitals, phones) |
| Homomorphic | Highest privacy (if speed not critical) |

## Related Topics
- [Data Governance](26-data-governance.md)
- [Differential Privacy](29-differential-privacy.md)

## Resources
- [Differential Privacy Library](https://github.com/tensorflow/privacy)
