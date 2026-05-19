# Federated Learning

## TL;DR
Train model across decentralized devices (phones, hospitals) without centralizing data. Devices compute locally, send gradient updates, server aggregates. Privacy + collaboration.

## Core Intuition
Hospital A, B, C have patient data. Can't share data (privacy). Solution: send model to each hospital, they train locally, send gradient updates, aggregate.

## How It Works

**Federated learning workflow:**
1. Server sends model to 100 devices
2. Each device trains locally for N iterations
3. Devices send gradient updates to server
4. Server averages gradients
5. Server sends updated model back
6. Repeat

Gradient updates ≠ raw data. Privacy preserved.

| Component | Detail |
|-----------|--------|
| Device | Hospital/phone, trains locally |
| Update | Gradient update (not raw data) |
| Server | Aggregates updates |
| Privacy | Data stays on device |

## Key Properties / Trade-offs
- Privacy: data never leaves device
- Communication: gradient updates sent over network (bandwidth cost)
- Accuracy: local training on limited data might reduce accuracy
- Complexity: coordination across devices is hard

## Common Mistakes / Gotchas
- Gradient updates leak data (privacy attack possible)
- Device dropout (devices go offline mid-training)
- Non-IID data (devices have different data distributions)
- Communication bottleneck (hundreds of devices × large gradients)

## Best Practices
- **Secure aggregation:** encrypt gradient updates (can't see individual updates)
- **Compression:** compress gradients before sending (reduce bandwidth)
- **Async update:** devices don't need to synchronized (handles dropout)
- **Privacy + federation:** combine differential privacy + federated learning

## Code Example
```python
# Simplified federated learning
def federated_training():
    server_model = load_model()
    
    for round in range(num_rounds):
        # Send model to devices
        device_updates = []
        
        for device in devices:
            local_model = copy(server_model)
            # Device trains locally
            local_update = device.train(local_model, local_data)
            device_updates.append(local_update)
        
        # Server aggregates
        avg_update = average(device_updates)
        server_model = apply_update(server_model, avg_update)
```

## Interview Q&A
**Q: Federated learning: 100 devices, 50% go offline. Training fails?**
A: No. Async aggregation: server aggregates updates from online devices, discards offline. Model still improves (slower, but works). Next round, different devices participate.

**Q: Gradient updates leak privacy. How to prevent?**
A: Differential privacy + secure aggregation. (1) Add noise to gradients on device. (2) Encrypt before sending. (3) Server aggregates encrypted updates (can't see individual gradients). Privacy preserved.

## Interview Quick-Reference
| Component | Purpose |
|-----------|---------|
| Device training | Privacy (data stays local) |
| Gradient update | Communication (efficient) |
| Aggregation | Model improvement |

## Related Topics
- [Privacy-Preserving ML](28-privacy-preserving-ml.md)
- [Differential Privacy](29-differential-privacy.md)

## Resources
- [Federated Learning: Collaborative ML without Centralizing Data](https://arxiv.org/abs/1602.05629)
