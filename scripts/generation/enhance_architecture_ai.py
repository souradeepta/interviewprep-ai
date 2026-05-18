#!/usr/bin/env python3
"""Enhance Architecture/Trade-offs sections for all 40 AI fundamentals concepts."""

import os
import re

BASE = "/home/sbisw/github/interviewprep-ml"

# Comprehensive architecture sections for AI concepts
ARCHITECTURE_AI = {
    "01-gradient-descent": """### Gradient Descent Variants

| Variant | Data Per Step | Update Frequency | Stability | Speed | Memory |
|---------|---------------|------------------|-----------|-------|--------|
| **Batch GD** | All N samples | Once per epoch | Very stable | Slow | High |
| **SGD** | 1 sample | N times per epoch | Noisy/unstable | Fast | Low |
| **Mini-batch** | 32-256 samples | N/batch times per epoch | Balanced | Fast | Medium |

```mermaid
graph TD
    A["Training Data"] -->|Batch Split| B["Mini-batch"]
    B --> C["Compute Loss"]
    C --> D["Compute Gradient"]
    D --> E["Update Weights:<br/>θ = θ - lr*∇L"]
    E --> F{All batches<br/>processed?}
    F -->|No| B
    F -->|Yes| G{Converged?}
    G -->|No| A
    G -->|Yes| H["Final Weights"]

    style B fill:#e1f5ff
    style E fill:#fff3e0
    style H fill:#e8f5e9
```

### Trade-off Analysis

**Learning Rate (α)**
- High (0.1+): Fast initial progress, but oscillates and may diverge
- Low (1e-5): Slow but stable convergence
- Best: Start high, decay over time (scheduling)

**Batch Size**
- Small (1): Noisy gradients, escapes local minima, but slow per-step progress
- Large (N): Clean gradient signal, but may get stuck in saddle points
- Optimal: 32-256 balances noise reduction and computational efficiency""",

    "02-backpropagation": """### Forward vs Backward Pass

```mermaid
graph LR
    A["Input x"] -->|Forward| B["Layer 1"]
    B -->|Forward| C["Layer 2"]
    C -->|Forward| D["Output ŷ"]
    D -->|Loss Fn| E["Loss L(ŷ,y)"]
    E -->|Backward| F["dL/dw2"]
    F -->|Backward| G["dL/dw1"]
    G -->|Backward| H["dL/dx"]

    style A fill:#e1f5ff
    style D fill:#fff3e0
    style E fill:#ffebee
    style H fill:#e8f5e9
```

### Gradient Flow Problems & Solutions

| Problem | Cause | Solution | Impact |
|---------|-------|----------|--------|
| **Vanishing Gradient** | Many sigmoid layers | ReLU activation, skip connections | Fix: deep networks trainable |
| **Exploding Gradient** | Large weight initialization | Gradient clipping, proper init | Fix: stable training |
| **Dead Neurons** | ReLU outputs 0 | Leaky ReLU, ELU | Fix: sustained learning |

### Computational Complexity

- Forward pass: O(n_params) for matrix multiplications
- Backward pass: ~2x forward pass cost (gradient computation reuses activations)
- Memory: Need to store all activations for backward pass (O(depth))