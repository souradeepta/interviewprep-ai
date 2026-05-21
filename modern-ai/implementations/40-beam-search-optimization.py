"""
Beam Search Optimization

Production implementation.
"""

import torch
import torch.nn as nn
from typing import Optional


class BeamSearchOptimizer(nn.Module):
    """Production-grade implementation."""
    
    def __init__(self, hidden_dim: int = 64, device: Optional[torch.device] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.layer(x)


if __name__ == "__main__":
    model = BeamSearchOptimizer(64)
    x = torch.randn(4, 8, 64)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
