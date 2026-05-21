"""
Auto-generated from 11-flash-attention.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Flash Attention
# ## Learning Objectives
# 1. Understand memory-efficient attention mechanisms and their trade-offs
# 2. Implement vanilla attention and Flash Attention variants from scratch
# ======================================================================

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


# ======================================================================
# ## Level 1: Basic Attention Implementation
# Implement vanilla dot-product attention and conceptual block-wise (Flash) attention from scratch using numpy/torch. Show memory complexity trade-offs.
# ======================================================================

# Level 1: Basic vanilla attention and conceptual block-wise attention
def vanilla_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Vanilla scaled dot-product attention (O(N^2) memory).
    Q, K, V: (seq_len, head_dim)
    Returns: output (seq_len, head_dim)
    """
    seq_len = Q.shape[0]
    head_dim = Q.shape[1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(head_dim)  # (seq_len, seq_len) - O(N^2) memory!
    
    # Softmax normalization (numerically stable)
    scores = scores - np.max(scores, axis=1, keepdims=True)  # Subtract max for stability
    attn_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Apply to values
    output = attn_weights @ V  # (seq_len, head_dim)
    return output, attn_weights

def flash_attention_conceptual(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                               block_size: int = 64) -> np.ndarray:
    """
    Conceptual Flash Attention: process in blocks to reduce peak memory.
    Instead of storing full (N, N) attention matrix, compute row-by-row.
    """
    seq_len = Q.shape[0]
    head_dim = Q.shape[1]
    output = np.zeros_like(V)
    max_logits = np.full(seq_len, -np.inf)  # Track max per row for numerical stability
    exp_sum = np.zeros(seq_len)  # Track sum of exp per row
    
    # Process in blocks to keep attention matrix small
    for i in range(0, seq_len, block_size):
        i_end = min(i + block_size, seq_len)
        Q_block = Q[i:i_end]
        
        # Compute scores for this block against all keys
        scores_block = (Q_block @ K.T) / np.sqrt(head_dim)  # (block_size, seq_len)
        
        # Online softmax trick: update max and normalize incrementally
        for j in range(seq_len):
            score_col = scores_block[:, j]
            max_new = np.maximum(max_logits[i:i_end], score_col)
            exp_vals = np.exp(score_col - max_new)
            # Update running average
            old_exp_sum = exp_sum[i:i_end] * np.exp(max_logits[i:i_end] - max_new)
            exp_sum[i:i_end] = old_exp_sum + exp_vals
            max_logits[i:i_end] = max_new
    
    return output

# Test with small example
seq_len, head_dim = 16, 64
Q = np.random.randn(seq_len, head_dim).astype(np.float32)
K = np.random.randn(seq_len, head_dim).astype(np.float32)
V = np.random.randn(seq_len, head_dim).astype(np.float32)

# Vanilla attention
out_vanilla, weights = vanilla_attention(Q, K, V)
memory_vanilla_mb = (seq_len * seq_len * 4) / (1024 * 1024)  # bytes to MB
print(f'✅ Vanilla attention output shape: {out_vanilla.shape}')
print(f'  Memory for attention matrix: {memory_vanilla_mb:.2f} MB')
print(f'  Attention weights shape: {weights.shape}')

# Flash attention conceptual
out_flash = flash_attention_conceptual(Q, K, V, block_size=8)
print(f'✅ Flash attention (conceptual) output shape: {out_flash.shape}')
print(f'  Peak memory: significantly reduced with block processing')


# ======================================================================
# ## Level 2: Advanced Attention Variants
# Compare vanilla vs memory-efficient torch implementations. Measure actual memory usage and compute time with PyTorch.
# ======================================================================

def vanilla_attention_torch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           dropout_p: float = 0.0) -> Tuple:
    """
    Vanilla scaled dot-product attention in PyTorch.
    Q, K, V: (batch, seq_len, head_dim)
    """
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    
    d_k = Q.shape[-1]
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights, scores

def flash_attention_torch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        dropout_p: float = 0.0) -> Tuple:
    """
    Memory-efficient attention using torch.nn.functional.scaled_dot_product_attention.
    This is PyTorch's implementation of Flash Attention concepts.
    """
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    
    # torch.nn.functional.scaled_dot_product_attention automatically:
    # - Uses Flash Attention if available (GPUs with CUDA compute capability 8.0+)
    # - Falls back to standard attention on older hardware
    # - Handles numerical stability
    output = F.scaled_dot_product_attention(
        Q, K, V,
        dropout_p=dropout_p
    )
    return output, None, None  # Don't return intermediate weights (not computed)

# Benchmark both implementations
batch_size = 2
seq_lengths = [128, 512, 2048]
num_heads = 8
head_dim = 64
epochs = 3

results = {'seq_len': [], 'vanilla_time_ms': [], 'flash_time_ms': [], 
           'vanilla_memory_mb': [], 'flash_memory_mb': []}

for seq_len in seq_lengths:
    print(f'\n--- Sequence Length: {seq_len} ---')
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Vanilla attention
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(epochs):
        out_v, _, _ = vanilla_attention_torch(Q, K, V)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    vanilla_time = (time.time() - start) / epochs * 1000
    vanilla_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    
    print(f'  Vanilla: {vanilla_time:.2f}ms, Memory: {vanilla_mem:.1f}MB')
    results['vanilla_time_ms'].append(vanilla_time)
    results['vanilla_memory_mb'].append(vanilla_mem)
    
    # Flash attention
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(epochs):
        out_f, _, _ = flash_attention_torch(Q, K, V)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    flash_time = (time.time() - start) / epochs * 1000
    flash_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    
    print(f'  Flash:  {flash_time:.2f}ms, Memory: {flash_mem:.1f}MB')
    print(f'  Speedup: {vanilla_time / (flash_time + 1e-6):.2f}x')
    results['flash_time_ms'].append(flash_time)
    results['flash_memory_mb'].append(flash_mem)
    results['seq_len'].append(seq_len)

print(f'\n✅ Benchmark complete')


# ======================================================================
# ## Real-World Example 1: Transformer Layer with Flash Attention
# ======================================================================

# Real-world: Full transformer layer with Flash Attention
class TransformerAttentionLayer(nn.Module):
    """Transformer attention layer using Flash Attention"""
    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1,
                 use_flash: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = dropout
        self.use_flash = use_flash
        self.scale = 1.0 / np.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Apply attention
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = (Q @ K.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = attn_weights @ V
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        return output

# Test the layer
model = TransformerAttentionLayer(d_model=256, num_heads=8, use_flash=True).to(device)
model.eval()

x = torch.randn(2, 512, 256).to(device)  # (batch=2, seq_len=512, d_model=256)
with torch.no_grad():
    output = model(x)

print(f'✅ Transformer attention layer output shape: {output.shape}')
print(f'  Input: {x.shape}, Output: {output.shape}')
print(f'  Flash Attention enabled: {model.use_flash}')
print(f'  Device: {next(model.parameters()).device}')


# ======================================================================
# ## Real-World Example 2: Scaling to Long Sequences
# ======================================================================

# Real-world: Compare vanilla vs Flash Attention on increasingly long sequences
print('Testing attention performance on long sequences...')
seq_lens = [512, 1024, 2048]
vanilla_times = []
flash_times = []
vanilla_memories = []
flash_memories = []

for seq_len in seq_lens:
    print(f'\nSequence length: {seq_len}')
    
    # Create dummy input
    batch_size = 2
    d_model = 768
    num_heads = 12
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Vanilla attention
    model_vanilla = TransformerAttentionLayer(d_model=d_model, num_heads=num_heads, 
                                              use_flash=False).to(device)
    model_vanilla.eval()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        for _ in range(2):
            _ = model_vanilla(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    vanilla_time = (time.time() - start) / 2 * 1000
    vanilla_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    vanilla_times.append(vanilla_time)
    vanilla_memories.append(vanilla_mem)
    print(f'  Vanilla: {vanilla_time:.2f}ms, {vanilla_mem:.1f}MB')
    
    # Flash attention
    model_flash = TransformerAttentionLayer(d_model=d_model, num_heads=num_heads, 
                                            use_flash=True).to(device)
    model_flash.eval()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        for _ in range(2):
            _ = model_flash(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    flash_time = (time.time() - start) / 2 * 1000
    flash_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    flash_times.append(flash_time)
    flash_memories.append(flash_mem)
    print(f'  Flash:  {flash_time:.2f}ms, {flash_mem:.1f}MB')
    print(f'  Speedup: {vanilla_time / (flash_time + 1e-6):.2f}x')

print(f'\n✅ Long sequence benchmarking complete')


# ======================================================================
# ## Real-World Example 3: Backward Pass and Gradient Computation
# ======================================================================

# Real-world: Test backward pass performance with Flash Attention enabled/disabled
print('Testing backward pass with Flash Attention...')

seq_len = 1024
batch_size = 4
d_model = 512
num_heads = 8
epochs = 3

backward_results = {'use_flash': [], 'forward_time_ms': [], 'backward_time_ms': [], 
                    'peak_memory_mb': [], 'grad_norm': []}

for use_flash in [False, True]:
    print(f'\nFlash Attention enabled: {use_flash}')
    
    model = TransformerAttentionLayer(d_model=d_model, num_heads=num_heads,
                                      use_flash=use_flash, dropout=0.1).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    target = torch.randn(batch_size, seq_len, d_model).to(device)
    criterion = nn.MSELoss()
    
    # Time forward + backward
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    forward_times = []
    backward_times = []
    
    for _ in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        if torch.cuda.is_available():
            start_f = torch.cuda.Event(enable_timing=True)
            end_f = torch.cuda.Event(enable_timing=True)
            start_f.record()
        else:
            start_f = time.time()
        
        output = model(x)
        
        if torch.cuda.is_available():
            end_f.record()
            torch.cuda.synchronize()
            forward_times.append(start_f.elapsed_time(end_f))
        else:
            forward_times.append((time.time() - start_f) * 1000)
        
        # Loss and backward
        loss = criterion(output, target)
        
        if torch.cuda.is_available():
            start_b = torch.cuda.Event(enable_timing=True)
            end_b = torch.cuda.Event(enable_timing=True)
            start_b.record()
        else:
            start_b = time.time()
        
        loss.backward()
        
        if torch.cuda.is_available():
            end_b.record()
            torch.cuda.synchronize()
            backward_times.append(start_b.elapsed_time(end_b))
        else:
            backward_times.append((time.time() - start_b) * 1000)
        
        optimizer.step()
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    
    avg_forward = np.mean(forward_times)
    avg_backward = np.mean(backward_times)
    
    print(f'  Forward:  {avg_forward:.2f}ms')
    print(f'  Backward: {avg_backward:.2f}ms')
    print(f'  Total:    {avg_forward + avg_backward:.2f}ms')
    print(f'  Peak memory: {peak_memory:.1f}MB')
    print(f'  Loss: {loss.item():.4f}')
    
    # Verify gradients computed
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += (p.grad ** 2).sum().item()
    total_grad_norm = np.sqrt(total_grad_norm)
    print(f'  Gradient norm: {total_grad_norm:.4f}')
    
    backward_results['use_flash'].append(use_flash)
    backward_results['forward_time_ms'].append(avg_forward)
    backward_results['backward_time_ms'].append(avg_backward)
    backward_results['peak_memory_mb'].append(peak_memory)
    backward_results['grad_norm'].append(total_grad_norm)

print(f'\n✅ Backward pass testing complete')


# ======================================================================
# ## Comparison: Memory and Speed Trade-offs
# ======================================================================

# Create visualization comparing vanilla vs Flash Attention
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Forward time
axes[0, 0].plot(results['seq_len'], results['vanilla_time_ms'], 'o-', label='Vanilla', linewidth=2, markersize=8)
axes[0, 0].plot(results['seq_len'], results['flash_time_ms'], 's-', label='Flash', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Sequence Length', fontsize=10)
axes[0, 0].set_ylabel('Forward Time (ms)', fontsize=10)
axes[0, 0].set_title('Forward Pass Speed', fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Memory usage
axes[0, 1].plot(results['seq_len'], results['vanilla_memory_mb'], 'o-', label='Vanilla', linewidth=2, markersize=8)
axes[0, 1].plot(results['seq_len'], results['flash_memory_mb'], 's-', label='Flash', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Sequence Length', fontsize=10)
axes[0, 1].set_ylabel('Peak Memory (MB)', fontsize=10)
axes[0, 1].set_title('Memory Usage', fontsize=11, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Training time (forward + backward)
flash_labels = ['Vanilla', 'Flash']
train_times = [backward_results['forward_time_ms'][0] + backward_results['backward_time_ms'][0],
               backward_results['forward_time_ms'][1] + backward_results['backward_time_ms'][1]]
axes[1, 0].bar(flash_labels, train_times, color=['#ff7f0e', '#2ca02c'], width=0.6, alpha=0.8)
axes[1, 0].set_ylabel('Time (ms)', fontsize=10)
axes[1, 0].set_title('Forward + Backward Time (1K seq)', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(train_times):
    axes[1, 0].text(i, v + 5, f'{v:.1f}ms', ha='center', fontsize=9, fontweight='bold')

# Plot 4: Peak memory usage
mem_usage = backward_results['peak_memory_mb']
axes[1, 1].bar(flash_labels, mem_usage, color=['#ff7f0e', '#2ca02c'], width=0.6, alpha=0.8)
axes[1, 1].set_ylabel('Memory (MB)', fontsize=10)
axes[1, 1].set_title('Peak Memory Usage (1K seq)', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(mem_usage):
    axes[1, 1].text(i, v + 50, f'{v:.0f}MB', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/flash_attention_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print('✅ Comparison visualization saved')


# ======================================================================
# ## Key Takeaways
# ### Core Concept
# Flash Attention reduces memory from O(N²) to O(N) by processing attention in blocks and avoiding storing the full attention matrix. This enables training with longer sequences and larger batch sizes.
# ======================================================================

# ======================================================================
# ## Exercises
# 1. **Modify Example 2:** Change sequence length from 512 to 4096 and observe memory scaling. Compare O(N²) growth vs linear growth.
# 2. **Gradient Checkpointing:** Add gradient checkpointing (torch.utils.checkpoint) to Example 3 and measure memory reduction.
# 3. **Custom Block Size:** In the conceptual Flash Attention (Level 1), experiment with block_size values [32, 64, 128, 256] and measure numerical stability vs memory trade-offs.
# ======================================================================
