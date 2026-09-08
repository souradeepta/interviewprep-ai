---
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: "Hu, Shen, Wallis, et al."
year: 2021
venue: "ICLR"
doi: "https://doi.org/10.48550/arXiv.2106.09685"
arxiv: "https://arxiv.org/abs/2106.09685"
domain: "nlp"
difficulty: "intermediate"
interview_frequency: "high"
related_concepts:
  - llm/concepts/01-transformers
  - llm/concepts/16-pre-training-fine-tuning
  - modern-ai/concepts/13-parameter-efficient-fine-tuning
---

# LoRA: Low-Rank Adaptation of Large Language Models

## Paper Overview

**Title:** LoRA: Low-Rank Adaptation of Large Language Models for Task-Specific Inference

**Authors:** Edward J. Hu, Yelong Shen, Philipp Wallis, et al. (Microsoft)

**Published:** ICLR 2021 | [arXiv](https://arxiv.org/abs/2106.09685)

**Citation:** 7,000+ (transformed how people fine-tune large models)

Fine-tuning massive models (7B-175B parameters) is expensive: requires storing optimizer states (4× model size), updating all parameters, storing entire fine-tuned model per task. LoRA solves this by training only **low-rank update matrices** added to frozen pre-trained weights. Instead of updating W (d×d), add ΔW = BA where B is d×r and A is r×d (r << d). LoRA fine-tunes on 7B parameter model using only ~1% of parameters (r=8-16), reducing training memory 10-100×, enabling fine-tuning on consumer GPUs. The trick: low-rank matrices capture task-specific adaptations; full-rank not needed. This sparked practical fine-tuning at scale and is now standard (used in LLaMA, Alpaca, Vicuna, etc.).

**Why this matters for interviews:** Fine-tuning large models is essential for deploying to new tasks. LoRA is the practical solution: cheap, fast, requires minimal storage. You'll use it in production systems. Understand: why low-rank works, rank selection, merging adapters, combining LoRA with quantization.

---

## Core Contribution

### The Problem: Fine-tuning Overhead

Fine-tuning GPT-3 (175B):
- Memory: 4× model size for optimizer states = 680GB (impractical)
- Per-task model: 175B × num_tasks storage (1000 tasks = 175TB)
- Training time: even on GPU, weeks
- Hardware: 8× A100 GPUs minimum

This made fine-tuning infeasible for most practitioners.

### The Solution: Low-Rank Adaptation (LoRA)

Key insight: task-specific weight updates W_new = W + ΔW, and ΔW is often low-rank.

Instead of updating W (d × d):
```
W_new = W + ΔW
      = W + BA  (B: d×r, A: r×d, r << d)
```

**Benefits:**
- Train B, A only (2 × d × r parameters vs. d² full W)
- Frozen W reduces memory (no gradients for W)
- Faster training (fewer parameters)
- Smaller artifact: save only B, A (not W), typically 1-10MB per task

### Key Innovation: Rank Selection

Empirically: r=8-16 works for most tasks on 7B models. r=4 for smaller models, r=32 for larger. With r=8 on 7B model:
- LoRA params: 2 × 7B × 8 / 7B = ~16 parameters per original parameter
- Memory savings: ~100× (only LoRA gradients + optimizer states)
- Accuracy: 99.5% of full fine-tune in many cases

---

## Key Ideas & Algorithm

### LoRA Mechanism

**Standard fine-tuning:**
```
y = W_original * x
Loss = L(y, target)
W_new = W_original - lr * dL/dW
```

Problem: dL/dW is d×d, needs gradient buffer.

**LoRA fine-tuning:**
```
y = (W_original + BA) * x
  = W_original * x + (B * (A * x))  ← rank-r update
Loss = L(y, target)
dL/dB, dL/dA computed and stored (much smaller)
B, A updated: B_new = B - lr * dL/dB, A_new = A - lr * dL/dA
W_original stays frozen
```

**Inference:**
```
y = (W + BA) * x  (combine at once, same speed as W alone)
```

### LoRA for Different Layer Types

| Layer Type | LoRA Application | Notes |
|------------|------------------|-------|
| Linear (attention Q,K,V) | Yes, directly | Low-rank works well |
| Linear (feed-forward) | Yes, optional | Can speed up or reduce quality |
| Embeddings | Yes | Lower priority than attention |
| Layer norm | No | Skip, parameters sparse |

### Rank Selection Strategy

| Rank (r) | 7B Model | 13B Model | 70B Model | Trade-off |
|----------|----------|-----------|-----------|-----------|
| 4 | Tiny (100K params) | Tiny | Tiny | Might underfit |
| 8 | Small (200K params) | Small | Small | Good default |
| 16 | Medium (400K params) | Medium | Medium | Better but slower |
| 32 | Large (800K params) | Large | Large | Approaches full fine-tune |
| 64 | Very large (1.6M params) | Large | Large | Usually unnecessary |

**Guidance:** Start with r=8, increase if validation loss plateaus. Diminishing returns after r=32.

### Combining LoRA Adapters

**Single task:** Use one LoRA (B, A)

**Multi-task inference:**
- Option 1: Train separate LoRA per task, swap at inference (no memory overhead)
- Option 2: Merge LoRA into model for single-file deployment (W_final = W + BA)

**Multi-task training (simultaneously):**
- Share frozen W, separate LoRA per task, add task-specific loss weights

---

## Architecture & Trade-offs

### LoRA vs. Full Fine-Tuning

| Aspect | Full Fine-Tune | LoRA (r=8) | LoRA (r=16) |
|--------|----------------|-----------|------------|
| **Params trained** | 7B | 100K | 200K |
| **Memory (training)** | 280GB (4× model) | 2.8GB | 5.6GB |
| **Training time** | 3 days (8xA100) | 2 hours (1xA100) | 4 hours (1xA100) |
| **Model artifact size** | 28GB | 100KB | 200KB |
| **Final accuracy** | 100% | 99-99.5% | 99.5% |

### LoRA vs. Other Parameter-Efficient Methods

| Method | Params Trained | Accuracy | Deployment |
|--------|----------------|----------|-----------|
| Full fine-tune | 100% | 100% | Large artifact |
| LoRA (r=8) | 0.1% | 99-99.5% | Small adapters |
| Prefix tuning | 0.1% | 95-98% | Small prefix |
| BitFit | 0.01% | 90-95% | Tiny, but worse |
| QLoRA | 0.01% (4-bit) | 98-99% | Tiny + quantized |

### When to Use Each

| Scenario | Best Method |
|----------|------------|
| Large labeled data (100K+), full GPU | Full fine-tune |
| Task-specific, limited data, need multiple tasks | LoRA (standard) |
| Very limited GPU memory | QLoRA (LoRA + 4-bit quantization) |
| Real-time adaptation needed | Prefix tuning |

---

## Interview Q&A

**Q: How does LoRA work? Why do weight updates have low rank?**

A: LoRA adds a low-rank update to frozen weights: W_new = W + BA where B is d×r, A is r×d, r << d. Intuition: task-specific adaptations don't need full rank—just a few directions matter. Like rotating a pre-trained model slightly for a new task, not overhauling it. Empirically: r=8-16 captures 99% of fine-tuning quality. Why low-rank exists: pre-trained models already capture most general patterns; task adaptation refines along few important directions (maybe sentiment for classification, factuality for QA, etc.). Memory savings: instead of storing gradients for d² parameters, store for 2dr ≈ 2d×8 = tiny.

**Q: When would you use LoRA vs. full fine-tuning?**

A: Use LoRA when: (1) deploying to many tasks (each LoRA is small, swap freely), (2) limited GPU memory (LoRA uses 100× less), (3) need fast iteration (LoRA trains in hours vs. days), (4) want to preserve pre-trained knowledge (frozen weights). Use full fine-tuning when: (1) task is very different from pre-training, (2) you have massive labeled data (100K+), (3) accuracy is critical and you have the compute. In practice: LoRA first (cheap, fast), full fine-tune only if LoRA hits accuracy ceiling.

**Q: How do you select the rank r in LoRA?**

A: Start with r=8 (default, good across most tasks). If validation loss plateaus (under-fitting), increase to r=16 or r=32. If it diverges (over-fitting), decrease to r=4. Measure compute-accuracy trade-off: r=8 vs. r=16 is ~2× slower training for ~1-2% accuracy gain. Diminishing returns after r=32. Different tasks have different optimal r: simple tasks (sentiment) fine with r=4-8, complex tasks (machine translation) benefit from r=16-32. A/B test on validation set: train two LoRAs with different r, measure final accuracy.

**Q: Can you merge LoRA into the original model for deployment?**

A: Yes! After training B, A, compute W_final = W_original + BA (single matrix multiply). Then save only W_final (same size as original model). At inference, use W_final directly (no LoRA overhead). This enables: (1) single-file deployment (no separate LoRA artifacts), (2) standard inference code (LoRA-unaware code works), (3) distribution (send one model, not model + adapter). Trade-off: you can't swap adapters anymore (lose multi-task flexibility). Use merging for: final deployment. Use separate LoRA for: experimentation, multi-task systems.

**Q: Can you combine LoRA with quantization (4-bit, 8-bit)?**

A: Yes! QLoRA: freeze a 4-bit quantized model, add LoRA on top. Benefits: (1) 4-bit model uses 1/4 memory, (2) LoRA trains only small adapters (another 1%), total ~5% of full fine-tune memory, (3) accuracy nearly matches full fine-tune. Trade-off: slower inference due to dequantization. Use QLoRA for: extreme memory constraints (fine-tune 70B model on single GPU). Use standard LoRA for: consumer GPUs with 12-16GB VRAM.

**Q: What happens if you train LoRA for too long or with wrong hyperparameters?**

A: (1) **Learning rate too high:** LoRA diverges, loss explodes. Impact: useless model. Fix: use smaller LR (5e-4, not 0.01) than full fine-tune. (2) **Too many epochs:** Over-fit to training data. Impact: high train loss, poor generalization. Fix: early stopping on validation loss. (3) **Rank too small:** Under-fit, validation loss plateaus. Impact: can't reach full fine-tune quality. Fix: increase r. (4) **Rank too large:** Slows down training, marginal improvement. Impact: wasted compute. Fix: reduce r if validation saturates.

---

## Best Practices

- **Rank selection:** Default r=8. Test r=16 if under-fitting. Rarely need r>32. For tiny models (125M), r=4 sufficient.

- **Learning rate:** 5e-4 to 5e-3 typical for LoRA. Higher than standard fine-tune (1e-5) because LoRA has fewer params. Reduce by 10× if diverging.

- **Warmup:** Use learning rate warmup for first 10% of steps (prevents divergence). Schedule: linear warmup, then cosine decay.

- **Batch size:** 8-64 depending on GPU memory. LoRA is memory-efficient, so larger batches possible. Larger batches often better (more stable gradients).

- **Epochs:** 3-5 epochs typical. Use early stopping based on validation loss. Train until validation loss stops improving.

- **LoRA on which layers:** Apply to query, key, value projections in attention (most important). Optional on feed-forward and embeddings. Skip layer norms.

- **Initialization:** Initialize A ∼ N(0, σ), B = 0. Why? At start, LoRA contributes nothing (B=0 means BA=0), so training is stable. Gradients flow through A immediately.

- **Merging:** After training, merge LoRA: W_final = W + BA. Verify accuracy unchanged (should be identical). Save only W_final for deployment.

- **Multi-task:** Train separate LoRA per task (same frozen W). Swap adapters at inference. Or merge all LoRA at once for single model.

---

## Common Pitfalls

- **Mistake: Using full fine-tuning learning rate for LoRA.** Full fine-tuning uses 1e-5 to 5e-5 (very small). LoRA needs 5e-4 to 5e-3 (100× larger) because fewer parameters. Impact: LoRA converges too slowly or not at all.
  → Fix: Use 5e-4 as default for LoRA, not 1e-5.

- **Mistake: Choosing rank too small without testing.** r=1 or r=2 might severely under-fit complex tasks. Impact: can't match full fine-tune.
  → Fix: Always test r=8 minimum. Increase if validation loss plateaus.

- **Mistake: Training LoRA with frozen encoder in embedding layer.** Some implementations train only decoder LoRA, forget encoder embedding. Impact: encoder embeddings can't adapt.
  → Fix: Apply LoRA to query/key/value in all transformer layers, not just decoder.

- **Mistake: Not initializing B = 0.** If B is random, training is unstable (LoRA immediately contributes noise). Impact: divergence.
  → Fix: Initialize A ~ N(0, σ), B = 0 always.

- **Mistake: Merging LoRA incorrectly.** Wrong merge formula: W_final = W * (1 + BA) instead of W + BA. Impact: accuracy drops.
  → Fix: Use W + BA. Verify merged model has same accuracy as LoRA model before merging.

- **Mistake: Thinking LoRA requires no tuning.** While simpler than full fine-tune, LoRA still needs: rank selection, LR tuning, epochs. Impact: suboptimal results.
  → Fix: Treat LoRA as tuning problem: test r ∈ {4,8,16}, LR ∈ {1e-4, 5e-4, 1e-3}, measure validation loss.

---

## Code Examples

### Example 1: LoRA Fine-tuning with HuggingFace + PEFT

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model
model_name = "meta-llama/Llama-2-7b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to apply LoRA
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
print(model.print_trainable_parameters())
# Output: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.06

# Training loop (standard PyTorch)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
model.train()

for epoch in range(3):
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Save LoRA adapter
model.save_pretrained("lora_adapter")

# Load and merge later
from peft import AutoPeftModelForCausalLM

model_with_adapter = AutoPeftModelForCausalLM.from_pretrained("lora_adapter")
merged_model = model_with_adapter.merge_and_unload()
merged_model.save_pretrained("merged_model")
```

### Example 2: Inference with LoRA Adapter

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load model with LoRA adapter
model = AutoPeftModelForCausalLM.from_pretrained(
    "lora_adapter",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Inference
prompt = "Translate English to French: Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))
# Output: "Translate English to French: Hello, how are you? Bonjour, comment allez-vous?"
```

### Example 3: Manual LoRA Implementation

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter"""
    
    def __init__(self, in_features, out_features, r=8, lora_alpha=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features)
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)  # d x r
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))  # r x d
    
    def forward(self, x):
        # x: (..., in_features)
        # Linear output
        out = self.linear(x)  # (..., out_features)
        
        # Add LoRA: (x @ A) @ B, scaled by alpha/r
        lora_out = (x @ self.lora_A) @ self.lora_B * (self.lora_alpha / self.r)
        
        return out + lora_out


# Test LoRA layer
layer = LoRALinear(in_features=768, out_features=3072, r=8)

# Count parameters
total = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print(f"Trainable parameters: {total:,}")  # 768*8 + 8*3072 = ~30K

# Forward pass
x = torch.randn(2, 10, 768)  # batch=2, seq_len=10, d=768
output = layer(x)
print(f"Output shape: {output.shape}")  # (2, 10, 3072)

# Gradient check
loss = output.sum()
loss.backward()
print(f"Gradient on lora_A: {layer.lora_A.grad is not None}")
print(f"Gradient on original weights: {layer.linear.weight.grad is None}")  # Should be None (frozen)
```

---

## Related Concepts

- [Attention Is All You Need](./01-attention-is-all-you-need.md) — Transformer architecture
- [BERT: Pre-training of Deep Bidirectional Transformers](./02-bert.md) — Pre-training paradigm
- [Language Models are Few-Shot Learners (GPT-3)](./03-gpt3.md) — Why fine-tuning large models matters
- [modern-ai/concepts/13-parameter-efficient-fine-tuning](../../../llm/concepts/11-parameter-efficient-finetuning.md) — LoRA variants and alternatives

