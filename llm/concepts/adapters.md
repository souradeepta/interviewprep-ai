# Adapters

## TL;DR
Small, task-specific modules inserted into pre-trained models. Update only adapters, freeze base. Similar to LoRA but different architecture (add FF layers vs low-rank).

## Core Intuition
Like LoRA: train small modules, not full model. Different structure: bottleneck layers instead of low-rank.

## How It Works
**Architecture:**
```
Hidden state → Adapter (down-project) 
           → ReLU 
           → Adapter (up-project) 
           → output
```

Similar parameters to LoRA, slightly different performance profile.

## Trade-offs
- vs Full FT: 99% parameter reduction
- vs LoRA: different architecture, similar efficiency
- Flexibility: can stack adapters (one per task)

## Interview Quick-Reference
**Adapters?** Task-specific modules, freeze base. Like LoRA, different architecture.

## Related Topics
- [LoRA](lora.md)
- [Parameter-Efficient Fine-tuning](parameter-efficient-finetuning.md)

## Resources
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
