# DPO (Direct Preference Optimization)

## TL;DR
Simpler alternative to RLHF: directly optimize for human preferences without training separate reward model. Fewer hyperparameters, faster training, comparable results.

## Core Intuition
RLHF is complex: train reward model, then RL. DPO directly uses preferences to fine-tune.

## How It Works
```
Traditional:
  Collect comparisons → Train reward model → PPO fine-tune

DPO:
  Collect comparisons → Directly optimize LLM preference objective
```

**Loss:**
```
Optimize: log(P(preferred) / P(dispreferred))
Simpler, direct, no separate reward model needed
```

## Trade-offs
- Simpler: fewer moving parts
- Faster: no reward model training
- Comparable accuracy to RLHF
- Less tested at scale

## Interview Quick-Reference
**DPO?** Direct preference optimization. Simpler than RLHF, no separate reward model.

## Related Topics
- [RLHF](rlhf.md)
- [Fine-tuning](finetuning.md)

## Resources
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
