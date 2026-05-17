# RLHF (Reinforcement Learning from Human Feedback)

## TL;DR
Fine-tune LLM using human preferences instead of ground truth. Humans rank outputs (good vs bad), train reward model, use to fine-tune. Enables alignment with human values.

## Core Intuition
Can't define "good response" formally. Ask humans to judge, learn what they prefer, optimize for that.

## How It Works
```
1. SFT: Fine-tune LLM on instruction-response pairs
2. Collect comparisons: humans rank outputs (A > B or B > A)
3. Train reward model: predict human preference
4. PPO: fine-tune LLM using reward model
   - High reward → reinforced
   - Low reward → discouraged
```

**Example:**
```
Prompt: "Write a poem about winter"
Output A: "Snow falls gently..."
Output B: "Cold bites deeply..."

Human judges: prefer A (more poetic)
Reward model: learns A scores higher
PPO: updates LLM to generate more like A
```

## Trade-offs
- Alignment: better follows human intent
- Cost: expensive (human annotation)
- Compute: PPO training is complex

## Common Mistakes / Gotchas
- **Reward hacking:** LLM exploits reward model (e.g., says "good" to everything)
- **Annotator bias:** rater preferences != user preferences
- **Scaling:** 10k comparisons takes weeks of annotation

## Interview Quick-Reference
**RLHF?** Use human preferences to train reward model, then RL to optimize LLM.

## Related Topics
- [Fine-tuning](finetuning.md)
- [DPO](dpo.md) — simpler alternative

## Resources
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
