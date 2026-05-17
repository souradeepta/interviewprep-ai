# Prompt Optimization

## TL;DR
Improve LLM outputs by refining prompts: clarify instructions, add examples, structure format. Iterative process: test → measure → refine.

## Core Intuition
Prompts matter hugely. "Classify" vs "Classify accurately into A/B/C" yield different results. Optimize iteratively.

## How It Works
```
Initial prompt: "Classify sentiment"
Measure accuracy: 70%

Refined: "Classify sentiment (positive/negative/neutral). Be careful with sarcasm."
Measure: 75%

Further: Add examples of sarcasm
Measure: 82%

Further: Add output format constraint
Measure: 85%
```

**Techniques:**
- Clarity: be specific
- Structure: use formats (JSON, XML)
- Examples: few-shot
- Constraints: "output only label"
- Tone: "be helpful", "think step-by-step"

## Interview Quick-Reference
**Prompt optimization?** Iteratively improve: clarify, add examples, structure, constrain output.

## Related Topics
- [Prompting](prompting.md)
- [Chain-of-Thought](chain-of-thought.md)
- [In-Context Learning](in-context-learning.md)

## Resources
- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
