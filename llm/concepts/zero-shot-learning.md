# Zero-Shot Learning

## TL;DR
No examples. Just instruction: "Classify sentiment: [text]" and model responds. LLM infers task from language alone. Works for simple tasks; weaker than few-shot. Trade: simplicity vs accuracy.

## Core Intuition
"Classify sentiment" tells LLM what to do without examples. Model knows "sentiment" is positive/negative/neutral from training. Works for intuitive tasks; fails on domain-specific or complex ones.

## How It Works

**Zero-Shot Prompt:**
```
Task: Classify sentiment
Text: "This product is amazing!"
Sentiment:
```

**How it Works:**
1. Model reads instruction ("classify sentiment")
2. Understands sentiment task from pre-training knowledge
3. Applies learned patterns to text
4. Outputs: "Positive"

**When it Works:**
- Tasks model has seen during training
- Simple, intuitive tasks (sentiment, language detection, classification)
- Well-known categories

**When it Fails:**
- Domain-specific tasks (medical coding, legal classification)
- Rare categories
- Complex reasoning

### Workflow Flowchart

```mermaid
graph LR
    A["Task Description<br/>No Examples"] -->|Zero-Shot| B["Model Output<br/>Accuracy: 40-70%"]
    A -->|Add 3-5 Examples| C["Few-Shot<br/>Accuracy: 70-90%"]
    A -->|Fine-Tune| D["Task-Specific<br/>Accuracy: 85-95%"]
    E["Cost/Speed"] -->|Fastest| B
    E -->|Medium| C
    E -->|Slowest| D
    style B fill:#ffebee
    style C fill:#fff3e0
    style D fill:#e8f5e9
```

## Key Properties / Trade-offs

| Aspect | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| Speed | Fastest | Slower (longer prompts) |
| Accuracy | Lower | Higher |
| Data needed | None | 3-10 examples |
| Flexibility | Good (new tasks fast) | Better (task-specific) |

## Common Mistakes / Gotchas

- **Assuming coverage:** Model may not know task from description alone. Test first.
- **Vague instructions:** "Analyze this" is too vague. Be specific: "Classify as A, B, or C."
- **Rare categories:** If category is niche (e.g., "Klingon sentiment"), zero-shot fails. Use few-shot.
- **Complex tasks:** Multi-step reasoning needs few-shot or fine-tuning.

## Code Example

```python
from anthropic import Anthropic

client = Anthropic()

# Zero-shot: no examples, just instruction
prompt = """Classify the language of this text: "Bonjour, comment allez-vous?"
Language:"""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=50,
    messages=[{"role": "user", "content": prompt}]
)
print("Zero-shot:", response.content[0].text)  # Output: "French"

# Compare with few-shot (for reference)
prompt_few_shot = """Classify language:
"Hello, how are you?" → English
"Hola, ¿cómo estás?" → Spanish
"Bonjour, comment allez-vous?" → French

"Guten Tag" → 
Language:"""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=50,
    messages=[{"role": "user", "content": prompt_few_shot}]
)
print("Few-shot:", response.content[0].text)  # Also "German" but more confident
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "Zero-shot?" | Task from instruction only, no examples. Fast but lower accuracy. Works for simple, well-known tasks. |
| "vs few-shot?" | Zero-shot: simpler prompt, faster. Few-shot: +10-30% accuracy, longer prompt. Use few-shot if accuracy critical. |
| "When use zero-shot?" | Simple tasks, fast iteration, or when examples hard to get. Otherwise use few-shot. |
| "Failure modes?" | Domain-specific tasks, rare categories, complex reasoning. Add examples if failing. |

## Real-World Examples

### Zero-Shot Language Detection
Model: multilingual. Task: detect language from text. No examples provided. Accuracy: 95% (works because language is distinct). vs few-shot: 98% (marginal gain, examples not needed).

### Zero-Shot Domain Transfer
Task: medical text classification. Model trained on general domain. Zero-shot: 45% accuracy (struggles with medical jargon). Few-shot (3 medical examples): 70%. Full fine-tune: 90%.

### Zero-Shot Generalization Test
Test model on completely unseen task (no fine-tuning, no examples). Success indicates good instruction following. Failure suggests task too specialized or model too narrow.

## Real-World Examples

### Zero-Shot Language Detection
Model: multilingual. Task: detect language from text. No examples provided. Accuracy: 95% (works because language is distinct). vs few-shot: 98% (marginal gain, examples not needed).

### Zero-Shot Domain Transfer
Task: medical text classification. Model trained on general domain. Zero-shot: 45% accuracy (struggles with medical jargon). Few-shot (3 medical examples): 70%. Full fine-tune: 90%.

### Zero-Shot Generalization Test
Test model on completely unseen task (no fine-tuning, no examples). Success indicates good instruction following. Failure suggests task too specialized or model too narrow.

## Real-World Examples

### Zero-Shot Language Detection
Model: multilingual. Task: detect language from text. No examples provided. Accuracy: 95% (works because language is distinct). vs few-shot: 98% (marginal gain, examples not needed).

### Zero-Shot Domain Transfer
Task: medical text classification. Model trained on general domain. Zero-shot: 45% accuracy (struggles with medical jargon). Few-shot (3 medical examples): 70%. Full fine-tune: 90%.

### Zero-Shot Generalization Test
Test model on completely unseen task (no fine-tuning, no examples). Success indicates good instruction following. Failure suggests task too specialized or model too narrow.

## Real-World Examples

### Zero-Shot Language Detection
Model: multilingual. Task: detect language from text. No examples provided. Accuracy: 95% (works because language is distinct). vs few-shot: 98% (marginal gain, examples not needed).

### Zero-Shot Domain Transfer
Task: medical text classification. Model trained on general domain. Zero-shot: 45% accuracy (struggles with medical jargon). Few-shot (3 medical examples): 70%. Full fine-tune: 90%.

### Zero-Shot Generalization Test
Test model on completely unseen task (no fine-tuning, no examples). Success indicates good instruction following. Failure suggests task too specialized or model too narrow.

## Related Topics
- [In-Context Learning](in-context-learning.md) — ICL includes both zero and few-shot
- [Few-Shot Learning](few-shot-learning.md) — adding examples to zero-shot
- [Prompting](prompting.md) — instruction quality matters for zero-shot

## Resources
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## Concept Relationships

```mermaid
graph TD
    A["Zero-Shot Learning"]
    B["Prompting"] -->|prerequisite| A
    A -->|used with| D["In-Context Learning"]
    
    style A fill:#fff3e0
```

## Interview Questions

**Q: What's zero-shot learning and when does it work?**
*A: No examples provided; model uses instructions only. 'Classify sentiment: positive/negative' (no examples). Works: simple, general tasks. Fails: complex reasoning, nuanced tasks. Baseline for few-shot comparison.*

**Q: Why is zero-shot accuracy often low?**
*A: Model never saw examples of task structure. Must infer from language alone. Without grounding, prone to misinterpretation. Few-shot adds grounding (examples show format/expectations).*

**Q: What model capabilities enable zero-shot?**
*A: Instruction following (understands task). Generalization (applies knowledge to new tasks). Reasoning (multi-step logic). Not all models strong at all three. Large models (100B+) better zero-shot.*

**Q: When would you use zero-shot vs few-shot?**
*A: Zero-shot: unknown task, can't afford annotation, exploring. Few-shot: better accuracy acceptable, data available, consistent task. Zero-shot faster (fewer tokens), few-shot more accurate.*

**Q: How do you prompt for effective zero-shot?**
*A: Be specific: 'Classify as positive/negative/neutral' (not just 'analyze'). Set format: 'Output: [label]'. Add context if helpful: 'Customer reviews: classify'. Still less effective than examples.*