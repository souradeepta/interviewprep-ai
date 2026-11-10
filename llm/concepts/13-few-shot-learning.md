# Few-Shot Learning

Humans learn fast from examples: show one translation, they get the pattern.

## Prerequisites

Familiarity with prompting. This note focuses on the LLM-specific mechanism and its engineering selection boundary.

## Learning Outcomes

- Explain the mechanism and assumptions behind few-shot learning.
- Compare its main alternative using task quality and one operational cost.
- Diagnose a failure using a metric or example tied to few-shot learning.

## Core Intuition

Humans learn fast from examples: show one translation, they get the pattern. Few-shot leverages LLMs' ability to recognize patterns from in-context examples. No weight updates needed.

## How It Works

**Structure:**
```
Task instruction (optional but helps)
Example 1: Input → Output
Example 2: Input → Output
Example 3: Input → Output
New input → [Model generates]
```

**Example: Sentiment Classification**
```
Classify sentiment (positive/negative):

Text: "I loved this movie!" → Sentiment: positive
Text: "Terrible experience." → Sentiment: negative
Text: "It was okay." → Sentiment: neutral

Text: "Amazing product, highly recommend!"
Sentiment:
```

**Effective Strategies:**
- **Diversity:** Mix different examples (long/short, obvious/subtle)
- **Order:** Put harder examples later (model performs better with warm-up)
- **Similarity:** Examples similar to test case → better few-shot performance
- **Explanations:** Include reasoning in examples (especially for complex tasks)

### Workflow Flowchart

```mermaid
graph TD
    A["Zero-Shot<br/>No Examples"] -->|Low Accuracy| B["30-50%"]
    C["Few-Shot<br/>3-5 Examples"] -->|Good Accuracy| D["70-80%"]
    E["Fine-Tune<br/>1000s Examples"] -->|High Accuracy| F["90%+"]
    B -.->|Cost: None| G["Cost"]
    D -.->|Cost: Minimal| G
    F -.->|Cost: High| G

    style A fill:#ffebee
    style C fill:#fff3e0
    style E fill:#e8f5e9
```

## Trade-offs

| Shots | Examples | Latency | Accuracy | Best For |
|-------|----------|---------|----------|----------|
| Zero | 0 | Fast | Lower | Simple tasks, general |
| One | 1 | Fast | Medium | Basic patterns |
| Few | 3-5 | Medium | High | Most use cases |
| Many | 10+ | Slow | Marginal gain | Complex, nuanced |

**Diminishing returns:** Typically 3-5 examples sufficient; 20+ shows little improvement.

## Implementation

```python
from anthropic import Anthropic

client = Anthropic()

# Few-shot prompt
prompt = """Classify sentiment (positive, negative, neutral):

Examples:
"I love this product!" → positive
"Waste of money" → negative
"It works fine" → neutral

Now classify: "This is amazing!"
Sentiment:"""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=50,
    messages=[{"role": "user", "content": prompt}]
)
print(response.content[0].text)
```

## Failure Modes

- **Bad example selection:** Unrepresentative or wrong examples confuse model. Choose carefully.
- **Too many examples:** Exceeds context window, dilutes signal. Sweet spot: 3-10.
- **No instructions:** Only examples, no task description. Add instruction for clarity.
- **Inconsistent formats:** If examples vary in format, model confused. Be consistent.

## Interview Quick Reference

| Question | What to say |
|---|---|
| "Few-shot?" | Show examples in prompt to teach task. 3-5 examples typical. Fast, no fine-tuning. |
| "vs fine-tuning?" | Few-shot: instant, flexible. Fine-tuning: better accuracy, requires data/compute. |
| "How many examples?" | 3-5 for most tasks. More helps complex tasks; diminishing returns >10. |
| "Bad performance?" | Check example quality and diversity. Adjust format, add instructions. |



**Design follow-up:** How would you monitor this choice in production and decide when to use an alternative?
## Related Concepts

- [In-Context Learning](15-in-context-learning.md) — broader ICL concept
- [Zero-Shot Learning](14-zero-shot-learning.md) — no examples, just instructions
- [Chain-of-Thought](16-chain-of-thought.md) — combine CoT with few-shot for better reasoning

## Real-World Examples

### Few-Shot for Customer Intent Classification
Chatbot: classify support tickets. Examples: 'Refund request' → returns, 'Can't login' → technical, 'Feedback' → general. Zero-shot: 45% accuracy. Few-shot (3 examples): 72% accuracy. Deployed in Zendesk integration.

### Few-Shot Semantic Matching
Task: match product descriptions to categories. Descriptions vary (informal, typos, abbreviations). Few-shot with diverse examples: 88% accuracy. Cost: <$0.01 per classification vs. $0.50 with fine-tuning.

### Multi-Language Few-Shot
Translate task instructions to 10 languages. Few-shot examples in each language. Model generalizes zero-shot to other languages through few-shot anchoring. Accuracy: 75% (vs 40% direct translation).

## Real-World Examples

### Few-Shot for Customer Intent Classification
Chatbot: classify support tickets. Examples: 'Refund request' → returns, 'Can't login' → technical, 'Feedback' → general. Zero-shot: 45% accuracy. Few-shot (3 examples): 72% accuracy. Deployed in Zendesk integration.

### Few-Shot Semantic Matching
Task: match product descriptions to categories. Descriptions vary (informal, typos, abbreviations). Few-shot with diverse examples: 88% accuracy. Cost: <$0.01 per classification vs. $0.50 with fine-tuning.

### Multi-Language Few-Shot
Translate task instructions to 10 languages. Few-shot examples in each language. Model generalizes zero-shot to other languages through few-shot anchoring. Accuracy: 75% (vs 40% direct translation).

## Real-World Examples

### Few-Shot for Customer Intent Classification
Chatbot: classify support tickets. Examples: 'Refund request' → returns, 'Can't login' → technical, 'Feedback' → general. Zero-shot: 45% accuracy. Few-shot (3 examples): 72% accuracy. Deployed in Zendesk integration.

### Few-Shot Semantic Matching
Task: match product descriptions to categories. Descriptions vary (informal, typos, abbreviations). Few-shot with diverse examples: 88% accuracy. Cost: <$0.01 per classification vs. $0.50 with fine-tuning.

### Multi-Language Few-Shot
Translate task instructions to 10 languages. Few-shot examples in each language. Model generalizes zero-shot to other languages through few-shot anchoring. Accuracy: 75% (vs 40% direct translation).

## Real-World Examples

### Few-Shot for Customer Intent Classification
Chatbot: classify support tickets. Examples: 'Refund request' → returns, 'Can't login' → technical, 'Feedback' → general. Zero-shot: 45% accuracy. Few-shot (3 examples): 72% accuracy. Deployed in Zendesk integration.

### Few-Shot Semantic Matching
Task: match product descriptions to categories. Descriptions vary (informal, typos, abbreviations). Few-shot with diverse examples: 88% accuracy. Cost: <$0.01 per classification vs. $0.50 with fine-tuning.

### Multi-Language Few-Shot
Translate task instructions to 10 languages. Few-shot examples in each language. Model generalizes zero-shot to other languages through few-shot anchoring. Accuracy: 75% (vs 40% direct translation).

## Real-World Examples

### Few-Shot for Customer Intent Classification
Chatbot: classify support tickets. Examples: 'Refund request' → returns, 'Can't login' → technical, 'Feedback' → general. Zero-shot: 45% accuracy. Few-shot (3 examples): 72% accuracy. Deployed in Zendesk integration.

### Few-Shot Semantic Matching
Task: match product descriptions to categories. Descriptions vary (informal, typos, abbreviations). Few-shot with diverse examples: 88% accuracy. Cost: <$0.01 per classification vs. $0.50 with fine-tuning.

### Multi-Language Few-Shot
Translate task instructions to 10 languages. Few-shot examples in each language. Model generalizes zero-shot to other languages through few-shot anchoring. Accuracy: 75% (vs 40% direct translation).

## Topic-specific notes

**Q: How many few-shot examples are optimal and how do you decide?**
A: More examples generally improve performance up to a point (diminishing returns after 8-16 examples for most tasks). But each example consumes context window tokens and adds latency. Test with 1, 3, 5, 8 examples on your eval set. For LLMs >10B, 3-5 examples typically captures most of the benefit. For complex tasks requiring diverse coverage, 8-16 may be better. When context is limited (e.g., long user inputs), prioritize quality over quantity—2 excellent examples > 8 mediocre ones.

**Q: How do you select which examples to include in few-shot prompts?**
A: Strategies in increasing sophistication: (1) random selection from a pool of curated examples; (2) diverse selection—cover different sub-types of the task; (3) similarity-based—retrieve examples most similar to the query using embedding search (dynamic few-shot); (4) hard example selection—prefer examples similar to cases the model struggles on. Dynamic few-shot (embedding retrieval) typically improves accuracy 5-15% over static examples for tasks with diverse inputs.

**Q: What is the difference between few-shot prompting and in-context learning?**
A: Few-shot prompting is a specific technique: provide examples in the prompt. In-context learning is the broader phenomenon: LLMs can learn new patterns from information in the context window without gradient updates. ICL includes few-shot examples but also single demonstrations, analogy-based reasoning, and task descriptions. The mechanism is debated—models may do implicit gradient-based learning in the forward pass, or they may use examples to identify the task distribution.

**Q: Why do example ordering and formatting matter in few-shot prompts?**
A: Recency bias: the model pays more attention to the last few examples before the query. Distributional skew: if early examples are all one class, the model may bias toward it. Consistent formatting across examples helps the model identify the pattern. Always ensure the final few examples before the query reflect the diversity you want in outputs. Randomizing example order and testing on multiple orderings gives a more robust evaluation.

**Q: When does few-shot prompting fail and what should you do instead?**
A: Fails when: the task requires knowledge the model doesn't have (no amount of examples helps); the format is so unusual that examples confuse rather than guide; or examples cover only a narrow slice of the actual input distribution. In these cases: fine-tune on domain data, use retrieval-augmented generation to provide factual context, or decompose the task into simpler subtasks that the model can handle individually.

**Q: How do you measure if your few-shot examples are actually helping?**
A: Compare: zero-shot performance vs. your few-shot configuration on a held-out eval set. If few-shot doesn't beat zero-shot by >5%, your examples may not be well-chosen. Run ablations: remove one example at a time and see which ones contribute most. Use leave-one-out cross-validation across your example pool. Track performance vs. number of examples to find the sweet spot for your context budget.

## Resources

- [In-Context Learning in Large Language Models](https://arxiv.org/abs/2301.00234)
- [What In-Context Learning "Learns"](https://arxiv.org/abs/2310.00867)

## Concept Relationships

```mermaid
graph TD
    A["Few-Shot Learning"]
    B["In-Context Learning"] -->|prerequisite| A
    A -->|used with| D["Chain-of-Thought"]
    
    style A fill:#fff3e0
```

## Interview Questions

**Q: What's few-shot learning and how does it differ from zero-shot?**
*A: Zero-shot: 'Classify sentiment: positive/negative'. Few-shot: '"Good product" → positive. "Bad product" → negative. "Good quality" → positive. Now classify: "Great service"'. Few-shot dramatically improves accuracy (sometimes 30-50% gain).*

**Q: How many examples do you need for effective few-shot?**
*A: Task-dependent: simple classification = 1-2 examples sufficient. Complex reasoning = 5-10 needed. Diminishing returns beyond 10 (20+ shows minimal improvement). Rule of thumb: start with 3, increase if accuracy low. Quality > quantity (good examples matter more than many).*

**Q: What makes a good few-shot example?**
*A: Diverse: cover range of inputs (easy + hard cases). Representative: similar to test distribution. Explained: include reasoning if helpful. Consistent: same format for all. Bad: unrepresentative or noisy examples confuse model.*

**Q: How do you select few-shot examples programmatically?**
*A: Random: simple, sometimes okay. Similarity-based: choose examples most similar to test input (use embeddings). Uncertainty sampling: examples model uncertain on. Diversity-based: examples covering feature space. Best: combination of similarity + diversity.*

**Q: When is few-shot insufficient and you need fine-tuning?**
*A: Few-shot works: general tasks, simple patterns, prompt-able behaviors. Fails: task requires significant internal model change, distribution shift, style transfer. Example: sentiment classification (few-shot fine) vs. writing style adaptation (needs fine-tuning).*
