# Agent Prompt Engineering

## Detailed Explanation

Prompt engineering for agents designs prompts that guide agent behavior. Techniques: role definition ("helpful assistant"), context (background info), instruction (explicit steps), examples (few-shot learning), constraints ("only use these tools"), feedback (guide toward correct behavior). Difference from user prompt engineering: agent prompts are long-lived (reused across many queries), stable (changes impact all uses), and behavior-optimized. Challenges: brittleness (small changes cause big behavioral shifts), testing (hard to measure prompt quality), latency (longer prompts = slower). Best for: steering agent toward specific behaviors, constraining action space, teaching via examples.

## Core Intuition

Job description + playbook for your team. Clear instructions = better work. Vague instructions = disappointing results.

## How It Works

System prompt + Task instructions + Context + Examples + Constraints:

1. **System Prompt** — Define role and baseline behavior
2. **Task Instruction** — What to do
3. **Context** — Relevant information
4. **Examples** — Few-shot good examples
5. **Constraints** — What agent cannot do
6. **Evaluation** — How to measure success

## Best Practices

1. Clear role definition
2. Explicit constraints
3. Examples of desired behavior
4. Output format specification
5. Fallback behavior
6. A/B testing changes
7. Version tracking
8. Iterative refinement

## Code Examples

### Example 1: System Prompt

```python
SYSTEM_PROMPT = """
You are a customer support agent. Your role:
1. Answer customer questions about products
2. Help resolve issues
3. Escalate complex issues to humans

Constraints:
- Only recommend in-stock products
- Don't promise delivery dates
- Be polite always

If unsure, ask for clarification.
"""
```

### Example 2: Few-Shot Examples

```python
EXAMPLES = [
    {
        "query": "What payment methods?",
        "response": "We accept credit cards, PayPal, and bank transfers."
    },
    {
        "query": "Product broken!",
        "response": "Sorry! Let me help. Can you describe the issue?"
    }
]
```

### Example 3: Output Constraints

```python
OUTPUT_FORMAT = """
Respond in JSON:
{
    "action": "approve|reject|escalate",
    "reason": "string",
    "confidence": 0-1
}
"""
```

## Related Concepts

- Agent Loops, Tool Use, Safety Alignment, Reflection
