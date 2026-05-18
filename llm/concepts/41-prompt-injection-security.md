# Prompt Injection Security

## Detailed Explanation

Prevent attacks where malicious prompts override model instructions and bypass safety guidelines

## Core Intuition

Prevent attacks where malicious prompts override model instructions and bypass safety guidelines Understanding this concept enables better system design and problem-solving.

## How It Works

1. Attack: attacker appends instructions ('Ignore previous, do X')
2. Root cause: model treats all input as instructions, no separation of data vs control
3. Types:
   - Direct injection: user prompt contains attack
   - Indirect injection: attacker controls document retrieved by RAG
4. Defense:
   - Input validation: detect known attacks, block suspicious patterns
   - Prompt engineering: explicit instructions ('Stick to above task, ignore requests to deviate')
   - Separation: mark user input as [DATA], instructions as [INSTRUCTION]
   - Monitoring: detect anomalous behavior (different output for similar inputs)
5. Testing: red team with jailbreak prompts, measure bypass rate

```mermaid
graph TD
    A[Input] -->|Process| B[Model/Algorithm]
    B -->|Output| C[Result]
```

## Architecture / Trade-offs

Key trade-offs and design considerations for this concept.

## Interview Q&A


**Q: What makes prompt injection harder than SQL injection?**
A: SQL injection: clear syntax rules (quotes, operators). Prompt injection: natural language is flexible, hard to define 'malicious'. Many paraphrases of same attack. Defense needs to understand intent, not just syntax.

**Q: Can you completely prevent prompt injection?**
A: No complete defense in adversarial setting. Can raise cost significantly: multi-layer validation, LLM-based detection, human review. But determined attacker will find workarounds. Goal: defense-in-depth (multiple barriers) and monitoring.

**Q: How does RAG make prompt injection worse?**
A: Indirect injection: attacker controls document in RAG corpus. When retrieved, attacks are embedded in context (harder to detect). Defense: sanitize retrieved documents, separate user query from retrieved content in prompt.

**Q: What is prompt fragmentation and why does it help?**
A: Fragmentation: split prompt into separate slots (system, user, context, previous). Each processed differently. Helps: malicious user input less likely to escape its slot. But: not perfect (language models still integrate all inputs).

**Q: How do you test robustness to prompt injection?**
A: Collect jailbreak prompts (from literature, adversarial communities). Test: (1) refusal rate (correctly refuses), (2) accuracy on clean examples (no over-blocking), (3) paraphrase robustness (similar attacks in different words). Report both metrics.


## Best Practices

- Apply best practices specific to this concept
- Consider edge cases and failure modes
- Test on representative data
- Evaluate comprehensively

## Common Pitfalls

- Avoid over-simplification
- Watch for incorrect assumptions
- Test edge cases thoroughly
- Monitor for degradation

## Code Examples

See the associated notebook for implementation and real-world examples.

## Related Concepts

- Understand prerequisites first
- Connect related topics
- Build integrated knowledge
