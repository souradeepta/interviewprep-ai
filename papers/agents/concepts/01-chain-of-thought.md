---
title: "Chain-of-Thought Prompting Elicits Reasoning in Language Models"
authors: "Wei, Wang, Schuurmans, et al."
year: 2022
venue: "NeurIPS"
arxiv: "https://arxiv.org/abs/2201.11903"
domain: "agents"
difficulty: "intermediate"
interview_frequency: "very_high"
related_concepts:
  - agentic-ai/concepts/XX-reasoning
  - agentic-ai/concepts/XX-prompt-engineering
---

# Chain-of-Thought Prompting Elicits Reasoning in Language Models

## Paper Overview

**Title:** Chain-of-Thought Prompting Elicits Reasoning in Language Models

**Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, et al. (Google Brain / DeepMind)

**Published:** NeurIPS 2022 | [arXiv](https://arxiv.org/abs/2201.11903)

**Citation:** 10,000+ (foundational paper for reasoning in LLMs)

Before Chain-of-Thought (CoT), large language models struggled with reasoning-heavy tasks: math problems, logical deduction, multi-step planning. A 540M parameter model would fail on problems that required multiple steps, even when the same model could handle each individual step. The breakthrough insight: **adding "Let's think step by step" before asking for the answer dramatically improves accuracy**. This paper showed that asking models to produce intermediate reasoning—verbalizing their thought process—unlocks capabilities that were previously hidden. CoT is now foundational to modern AI systems and appears in virtually every production agent.

**Why this matters for interviews:** CoT is one of the most frequently asked topics in LLM/agent interviews. Interviewers ask about it constantly: "Why does step-by-step help?", "When does CoT work?", "Trade-offs with inference cost?", "How to optimize prompts?". You need to understand not just that CoT works, but *why*—and how to apply it effectively.

---

## Core Contribution

### The Problem: Models Struggle With Reasoning

In early 2022, GPT-3 and other large language models showed impressive few-shot learning on many tasks. But on reasoning-heavy problems, they failed:

- **Math word problems (SVAMP, MAWPS):** Accuracy 17.7% without reasoning, but the model *could* do arithmetic
- **Logical reasoning (CRWQ):** Models failed to trace through deductions step-by-step
- **Multi-hop reasoning (HotpotQA):** Models could find relevant facts but struggled to combine them

The hypothesis: models *have* the capability to reason, but generating the final answer directly (without intermediate steps) causes them to lose track of the reasoning process.

### The Solution: Elicit Intermediate Reasoning

Instead of asking directly for the answer, ask the model to "think step by step":

**Prompt without CoT:**
```
Question: If there are 3 apples and Mary eats 1, how many remain?
Answer:
```
→ Model might say "2" or fail entirely

**Prompt with CoT:**
```
Question: If there are 3 apples and Mary eats 1, how many remain?
Let's think step by step.
Step 1: Start with 3 apples.
Step 2: Mary eats 1, so we subtract 1.
Step 3: 3 - 1 = 2.
Answer: 2
```
→ Model generates intermediate reasoning, arrives at correct answer

### Key Innovation: Zero-Shot CoT

Even more surprising: **you don't need examples**. Just adding "Let's think step by step" (zero-shot) works nearly as well as few-shot examples:

- **Zero-shot CoT:** "Question: ... Let's think step by step." → 50-80% improvement
- **Few-shot CoT:** Provide examples with reasoning → 60-90% improvement

This is powerful because you don't need to construct example chains—the model figures it out from the instruction alone.

---

## Key Ideas & Algorithm

### How Chain-of-Thought Works

**Step 1: Invoke Reasoning Mode**
- Add instruction: "Let's think step by step."
- Model switches from direct-answer mode to reasoning mode

**Step 2: Generate Intermediate Thoughts**
- Model produces step-by-step reasoning
- Each step builds on previous reasoning
- Model maintains internal consistency

**Step 3: Arrive at Final Answer**
- After reasoning chain, model produces final answer
- Answer is typically more accurate than without reasoning

**Step 4: Why This Works (Mechanistic Explanation)**
- **Extended compute:** More tokens = more opportunities for model to find right answer
- **Process transparency:** Model can backtrack if reasoning is wrong
- **Activation patterns:** Intermediate tokens activate correct reasoning circuits
- **Gradient alignment:** Model learns to associate reasoning chains with correct answers

### Variants of Chain-of-Thought

| Variant | Prompt Template | Use Case |
|---------|-----------------|----------|
| **Zero-shot CoT** | "Let's think step by step." | Generic reasoning, no examples available |
| **Few-shot CoT** | Provide 1-5 examples with reasoning | Domain-specific, need consistent style |
| **Self-consistency** | Generate multiple CoT chains, take majority vote | When uncertainty is high, compute budget available |
| **Least-to-most prompting** | Decompose into simpler subproblems first | Very complex problems requiring many steps |
| **Plan-and-solve** | "Let's break this down: 1) Plan, 2) Solve" | Problems needing explicit planning |

### Scaling Laws of CoT

**Empirical finding:** CoT effectiveness *increases* with model size.

| Model Size | SVAMP Accuracy (Direct) | SVAMP Accuracy (CoT) | Improvement |
|------------|------------------------|----------------------|-------------|
| 8B parameters | 10% | 25% | +15% |
| 62B parameters | 17.7% | 40% | +22% |
| 540B parameters (PaLM) | 58% | 79% | +21% |

**Interpretation:** Larger models are better at reasoning *and* benefit more from being asked to reason explicitly. This is why CoT became standard practice: it unlocks latent reasoning abilities in large models.

---

## Architecture & Trade-offs

### Explicit vs. Implicit Reasoning

| Aspect | Explicit CoT | Implicit (Fine-tuning) |
|--------|-------------|----------------------|
| **How** | Prompt the model to reason | Train model with reasoning data |
| **Cost** | Low (just prompt, but longer generation) | High (fine-tuning data + compute) |
| **Transparency** | Human can read reasoning | Model learned it, hard to inspect |
| **Generalization** | Works across tasks with same prompt | Task-specific, may not generalize |
| **When to use** | Quick iteration, zero-shot tasks | Task is fixed, need latency < reasoning time |

### Zero-Shot vs. Few-Shot CoT

**Zero-shot:** Just say "Let's think step by step"

**Pros:**
- No need to construct examples
- Works for novel domains
- Generic, task-agnostic

**Cons:**
- Lower accuracy than few-shot
- Model may use wrong reasoning style
- May miss domain conventions

**Few-shot:** Provide 1-5 examples with reasoning

**Pros:**
- Higher accuracy (60-90%)
- Models reasoning style from examples
- Better for critical tasks

**Cons:**
- Need high-quality examples
- Takes more prompt tokens
- Harder to scale across many tasks

**Trade-off:** Use zero-shot for exploration; use few-shot for production if examples are available.

### Computational Cost

**Chain-of-Thought increases inference cost:**

- **Direct answer:** "Question: ... Answer:" → ~5-20 tokens
- **With CoT:** "Question: ... Let's think step by step. [reasoning] Answer:" → ~50-500 tokens

**Cost multiplier:** 5-50× longer generation

**When this matters:**
- Real-time systems (latency critical): Avoid CoT or use shorter chains
- High-throughput APIs (cost sensitive): Cache reasoning if possible
- One-time queries: CoT is fine

**Optimization:**
- Use shorter reasoning when possible ("Briefly explain reasoning:")
- Cache reasoning for similar problems
- Use self-consistency only when high accuracy is critical
- Fall back to direct answer for simple problems

### Self-Consistency: Averaging Multiple Chains

**Problem:** CoT still fails sometimes (wrong reasoning path)

**Solution:** Generate *multiple* reasoning chains, take majority vote

```
Generate 5 different CoT chains:
Chain 1: ... → Answer A
Chain 2: ... → Answer A
Chain 3: ... → Answer B
Chain 4: ... → Answer A
Chain 5: ... → Answer C

Vote: Answer A (3/5) wins → more confident
```

**Improvement:** +5-15% accuracy over single chain

**Cost:** 5-10× longer (generate multiple chains)

**When to use:** High-stakes decisions, when accuracy > latency

---

## Interview Q&A

**Q: Why does step-by-step reasoning improve LLM performance? What's the mechanism?**

A: There are multiple theories. First, extended reasoning gives the model more opportunities to correct mistakes—more tokens means more computation. Second, generating intermediate steps forces the model to *articulate* reasoning in a way that activates the right neural circuits. Third, it prevents the model from jumping to conclusions. For very complex problems, the model might not have "seen" a direct path from input to output during training, so it falls back to simpler step-by-step patterns it learned from reasoning data.

**Q: When does Chain-of-Thought NOT work? What are the failure cases?**

A: CoT works best on reasoning tasks (math, logic, multi-hop QA) and fails or adds little on tasks that are primarily about *knowledge* or *pattern matching*. For example: "What's the capital of France?" doesn't benefit from CoT—the model either knows it or doesn't. CoT can actually hurt if the intermediate reasoning contradicts the correct answer or if the model generates plausible-sounding but wrong reasoning.

**Q: What's the trade-off between zero-shot and few-shot CoT?**

A: Zero-shot CoT works surprisingly well and requires no examples, making it quick to deploy. Few-shot CoT (1-5 examples) usually achieves 5-15% higher accuracy because the model can imitate the reasoning *style* in your examples. For production, if you have high-quality examples and the task is stable, use few-shot. If you're iterating or need to handle novel domains, start with zero-shot.

**Q: How do you optimize CoT prompts for production?**

A: Start with "Let's think step by step"—it's surprisingly effective. If accuracy is still low, add a few (1-3) high-quality examples showing the desired reasoning style. Use self-consistency (generate multiple chains) only if you have compute budget and need very high confidence. Monitor latency: if CoT chains are too long, use shorter prompts like "Briefly reason:" or break the problem into smaller pieces that don't need reasoning.

**Q: Does CoT work equally well for all model sizes?**

A: No—larger models benefit *more*. A 540B model might improve 20% with CoT, while an 8B model might improve only 5%. This is because larger models have stronger reasoning capabilities to elicit. For very small models (<1B), CoT may not help much. There's also a threshold effect: models seem to need to be at least 60B+ parameters for CoT to consistently help on hard reasoning tasks.

**Q: How does CoT compare to fine-tuning the model on reasoning examples?**

A: Fine-tuning is more expensive (data collection, training time, GPU cost) but can be cheaper at inference time if you don't need CoT. CoT works immediately with zero data collection. For production systems, you often combine both: use CoT prompting for out-of-distribution problems, but fine-tune on your specific reasoning patterns for frequently used tasks.

**Q: In production, how do you handle the cost of longer CoT chains?**

A: Several strategies: (1) Only use CoT for hard problems—classify problem difficulty first, use direct answer for easy ones. (2) Use shorter reasoning ("Briefly:") to save tokens. (3) Cache reasoning for similar problems. (4) Use streaming to show reasoning to users (longer response feels fast). (5) For very cost-sensitive systems, fine-tune on reasoning examples instead of prompting with CoT.

---

## Best Practices

- **Start with zero-shot:** Just add "Let's think step by step" as a test. It's remarkably effective and costs nothing to try.

- **Few-shot examples should show correct reasoning:** If you provide examples, ensure the reasoning is actually correct—models will imitate bad reasoning too.

- **Format matters:** Structure reasoning consistently. "Step 1: ... Step 2: ... Answer:" works better than free-form reasoning.

- **Use for complex tasks only:** Don't add CoT to simple tasks like "What color is X?" that don't need reasoning. It wastes tokens and may confuse the model.

- **Monitor latency:** CoT typically adds 2-10s to inference time (longer generation). In real-time systems, measure actual impact on user experience.

- **Combine with self-consistency when needed:** If accuracy is critical and you have compute budget, generate 5-10 chains and vote. But this is expensive—use sparingly.

- **Break down hard problems:** Instead of one long CoT chain, decompose into simpler subproblems with shorter reasoning chains.

- **Validate reasoning quality:** Sample outputs and check if reasoning is actually correct. Models can generate plausible-sounding wrong reasoning.

---

## Common Pitfalls

- **Mistake: Using CoT for knowledge-based questions.** Questions like "Who wrote 1984?" don't benefit from reasoning—the model either knows or doesn't. CoT may generate plausible but wrong chains of "reasoning" that sound good but are incorrect.

- **Mistake: Long, unstructured reasoning.** Free-form reasoning is harder to parse. Use structured formats like "Step 1: ... Step 2: ..." for better results.

- **Mistake: Expecting CoT to fix completely broken models.** CoT helps reasoning, but if a model fundamentally lacks knowledge, no amount of reasoning helps. Use the largest model available; CoT works better on large models.

- **Mistake: Not validating the intermediate reasoning.** Models can generate plausible-sounding chains that lead to wrong answers. Always check that reasoning is actually correct.

- **Mistake: Overusing self-consistency.** Generating 10 chains for every query is expensive. Reserve self-consistency for high-stakes decisions or use adaptive strategies (generate chains until confidence is high).

---

## Code Examples

### Example 1: Basic Chain-of-Thought with OpenAI API

```python
import openai

def chain_of_thought_prompt(question: str, model: str = "gpt-4") -> str:
    """Invoke chain-of-thought reasoning without examples."""
    prompt = f"""Question: {question}

Let's think step by step."""
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# Example: Math problem
question = "Sarah has 5 apples. She buys 3 more and gives 2 to her friend. How many does she have?"
result = chain_of_thought_prompt(question)
print(result)
# Output: "Let's think step by step.
# Step 1: Sarah starts with 5 apples.
# Step 2: She buys 3 more, so 5 + 3 = 8 apples.
# Step 3: She gives 2 to her friend, so 8 - 2 = 6 apples.
# Answer: Sarah has 6 apples."
```

### Example 2: Few-Shot Chain-of-Thought

```python
def few_shot_cot(question: str, model: str = "gpt-4") -> str:
    """Provide examples to guide reasoning style."""
    examples = """
Example 1:
Question: If John has 10 dollars and spends 3, how much is left?
Let's think step by step.
Step 1: John starts with $10.
Step 2: He spends $3.
Step 3: $10 - $3 = $7.
Answer: John has $7 left.

Example 2:
Question: A store has 20 items. They sell 5 in the morning and 3 in the afternoon. How many are left?
Let's think step by step.
Step 1: The store starts with 20 items.
Step 2: They sell 5 in the morning, so 20 - 5 = 15 items remain.
Step 3: They sell 3 in the afternoon, so 15 - 3 = 12 items remain.
Answer: 12 items are left.

"""
    
    prompt = examples + f"""
Question: {question}
Let's think step by step."""
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']
```

### Example 3: Self-Consistency with Voting

```python
def self_consistent_cot(question: str, num_chains: int = 5, model: str = "gpt-4") -> dict:
    """Generate multiple reasoning chains and vote on answer."""
    chains = []
    answers = []
    
    for i in range(num_chains):
        prompt = f"""Question: {question}

Let's think step by step."""
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,  # Higher temp for diversity
            max_tokens=500
        )
        
        chain = response['choices'][0]['message']['content']
        chains.append(chain)
        
        # Extract final answer (simple heuristic: last line often contains answer)
        last_line = chain.strip().split('\n')[-1]
        answers.append(last_line)
    
    # Vote on most common answer
    from collections import Counter
    answer_counts = Counter(answers)
    best_answer = answer_counts.most_common(1)[0][0]
    confidence = answer_counts.most_common(1)[0][1] / num_chains
    
    return {
        "chains": chains,
        "answer": best_answer,
        "confidence": confidence,
        "num_unique_answers": len(answer_counts)
    }

# Example usage
result = self_consistent_cot("What's 47 + 38?", num_chains=5)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Unique answers seen: {result['num_unique_answers']}")
```

---

## Related Concepts

- [ReAct: Synergizing Reasoning and Acting](./02-react.md) — Extend CoT with tool use (reasoning + actions)
- [Tree of Thoughts](./03-tree-of-thoughts.md) — Explore multiple reasoning paths, not just one linear chain
- [agentic-ai/concepts/XX-prompt-engineering](../../../agentic-ai/concepts/35-agent-prompt-engineering.md) — Broader context for prompt design
- [agentic-ai/concepts/XX-few-shot-learning](../../../llm/concepts/13-few-shot-learning.md) — How few-shot examples work

---

**Last Updated:** 2026-05-31
