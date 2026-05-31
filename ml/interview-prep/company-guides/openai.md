# ML Interview at OpenAI

## What They're Looking For
OpenAI expects the deepest LLM knowledge in the industry. You should be able to discuss recent papers from memory, explain RLHF tradeoffs at implementation depth, and reason about safety and alignment. This is the frontier — interviewers assume you've read the GPT-4, InstructGPT, and Constitutional AI papers. Research curiosity and safety awareness are as important as engineering skill.

## Interview Rounds
| Round | Type | Duration | What's Tested |
|-------|------|----------|--------------|
| Phone Screen | Technical Deep Dive | 60 min | LLM architecture or RLHF depth |
| Onsite 1 | ML Research Discussion | 60 min | Discuss a recent paper, extend it |
| Onsite 2 | ML System Design | 60 min | LLM training or serving infrastructure |
| Onsite 3 | ML Coding | 60 min | Implement attention, RL algorithm, or training component |
| Onsite 4 | Safety & Alignment | 45 min | How to make LLMs safer, evaluate alignment |
| Onsite 5 | Behavioral | 45 min | Mission alignment, collaboration |

## Most Common Question Topics
1. **RLHF deep dive** — Reward model training, PPO mechanics, KL penalty, reward hacking
2. **Scaling laws** — Chinchilla optimal, compute-optimal training, emergent capabilities
3. **Safety and alignment** — Constitutional AI, red-teaming, evaluation of harmful outputs
4. **LLM training infrastructure** — Distributed training at 1000s of GPUs, mixed precision, gradient checkpointing
5. **Inference optimization** — Speculative decoding, KV cache management, continuous batching
6. **Evaluation** — How to evaluate frontier models, LLM-as-judge, human preference data

## Their ML Tech Stack (Known)
- **Framework:** PyTorch (primary), Triton (custom CUDA kernels for performance)
- **Training:** Internal distributed training on Azure (thousands of A100s)
- **Serving:** Internal inference infrastructure, vLLM-influenced designs
- **Safety:** Red-teaming tools, Constitutional AI classifiers, internal eval frameworks

## Sample Questions from This Company

### Q: How would you detect reward hacking before it damages a model in production RLHF training?
**What they're testing:** RLHF implementation depth and safety awareness

**Green flags:** Monitor KL divergence from reference policy (alert at >10 nats), track response length distribution (gaming length = hacking), separate RM evaluation set not used in training, hold out golden human evals run weekly, ensemble reward models to detect disagreement

**Red flags:** "Just look at loss curves" — doesn't catch reward hacking specifically

### Q: Explain the tradeoff between KL penalty strength and reward maximization in PPO fine-tuning.
**What they're testing:** RLHF math and intuition

**Green flags:** Beta too low means reward hacking, reward model exploitation, catastrophic forgetting of language model capability; beta too high means model barely moves from reference policy, reward doesn't improve; optimal beta found empirically, usually 0.1-0.5; adaptive KL control (adjust beta based on KL drift)

**Red flags:** Can't explain the role of KL in the RLHF objective

### Q: How do scaling laws inform your decision to train larger vs train longer?
**What they're testing:** Research intuition and frontier knowledge

**Green flags:** Chinchilla (Hoffmann et al.): compute-optimal means model size proportional to training tokens proportional to sqrt(C). Before Chinchilla, models were undertrained. Now: 7B model needs ~140B tokens (20x params). Training longer on fixed model is often better than going larger. Also: inference cost matters — smaller model at inference times many queries may justify undertrained large model.

**Red flags:** "Bigger is always better" — misses compute-optimal training

### Q: What are the failure modes of Constitutional AI compared to human RLHF?
**What they're testing:** Safety research depth

**Green flags:** CAI uses AI feedback (faster, cheaper, scalable) but inherits biases of the critique model; may miss novel jailbreaks a human rater would catch; harder to audit than human preference data; circular if critique model has same blind spots as target model; human RLHF is ground truth but expensive and subject to annotator fatigue/inconsistency

**Red flags:** "CAI is strictly worse" or "CAI is strictly better" — nuanced tradeoffs expected

### Q: How would you evaluate an o1-style chain-of-thought reasoning model?
**What they're testing:** Evaluation methodology for frontier models

**Green flags:** Task-specific eval (MATH, AIME, competitive programming); process reward model (grade intermediate reasoning steps, not just final answer); rejection sampling quality (how often does the best-of-N solution improve over greedy?); evaluation contamination audit (ensure test problems not in training data)

**Red flags:** "Use MMLU" — generic benchmark doesn't test chain-of-thought specifically

### Q: Design a speculative decoding system. What are the conditions under which it helps?
**What they're testing:** Inference optimization depth

**Green flags:** Draft model generates k tokens, target model verifies in parallel (one forward pass); speedup only when draft model acceptance rate is high (tokens match target); works best on repetitive or predictable text (code, structured output); bottleneck is memory bandwidth not compute for small batch inference; typically 2-3x speedup at batch size 1

**Red flags:** Speculative decoding always helps — acceptance rate dependency is the key nuance

## Key Papers to Know Cold
| Paper | Key Contribution | What to Know |
|-------|-----------------|-------------|
| InstructGPT (2022) | RLHF for instruction following | RM training, PPO objective, KL penalty, evaluation |
| Constitutional AI (2022) | AI feedback instead of human feedback | CAI pipeline, critique-revision, advantages/limits |
| Chinchilla (2022) | Compute-optimal training | 20x tokens per parameter rule, Chinchilla loss |
| FlashAttention (2022) | IO-aware attention | Tile-based computation, memory bandwidth bottleneck |
| GPT-4 Technical Report (2023) | Frontier model capabilities | Evaluation methodology, safety measures |
| LLaMA (2023) | Open-source competitive LLM | Training recipe, tokenizer, architecture choices |

## 3-Week Prep Strategy
**Week 1:** Read foundational papers: InstructGPT (RLHF), Constitutional AI, GPT-4 technical report, Chinchilla. Take notes on methods, not just results.

**Week 2:** Implement RLHF from scratch (toy LM + reward model + PPO). Study speculative decoding, vLLM, FlashAttention papers. Understand inference optimization hierarchy.

**Week 3:** Safety deep dive: red-teaming methodology, alignment evaluation. Practice explaining complex tradeoffs clearly (interviewers will push back to test depth).

## Insider Tips
- OpenAI cares deeply about mission alignment — be genuine about why you want to work on safe, beneficial AI
- Expect to be asked "what's wrong with current approach X?" — they value critical thinking about their own work
- Know the InstructGPT paper cold: reward model training, PPO objective, KL penalty, evaluation methodology
- Safety/alignment is a first-class concern, not an afterthought — integrate it into every system design answer
- Research discussion round is unique to OpenAI: pick a recent paper you genuinely find interesting and have opinions on
