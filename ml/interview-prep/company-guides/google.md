# ML Interview at Google

## What They're Looking For
Google values ML theory depth above all — expect to derive algorithms on a whiteboard. Research mindset is rewarded: knowing recent papers, thinking about scaling, connecting theory to practice. Systems must work at Google scale (billions of queries/day). Googleyness: collaborative, data-driven, intellectually curious. More academic culture than Meta or Amazon.

## Interview Rounds
| Round | Type | Duration | What's Tested |
|-------|------|----------|--------------|
| Phone Screen | ML Theory + Coding | 45 min | Theory + LC medium |
| Onsite 1 | ML Coding | 60 min | Implement models, numpy manipulations |
| Onsite 2 | ML System Design | 60 min | Large-scale ML infrastructure |
| Onsite 3 | ML Theory | 60 min | Deep math: optimization, statistics, Bayesian |
| Onsite 4 | General Coding | 60 min | Standard LC medium-hard |
| Onsite 5 | Googleyness/Leadership | 45 min | Behavioral |

## Most Common Question Topics
1. **Optimization algorithms** — Derive SGD, Adam, second-order methods. Know the math.
2. **Transformer architecture** — Attention mechanism derivation, positional encoding, scaling
3. **Distributed training** — Data parallelism vs model parallelism, AllReduce, gradient compression
4. **Recommendation systems** — YouTube-style two-tower, collaborative filtering at scale
5. **Evaluation** — AUC, NDCG, offline vs online metrics, experiment design
6. **TPU/Hardware-aware ML** — XLA compilation, operator fusion, memory layout

## Their ML Tech Stack (Known)
- **Framework:** JAX (primary for research), TensorFlow (production legacy), PyTorch
- **Hardware:** TPUs (v4/v5), custom TensorFlow XLA compiler
- **Training Platform:** Borg cluster scheduler, internal distributed training
- **ML Platform:** Vertex AI, TFX (TensorFlow Extended) pipelines
- **Data:** Spanner, BigQuery, internal Flume/Dataflow

## Sample Questions from This Company

### Q: Derive the gradient for cross-entropy loss with softmax output.
**What they're testing:** Math fluency and comfort deriving from first principles

**Green flags:** Start from definition, show chain rule cleanly, arrive at y-hat minus y result, note numerical stability trick (subtract max before exp)

**Red flags:** Can't derive it, jump to "the gradient is y-hat - y" without showing work

### Q: How does BERT's pre-training differ from GPT's? When would you use each?
**What they're testing:** Architecture knowledge and judgment

**Green flags:** BERT = bidirectional MLM (sees both sides, better for understanding); GPT = autoregressive (left-to-right, natural for generation). Use BERT for classification/NER/QA; GPT for generation/completion

**Red flags:** Confusing bidirectional vs unidirectional; can't name a concrete use case for each

### Q: Design a distributed training system for a 500B parameter model across 1024 TPUs.
**What they're testing:** Systems + ML depth

**Green flags:** Tensor parallelism (split weight matrices), pipeline parallelism (assign layers), ZeRO-style optimizer state sharding, gradient checkpointing, bf16 training

**Red flags:** "Just use data parallelism" — doesn't work for models larger than GPU memory

### Q: How do you reduce latency for a ranking model from 500ms to <100ms?
**What they're testing:** Systematic optimization thinking

**Green flags:** Profile first (where is time spent?), candidate generation vs ranking decomposition, approximation (ANN instead of exact), quantization, feature caching, lighter model architecture

**Red flags:** Jump to one solution without profiling

### Q: Walk through the attention mechanism from scratch. Why does it scale as O(n^2) in sequence length?
**What they're testing:** Transformer internals at derivation depth

**Green flags:** Q = XW_Q, K = XW_K, V = XW_V; Attention = softmax(QK^T / sqrt(d_k)) * V; O(n^2) because every token attends to every other token; sparse attention / FlashAttention as mitigations

**Red flags:** Describe attention at a high level without the matrix operations; can't explain the quadratic complexity

## 3-Week Prep Strategy
**Week 1:** Math review. Derive: gradient descent, backprop, attention mechanism, Bayes' theorem. Practice explaining derivations out loud, cleanly.

**Week 2:** Systems. Study transformer architecture in depth, distributed training (data/model/pipeline parallelism), Google's ML papers (Word2Vec, BERT, T5, ViT).

**Week 3:** Mock interviews. Focus on thinking out loud. Practice whiteboard derivations. Do 5+ ML system design mocks.

## Insider Tips
- Write clean math on the whiteboard — Google interviewers judge mathematical maturity
- Reference papers when relevant: "In the original Attention is All You Need paper..."
- Always sanity-check answers: "If gradient is y-hat minus y, does it have the right sign?"
- Googleyness: be collaborative, acknowledge uncertainty, invite discussion
- Google has more rounds than most companies — prepare for 5-6 hour onsite stamina
