# Pretraining

## TL;DR
Pretraining a large language model on massive unlabeled text teaches it language patterns, knowledge, and reasoning. The foundation of all modern LLMs (GPT, BERT, Llama, Claude). Two paradigms: causal (GPT) predicts next token; masked (BERT) predicts masked tokens.

## Core Intuition
A student learns by reading books (pretraining) before taking exams (downstream tasks). By predicting the next word billions of times, the model internalizes grammar, facts, logic, and reasoning. This learned foundation transfers to new tasks via fine-tuning.

## How It Works

**Causal LM (GPT-style):** predict next token given all previous tokens.
- Objective: $\max_\theta \sum_i \log p(x_i | x_{<i}; \theta)$
- Decoding: autoregressive (generate one token at a time)
- Uses causal masking: attend only to past positions

**Masked LM (BERT-style):** predict masked tokens given context on both sides.
- Objective: $\max_\theta \sum_i \mathbb{1}[\text{masked}_i] \log p(x_i | x_{\setminus i}; \theta)$
- Bidirectional attention (better for classification tasks)
- Non-autoregressive (encode once, extract representations)

**Scale laws:** performance improves predictably with model size, data, and compute (Chinchilla: ~20 tokens per parameter for optimal training).

## Key Properties / Trade-offs
- Causal: better for generation, harder to train (language modeling is harder than masked prediction)
- Masked: better for understanding (classification, retrieval), can't generate directly
- Compute: pretraining is 10–100× more expensive than fine-tuning

## Common Mistakes / Gotchas
- Confusing pretraining (unsupervised) and fine-tuning (supervised) — both matter
- Assuming a larger model always generalizes better — true to a point, but architectural choices matter
- Ignoring data quality in pretraining — garbage in, garbage out at scale

## Code Example
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
# model already pretrained on 40GB of text
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What is pretraining?" | Unsupervised learning on massive text to predict next tokens (causal) or masked tokens (masked). Teaches language and world knowledge. |
| "Causal vs masked?" | Causal: unidirectional, better for generation. Masked: bidirectional, better for understanding/classification. |
| "How does pretraining help downstream tasks?" | Learned patterns transfer. Fine-tuning on small labeled data is much faster than training from scratch. |

## Related Topics
- [Tokenization](tokenization.md) — [Transformers](../../ml/concepts/deep-learning/transformers.md) — [Fine-tuning](finetuning.md)

## Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2 paper)](https://openai.com/research/language-models-are-unsupervised-multitask-learners)
