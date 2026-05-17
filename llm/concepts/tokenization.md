# Tokenization

## TL;DR
Converting text into tokens (integers) that models can process. Tokenization strategies (BPE, WordPiece, SentencePiece) fundamentally affect model capability and efficiency. Understanding tokenization is essential for prompt engineering and production LLM systems.

## Core Intuition
Models can't process raw text — they work with numbers. Tokenization bridges that gap by mapping text → integers. The granularity matters: too coarse (full words) and you have huge vocabularies; too fine (characters) and you have long sequences. BPE finds the sweet spot.

## How It Works

**Byte-Pair Encoding (BPE):** iteratively merge the most frequent adjacent bytes/tokens.
1. Start with character-level tokens
2. Find the most common adjacent pair (A, B)
3. Merge them into a single token (AB)
4. Repeat until you reach vocab size

**WordPiece:** similar to BPE but merges based on likelihood gain (used in BERT).

**SentencePiece:** operates on raw bytes, no assumption of spaces → better for non-Latin scripts.

**Vocabulary size:** typically 50k–250k tokens. Larger = more semantic compression but harder to train.

## Key Properties / Trade-offs
- BPE is lossless: any text can be tokenized and recovered (mostly)
- Token efficiency: "tokenization" might be 1 token (WordPiece) or 3 tokens (BPE)
- Context length: with fixed context, token count directly limits text length
- Language effects: English ≈ 4 chars per token; Chinese ≈ 1.3 chars per token

## Common Mistakes / Gotchas
- Assuming tokens = words — they're usually subwords
- Not understanding token length affects cost (API billing per token)
- Tokenizing on the wrong side of train/test split → data leakage

## Code Example
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [15496, 11, 995, 0]
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What is tokenization?" | Converting text to token IDs. Different schemes (BPE, WordPiece) trade off vocabulary size vs sequence length. |
| "Why not just use words?" | Huge vocabulary, can't handle rare words. Subword tokens compress better. |
| "Effect on context length?" | More tokens per text → shorter effective context. Impacts max input length. |

## Related Topics
- [Transformers](../../ml/concepts/deep-learning/transformers.md) — [Pretraining](pretraining.md)

## Resources
- [Neural Text Generation With Transformers (OpenAI blog)](https://openai.com/research/tokenizers) — explains BPE
- [SentencePiece: A simple and language independent approach](https://arxiv.org/abs/1808.06226)
