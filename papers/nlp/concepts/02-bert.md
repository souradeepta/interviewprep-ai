---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
authors: "Devlin, Chang, Lee, Toutanova"
year: 2018
venue: "NAACL"
doi: "https://doi.org/10.48550/arXiv.1810.04805"
arxiv: "https://arxiv.org/abs/1810.04805"
domain: "nlp"
difficulty: "intermediate"
interview_frequency: "very_high"
related_concepts:
  - llm/concepts/01-transformers
  - llm/concepts/15-masked-language-modeling
  - llm/concepts/16-pre-training-fine-tuning
---

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Paper Overview

**Title:** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)

**Published:** NAACL 2018 | [arXiv](https://arxiv.org/abs/1810.04805)

**Citation:** 70,000+ (transformed NLP industry)

Before BERT, language models were unidirectional: GPT trained left-to-right, missing context from the right. BERT introduced **bidirectional pre-training**—the model sees both left and right context simultaneously—using a clever training objective called **masked language modeling** (MLM). Instead of predicting the next token, BERT randomly masks tokens and predicts them from surrounding context. This forces the model to develop deep bidirectional understanding. BERT pre-trained on massive text (100B+ tokens) then fine-tuned on downstream tasks (classification, QA, NER) achieved state-of-the-art performance across NLP benchmarks. The pre-training → fine-tuning paradigm became the standard for NLP and influenced all subsequent large models.

**Why this matters for interviews:** BERT is the bridge between Transformers and modern NLP. You need to understand: (1) why bidirectional is better than unidirectional, (2) masked language modeling as a training objective, (3) the pre-training → fine-tuning paradigm, (4) how to fine-tune for your task. This is asked constantly in NLP roles.

---

## Core Contribution

### The Problem: Limited Bidirectional Context

Pre-2018 language models were mostly unidirectional:
- GPT: predicts left-to-right (uses only context to the left)
- Consequence: doesn't see right context for understanding

Example: "I went to the bank to withdraw money."
- Left-to-right (GPT): at "bank", doesn't know if it's "financial institution" or "river bank"
- Bidirectional (BERT): sees "withdraw money" to the right → understands "financial institution"

### The Solution: Masked Language Modeling (MLM)

Instead of predicting the next token:
1. Randomly mask 15% of tokens
2. Predict masked tokens from surrounding context (left AND right)
3. Forces model to learn bidirectional understanding

**Why this works:**
- Both left and right context available → deep understanding
- Tasks like QA benefit from seeing full context before predicting
- No "exposure bias" (training vs. inference mismatch for autoregressive models)

### Key Innovation: Next Sentence Prediction (NSP)

Additionally, train on:
1. Two concatenated sentences: [CLS] sentence_A [SEP] sentence_B [SEP]
2. Predict if B follows A (50% yes, 50% no)
3. Learn sentence-level relationships

---

## Key Ideas & Algorithm

### BERT Pre-training Objectives

**Objective 1: Masked Language Modeling (MLM)**

1. Tokenize text: [CLS] token1 token2 [MASK] token4 ... [SEP]
2. For 15% of tokens:
   - 80% replace with [MASK]
   - 10% replace with random token
   - 10% keep original (teach model to handle noisy input)
3. Train to predict original token from context
4. Loss: cross-entropy on masked positions only

**Objective 2: Next Sentence Prediction (NSP)**

1. 50% of training: B is actual next sentence (label = 1)
2. 50% of training: B is random sentence (label = 0)
3. Train [CLS] token to predict IsNext (binary classification)

**Combined loss:**
```
Loss = MLM_loss + NSP_loss
```

### BERT Architecture

```
Input: [CLS] token1 token2 [MASK] token4 [SEP]
  ↓
Token Embedding + Positional Embedding + Segment Embedding
  ↓
Transformer Block (12-24 layers, 12-16 heads)
  ↓
MLM Head: Linear(d_model → vocab_size) for masked positions
NSP Head: Linear(d_model → 2) for [CLS] token classification
```

### BERT Variants & Sizes

| Model | Layers | Heads | d_model | Parameters | Pre-training |
|-------|--------|-------|---------|-----------|--------------|
| BERT-base | 12 | 12 | 768 | 110M | 4 TPUs, 4 days |
| BERT-large | 24 | 16 | 1024 | 340M | 64 TPUs, 4 days |
| RoBERTa | 12-24 | 12-16 | 768-1024 | 125M-355M | Better pre-training |
| ALBERT | 12 | 12 | 768 | 12M | Parameter sharing |

---

## Architecture & Trade-offs

### Bidirectional vs. Unidirectional

| Aspect | Unidirectional (GPT) | Bidirectional (BERT) |
|--------|---------------------|---------------------|
| **Context** | Left context only | Left + right |
| **Training** | Predict next token | Predict masked tokens |
| **Task fit** | Generation, LM | Understanding, classification |
| **Efficiency** | O(N) inference (autogressive) | O(1) inference (can encode all at once) |
| **When to use** | Text generation | NLP tasks (QA, NER, classification) |

### Pre-training vs. Fine-tuning

**BERT Pre-training Paradigm:**

```
Large unlabeled data (100B+ tokens)
  ↓
Pre-train with MLM + NSP (months on large cluster)
  ↓
Save learned weights
  ↓
For each downstream task:
  - Add task-specific head (Linear layer)
  - Fine-tune on labeled data (hours on 1-2 GPUs)
  - Freezes pre-trained weights or use small LR (1e-5)
```

**Benefits:**
- Transfer learning: learn general patterns on large data, specialize on task
- Efficiency: 1000× less labeled data needed vs. training from scratch
- State-of-the-art: BERT-based models dominate NLP benchmarks

### Masking Strategy Trade-offs

| Strategy | MLM Rate | Mask Type | Pros | Cons |
|----------|----------|-----------|------|------|
| Standard | 15% | [MASK], random, keep | Balanced | Some data waste |
| ELECTRA | 15% | Realistic corruption | Efficient | More complex |
| RoBERTa | 15% | [MASK] only | Simpler | Slight worse performance |

---

## Interview Q&A

**Q: Why is BERT bidirectional? What's the advantage over GPT's left-to-right?**

A: BERT sees both left and right context simultaneously, allowing it to understand words based on full surrounding context. Example: "I went to the bank to withdraw money"—BERT sees "withdraw money" to the right, so it understands "bank" = financial institution. GPT only sees left context, so it can't use that cue. Bidirectional works for understanding tasks (classification, QA, NER) where you have the full input. It doesn't work for generation because you generate left-to-right—can't attend to tokens that don't exist yet. Trade-off: bidirectional for understanding, unidirectional for generation.

**Q: How does masked language modeling work? Why is it better than predicting the next token?**

A: MLM randomly masks 15% of tokens and predicts them from context (left + right). This is better than next-token prediction because: (1) uses both left and right context, (2) no exposure bias (training vs. inference mismatch—at test time, you see full context), (3) works for understanding tasks (you have full input). The masking strategy: 80% [MASK] token, 10% random token, 10% original. Why? If always [MASK], model learns to recognize [MASK] and attends specially to it. Random/original tokens force model to attend to actual content. Cost: less efficient than next-token prediction (only 15% of tokens predict, not 100%), but better performance on downstream tasks.

**Q: When would you use BERT vs. GPT? What's the use case for each?**

A: BERT for understanding tasks where you have full input: classification, NER, QA, semantic similarity, information extraction. You add a task-specific head and fine-tune. GPT for generation tasks: language modeling, translation, summarization, dialogue. You generate token-by-token autoregressively. BERT can do generation (e.g., masked LM generates multiple tokens), but it's inefficient (need to mask, predict, unmask, repeat). GPT generates naturally left-to-right. In practice: use BERT for understanding, GPT for generation. Modern hybrid approaches (encoder-decoder, like T5) combine both: BERT-like encoder for understanding, GPT-like decoder for generation.

**Q: How do you fine-tune BERT for a new task?**

A: Add a task-specific head (e.g., linear layer) on top of BERT and train on labeled data for your task. Example for classification: take [CLS] token output (shape: batch × d_model), pass through linear layer (d_model → num_classes), predict class. Use small learning rate (1e-5 to 5e-5) to avoid catastrophic forgetting of pre-trained weights. Typically fine-tunes in 3-10 epochs with 10K-100K labeled examples. Freezing early layers and fine-tuning only last 2-3 layers can sometimes help if labeled data is very small. Common tasks: classification, QA (predict start/end token), NER (per-token classification), semantic similarity (fine-tune on paired sentences).

**Q: Why does masked language modeling work better than standard language modeling for pre-training?**

A: Standard LM (next-token prediction) only predicts using left context. Masked LM forces the model to use both left and right context. Empirically, masked LM converges faster and achieves better performance on downstream tasks. Intuition: bidirectional models are harder to train with standard loss (you can't attend to future tokens you don't know), so MLM sidesteps this by directly training on bidirectional understanding. Cost: only 15% of tokens generate loss (vs. 100% for standard LM), so you need more data/steps. Worth it for downstream task performance.

**Q: What happens if you pre-train on one domain and fine-tune on another?**

A: Transfer learning works but performance drops if domains are very different. BERT pre-trained on general text (Wikipedia, books) transfers well to most NLP tasks. For specialized domains (biomedical, legal, code), domain-specific pre-training or continued pre-training helps. SciBERT pre-trained on scientific papers > general BERT on scientific QA. Cost: domain-specific pre-training requires 10-100× less data than general pre-training but still months on large clusters. In practice: use general BERT for most tasks, domain BERT if labeled task data is very limited and domain is very different.

---

## Best Practices

- **Fine-tuning learning rate:** Use 2e-5 to 5e-5 for most tasks. Larger LR (1e-4) if you have lots of labeled data. Smaller LR (1e-5) if data is very limited. Use learning rate warmup for stability.

- **Fine-tuning epochs:** 3-5 epochs typical. More epochs for small datasets (100-1K examples). Fewer for large datasets (10K+). Monitor validation loss to prevent overfitting.

- **Batch size:** 16-32 for fine-tuning on single GPU. Larger batches (128-256) help with optimization but need more memory. Gradient accumulation simulates larger batches.

- **Maximum sequence length:** 512 tokens standard (BERT max length). Longer sequences need RoPE or other positional encoding. Shorter sequences are faster. Truncate or split long documents.

- **Task-specific head design:** Typically simple (1-2 linear layers) to preserve pre-trained knowledge. Complex heads can overfit on small datasets. For classification, add dropout on [CLS] before final linear layer.

- **Frozen vs. fine-tuned weights:** Fine-tuning all weights works for most tasks. Freezing early layers helps for tiny datasets (<100 examples). Differential learning rates (smaller LR for early layers, larger for later) can help.

- **Domain adaptation:** For domain shift, continue pre-training on domain data with MLM before fine-tuning on task. Or use domain-specific BERT variants (SciBERT, BioBERT, FinBERT, CodeBERT).

---

## Common Pitfalls

- **Mistake: Using very high learning rate when fine-tuning.** BERT's pre-trained weights are sensitive. High LR (0.001-0.01) causes divergence or catastrophic forgetting. Impact: loss spikes or diverges.
  → Fix: Use small LR (1e-5 to 5e-5) for fine-tuning. Use warmup scheduler for first 10% of steps.

- **Mistake: Training for too many epochs.** Fine-tuning on small datasets (100-1K examples) for 10+ epochs causes overfitting to training set, poor validation/test performance. Impact: gap between train and validation loss grows.
  → Fix: Use early stopping based on validation loss. Typically 3-5 epochs is enough.

- **Mistake: Not using [CLS] token for classification.** [CLS] is specially trained to aggregate sentence meaning. Using other tokens (mean pooling, max pooling) loses this learned structure. Impact: worse classification accuracy.
  → Fix: Always use [CLS] token output for classification. For other tasks (NER, QA), use per-token outputs.

- **Mistake: Fine-tuning with very large batch size.** Large batches (512+) can diverge during fine-tuning or converge to poor local minimum. Impact: training loss is high or noisy.
  → Fix: Use batch size 16-32. If you need larger effective batch size, use gradient accumulation.

- **Mistake: Ignoring input preprocessing (tokenization, padding).** BERT tokenizer is subword tokenizer (WordPiece). If you use wrong tokenizer or skip special tokens [CLS], [SEP], attention masks don't work correctly. Impact: model receives corrupted input.
  → Fix: Always use AutoTokenizer from Hugging Face. Ensure [CLS] and [SEP] tokens are present. Use attention_mask for padded sequences.

---

## Code Examples

### Example 1: Basic BERT Tokenization and Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2  # Binary classification
)

# Tokenize sample texts
texts = [
    "This movie is great!",
    "I didn't like it at all.",
]

# Tokenize: returns input_ids, attention_mask, token_type_ids
encodings = tokenizer(
    texts, 
    max_length=128, 
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(f"Input IDs shape: {encodings['input_ids'].shape}")
print(f"Attention mask shape: {encodings['attention_mask'].shape}")

# Forward pass
with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits
    predictions = logits.argmax(dim=-1)

print(f"Predictions: {predictions}")
```

### Example 2: Fine-tuning BERT for Sentiment Classification

```python
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            max_length=max_length, 
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# Sample data
train_texts = ["Great movie!", "Terrible film.", "I loved it!", "Waste of time."]
train_labels = [1, 0, 1, 0]

# Create dataset and dataloader
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
)

# Initialize trainer
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune
trainer.train()
```

### Example 3: Masked Language Modeling with BERT

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load BERT for masked language modeling
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Create masked input
text = "I went to the [MASK] to withdraw money."
inputs = tokenizer(text, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Find [MASK] token position
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1][0]

# Get predictions for masked token
mask_logits = logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_logits, 5, dim=0).indices

print("Top 5 predictions for [MASK]:")
for token_id in top_5_tokens:
    word = tokenizer.decode([token_id])
    print(f"  - {word}")

# Output: bank, river, institution, etc.
```

---

## Related Concepts

- [Attention Is All You Need](./01-attention-is-all-you-need.md) — Transformer architecture foundation
- [GPT-3: Language Models are Few-Shot Learners](./03-gpt3.md) — Unidirectional autoregressive alternative
- [Scaling Laws for Neural Language Models](./04-scaling-laws.md) — How pre-training scale affects performance
- [LoRA: Low-Rank Adaptation](./05-lora.md) — Efficient fine-tuning of BERT

