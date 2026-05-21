"""
Auto-generated from 01-tokenization.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Tokenization
# ## Learning Objectives
# 1. Understand core concepts and applications of tokenization
# 2. Implement tokenization with HuggingFace Transformers
# ======================================================================

# ======================================================================
# ## Level 1: Basic Implementation
# ======================================================================

import numpy as np
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Basic tokenization
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(f"Decoded: {tokenizer.decode(token_ids)}")

# Special tokens
print(f"\nSpecial tokens:")
print(f"  BOS: {tokenizer.bos_token_id}")
print(f"  EOS: {tokenizer.eos_token_id}")
print(f"  PAD: {tokenizer.pad_token_id}")
print(f"  UNK: {tokenizer.unk_token_id}")



# ======================================================================
# ## Level 2: Advanced Implementation
# ======================================================================

import numpy as np
from transformers import AutoTokenizer
import torch

class TokenizationPipeline:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(self, texts, max_length=128):
        # Tokenize with padding and truncation
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encodings

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_token_frequency(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        return dict(zip(tokens, token_ids))

# Usage
pipeline = TokenizationPipeline()

texts = ["Hello world", "How are you?", "Great to see you"]
encodings = pipeline.tokenize_batch(texts)

print(f"Vocabulary size: {pipeline.get_vocab_size()}")
print(f"Input IDs shape: {encodings['input_ids'].shape}")
print(f"Attention mask shape: {encodings['attention_mask'].shape}")

# Show one example
print(f"\nExample tokens: {pipeline.tokenizer.convert_ids_to_tokens(encodings['input_ids'][0])}")



# ======================================================================
# ## Real-World Example 1: Production Pattern
# ======================================================================

# Real-World: Multi-language Tokenization
from transformers import AutoTokenizer

# Different tokenizers for different languages
tokenizers = {
    "english": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "german": AutoTokenizer.from_pretrained("bert-base-german-uncased"),
    "multilingual": AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
}

texts = {
    "en": "How are you doing today?",
    "de": "Wie geht es dir heute?",
    "multi": "Hello مرحبا Привет"
}

for lang, text in texts.items():
    tokenizer = tokenizers["multilingual"]  # Use multilingual for all
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    print(f"{lang:5} | {text:30} | Tokens: {len(tokens)}")



# ======================================================================
# ## Real-World Example 2: Advanced Usage
# ======================================================================

# Real-World: Streaming Tokenization
from transformers import AutoTokenizer

class StreamingTokenizer:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.buffer = ""

    def process_chunk(self, text_chunk):
        self.buffer += text_chunk
        # Tokenize buffer when we have complete words
        last_space = self.buffer.rfind(" ")
        if last_space > 0:
            to_tokenize = self.buffer[:last_space]
            self.buffer = self.buffer[last_space+1:]
            return self.tokenizer.encode(to_tokenize, add_special_tokens=False)
        return []

    def flush(self):
        if self.buffer:
            token_ids = self.tokenizer.encode(self.buffer, add_special_tokens=False)
            self.buffer = ""
            return token_ids
        return []

# Simulate streaming input
streamer = StreamingTokenizer()
stream = ["Hello ", "world, ", "this is ", "streaming tokenization"]

all_tokens = []
for chunk in stream:
    tokens = streamer.process_chunk(chunk)
    all_tokens.extend(tokens)

final = streamer.flush()
all_tokens.extend(final)

print(f"Total tokens from stream: {len(all_tokens)}")
print(f"Token IDs: {all_tokens[:20]}")  # First 20 tokens



# ======================================================================
# ## Real-World Example 3: Optimization
# ======================================================================

# Real-World: Custom Tokenizer Training
from transformers import AutoTokenizer
import numpy as np

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Analyze token distribution on a corpus
texts = [
    "Machine learning is transforming industries",
    "Transformers have revolutionized NLP",
    "BERT and GPT are foundational models",
    "Tokenization splits text into tokens",
] * 100  # Repeat for analysis

# Encode all texts
all_token_ids = []
for text in texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    all_token_ids.extend(ids)

# Token frequency analysis
unique, counts = np.unique(all_token_ids, return_counts=True)
top_tokens = sorted(zip(counts, unique), reverse=True)[:10]

print("Top 10 tokens in corpus:")
for count, token_id in top_tokens:
    token_text = tokenizer.decode([token_id])
    print(f"  Token {token_id:5d}: '{token_text:15s}' (freq: {count})")



# ======================================================================
# ## Key Takeaways
# **When to use tokenization:**
# - For NLP tasks with sequence data
# - When transfer learning from pre-trained models saves time
# ======================================================================
