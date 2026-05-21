"""
Auto-generated from 29-token-optimization.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Token Optimization
# ## Learning Objectives
# 1. Understand core concepts and applications of token optimization
# 2. Implement token optimization with HuggingFace Transformers
# ======================================================================

# ======================================================================
# ## Level 1: Basic Implementation
# ======================================================================

# Token Optimization
# Level 1: Basic implementation

import torch
from transformers import AutoTokenizer, AutoModel

# Load model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Basic usage
text = "Example text for token-optimization"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("Model loaded and inference completed")
print(f"Output shape: {outputs.last_hidden_state.shape}")



# ======================================================================
# ## Level 2: Advanced Implementation
# ======================================================================

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

class TokenOptimizationPipeline:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def process(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def batch_process(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

# Usage
pipeline = TokenOptimizationPipeline()
result = pipeline.process("Example text")
print(f"Result shape: {result.shape}")



# ======================================================================
# ## Real-World Example 1: Production Pattern
# ======================================================================

# Example 1 for Token Optimization
# See concept file for details



# ======================================================================
# ## Real-World Example 2: Advanced Usage
# ======================================================================

# Example 2 for Token Optimization
# See concept file for details



# ======================================================================
# ## Real-World Example 3: Optimization
# ======================================================================

# Example 3 for Token Optimization
# See concept file for details



# ======================================================================
# ## Key Takeaways
# **When to use token optimization:**
# - For NLP tasks with sequence data
# - When transfer learning from pre-trained models saves time
# ======================================================================
