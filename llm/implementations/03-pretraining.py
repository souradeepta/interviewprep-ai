"""
Auto-generated from 03-pretraining.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Pretraining
# ## Learning Objectives
# 1. Understand core concepts and applications of pretraining
# 2. Implement pretraining with HuggingFace Transformers
# ======================================================================

# ======================================================================
# ## Level 1: Basic Implementation
# ======================================================================

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, Trainer, TrainingArguments
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Simple generation
prompt = "Machine learning is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate
output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")



# ======================================================================
# ## Level 2: Advanced Implementation
# ======================================================================

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.inputs = tokenizer(texts, truncation=True, max_length=max_length,
                                 padding="max_length", return_tensors="pt")

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.inputs['input_ids'][idx]
        }

# Mock pretraining data
texts = [
    "Natural language processing enables machines to understand text",
    "Deep learning models learn patterns from large datasets",
    "Transfer learning reuses knowledge from one task to another",
] * 100

# Setup training
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
dataset = SimpleTextDataset(texts, tokenizer)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# NOTE: Actual training would run with: trainer.train()
print(f"Dataset size: {len(dataset)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print("(Training would begin with: trainer.train())")



# ======================================================================
# ## Real-World Example 1: Production Pattern
# ======================================================================

# Real-World: Continued Pretraining on Domain Data
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a base pretrained model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Domain-specific texts (e.g., medical domain)
domain_texts = [
    "The patient presents with symptoms of diabetes",
    "Diagnosis requires blood glucose testing",
    "Treatment options include medication and lifestyle changes",
] * 50

# Encode domain data
inputs = tokenizer(domain_texts, return_tensors="pt", padding=True, truncation=True)

# Continued pretraining setup
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop (simplified)
for epoch in range(2):
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

print(f"\nDomain-adapted model ready for fine-tuning")



# ======================================================================
# ## Real-World Example 2: Advanced Usage
# ======================================================================

# Real-World: Distributed Pretraining Setup
from transformers import GPT2Config, GPT2LMHeadModel
import torch

# Create custom model architecture
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
)

model = GPT2LMHeadModel(config)

# Would use distributed training in practice
def setup_distributed_training():
    '''
    Production setup:
    1. Use torch.distributed.launch
    2. Split data across GPUs
    3. Use DistributedDataParallel
    4. Sync gradients
    '''
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (distributed training on CPU not recommended)")

setup_distributed_training()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")



# ======================================================================
# ## Real-World Example 3: Optimization
# ======================================================================

# Real-World: Pretraining Curriculum Learning
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Curriculum learning: easy data first, then harder
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Stage 1: Simple English (high-quality data)
stage1_data = [
    "The cat sat on the mat",
    "Dogs are loyal animals",
] * 50

# Stage 2: Mixed language (code + text)
stage2_data = [
    "def hello(): return 'world'",
    "Machine learning uses data",
] * 50

# Stage 3: Noisy/domain-specific data
stage3_data = [
    "@user awesome #ML post",
    "C++ programming langauge",  # Intentional typo
] * 50

stages = [
    ("Stage 1: Simple", stage1_data),
    ("Stage 2: Mixed", stage2_data),
    ("Stage 3: Noisy", stage3_data),
]

print("Curriculum pretraining stages:")
for stage_name, data in stages:
    inputs = tokenizer(data[:10], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    print(f"  {stage_name}: Loss = {outputs.loss:.4f} (Data points: {len(data)})")



# ======================================================================
# ## Key Takeaways
# **When to use pretraining:**
# - For NLP tasks with sequence data
# - When transfer learning from pre-trained models saves time
# ======================================================================
