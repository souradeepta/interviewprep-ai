"""
Auto-generated from 08-lora.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # LoRA (Low-Rank Adaptation) - Production Implementation
# **Complete guide with real HuggingFace libraries and production patterns.**
# This notebook uses:
# ======================================================================

# ======================================================================
# ## Setup
# ======================================================================

# Install required packages
# !pip install transformers torch sentence-transformers datasets peft bitsandbytes

import warnings
warnings.filterwarnings('ignore')

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")


# ======================================================================
# ## Quick Start
# ======================================================================

# LoRA with HuggingFace
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ======================================================================
# ## Production Implementation
# ======================================================================

# LoRA Training with HuggingFace Trainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import torch

# Load model and data
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)

# Training setup
training_args = TrainingArguments(
    output_dir="./lora_output",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01
)

# Load small dataset
dataset = load_dataset("glue", "sst2")

# Preprocess
def preprocess(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length")

dataset = dataset.map(preprocess, batched=True)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle().select(range(100))
)

# trainer.train()
print("LoRA training ready")


# ======================================================================
# ## Real-World: Inference
# ======================================================================

# Real-World: LoRA Inference at Scale
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List

class LoRAInferenceService:
    """Production LoRA inference service"""

    def __init__(self, base_model_id="gpt2", lora_id=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        if lora_id:
            self.base_model = PeftModel.from_pretrained(self.base_model, lora_id)

        self.base_model.to(self.device)
        self.base_model.eval()

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using LoRA model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], max_length: int = 100):
        """Batch generation for efficiency"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_length)
            results.append(result)
        return results

# Usage
service = LoRAInferenceService(base_model_id="gpt2")
# service = LoRAInferenceService(base_model_id="gpt2", lora_id="your-lora-model")

output = service.generate("Once upon a time")
print(f"Generated: {output}")


# ======================================================================
# ## Real-World: Merging
# ======================================================================

# Real-World: LoRA Merging for Deployment
from peft import PeftModel
from transformers import AutoModelForCausalLM

class LoRAMergeService:
    """Merge LoRA weights into base model for deployment"""

    def __init__(self, base_model_id, lora_model_id):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
        self.lora_model = PeftModel.from_pretrained(self.base_model, lora_model_id)

    def merge_and_export(self, output_path):
        """Merge LoRA into base weights and save"""
        # Merge
        merged_model = self.lora_model.merge_and_unload()

        # Save as standalone model
        merged_model.save_pretrained(output_path)
        print(f"Merged model saved to {output_path}")

        return merged_model

    def get_size_comparison(self):
        """Compare model sizes"""
        base_size = sum(p.numel() for p in self.base_model.parameters())
        lora_size = sum(p.numel() for p in self.lora_model.parameters()
                       if 'lora' in str(p))

        return {
            "base_params": base_size,
            "lora_params": lora_size,
            "merged_params": base_size
        }

# Production usage
# merger = LoRAMergeService("gpt2", "your-lora")
# merged = merger.merge_and_export("./merged_model")
# sizes = merger.get_size_comparison()
# print(f"LoRA params: {sizes['lora_params']:,}")


# ======================================================================
# ## Production Checklist
# - [ ] Load models from HuggingFace Hub
# - [ ] Set up GPU device handling
# - [ ] Implement batch processing
# ======================================================================
