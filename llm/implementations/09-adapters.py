"""
Auto-generated from 09-adapters.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Adapters - Production Implementation
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

# LoRA Adapters - Quick Start
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA (adapter pattern)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ======================================================================
# ## Production Implementation
# ======================================================================

# Production LoRA Setup with Training
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch
from datasets import Dataset

# Load pretrained model for fine-tuning
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA for classification task
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Setup training
training_args = TrainingArguments(
    output_dir="./lora_results",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=100
)

# Example data
texts = ["This is great!", "This is terrible!"]
labels = [1, 0]

tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
dataset = Dataset.from_dict({
    "input_ids": tokenized["input_ids"],
    "attention_mask": tokenized["attention_mask"],
    "labels": torch.tensor(labels)
})

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Train
# trainer.train()

print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# ======================================================================
# ## Real-World: Huggingface
# ======================================================================

# Real-World: HuggingFace Model Hub Integration
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceAdapterService:
    """Load and use adapters from HuggingFace Hub"""

    def __init__(self, base_model_id="gpt2"):
        self.base_model_id = base_model_id
        self.model = None
        self.tokenizer = None

    def load_base_model(self):
        """Load base model from HF Hub"""
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        print(f"Loaded base model: {self.base_model_id}")

    def load_adapter(self, adapter_id):
        """Load adapter from HF Hub"""
        # Example: "username/gpt2-sentiment-adapter"
        self.model = PeftModel.from_pretrained(self.model, adapter_id)
        print(f"Loaded adapter: {adapter_id}")

    def inference(self, text, max_length=50):
        """Run inference with adapter"""
        if self.model is None:
            self.load_base_model()

        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0])

# Production usage
service = HuggingFaceAdapterService(base_model_id="gpt2")
service.load_base_model()

# In production: load from Hub
# service.load_adapter("your-org/your-adapter")
# result = service.inference("Your prompt here")

print("Adapter service ready")


# ======================================================================
# ## Real-World: Multilora
# ======================================================================

# Real-World: Multi-Adapter Hub with Switch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MultiAdapterHub:
    """Manage multiple task adapters on shared base model"""

    def __init__(self, base_model_name="gpt2"):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.adapters = {}
        self.active_adapter = None

    def add_adapter(self, task_name, adapter_config=None):
        """Add new task adapter"""
        if adapter_config is None:
            adapter_config = LoraConfig(
                r=8, lora_alpha=16, target_modules=["c_attn"],
                lora_dropout=0.1, bias="none"
            )

        # Create model copy with adapter
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model.config.model_type
        )
        model = get_peft_model(model, adapter_config)
        self.adapters[task_name] = model

    def switch_adapter(self, task_name):
        """Switch to different task"""
        if task_name not in self.adapters:
            raise ValueError(f"Adapter '{task_name}' not found")
        self.active_adapter = task_name
        print(f"Switched to adapter: {task_name}")

    def generate(self, prompt, task_name=None, max_length=100):
        """Generate using selected adapter"""
        if task_name:
            self.switch_adapter(task_name)

        if not self.active_adapter:
            raise ValueError("No adapter selected")

        model = self.adapters[self.active_adapter]
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)

        return self.tokenizer.decode(outputs[0])

# Usage
hub = MultiAdapterHub("gpt2")

# Add task-specific adapters
hub.add_adapter("sentiment")
hub.add_adapter("summarization")
hub.add_adapter("translation")

# Switch and generate
prompt = "This movie was amazing"
# result = hub.generate(prompt, task_name="sentiment")
print("Multi-adapter hub ready for production")


# ======================================================================
# ## Production Checklist
# - [ ] Load models from HuggingFace Hub
# - [ ] Set up GPU device handling
# - [ ] Implement batch processing
# ======================================================================
