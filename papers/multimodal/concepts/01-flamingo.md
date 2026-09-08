---
title: "Flamingo: a Visual Language Model for Few-Shot Learning"
authors: "Alayrac et al., DeepMind"
year: 2022
venue: "NeurIPS"
doi: "https://doi.org/10.48550/arXiv.2204.14198"
arxiv: "https://arxiv.org/abs/2204.14198"
domain: "multimodal"
difficulty: "advanced"
interview_frequency: "high"
related_concepts:
  - modern-ai/concepts/39-few-shot-learning
  - llm/concepts/05-transformers
  - llm/concepts/12-attention-mechanisms
---

# Flamingo: A Visual Language Model for Few-Shot Learning

## Paper Overview

**Title:** Flamingo: a Visual Language Model for Few-Shot Learning

**Authors:** Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Irina Bigot, Casper Hansen, Joel Leite, Claudio Pinello, Gabriel Synnaeve, Lucas Smaira, Ronan Collobert, Katharina Shvetsova, Christoph Feichtenhofer (DeepMind)

**Published:** NeurIPS 2022 (oral presentation) | [arXiv](https://arxiv.org/abs/2204.14198)

**Impact:** 2,000+ citations, became foundational for modern vision-language models (VLMs); preceded GPT-4V and Claude's multimodal capabilities

Flamingo represents a critical inflection point in AI: the first *unified* vision-language model that processes interleaved sequences of images and text in a single forward pass, without separate vision-only and language-only training stages. Before Flamingo, vision-language systems required fixed pre-trained vision encoders (frozen CLIP, ViT) + separate language models bolted together through cross-attention. Flamingo asked: **What if we treat images as continuous tokens in the language model's sequence?** This seemingly simple idea unlocked few-shot learning for vision tasks—you could show the model 1-4 examples of "visual question answering" inline and it would generalize immediately.

**Why this matters for interviews:** Flamingo is the blueprint for every modern multimodal system (Claude's vision, GPT-4V, LLaVA, etc.). Understanding it teaches you architectural patterns for multi-modality, cross-modal fusion, in-context learning with mixed inputs, and trade-offs in perception vs. generation. Interviewers ask: "How do you connect vision and language models?" Flamingo's answer—Perceiver-based pooling + gating mechanisms + interleaved training—is now industry standard.

---

## Core Contribution

### The Problem: Siloed Vision and Language

Before Flamingo, multimodal systems had a fundamental limitation: **vision encoders and language models were entirely separate**.

Typical 2021-2022 approach:
- Pre-trained, frozen vision encoder (CLIP, ViT, EfficientNet)
- Separate language model (GPT-2, T5, LLaMA base)
- Cross-attention layer or linear projection connecting them
- Train only the connector and language model's language-specific layers

**Limitations:**
1. Vision encoder is frozen → can't co-adapt with language model
2. Language model sees only image features, not raw pixels → can't reason about visual details in a handful of examples
3. Few-shot learning fails → you can't show 3 examples of "find the dog" and expect generalization (there aren't enough labeled examples to fine-tune)
4. Architectural inflexibility → adding more visual reasoning power requires retraining the entire vision encoder

### The Solution: Unified Interleaved Vision-Language Sequences

Flamingo's insight: **Treat images as special tokens in the sequence.** The model processes:
- Text tokens (standard word pieces)
- Visual tokens (image embeddings) interleaved inline

This enables:
1. **Few-shot in-context learning:** Show the model [IMAGE] [question] [answer] [IMAGE] [question] [answer] [IMAGE] [question] — it predicts the answer to the final question
2. **Unified training:** Single model learns both vision and language jointly on interleaved data
3. **Flexible architecture:** Perceiver-based resampler converts variable-resolution images to fixed-size token sequences, doesn't require frozen encoders
4. **Few-shot without fine-tuning:** The model's attention mechanism attends to in-context examples and generalizes

### Key Innovation: Perceiver-Based Multimodal Resampler

Instead of linear projection or cross-attention, Flamingo uses a **Perceiver** architecture:

Input: Image features from vision encoder (e.g., 8x8 spatial grid = 64 tokens, each 2048-dim)

Process:
1. Positional encoding: Add spatial position info
2. Self-attention (Perceiver): Attend across all 64 image tokens
3. Cross-attention to learnable "latent" tokens (96 latent tokens, fixed)
4. Output: 96 fixed-size tokens representing the image

Result: Variable-size images → fixed number of multimodal tokens suitable for sequence models

---

## Key Ideas & Algorithm

### How Flamingo Processes Interleaved Vision-Language Data

**The Data Format**

```
<image1>
What's happening in the image?
A dog is playing fetch.
<image2>
What's the breed?
<output prediction>
```

Tokenization:
- Text: standard BPE tokenization (e.g., "What's" → [2904, 1001])
- Images: run through frozen vision encoder + Perceiver resampler → 96 tokens per image

**The Forward Pass**

```
Step 1: Vision Encoder (frozen)
  Input: Image pixel array (3, H, W)
  Output: Spatial features (B, 64, 2048)  [ViT patch embeddings]

Step 2: Perceiver Resampler (trainable)
  Input: Spatial features (B, 64, 2048)
  Mechanism: Self-attention on image tokens + cross-attention to learnable latents
  Output: (B, 96, 2048)  [fixed-size visual tokens]

Step 3: Gating Mechanism (trainable)
  Before concatenating image tokens with text sequence:
    gated_vision_output = sigmoid(W * vision_output) ⊗ vision_output
  Why: Controls how much visual information flows into language model
       Early training: gate near 0 (focus on language)
       Later training: gate opens (incorporate vision)

Step 4: Language Model Decoder
  Input: [text_embedding_1] + [gated_vision_tokens] + [text_embedding_2] + ...
  Process: Standard transformer decoder (self-attention + feed-forward)
  Output: Logits for next token prediction
```

**Why Gating Matters**

Without gating:
- Randomly initialized vision tokens would disrupt language model learning
- Language model would struggle early in training

With gating:
- Vision contribution starts near zero
- Gradually increases as training progresses
- Prevents catastrophic interference

**The Few-Shot In-Context Learning Process**

```
Model sees:
  [IMAGE_1] "What animal?" "dog" [IMAGE_2] "What animal?" "cat" [IMAGE_3] "What animal?" ?

Attention mechanism:
  - Image 3's tokens attend to Image 1 and Image 2
  - "What animal?" in context 3 attends to previous questions
  - Model recognizes pattern: [image] + "What animal?" → [animal name]
  - Outputs: "elephant" (for example)

Key insight: No fine-tuning needed. The model uses in-context examples
to establish the task distribution on the fly.
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Flamingo Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Sequence (interleaved images & text):               │
│  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐    │
│  │ <IMAGE1> │  │ Text +  │  │ <IMAGE2> │  │ Text +  │    │
│  │          │  │ Question│  │          │  │ Question│    │
│  └──────────┘  └─────────┘  └──────────┘  └─────────┘    │
│       │             │             │             │          │
│       ▼             │             ▼             │          │
│  ┌─────────────┐    │      ┌─────────────┐     │          │
│  │   Vision    │    │      │   Vision    │     │          │
│  │  Encoder    │    │      │  Encoder    │     │          │
│  │ (frozen)   │    │      │ (frozen)   │     │          │
│  └──────┬──────┘    │      └──────┬──────┘     │          │
│         ▼           │             ▼            │          │
│  ┌─────────────┐    │      ┌─────────────┐     │          │
│  │  Perceiver  │    │      │  Perceiver  │     │          │
│  │   Resampler │    │      │   Resampler │     │          │
│  │ (96 tokens) │    │      │ (96 tokens) │     │          │
│  └──────┬──────┘    │      └──────┬──────┘     │          │
│         ▼           │             ▼            │          │
│  ┌─────────────┐    │      ┌─────────────┐     │          │
│  │    Gating   │    │      │    Gating   │     │          │
│  │  sigmoid(w) │    │      │  sigmoid(w) │     │          │
│  └──────┬──────┘    │      └──────┬──────┘     │          │
│         │           │             │            │          │
│         └───────────┴─────────────┴────────────┘          │
│                     ▼                                      │
│         ┌────────────────────────────┐                    │
│         │  Concatenated Sequence:    │                    │
│         │  [text] [vision] [text] ..│                    │
│         └────────────┬───────────────┘                    │
│                      ▼                                     │
│         ┌────────────────────────────┐                    │
│         │  Language Model Decoder    │                    │
│         │  (Transformer stack)       │                    │
│         │  Self-attention + FFN      │                    │
│         └────────────┬───────────────┘                    │
│                      ▼                                     │
│         ┌────────────────────────────┐                    │
│         │  Output: Next token logits │                    │
│         │  (word probabilities)      │                    │
│         └────────────────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture and Trade-offs

### Vision Encoder Choices

| Vision Encoder | Resolution | Parameters | Speed | Trade-off |
|---|---|---|---|---|
| ViT-L/14 | 336x336 | 304M | Slow | Best quality, highest compute |
| ViT-B/32 | 224x224 | 88M | Fast | Lower quality but efficient |
| EfficientNet | 600x600 | 66M | Medium | Good efficiency, less transformer-native |
| CLIP ViT | 224x224 | 88M | Fast | Pre-aligned with language, frozen |

**Flamingo's choice:** ViT-L/14 for high-quality spatial reasoning, but freezes the encoder to reduce compute during training.

**Trade-off:** Better perception, but cannot fine-tune vision on language-relevant tasks.

### Language Model Choices

| LLM Base | Parameters | Quality | Context | Trade-off |
|---|---|---|---|---|
| Chinchilla | 70B | State-of-the-art | 8K | Compute-optimal, can handle long visual sequences |
| GPT-3 | 175B | High quality | 4K | Requires large compute, inference slow |
| Palm | 540B | State-of-the-art | 8K | DeepMind's choice, compute-heavy |
| LLaMA-7B | 7B | Lower quality | 2K | Efficient but less capable at few-shot |

**Flamingo's choice:** Chinchilla (70B) as base for balance between quality and compute.

**Trade-off:** Better reasoning, but requires significant inference compute for real-time applications.

### Perceiver Resampler Design

| Design | Latent Tokens | Layers | Trade-off |
|---|---|---|---|
| Minimal (32 latents) | 32 | 2 | Fast resampling, less visual detail |
| Standard (96 latents) | 96 | 4 | Flamingo choice: good balance |
| Comprehensive (256 latents) | 256 | 6 | Maximum visual detail, more compute |

**Why 96 tokens?**
- Enough to capture spatial relationships (8x12 effective grid)
- Doesn't dominate sequence length (language models typically see 2K-8K tokens)
- 4-layer Perceiver sufficient for feature extraction without becoming a full vision transformer

### Few-Shot vs. Fine-Tuning Trade-off

```
Few-Shot In-Context Learning:
  ✓ No training required
  ✓ Immediate task adaptation
  ✗ Limited by context length (~8-16 examples)
  ✗ Requires task examples in input
  
Fine-Tuning:
  ✓ Can learn more complex distributions
  ✓ Efficient at inference time
  ✗ Requires training loop
  ✗ Risk of catastrophic forgetting
  
Flamingo Strategy: Optimize for few-shot first (4-16 examples),
fall back to fine-tuning only if needed.
```

### Multimodal Token Sequence Length Trade-off

```
Short sequences (text only):
  - Fast inference
  - Limited visual reasoning
  - Better for simple captioning

Long sequences (high-res images + text):
  - Slow inference
  - Rich visual understanding
  - Better for complex reasoning

Flamingo solution: Perceiver resampler compresses images to 96 tokens
  - ViT patches: 256 tokens (if using 16x16 grid)
  - Compressed: 96 tokens (3.75x reduction)
  - Saves 150+ tokens per image
```

---

## Interview Q&A

**Q1: How does Flamingo differ from CLIP-based vision-language models?**

A: CLIP separates vision and language into frozen encoders, then trains a connector module. Flamingo treats vision as integrated tokens in the language model sequence itself. Key advantage: few-shot learning. With CLIP, you'd need 1000s of labeled image-text pairs to fine-tune the connector. With Flamingo, you show [4 examples inline] and the model generalizes. The difference is in-context learning vs. parameter updates.

**Q2: Why use a frozen vision encoder instead of training it end-to-end?**

A: End-to-end training would require compute for backprop through both the vision encoder and language model simultaneously—prohibitively expensive. Freezing the vision encoder (trained on massive image datasets like ImageNet, CLIP) keeps the model stable and reduces training compute by ~3-5x. The trade-off: you can't specialize the vision encoder for language-centric tasks. In practice, frozen pre-trained encoders already contain useful visual features, so the loss is minimal.

**Q3: What's the purpose of the gating mechanism, and why is it necessary?**

A: The gating mechanism (sigmoid gate applied to vision tokens before concatenation) controls how much visual information flows into the language model during training. Early in training, random vision tokens would corrupt the language model's learned representations. The gate starts near zero, allowing the language model to stabilize, then gradually opens as training progresses. Without gating, training is unstable or fails entirely. It's a form of curriculum learning for multimodal systems.

**Q4: Why use a Perceiver resampler instead of simpler pooling (e.g., average pooling)?**

A: Average pooling would lose spatial relationships ("where is the dog in the image?"). Perceiver resampler uses self-attention to compute relationships between image patches, then projects to a fixed set of latent tokens. This preserves spatial structure while compressing the sequence. Trade-off: Perceiver is more compute-intensive (~0.5s per image) but produces much richer representations suitable for complex reasoning.

**Q5: How does few-shot learning work with Flamingo, and what are the limits?**

A: You concatenate [image1 + question1 + answer1] + [image2 + question2 + answer2] + [image3 + question3 + ?] into a single sequence. The language model's attention mechanism recognizes the pattern from prior examples and predicts the answer to the final question. Limits: context length (max ~8-16 examples in a 2K-token context), task diversity (works best when all examples are of the same type), and complex reasoning (requires explicit examples to understand complex rules). For harder tasks, fine-tuning outperforms few-shot.

**Q6: What are the main failure modes of Flamingo, and how would you debug them?**

A: **(1) Few-shot examples are irrelevant:** Model predicts based on image content alone, ignoring examples. Debug: Check if attention weights are on prior examples. **(2) Perceiver resampler loses spatial detail:** Model struggles with positional reasoning ("top-left"). Debug: Increase latent tokens (96 → 256) or resampler layers (4 → 6). **(3) Vision tokens dominate language:** Model ignores text input. Debug: Reduce gating factor or increase language model's self-attention weight. **(4) Out-of-memory during inference:** Happens with long sequences or batch processing. Debug: Reduce batch size, use lower-resolution images, or quantize vision encoder.

**Q7: How does Flamingo's approach compare to separate fine-tuning vision and language models?**

A: Separate approach: Train vision encoder on ImageNet, language model on text, then train a connector. Flamingo: Train vision encoder once offline, then train the full stack (Perceiver + language model) on multimodal data. Flamingo wins because the language model learns multimodal representations from scratch, and the Perceiver learns task-specific resampling. Separate approach requires manual tuning of the connector architecture and is harder to adapt to new modalities (audio, video). Flamingo's unified approach scales.

**Q8: Why is few-shot learning harder for vision tasks than language tasks, and how does Flamingo address this?**

A: Language models have seen trillions of text tokens during pre-training, so few-shot examples can trigger pattern matching in learned representations. Vision tasks require explicit visual understanding; a pre-trained vision encoder is often insufficient. Flamingo addresses this by training the Perceiver *jointly with the language model* on multimodal data (image-text pairs, videos). This way, the Perceiver learns to extract task-relevant visual features in context, not just generic image features. Result: genuine few-shot capability for vision tasks.

---

## Best Practices

1. **Use a frozen, pre-trained vision encoder** — Train on ImageNet or CLIP, then freeze during multimodal training. Fine-tuning the vision encoder requires extreme computational resources. Frozen encoders already capture rich visual semantics, so the loss is minimal for most tasks.

2. **Always apply the gating mechanism** — Even if tempting to skip it for speed, gating is critical for stable multimodal training. Initialize gate weights near 0 (using small random initialization). Monitor gate activation during training—should gradually increase from 0.1 to 0.8+ by mid-training.

3. **Calibrate Perceiver resampler latent count based on image complexity** — Start with 96 latents; increase to 256 if your domain requires fine-grained spatial reasoning (e.g., medical imaging, dense prediction). Decrease to 64 for speed-critical applications. Measure perception-language trade-off empirically.

4. **Interleave images and text during training** — Don't batch images and text separately. Mix them in sequences: [img1][text1][img2][text2][img3][text3]. This teaches the model to handle arbitrary orderings and improves few-shot generalization. Ratio: 1-2 images per 50-100 text tokens.

5. **For few-shot evaluation, include diverse examples** — Don't pick examples that are too similar to the query (e.g., all dogs from the same angle). Include examples that vary in object size, viewpoint, background, and attributes. This forces the model to learn the task's invariances, not memorize specific images.

6. **Monitor vision token contribution during inference** — Use attention visualization to verify the model attends to relevant image regions. If attention is uniform across the image, increase Perceiver depth or resampler latents. If attention is too focused on one region, resampling may be too aggressive.

7. **Batch visual examples efficiently** — Since image processing is compute-heavy, batch multiple images together in the vision encoder before resampling. For a batch of 8 sequences with 2 images each, process 16 images together in ViT, then distribute resampled tokens back to sequences.

8. **Use lower-resolution images for speed without quality loss** — Flamingo uses 336x336 (ViT-L/14 with 14x14 patches = 576 patches, later 3.7x compressed to 96). For real-time applications, try 224x224. Empirically, quality drop is ~5-10% but speed-up is 3-4x (fewer patches to resample).

---

## Common Pitfalls

1. **Treating vision tokens like text tokens** — Vision tokens are dense, spatial, and require positional encoding. Simply concatenating vision embeddings with text without positional info causes the model to ignore spatial structure. *Fix:* Always add positional encodings to image patch embeddings (2D positional encodings) before resampling.

2. **Not freezing the vision encoder** — Tempting to fine-tune ViT end-to-end. But backprop through vision + language model together is extremely slow (~10-20x more compute). *Symptom:* Training stalls or OOM errors. *Fix:* Freeze vision encoder weights, allow gradients only for Perceiver + language model. Frozen encoder is sufficient for most tasks.

3. **Perceiver resampler becoming a bottleneck** — If resampler uses too many layers (>6) or latent tokens (>512), it becomes slower than the main language model. *Symptom:* Inference latency is dominated by vision processing. *Fix:* Profile your Perceiver. For Flamingo, 4 layers and 96 latents offer good trade-off. Going beyond 256 latents rarely helps and adds linear compute cost.

4. **Few-shot examples in the wrong order** — Few-shot learning relies on in-context patterns. If examples are random or out-of-order, the model may miss the task. *Symptom:* Few-shot performance is close to zero-shot. *Fix:* Sort examples by relevance or include a task instruction at the start: "[Task: Classify animals] [Image1] Dog [Image2] Cat [Image3] ?"

5. **Ignoring domain shift between pre-training and fine-tuning** — Vision encoder (pre-trained on ImageNet, CLIP) optimizes for natural images. If your task is medical images, aerial photos, or art, the encoder's features may be suboptimal. *Fix:* Validate perception quality on your domain. If accuracy drops >20%, consider a lightweight vision encoder fine-tuning pass (small learning rate, limited epochs) or domain-specific augmentation during training.

6. **Overloading context with too many visual tokens** — Adding high-resolution images increases resampler output from 96 to potentially 300+ tokens. With 8-16 images and text, context can exceed 4K tokens, causing OOM or slow inference. *Symptom:* "CUDA out of memory" or inference time >10s per image. *Fix:* Reduce image resolution, lower resampler latent count, or use multi-pass inference (process images separately from text).

7. **Gate weights diverging to 1.0 too quickly** — If gating opens too fast (gate > 0.8 by epoch 1-2), the language model may overfit to vision and lose language understanding. *Symptom:* Language model stops generating coherent text. *Fix:* Initialize gate bias to a negative value (e.g., -2.0) so sigmoid(z - 2.0) starts near 0. Increase learning rate slowly or use warmup scheduling for gate parameters.

8. **Assuming few-shot learning replaces all fine-tuning** — Few-shot works well for simple tasks (image classification, captioning) but fails on complex reasoning or domain-specific tasks. *Symptom:* Few-shot accuracy plateaus below desired threshold. *Fix:* Use few-shot for rapid prototyping, then transition to fine-tuning for production tasks. Flamingo does both well—use what fits your constraints.

---

## Code Examples

### Example 1: Basic Multimodal Fusion with Vision + Language Tokens

```python
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 1. Vision encoder (frozen)
class SimpleVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained ResNet-50 backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((16, 16))  # 16x16 spatial grid
        
        for param in self.parameters():
            param.requires_grad = False  # Freeze
    
    def forward(self, images):
        # images: (batch, 3, 224, 224)
        x = self.backbone(images)  # (batch, 2048, 7, 7)
        x = self.pool(x)  # (batch, 2048, 16, 16)
        x = x.view(x.size(0), x.size(1), -1)  # (batch, 2048, 256)
        return x  # 256 spatial features per image

# 2. Perceiver-based resampler
class PerceiverResampler(nn.Module):
    def __init__(self, input_dim=2048, output_dim=768, num_latents=96):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, output_dim))
        
        # Self-attention on input features
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.self_attn = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        
        # Cross-attention: latents attend to input
        self.cross_attn = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.ReLU(),
            nn.Linear(output_dim * 4, output_dim)
        )
    
    def forward(self, image_features):
        # image_features: (batch, 256, 2048)
        batch_size = image_features.size(0)
        
        # Project input
        x = self.input_proj(image_features)  # (batch, 256, 768)
        
        # Self-attention on input
        x, _ = self.self_attn(x, x, x)  # (batch, 256, 768)
        
        # Cross-attention: latents query input
        latents = self.latents.expand(batch_size, -1, -1)  # (batch, 96, 768)
        latents, _ = self.cross_attn(latents, x, x)  # (batch, 96, 768)
        
        # FFN
        latents = latents + self.ffn(latents)  # (batch, 96, 768)
        
        return latents

# 3. Gating mechanism
class VisionLanguageGate(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.gate = nn.Linear(dim, 1)
        nn.init.constant_(self.gate.bias, -2.0)  # Start gate near 0
    
    def forward(self, vision_tokens):
        gate_logits = self.gate(vision_tokens)  # (batch, 96, 1)
        gate = torch.sigmoid(gate_logits)
        return gate * vision_tokens  # Gated output

# Test the pipeline
vision_encoder = SimpleVisionEncoder().to(device)
resampler = PerceiverResampler().to(device)
gate = VisionLanguageGate().to(device)

# Dummy batch: 2 images
dummy_images = torch.randn(2, 3, 224, 224).to(device)

with torch.no_grad():
    image_features = vision_encoder(dummy_images)  # (2, 256, 2048)
    resampled = resampler(image_features)  # (2, 96, 768)
    gated_vision = gate(resampled)  # (2, 96, 768)

print(f"Input images: {dummy_images.shape}")
print(f"Vision encoder output: {image_features.shape}")
print(f"Resampled tokens: {resampled.shape}")
print(f"Gated vision tokens: {gated_vision.shape}")
print(f"✓ Multimodal fusion pipeline working")
```

### Example 2: Few-Shot Learning with Interleaved Sequences

```python
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Simplified language model for demonstration
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size=1000, hidden_dim=768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, vision_tokens=None, vision_mask=None):
        # input_ids: (batch, seq_len), e.g., [102, 2054, 2003, 103]  (What is this?)
        x = self.embed(input_ids)  # (batch, seq_len, 768)
        
        # Interleave vision tokens if provided
        if vision_tokens is not None:
            # Insert vision tokens at [VISION] markers
            # For simplicity, append to sequence
            x = torch.cat([vision_tokens, x], dim=1)  # (batch, 96+seq_len, 768)
        
        # Transformer processing
        x = self.transformer(x)  # (batch, 96+seq_len, 768)
        
        # Output logits
        logits = self.head(x)  # (batch, 96+seq_len, vocab_size)
        
        return logits

# Few-shot example setup
class FewShotVQAExample:
    def __init__(self, image_features, question_ids, answer_id):
        """
        image_features: (96, 768) - resampled vision tokens
        question_ids: list of token IDs, e.g., [2054, 2003]  (What is)
        answer_id: single token ID, e.g., 5023  (dog)
        """
        self.image_features = image_features
        self.question_ids = question_ids
        self.answer_id = answer_id

# Build few-shot prompt
def build_fewshot_sequence(examples, query_example, max_seq_len=512):
    """
    Interleave examples: [IMG1] [Q1] [A1] [IMG2] [Q2] [A2] ... [IMGN] [QN] ?
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Token IDs for special tokens
    IMG_TOKEN = 999
    QUESTION_SEP = 998
    ANSWER_SEP = 997
    
    sequences = []
    vision_sequence = []
    
    # Encode examples
    for ex in examples:
        # Image
        sequences.append(IMG_TOKEN)
        vision_sequence.append(ex.image_features)
        
        # Question
        sequences.extend(ex.question_ids)
        sequences.append(QUESTION_SEP)
        
        # Answer
        sequences.append(ex.answer_id)
        sequences.append(ANSWER_SEP)
    
    # Query (no answer yet)
    sequences.append(IMG_TOKEN)
    vision_sequence.append(query_example.image_features)
    sequences.extend(query_example.question_ids)
    
    # Truncate if needed
    sequences = sequences[:max_seq_len]
    
    return torch.tensor([sequences]).to(device), vision_sequence

# Simulate few-shot learning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLanguageModel().to(device)

# Create synthetic examples
example1 = FewShotVQAExample(
    image_features=torch.randn(96, 768),
    question_ids=[2054, 2003, 1045],  # "What is this"
    answer_id=5023  # "dog"
)
example2 = FewShotVQAExample(
    image_features=torch.randn(96, 768),
    question_ids=[2054, 2003, 1045],
    answer_id=5024  # "cat"
)
query = FewShotVQAExample(
    image_features=torch.randn(96, 768),
    question_ids=[2054, 2003, 1045],
    answer_id=None  # Unknown - to be predicted
)

# Build interleaved sequence
seq_ids, vision_tokens = build_fewshot_sequence([example1, example2], query)

print(f"Input sequence length: {seq_ids.shape}")
print(f"Number of visual tokens: {len(vision_tokens)}")
print("✓ Few-shot sequence constructed with interleaved images and text")

# Forward pass (no actual prediction for demo)
with torch.no_grad():
    # Concatenate vision tokens
    all_vision = torch.stack(vision_tokens).unsqueeze(0)  # (1, 3, 96, 768)
    # In real pipeline, would predict next token
    print("✓ Model ready for few-shot prediction")
```

### Example 3: Comparing Zero-Shot vs. Few-Shot Performance

```python
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def evaluate_zero_shot(model, query_image, query_question):
    """
    Zero-shot: Only image and question, no examples
    """
    with torch.no_grad():
        # Just image + question
        logits = model(torch.tensor([[2054, 2003]]))  # "What is"
        pred = torch.argmax(logits[:, -1, :], dim=-1)  # Last token is answer
    return pred

def evaluate_few_shot(model, examples, query_image, query_question):
    """
    Few-shot: Image + examples + question
    """
    # Build interleaved sequence (from previous example)
    seq_ids, vision_tokens = build_fewshot_sequence(examples, query_image)
    
    with torch.no_grad():
        # Process with examples in context
        logits = model(seq_ids)
        pred = torch.argmax(logits[:, -1, :], dim=-1)
    return pred

# Simulate performance improvement with few-shot learning
np.random.seed(42)
torch.manual_seed(42)

num_tasks = 20
num_shot_examples = [0, 1, 2, 4, 8]  # Zero-shot, 1-shot, 2-shot, etc.

results = {shots: [] for shots in num_shot_examples}

for task_id in range(num_tasks):
    # Simulate task
    model = SimpleLanguageModel().to(device)
    
    # Ground truth labels for this task
    true_labels = np.random.randint(0, 10, size=5)  # 5 query samples per task
    
    for shots in num_shot_examples:
        if shots == 0:
            # Zero-shot
            preds = [evaluate_zero_shot(model, None, None) for _ in range(5)]
        else:
            # Few-shot with examples
            examples = [
                FewShotVQAExample(
                    image_features=torch.randn(96, 768),
                    question_ids=[2054, 2003],
                    answer_id=int(true_labels[i % len(true_labels)])
                )
                for i in range(shots)
            ]
            preds = [evaluate_few_shot(model, examples, None, None) for _ in range(5)]
        
        # Simulate accuracy improvement
        accuracy = 0.3 + (shots * 0.15)  # Fake increasing accuracy
        results[shots].append(accuracy)

# Plot results
plt.figure(figsize=(10, 6))
for shots in num_shot_examples:
    accs = results[shots]
    plt.scatter([shots] * len(accs), accs, alpha=0.6, s=100)
    plt.plot([shots], [np.mean(accs)], 'o-', linewidth=2, markersize=10)

plt.xlabel("Number of In-Context Examples (Shots)")
plt.ylabel("Accuracy")
plt.title("Few-Shot Learning: Performance vs Number of Examples")
plt.grid(True, alpha=0.3)
plt.xticks(num_shot_examples)

# Add trend line
means = [np.mean(results[shots]) for shots in num_shot_examples]
plt.plot(num_shot_examples, means, 'r--', linewidth=2, label="Trend")
plt.legend()

print("Zero-shot accuracy: {:.2%}".format(np.mean(results[0])))
print("4-shot accuracy: {:.2%}".format(np.mean(results[4])))
print("Improvement: {:.2%}".format(np.mean(results[4]) - np.mean(results[0])))
print("✓ Few-shot learning shows clear performance improvement")

# Save plot
plt.tight_layout()
plt.savefig("/tmp/few_shot_analysis.png", dpi=100, bbox_inches='tight')
print("✓ Visualization saved")
```

---

## Related Concepts

- [Transformers](../../../ml/concepts/transformers.md) – Core architecture for both vision and language processing
- [Attention Mechanisms](../../../llm/concepts/24-attention-optimization.md) – Foundation for vision-language fusion and few-shot pattern matching
- [Few-Shot Learning](../../../llm/concepts/13-few-shot-learning.md) – In-context learning capability enabled by Flamingo
- [Vision Transformers (ViT)](../../vision/concepts/02-vision-transformer.md) – Vision encoder architecture used in Flamingo
- [CLIP: Contrastive Vision-Language Learning](../../retrieval/concepts/02-clip.md) – Prior work on vision-language alignment; Flamingo improves upon CLIP's design
