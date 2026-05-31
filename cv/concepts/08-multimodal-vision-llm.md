# Multimodal Vision-Language Models

## Detailed Explanation

Multimodal vision-language models (VLMs) combine visual encoders with large language
models to enable image understanding, visual question answering (VQA), image captioning,
and grounded reasoning. They represent a convergence of computer vision and natural
language processing into a unified model that can reason about both modalities.

The dominant architecture is exemplified by LLaVA (Liu et al., 2023): a pre-trained
visual encoder (typically ViT-L/14 from CLIP) produces a sequence of visual tokens from
image patches, a lightweight linear projection (or a small MLP) maps visual token
embeddings into the LLM's token embedding space, and these visual tokens are prepended
to the text token sequence before being fed into the LLM. The LLM then attends to both
visual and text tokens using its standard causal attention mechanism.

Training proceeds in two stages: (1) pretraining — freeze the LLM and visual encoder,
train only the projection layer on 600k image-text pairs from CC3M to align modalities;
(2) instruction fine-tuning — unfreeze the LLM and fine-tune on 150k visual instruction
following examples to teach the model to respond to visual questions.

Key open challenges include:

**Hallucination**: The model confidently describes objects not present in the image,
inherited from LLM's tendency to generate plausible-sounding continuations. Mitigation
includes RLHF with visual feedback and contrastive decoding.

**Resolution**: Standard ViT is trained at 224x224. Text-heavy images (documents, charts)
require high resolution; LLaVA-HD and InternVL use dynamic resolution to handle this.

**Modality alignment**: Ensuring visual tokens carry semantically meaningful information
in the LLM's embedding space requires careful projection design and sufficient pretraining.

## Core Intuition

A multimodal VLM is an LLM that has learned to "read" images by converting them into a
foreign language it already knows — the vector space of word embeddings. The projection
layer is the dictionary that translates visual patch features into tokens the LLM can
process alongside words. Once trained, the LLM does not distinguish whether a token
came from text or an image.

## How It Works

```
graph TD
    A[Input Image 224x224] --> B[Patch extraction 16x16 patches]
    B --> C[ViT visual encoder]
    C --> D[196 visual tokens D_v each]
    D --> E[Linear projection W]
    E --> F[196 visual tokens D_llm each]
    G[Text tokens input ids] --> H[LLM embedding layer]
    H --> I[text token embeddings]
    F --> J[Concatenate visual + text tokens]
    I --> J
    J --> K[LLM transformer layers causal attention]
    K --> L[Output text tokens]
    L --> M[Generated answer]
```

1. **Patch tokenization**: Divide input image into non-overlapping 16x16 patches.
   For a 224x224 image, this yields 196 patches. Each patch is linearly projected to a
   D_v-dimensional embedding by the ViT.

2. **Visual encoding**: The ViT (e.g., CLIP ViT-L/14 with D_v=1024) processes all 196
   patch tokens plus a class token through L transformer layers with self-attention.
   Produces 196 visual feature vectors capturing local and global image context.

3. **Projection**: A learned linear layer W maps each visual token from D_v to D_llm
   (the LLM's embedding dimension, e.g., 4096 for LLaMA-7B). This is the "translation
   layer" between vision and language spaces.

4. **Token concatenation**: Visual tokens are prepended to text input tokens:
   [VIS_1, VIS_2, ..., VIS_196, TEXT_1, TEXT_2, ..., TEXT_Q, ANSWER_1, ...]. The LLM
   processes this as a standard token sequence with causal attention.

5. **Two-stage training**:
   Stage 1 (pretraining): Only W is trainable. Loss computed on caption tokens only.
   Stage 2 (instruction tuning): W + LLM are trainable. Loss on answer tokens only.
   Visual encoder stays frozen in both stages.

6. **Inference**: Feed image + question, generate answer tokens autoregressively.
   Temperature sampling or beam search for generation. KV-cache for visual tokens
   speeds up multi-turn conversations by 5-10x.

## Architecture and Trade-offs

### Visual Encoder Options

| Encoder | Dimension | Visual Tokens | Pretrain Data | Strengths |
|---------|-----------|---------------|---------------|-----------|
| CLIP ViT-L/14 | 1024 | 256 | 400M image-text pairs | Strong semantic understanding |
| CLIP ViT-L/14@336 | 1024 | 576 | 400M | Higher resolution |
| SigLIP-400M | 1152 | 729 | 4B pairs | Better zero-shot |
| InternViT-6B | 3200 | 256 | Proprietary | Very high capacity |
| DINOv2-L | 1024 | 256 | 142M images (SSL) | Strong spatial features |

### Projection Layer Options

| Method | Parameters | Alignment Speed | Notes |
|--------|------------|-----------------|-------|
| Linear (W matrix) | D_v * D_llm | Fast | LLaVA v1.0 original |
| 2-layer MLP | 2 * D_v * D_llm | Slower | LLaVA v1.5, +3-5% accuracy |
| Q-Former (cross-attn) | ~200M | Very slow | BLIP-2, compresses to 32 tokens |
| MLP + layer norm | 2 * D_v * D_llm | Moderate | Balanced, most common |

### Key VLM Models Comparison

| Model | Visual Encoder | LLM | VQA Acc | MMBench | Notes |
|-------|----------------|-----|---------|---------|-------|
| LLaVA-1.5-7B | CLIP ViT-L | LLaMA-3-8B | 80% | 67% | Baseline open VLM |
| LLaVA-1.5-13B | CLIP ViT-L | LLaMA-3-13B | 83% | 70% | Standard research |
| InternVL-2-26B | InternViT-6B | InternLM-20B | 89% | 82% | State-of-art open |
| GPT-4V | Unknown | GPT-4 | 93%+ | 87%+ | Proprietary |
| Gemini Ultra | Unknown | Gemini | 90%+ | 88%+ | Proprietary |

## Interview Q&A

**Q: Why are visual tokens prepended rather than appended to text tokens?**

A: Causal attention means each token attends to all previous tokens. If visual tokens
come first, every text token can attend to every visual token — enabling full visual
grounding for each word. If appended, the visual tokens only attend to the full question
but text tokens never see the image. Empirically, prepending gives 2-5% higher VQA
accuracy. For multi-image inputs, interleaving visual and text tokens at image positions
(as in Flamingo) generalizes this principle.

**Q: LLaVA hallucination rate is around 30% on POPE benchmark. What causes this and
how would you reduce it?**

A: LLMs generate fluent, plausible-sounding text conditioned on context; when visual
grounding is weak, they default to prior knowledge. The projection layer may not
sufficiently ground language generation in visual features. Mitigation: (1) RLHF with
visual grounding reward — score responses by whether described objects are actually
present; (2) contrastive decoding — subtract logits from a text-only model run in
parallel; (3) include negative examples in instruction tuning (teach the model to say
"no" when asked about absent objects).

**Q: What happens if you skip the pretraining stage and go directly to instruction tuning?**

A: The projection layer starts with random weights, meaning visual tokens look like noise
to the LLM. Instruction tuning must simultaneously learn the projection AND task-following,
causing unstable training. Final accuracy drops by 10-20% on VQA. The pretraining stage
is a critical warmup that teaches the projection to map visual features to the LLM's
language space before any question-answering structure is imposed.

**Q: How would you extend a standard 224x224 VLM to handle high-resolution images like
scanned documents?**

A: Two strategies: (1) Dynamic resolution — divide the high-res image into 224x224 tiles,
encode each tile independently, concatenate tile tokens, and feed to LLM (LLaVA-HD,
InternVL approach). Tokens scale quadratically with tiles, so limit to 4-6 tiles.
(2) Any-resolution ViT — train with variable resolution and positional embedding
interpolation (NaViT approach). Option 1 is practical for immediate adaptation of
existing models without retraining the visual encoder.

**Q: Why does the Q-Former in BLIP-2 compress 196 visual tokens to 32 learned queries?
What does this trade-off cost?**

A: Q-Former reduces visual tokens from 196 to 32 via cross-attention between learned
query vectors and visual patch features. This dramatically reduces LLM sequence length
and thus compute. The cost is spatial information loss — queries learn to summarize
scene semantics but discard spatial layout, making the model worse at grounding tasks
that require "where is object X" reasoning. BLIP-2 excels at captioning (semantics-first)
but underperforms LLaVA on spatial reasoning benchmarks.

**Q: At inference, a user sends 10 messages in a conversation with the same image. How
would you optimize throughput?**

A: Cache the key-value pairs for visual tokens after the first forward pass. Visual tokens
are the same across all 10 messages, so their K,V projections can be computed once and
reused. This reduces per-message compute from O(V+T) to O(T) where V=196 visual tokens
and T=text tokens. In practice, this gives 3-5x throughput improvement for multi-turn
conversations. Implement as a KV-cache prefix that is pinned in GPU memory between turns.

## Best Practices

- Use float16 for both the visual encoder and LLM during inference; the projection layer
  can stay in float32 without memory cost (it is tiny). Never use int8 quantization for
  the projection layer — alignment quality is too sensitive to quantization error.
- During instruction tuning, mask the loss on visual tokens and instruction tokens —
  only compute loss on answer tokens. Training on visual tokens adds noise and does not
  improve downstream VQA performance.
- Batch visual token encoding across multiple images in parallel before feeding to the
  LLM — the visual encoder is the throughput bottleneck, not the LLM, in most pipelines.
- Use at least 100k visual instruction examples for stage 2 fine-tuning. Below 50k, the
  model follows instructions inconsistently. Quality (diverse task types, accurate labels)
  matters more than raw count above 500k.
- Monitor the visual-text attention entropy — if text tokens consistently have near-zero
  attention on visual tokens, the model is ignoring the image. Inspect attention maps
  during early training.
- For production serving, separate the visual encoder as a separate microservice that
  computes and caches visual features. The LLM server then only receives 196-token
  visual embeddings, not raw images.

## Common Pitfalls

- **Not freezing the visual encoder during stage 1**: Updating CLIP during projection
  pretraining destroys the semantic structure that makes visual features useful. CLIP
  learned from 400M pairs — overwriting it with 600k pairs causes catastrophic forgetting.
  Fix: always freeze the visual encoder in stage 1; only unfreeze in stage 2 if you have
  at least 10M instruction pairs.

- **Evaluating on benchmark tasks the model was explicitly trained on**: Many instruction
  tuning datasets include VQAv2, GQA, and TextVQA samples. Reporting accuracy on these
  benchmarks without held-out evaluation overstates generalization. Always evaluate on a
  held-out benchmark (MMBench, POPE, SeedBench) that was not in the fine-tuning mix.

- **Ignoring positional encoding interpolation at new resolutions**: ViT positional
  encodings are trained at 224x224. Feeding a 336x336 image without interpolating
  positional embeddings causes the model to see misaligned spatial structure. Fix: use
  bicubic interpolation of the 2D positional embedding grid to match new resolution.

- **Computing loss on visual tokens during instruction tuning**: Visual tokens don't
  have a "correct" next-token prediction target in the instruction-following context.
  Including them in the loss adds noise and can diverge the projection layer from its
  alignment role. Always mask visual token positions in the cross-entropy loss computation.

- **Using a single resolution for diverse image types**: Photos, charts, and documents
  require different resolutions. A single-resolution model performs poorly on text-heavy
  images. Fix: implement dynamic tiling for high-res inputs or use LLaVA-HD style
  multi-tile encoding.

## Related Concepts

- [05-contrastive-learning-vision.md](./05-contrastive-learning-vision.md) — CLIP's
  contrastive pre-training is the foundation for most VLM visual encoders
- [06-diffusion-models.md](./06-diffusion-models.md) — text conditioning in diffusion
  models uses cross-modal attention similar to VLM architectures
- [07-video-understanding.md](./07-video-understanding.md) — video VLMs extend image
  VLMs by tokenizing temporal sequences of frames
