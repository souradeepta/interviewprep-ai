# LLM Models Quick Reference

> Parameter counts for closed models (GPT-4o, Claude, Gemini) are **not publicly disclosed**.
> Cost figures are approximate as of mid-2025 and change frequently — always check provider pricing pages.

---

## Model Comparison Table

| Model | Params | Context | Strengths | Approx Cost | Best Use |
|-------|--------|---------|-----------|-------------|---------|
| GPT-4o | Undisclosed | 128K tokens | Strong reasoning, multimodal (text+image+audio), fast for its class | ~$2.50/1M in, ~$10/1M out | Complex reasoning, agentic tasks, multimodal apps |
| GPT-4o mini | Undisclosed | 128K tokens | Very fast, cheap, solid at instruction following | ~$0.15/1M in, ~$0.60/1M out | High-throughput classification, extraction, RAG retrieval step |
| Claude 3.5 Sonnet | Undisclosed | 200K tokens | Top coding performance, long context, strong reasoning | ~$3/1M in, ~$15/1M out | Coding assistants, document analysis, agentic tool use |
| Claude 3 Haiku | Undisclosed | 200K tokens | Fastest Claude, very low cost | ~$0.25/1M in, ~$1.25/1M out | Real-time applications, high-volume RAG, lightweight tasks |
| Llama 3.1 70B | 70B | 128K tokens | Best open-weight quality at 70B scale, permissive license | Self-hosted (compute cost) | Private data use cases, fine-tuning for domain adaptation |
| Llama 3.1 8B | 8B | 128K tokens | Fits on single GPU, fast inference, open weight | Self-hosted | Edge inference, fine-tuning, low-resource experiments |
| Mistral 7B | 7B | 32K tokens | Efficient, strong for size, Apache 2.0 license | Self-hosted | On-device inference, fine-tuning baseline, instruction following |
| Mixtral 8x7B | 46.7B (12.9B active) | 32K tokens | MoE architecture: 8B active params, 46.7B total | Self-hosted | Better than 13B dense, cheaper than 70B, multilingual |
| Gemini 1.5 Pro | Undisclosed | 1M tokens (!) | Longest context of any commercial model, multimodal | ~$3.50/1M in, ~$10.50/1M out | Very long document analysis, video understanding, RAG over large corpora |
| Gemini 1.5 Flash | Undisclosed | 1M tokens | Fast + long context at lower cost | ~$0.075/1M in, ~$0.30/1M out | Long-context at scale, real-time pipelines |

---

## Fine-Tuning Support

| Model | Fine-Tuning | Method | When Available |
|-------|------------|--------|----------------|
| GPT-4o | Yes (OpenAI API) | Supervised fine-tuning | Generally available |
| GPT-4o mini | Yes (OpenAI API) | Supervised fine-tuning | Generally available |
| Claude 3.5 Sonnet | No (as of mid-2025) | — | Not publicly available |
| Llama 3.1 70B | Yes (self-hosted) | Full fine-tune, LoRA, QLoRA | Open weights |
| Llama 3.1 8B | Yes (self-hosted) | Full fine-tune, LoRA, QLoRA | Open weights |
| Mistral 7B | Yes (self-hosted) | Full fine-tune, LoRA | Open weights |
| Mixtral 8x7B | Yes (self-hosted) | LoRA (adapter per expert or shared) | Open weights |
| Gemini 1.5 Pro | Yes (Vertex AI) | Supervised fine-tuning | Via Google Cloud |

---

## API Availability

| Model | API Provider | SDK | Self-Host Option |
|-------|-------------|-----|-----------------|
| GPT-4o / GPT-4o mini | OpenAI | openai Python / JS | No |
| Claude 3.5 Sonnet / Haiku | Anthropic | anthropic Python / JS | No |
| Llama 3.1 (all sizes) | Meta (weights) + Groq, Together, Fireworks | Provider-specific | Yes (vLLM, Ollama, llama.cpp) |
| Mistral 7B / Mixtral | Mistral AI + Together, Fireworks | mistralai / OpenAI-compatible | Yes |
| Gemini 1.5 Pro / Flash | Google AI / Vertex AI | google-generativeai | No (managed only) |

---

## Reasoning Capability Ranking (approximate, as of mid-2025)

| Tier | Models | Reasoning Tasks |
|------|--------|----------------|
| Tier 1 (Frontier) | GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro | Complex multi-step reasoning, math olympiad, agentic planning |
| Tier 2 (Strong) | GPT-4o mini, Llama 3.1 70B, Mixtral 8x7B | Most production tasks, RAG, structured extraction |
| Tier 3 (Efficient) | Llama 3.1 8B, Mistral 7B, Claude 3 Haiku | Simple classification, extraction, instruction following |

---

## Model Selection Decision Guide

```
Need best overall quality + API reliability?
  -> GPT-4o or Claude 3.5 Sonnet (use benchmarks for your specific task)

Need best cost-performance for high-volume tasks?
  -> GPT-4o mini or Gemini 1.5 Flash

Need very long context (>128K tokens)?
  -> Gemini 1.5 Pro (1M tokens) or Claude 3.5 Sonnet (200K tokens)

Need to fine-tune on private/proprietary data?
  -> Llama 3.1 70B (self-hosted) or GPT-4o mini via OpenAI fine-tuning API

Need to run entirely on your hardware (data privacy, air-gapped)?
  -> Llama 3.1 8B (single GPU) or Llama 3.1 70B (multi-GPU or quantized)

Need open-source with permissive license for commercial use?
  -> Llama 3.1 (Meta license, commercial use allowed), Mistral 7B (Apache 2.0)

Building a real-time interactive product?
  -> Claude 3 Haiku or GPT-4o mini (lowest latency at acceptable quality)
```

---

## Key Technical Differences

| Property | GPT-4o | Claude 3.5 Sonnet | Llama 3.1 70B | Gemini 1.5 Pro |
|----------|--------|------------------|---------------|----------------|
| Architecture | Transformer (MoE suspected) | Transformer | Dense Transformer | Transformer (MoE) |
| Tokenizer | tiktoken (BPE) | Custom BPE | tiktoken-style (BPE) | SentencePiece |
| Context handling | Sliding window + RoPE | Extended attention | RoPE + grouped-query | Very long context (1M) |
| Multimodal | Text, image, audio | Text, image | Text only | Text, image, video, audio |
| Output format | JSON mode, function calling | Tool use, structured output | Depends on serving framework | Function calling, JSON mode |

---

## Cost Optimization Patterns

1. **Cascade / routing:** Use cheap model (GPT-4o mini) for easy queries; route complex ones to GPT-4o. Can cut costs 60–80%.
2. **Prompt caching:** Anthropic and OpenAI support caching repeated prefix (system prompt). For same-prefix queries, cache discount is ~90% on cached tokens.
3. **Batching:** Offline tasks (classification, extraction) at batch API endpoints typically 50% cheaper with 24h SLA.
4. **Context pruning:** Trim retrieved chunks, conversation history, and system prompt. Each 1K tokens saved = cost reduction at scale.
5. **Open weights self-hosting:** For >10M tokens/day, hosting Llama 3.1 70B on 4×A100s costs less than API pricing.
