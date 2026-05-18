#!/usr/bin/env python3
"""
Insert Interview Q&A sections into 32 LLM concept markdown files.
Inserts before ## Related Topics (or ## Resources if no Related Topics).
Skips files that already have ## Interview Q&A.
"""

import os
import re

CONCEPTS_DIR = "/home/sbisw/github/interviewprep-ml/llm/concepts"

QA_SECTIONS = {
    "01-tokenization.md": """## Interview Q&A

**Q: Why do different tokenizers produce different token counts for the same text, and why does this matter for cost?**
A: Tokenizers differ in their vocabulary and merging rules (BPE vs WordPiece vs SentencePiece). A word like "unbelievable" might be 1 token in one tokenizer but 4 in another. This directly affects API cost (billed per token), context window utilization, and model behavior since the model sees different granularity of input.

**Q: When would you choose BPE over WordPiece tokenization?**
A: BPE is preferred for general-purpose models (GPT family) because it's language-agnostic and handles rare words gracefully by splitting into subwords. WordPiece (BERT family) is preferred when you have a fixed domain vocabulary because it optimizes for maximum likelihood of training data. In practice the difference is small—the tokenizer choice is usually inherited from the pretrained model you use.

**Q: How does tokenization affect multilingual model performance?**
A: Most tokenizers are trained predominantly on English text, so non-Latin scripts get tokenized into many more tokens per word. A Chinese character might be 3-4 tokens vs 1 in a Chinese-optimized tokenizer. This wastes context window, increases cost, and degrades performance because the model sees noisier representations of the input.

**Q: What is the fertility metric and when do you care about it?**
A: Fertility is the average number of tokens per word for a given language/corpus. Low fertility (1.0-1.5) means the tokenizer handles that language well. High fertility (3.0+) means the model is inefficient on that language. You care when deploying multilingual systems—high-fertility languages cost more and perform worse, so you might need a domain-specific tokenizer or a model trained on that language.

**Q: Why can tokenization artifacts cause unexpected model behavior?**
A: Token boundaries create hard splits that the model treats as semantic units. "New York" tokenized as ["New", " York"] vs ["New", "York"] produces different embeddings. Numbers are especially tricky—"100" might be ["1", "00"] making arithmetic harder. Whitespace handling differs across tokenizers, causing issues when prompts are copy-pasted from different sources.

**Q: How would you debug a model that performs poorly on a specific type of input?**
A: First tokenize the input and inspect the tokens. Check fertility (tokens per word), check if domain terms are split unexpectedly, check for unusual whitespace or encoding artifacts. Compare tokenization of working vs failing examples. If domain terms are split, consider adding them to a custom tokenizer vocabulary or preprocessing to normalize them before tokenization.

""",

    "02-embeddings.md": """## Interview Q&A

**Q: When should you use a sentence embedding model vs computing embeddings from a base LLM?**
A: Sentence embedding models (sentence-transformers) are fine-tuned with contrastive losses specifically to make semantically similar texts cluster together. Base LLM embeddings from the last hidden state are often poor for retrieval because they're optimized for next-token prediction, not similarity. Use sentence-transformers for RAG/search; use base LLM embeddings only if you're fine-tuning on a downstream task.

**Q: What is the curse of dimensionality and how does it affect embedding similarity search?**
A: In high dimensions, cosine similarity between random vectors concentrates near 0—everything looks equidistant. This means your top-k nearest neighbors may not be meaningfully more similar than random documents. Mitigate by reducing dimensions (PCA, quantization), using learned metrics, or ensuring your embedding space is well-calibrated through fine-tuning on your domain.

**Q: How do you evaluate whether embeddings are good for your use case?**
A: Run retrieval evaluation: take labeled query-document pairs, embed both, measure recall@k (what fraction of relevant docs appear in top-k results). MTEB (Massive Text Embedding Benchmark) scores are a good starting point. But always evaluate on your domain—a model top on MTEB may underperform a smaller domain-specific model on your data.

**Q: When would you fine-tune embedding models vs use them off-the-shelf?**
A: Fine-tune when off-the-shelf retrieval recall@10 is below ~70% on your eval set, when your domain has specialized vocabulary (legal, medical, code), or when you have labeled query-document pairs from user interactions. Off-the-shelf works well for general English text. Fine-tuning with contrastive loss (positive/negative pairs) typically gives 10-30% recall improvement in specialized domains.

**Q: What are the trade-offs between larger and smaller embedding models?**
A: Larger models (768+ dim, 300M+ params) have higher quality but slower inference and higher storage. Small models (384 dim, 22M params like all-MiniLM-L6-v2) are 5-10x faster with only ~5-10% quality drop for general tasks. In production, embed offline where possible; for real-time queries, use the smallest model that meets your recall threshold.

**Q: How does chunking strategy affect retrieval quality in RAG?**
A: Chunk size creates a precision-recall trade-off. Small chunks (128 tokens) give precise retrieval but lose context. Large chunks (512+ tokens) preserve context but dilute relevance scores. Overlapping chunks (50% overlap) help with boundary sentences. Best practice: chunk at natural boundaries (paragraphs, sections), use hierarchical retrieval (retrieve parent chunk after matching child chunk).

""",

    "03-pretraining.md": """## Interview Q&A

**Q: Why is pretraining data quality more important than quantity for LLMs?**
A: Models trained on high-quality curated data (like books, papers, filtered web) consistently outperform models trained on larger but noisier corpora. GPT-3 used 570GB filtered Common Crawl; Llama used 1.4T tokens but heavily filtered. Deduplication alone often improves downstream performance by 5-15% because the model doesn't waste capacity memorizing duplicated boilerplate.

**Q: What is the role of the data mixture during pretraining and how do you tune it?**
A: Data mixture ratios (e.g., 70% web, 15% code, 10% books, 5% papers) control what capabilities the model develops. More code data improves reasoning and structured output. More math improves arithmetic. Tune mixture by holding out domain benchmarks and ablating ratios on small models first, then scaling. Changing the mixture late in training (cooldown phase) can shift capabilities without full retraining.

**Q: What is the difference between next-token prediction and masked language modeling as pretraining objectives?**
A: Causal LM (GPT-style) predicts each next token given all prior tokens—naturally suited for generation. MLM (BERT-style) predicts masked tokens using bidirectional context—better for understanding/classification tasks. Causal LM has become dominant because generative models can be fine-tuned for classification too, but bidirectional models still outperform on tasks requiring full-context understanding like NER.

**Q: How does learning rate warmup affect pretraining stability?**
A: Without warmup, large learning rates at initialization cause exploding gradients because weights are random and gradients are large. Linear warmup over 1-4% of total steps (e.g., 2000 steps for 100K total) starts with near-zero LR and ramps up, letting the model stabilize before aggressive updates. Combined with cosine decay, this is the standard schedule for all large model training.

**Q: Why do LLMs exhibit emergent capabilities and what does this mean for scaling?**
A: Emergent capabilities (few-shot learning, chain-of-thought reasoning) appear suddenly at certain model sizes because they require multiple sub-capabilities to work together—none individually sufficient. This means you can't predict from small-scale experiments whether a capability will emerge at larger scale. Practically: don't conclude a model "can't do X" based on small model evaluations; test at the scale you actually deploy.

**Q: What is the chinchilla scaling law and how should it guide training decisions?**
A: Chinchilla showed optimal training requires roughly 20 tokens per parameter (a 7B model should see ~140B tokens). Previously, models were often undertrained (GPT-3: 300B tokens for 175B params = ~1.7 tokens/param). The law implies: for a fixed compute budget, train a smaller model for longer rather than a large model briefly. This insight led to Llama (7B trained on 1T+ tokens) outperforming larger but undertrained models.

""",

    "04-finetuning.md": """## Interview Q&A

**Q: When should you fine-tune vs use few-shot prompting?**
A: Fine-tune when: you have 500+ labeled examples, consistent format/style requirements, latency constraints (shorter prompts), or need behavior that's hard to specify in a prompt. Use few-shot prompting when: you have few examples, the task changes frequently, or you need interpretability into what examples drive performance. Fine-tuning wins on consistency and cost at scale; prompting wins on flexibility and speed of iteration.

**Q: What are the signs that your fine-tuned model is catastrophically forgetting?**
A: Performance on tasks outside your fine-tuning domain drops significantly. The model stops following safety guidelines. It loses capabilities that the base model had (e.g., multilingual ability, code generation). Mitigate with: lower learning rate (1e-5 vs 1e-4), smaller number of training steps, regularization (weight decay), or mixing in base model data during fine-tuning.

**Q: How do you construct a good fine-tuning dataset?**
A: Quality over quantity: 500 carefully curated examples often outperforms 5000 noisy ones. Ensure diversity across your input space—don't just collect easy cases. Format data exactly as you'll format inference prompts. Include negative examples or rejection samples if you want the model to decline certain requests. Validate with held-out eval set; track performance vs dataset size to know when you have enough.

**Q: What learning rate should you use for fine-tuning and how do you tune it?**
A: Start with 1e-5 to 5e-5 for full fine-tuning, 1e-4 for LoRA. Lower than pretraining because weights are already meaningful—large updates cause forgetting. Use linear warmup (5-10% of steps) then cosine decay. Run a learning rate range test: train for 100 steps each at [1e-6, 5e-6, 1e-5, 5e-5, 1e-4], pick the LR just before loss stops decreasing.

**Q: How do you evaluate whether fine-tuning improved the model?**
A: Always evaluate on a held-out test set that mirrors production distribution. Use task-specific metrics (ROUGE for summarization, exact match for QA, human eval for open-ended). Compare against: base model + same prompt, base model + few-shot, production baseline. Watch for mode collapse: fine-tuned model that always outputs one pattern may have high automatic metrics but fail in production.

**Q: When is full fine-tuning worth the cost vs PEFT methods like LoRA?**
A: Full fine-tuning is worth it when: the task requires changes to the model's core representations (not just surface behavior), you have large high-quality datasets (100K+ examples), or you'll serve many users sharing one fine-tuned model. PEFT/LoRA is sufficient for most production cases—it achieves 90-95% of full fine-tuning quality at 1-5% of the compute cost. Full fine-tuning rarely justifies cost unless you're training a foundation model.

""",

    "05-instruction-tuning.md": """## Interview Q&A

**Q: What makes instruction tuning data high quality vs low quality?**
A: High quality: diverse task coverage, clear instructions, correct and detailed responses, natural variation in phrasing. Low quality: narrow task distribution (all the same type), short one-word answers, instructions that don't match the response, or machine-generated responses from weak models. The LIMA paper showed 1000 high-quality examples outperforms 50K lower-quality ones—quality of demonstrations matters more than scale.

**Q: How does instruction tuning differ from RLHF?**
A: Instruction tuning is supervised fine-tuning on (instruction, response) pairs—it teaches the model to follow instructions but doesn't explicitly optimize for human preferences. RLHF adds a reward model trained on human preference comparisons and optimizes the LLM policy against it, teaching the model not just to respond but to respond in ways humans prefer. RLHF requires more infrastructure but produces better-calibrated, safer outputs.

**Q: Why do instruction-tuned models sometimes refuse requests they should fulfill?**
A: Instruction tuning with safety data teaches refusal patterns that can generalize too broadly. If refusal examples are over-represented or if the refusal classifier operates on surface features (certain keywords), the model learns to refuse similar-sounding but benign requests. Fix by adding targeted examples of similar requests with appropriate completions, and using held-out eval cases for borderline requests.

**Q: How do you maintain capabilities after instruction tuning?**
A: Instruction tuning on narrow task distributions can hurt performance on tasks not in the fine-tuning set. Mix in general instruction-following data during tuning. Use a lower learning rate. Include examples from the base model's training distribution. Evaluate on a broad benchmark suite (MMLU, HumanEval, etc.) before and after—regressions on these are a sign of capability degradation.

**Q: What is the role of the system prompt in instruction tuning?**
A: The system prompt sets persistent context and persona that the model follows throughout a conversation. Instruction-tuned models learn to prioritize system prompt instructions over user instructions, enabling enterprise customization without retraining. It's important to include diverse system prompts in training data so the model generalizes to novel ones rather than memorizing a handful of templates.

**Q: How would you create instruction tuning data for a domain-specific use case with no existing labeled data?**
A: Start with self-instruct: prompt GPT-4 to generate diverse instructions for your domain, filter for quality and novelty, generate responses. Use domain experts to write 50-100 seed examples that capture the desired style and depth, then expand with GPT-4. Validate a sample manually. For specialized domains (legal, medical), have subject-matter experts review generated responses before including them in training data.

""",

    "06-rlhf.md": """## Interview Q&A

**Q: Why does RLHF improve model behavior beyond what supervised fine-tuning achieves?**
A: SFT teaches the model to mimic demonstrations but can't distinguish "pretty good" from "excellent" responses—it treats all training examples equally. RLHF optimizes explicitly for human preferences by training a reward model on pairwise comparisons, then using RL to maximize that reward. This allows the model to explore response variations not in the training data and learn nuanced preferences like helpfulness vs verbosity trade-offs.

**Q: What is reward hacking and how do you mitigate it?**
A: Reward hacking occurs when the model finds responses that score high on the reward model but are not actually preferred by humans—e.g., very long hedged responses that seem thorough, or sycophantic agreement. Mitigate with: KL penalty against the SFT model (prevents too much deviation), diverse reward model ensemble, regular human eval of high-reward outputs, and updating the reward model periodically with new comparisons.

**Q: How many human preference comparisons are needed for RLHF?**
A: Depends on task complexity and desired quality. InstructGPT used ~13K comparisons for the reward model—a surprisingly small number. For domain-specific applications, 1000-5000 comparisons often suffice to learn the most important preferences. Quality matters: expert annotators with clear guidelines outperform crowd workers. Diminishing returns set in around 10-20K; invest in annotation guidelines rather than raw scale.

**Q: What are the failure modes of preference data collection?**
A: Annotator disagreement: different annotators have different preferences—high inter-annotator disagreement means noisy reward signal. Presentation bias: annotators prefer longer, more confident-sounding responses regardless of correctness. Position bias: first response shown tends to get favored. Selection bias: if you only compare model outputs, you miss the space of human-written ideal responses. Mitigate with clear guidelines, multiple annotators per pair, randomized presentation order.

**Q: How does PPO differ from other RL algorithms for RLHF?**
A: PPO (Proximal Policy Optimization) clips the policy update to prevent large steps that destabilize training—crucial because small language model deviations can cause catastrophic behavior. The clipping ratio (typically 0.2) limits how much the policy can change per update. Other RL algorithms like REINFORCE have higher variance and are less stable at LLM scale. PPO's compute cost is high (requires running inference for rollouts), which is why DPO emerged as a simpler alternative.

**Q: When would you choose DPO over PPO for preference optimization?**
A: DPO when: you have a fixed preference dataset and don't need online data collection, you want simpler implementation (no reward model, no RL loop), or your compute budget is limited. PPO when: you need online data collection (model generates rollouts, humans rate them), you want to optimize a non-differentiable reward (e.g., from a separate classifier), or you need fine-grained control over the optimization process. In practice, DPO has become the default for most fine-tuning workflows.

""",

    "07-dpo.md": """## Interview Q&A

**Q: Why does DPO not require a separate reward model while RLHF does?**
A: DPO mathematically shows that the optimal policy under RLHF can be expressed directly in terms of the ratio of chosen to rejected completion probabilities. This means the language model implicitly represents the reward through its log-probabilities. The DPO loss directly updates these log-probs using preference pairs, bypassing the need to train and maintain a separate reward model while achieving equivalent results.

**Q: What happens if the preference data has a high tie rate or low confidence?**
A: Low-confidence preference pairs (annotators frequently disagree or mark ties) inject noisy gradients into DPO training. The model may not converge to a clear preference direction, or it may overfit to noise rather than genuine preferences. Filter out low-confidence examples (keep only clear wins/losses), use annotator confidence scores to weight the loss, or collect higher-quality data with clearer preference signals.

**Q: How does the beta parameter in DPO affect model behavior?**
A: Beta controls the KL divergence penalty against the reference model. High beta (0.5+): model stays close to the SFT reference, less preference optimization. Low beta (0.01): aggressive preference learning, higher risk of reward hacking or capability degradation. Typical values: 0.1-0.2. Tune beta by monitoring: KL divergence from reference model, performance on held-out preference eval, and capability benchmarks to detect regression.

**Q: Can DPO be applied iteratively, and what are the risks?**
A: Yes—iterative DPO trains a model, generates new preference data using that model (online DPO), then fine-tunes again. This improves performance because preference data from the current model is more informative. Risk: distribution shift—if the model drifts, the reference model becomes a poor anchor and KL regularization loses meaning. Mitigate by resetting the reference model each iteration or using a conservative beta.

**Q: How do you construct preference pairs for DPO when you don't have human annotators?**
A: Use AI feedback (RLAIF): prompt GPT-4 to generate both a "better" and "worse" response, or to rank model-generated outputs. Use Constitutional AI principles: have a model critique and revise its own outputs, forming (original, revised) pairs. Use best-of-N sampling: generate N completions, pick the best and worst by some metric (ROUGE, reward model score) as the preference pair. Quality of the proxy judge directly limits DPO quality.

**Q: What evaluation metrics tell you if DPO actually improved the model?**
A: Win rate against the SFT reference model on a held-out preference test set (should be >50%). Reward model score on the test set (should increase). MT-Bench or AlpacaEval for general instruction following. Crucially, also check capability regressions: MMLU, HumanEval, TruthfulQA—DPO can degrade factual accuracy if preference data favors confident-sounding but wrong answers.

""",

    "08-lora.md": """## Interview Q&A

**Q: Why does LoRA work — what is the theoretical justification for low-rank updates?**
A: The hypothesis is that the change in weights during fine-tuning (ΔW) has low intrinsic dimensionality—the task-specific adaptation can be expressed in a low-dimensional subspace. Empirically, this holds for most downstream tasks: fine-tuning a 768×768 attention weight matrix requires rank 4-16 (vs 768) to achieve near-full fine-tuning performance. The low-rank constraint also acts as implicit regularization.

**Q: How do you choose the LoRA rank and what are the consequences of choosing wrong?**
A: Start with rank 4-8 for simple tasks (classification, formatting), 16-32 for complex tasks (instruction following, reasoning). Too low: underfitting—the adapter can't capture the needed task variation. Too high: overfitting to small datasets, more parameters, slower training. Monitor: train/val loss gap and downstream task accuracy. If train loss falls but val accuracy plateaus, you may be overfitting—reduce rank or add dropout.

**Q: What is the difference between LoRA and QLoRA?**
A: LoRA keeps the base model in full precision (float32 or float16) and adds low-rank adapters in the same precision. QLoRA quantizes the base model to 4-bit (NF4) and uses paged optimizers to reduce memory, while keeping the LoRA adapters in 16-bit for gradient stability. QLoRA enables fine-tuning 70B+ models on a single GPU at the cost of ~20% slower training. Use QLoRA when GPU memory is the bottleneck.

**Q: After LoRA fine-tuning, should you merge the adapters before deployment?**
A: Yes, almost always. Merging computes W_merged = W_base + A×B and saves a single model file. Benefits: no adapter loading overhead at inference, no separate adapter management, simpler deployment. Keep un-merged weights only if you need to serve multiple task-specific adapters on the same base model (adapter switching), which requires the PEFT library and adds ~2-3ms latency per request.

**Q: How does LoRA compare to full fine-tuning for very small datasets?**
A: For very small datasets (<100 examples), LoRA's implicit regularization (low-rank constraint) can outperform full fine-tuning because it prevents overfitting. Full fine-tuning with 7B parameters and 50 examples will overfit severely. LoRA with rank 4 has only ~1M trainable parameters, making it much less prone to memorizing the training set. Add LoRA dropout (0.1) for extra regularization.

**Q: Can you apply LoRA to non-attention layers and should you?**
A: By default, LoRA is applied to query and value projection matrices in attention. You can apply it to all linear layers (including FFN layers) by setting target_modules to all linear layer names. Applying to more layers uses more parameters but can improve quality for tasks requiring deeper representation changes. For most fine-tuning tasks, attention-only LoRA is sufficient. Apply to FFN layers for tasks requiring significant knowledge injection.

""",

    "09-adapters.md": """## Interview Q&A

**Q: How do adapter layers differ architecturally from LoRA, and when does that matter?**
A: Adapters insert new layers (down-proj → activation → up-proj) sequentially into the model's forward pass, adding computational overhead at inference. LoRA adds parallel low-rank matrices that are merged into existing weights before inference, eliminating overhead. For latency-sensitive applications, prefer LoRA (zero inference overhead after merging). Prefer adapters for multi-task serving where you need to switch tasks without reloading the model.

**Q: What bottleneck dimension should you use for adapter layers?**
A: Typical bottleneck dimensions: 16-128 for encoder-only models (BERT), 32-256 for decoder models. The bottleneck dimension controls the number of trainable parameters: roughly 2 × hidden_dim × bottleneck_dim per adapter. Start at 64 and reduce if you see overfitting. The original Adapter paper used 64 for BERT-base (768 hidden dim), yielding ~98K parameters per task.

**Q: How do you serve multiple tasks with adapter-based fine-tuning?**
A: Keep the base model in shared GPU memory, load only the task-specific adapter weights per request (a few MB each). At inference, route requests to the appropriate adapter based on task classification. PEFT's `set_adapter` method allows hot-swapping adapters in ~1ms. This pattern enables serving 100+ task-specific models on the same GPU that would otherwise require 100× the memory for separate full fine-tuned models.

**Q: Why might adapters underperform LoRA for the same parameter budget?**
A: Adapters modify the residual stream sequentially—their effect must pass through subsequent layers before contributing to the final output. LoRA modifies weights directly in the attention mechanism, allowing more direct gradient flow. Additionally, adapters add depth (sequential non-linearity) which can be harder to optimize. In practice the gap is small (1-3% on most benchmarks), but LoRA is now the dominant choice due to zero inference overhead.

**Q: What happens to the base model's representations when adapters are trained?**
A: Base model weights remain frozen—only adapter parameters update. The base model's representations are preserved, enabling catastrophic forgetting resistance. Adapters learn to reroute activations to achieve task-specific behavior through the bottleneck transformations. This means adapters work best when the base model's representations are already relevant to the task; for very out-of-domain tasks, adapter capacity may be insufficient.

**Q: How do you evaluate whether adapters have learned the target task vs. just memorized training data?**
A: Measure: train vs. test performance gap (large gap = overfitting). Test on inputs with slight paraphrasing of training examples. Check performance on a held-out subset of the task distribution. If the model performs well only on near-duplicates of training examples, increase adapter dropout, reduce bottleneck dimension, or collect more diverse training data.

""",

    "10-prefix-tuning.md": """## Interview Q&A

**Q: What is the fundamental difference between prefix tuning and prompt tuning?**
A: Prefix tuning prepends trainable continuous vectors to every attention layer's key/value tensors throughout the entire model. Prompt tuning only prepends trainable tokens to the input embedding layer. Prefix tuning has more parameters and affects every layer—better for complex tasks. Prompt tuning is more parameter-efficient (10-100x fewer params) but works best for large models (>10B) where it can achieve comparable performance to full fine-tuning.

**Q: Why does prefix tuning use a reparameterization trick during training?**
A: Directly optimizing prefix parameters is unstable—the loss landscape is non-smooth and gradients are noisy. The reparameterization uses an MLP to generate prefix parameters from a smaller latent space, then discards the MLP at inference. This provides a smooth optimization surface. Without it, prefix tuning often diverges or converges to poor local minima.

**Q: When is prefix tuning preferable to LoRA?**
A: Prefix tuning is preferable for: tasks that require modifying the model's attention pattern globally (not just weight matrices), multi-task serving where you need minimal memory overhead per task (prefix vectors are tiny), and generation tasks where steering at every layer helps (creative writing, style transfer). LoRA is generally preferred for tasks closer to fine-tuning—prefix tuning has a lower performance ceiling for complex tasks.

**Q: How does the prefix length affect performance and what are the trade-offs?**
A: Longer prefixes (100-200 tokens): more expressive, can encode complex task instructions, but take up context window space and add computation. Shorter prefixes (10-20 tokens): efficient but may lack expressivity for complex tasks. For classification tasks, 10-20 prefix tokens suffice. For generation tasks requiring style or format control, 50-100 is typical. The prefix length adds to every input—factor this into context window calculations.

**Q: Why does prefix tuning work better with frozen vs. fine-tuned base models?**
A: When the base model is frozen, the prefix learns to steer fixed representations that remain consistent across training. When the base model is also fine-tuned, the prefix and base model parameters co-adapt in complex ways that can cause overfitting or instability. The efficiency advantage of prefix tuning (few parameters) disappears if you also fine-tune the base model.

**Q: How would you debug a prefix-tuned model that generates repetitive or low-quality output?**
A: First check: is the prefix length appropriate? Too short prefixes may not have enough capacity. Second: is the reparameterization MLP too small? Increase hidden size. Third: check learning rate—prefix tuning is sensitive; try 1e-4 to 1e-3 range. Fourth: inspect the prefix vectors with PCA to check if they've collapsed to a low-dimensional manifold (sign of poor optimization). Finally, ensure the training data has enough diversity to prevent mode collapse.

""",

    "11-parameter-efficient-finetuning.md": """## Interview Q&A

**Q: How do you choose between LoRA, adapters, and prefix tuning for a new task?**
A: LoRA: default choice for most tasks—zero inference overhead after merging, works well at ranks 4-32, broad community support. Adapters: when you need to serve many tasks on one base model without merging (hot-swappable). Prefix tuning: when you want to minimize trainable parameters and the task is generation-focused. In practice, run a quick experiment with LoRA first; switch to alternatives only if you have specific constraints (multi-task serving → adapters, extreme parameter budget → prompt tuning).

**Q: What is the relationship between PEFT methods and the size of the base model?**
A: Larger base models benefit more from PEFT—a 70B model already has enormous representational capacity, so small adapters can steer it effectively. Smaller models (1-3B) may need higher LoRA rank or full fine-tuning to achieve target quality. Prompt tuning shows this clearly: it matches full fine-tuning for models >10B but underperforms significantly for smaller models. Choose PEFT method based on both task requirements and model scale.

**Q: How do you stack multiple PEFT adapters for multi-task scenarios?**
A: Methods include: (1) task routing—train separate adapters, select at inference based on input classification; (2) adapter fusion—train task-specific adapters, then train a small fusion network that combines them; (3) LoRA composition—add LoRA adapters sequentially using different target modules. Adapter fusion can outperform individual adapters by combining knowledge, but adds inference overhead. Task routing is simpler and more interpretable.

**Q: What are the memory requirements during PEFT training vs. inference?**
A: Training: base model weights (frozen, in float16/int8/int4) + adapter weights + activations + optimizer states for adapter parameters only. QLoRA (int4 base) reduces base model memory by 4-8x vs float16. Inference: base model + adapter weights (merged or separate). A 7B model in float16 requires ~14GB; with QLoRA int4, ~4-5GB. Adapter weights add negligible memory (MB scale).

**Q: Can PEFT methods match full fine-tuning quality and when do they fall short?**
A: For most standard NLP tasks (classification, QA, summarization), LoRA at rank 16-32 achieves 95-99% of full fine-tuning quality. PEFT falls short when: the task requires large distribution shift from the pretraining domain (specialized medical/legal reasoning), training dataset is very large (>1M examples where the full model can benefit from all the signal), or the task requires updating core factual knowledge (PEFT struggles to inject new facts efficiently).

**Q: How do you set up PEFT training with Hugging Face and what are the key configuration choices?**
A: Use `peft.LoraConfig` (rank r, target_modules, lora_alpha, lora_dropout), wrap with `get_peft_model`. Key choices: target_modules (q_proj+v_proj is default; add k_proj and o_proj for more capacity); lora_alpha (typically 2×rank for stable training); use `prepare_model_for_kbit_training` before applying PEFT for quantized base models. Always call `model.print_trainable_parameters()` to verify you're training the right number of parameters.

""",

    "12-prompting.md": """## Interview Q&A

**Q: How do you systematically improve a prompt that is producing inconsistent results?**
A: First, collect 20-30 failure examples and categorize them (wrong format, wrong content, missing steps). For each failure category, add a few-shot example that demonstrates the correct behavior, or add an explicit instruction addressing that failure mode. Test on a held-out eval set after each change—adding instructions can fix one problem while breaking another. Use a version-controlled prompt registry to track changes.

**Q: What is the difference between zero-shot, few-shot, and chain-of-thought prompting and when do you use each?**
A: Zero-shot: just instructions, no examples—use when you have no examples or the task is simple. Few-shot: add 3-5 input/output examples before the query—use for tasks with specific formats, edge cases, or non-obvious behavior. Chain-of-thought: add "think step by step" or example reasoning chains—use for multi-step reasoning, math, or logic tasks. COT adds tokens but dramatically improves accuracy on reasoning tasks (often 20-40% on benchmarks).

**Q: When does a longer, more detailed prompt hurt rather than help?**
A: Longer prompts can hurt when: they exceed the model's effective attention span (important instructions in the middle of long prompts are often ignored—the "lost in the middle" effect), they introduce contradictory instructions, or they narrow the model's response too much when you want creative variation. Keep critical instructions near the beginning or end of the prompt. Test systematically—sometimes a single precise sentence beats a paragraph of explanation.

**Q: How do you prevent prompt injection attacks in production systems?**
A: Use a system prompt that explicitly instructs the model to ignore attempts to override its instructions. Sanitize user inputs by filtering known injection patterns. Use a separate classifier to detect prompt injection attempts before passing to the LLM. Keep system instructions in a separate message rather than concatenated with user input. For high-security applications, use guardrail models to validate outputs regardless of what the LLM was prompted.

**Q: What is temperature and how does it interact with prompting strategies?**
A: Temperature scales the logit distribution before sampling: low temperature (0.1-0.3) concentrates probability on high-likelihood tokens, producing consistent, predictable outputs. High temperature (0.8-1.0) flattens the distribution, producing more varied/creative outputs. For factual Q&A, use low temperature. For creative tasks, use 0.7-0.9. For chain-of-thought reasoning where you want deterministic steps, use temperature=0. Temperature interacts with few-shot examples: low-temp + good examples = very consistent output.

**Q: How do you structure a prompt for a multi-step workflow where the model must maintain state?**
A: Use structured XML or markdown tags to separate components (instructions, context, current state, task). Pass relevant state explicitly in the prompt—don't rely on the model to infer it. For long workflows, summarize completed steps rather than passing full history. Use explicit step markers ("Step 3 of 5:") to help the model track progress. Test with both short and long accumulated contexts to check for performance degradation.

""",

    "13-few-shot-learning.md": """## Interview Q&A

**Q: How many few-shot examples are optimal and how do you decide?**
A: More examples generally improve performance up to a point (diminishing returns after 8-16 examples for most tasks). But each example consumes context window tokens and adds latency. Test with 1, 3, 5, 8 examples on your eval set. For LLMs >10B, 3-5 examples typically captures most of the benefit. For complex tasks requiring diverse coverage, 8-16 may be better. When context is limited (e.g., long user inputs), prioritize quality over quantity—2 excellent examples > 8 mediocre ones.

**Q: How do you select which examples to include in few-shot prompts?**
A: Strategies in increasing sophistication: (1) random selection from a pool of curated examples; (2) diverse selection—cover different sub-types of the task; (3) similarity-based—retrieve examples most similar to the query using embedding search (dynamic few-shot); (4) hard example selection—prefer examples similar to cases the model struggles on. Dynamic few-shot (embedding retrieval) typically improves accuracy 5-15% over static examples for tasks with diverse inputs.

**Q: What is the difference between few-shot prompting and in-context learning?**
A: Few-shot prompting is a specific technique: provide examples in the prompt. In-context learning is the broader phenomenon: LLMs can learn new patterns from information in the context window without gradient updates. ICL includes few-shot examples but also single demonstrations, analogy-based reasoning, and task descriptions. The mechanism is debated—models may do implicit gradient-based learning in the forward pass, or they may use examples to identify the task distribution.

**Q: Why do example ordering and formatting matter in few-shot prompts?**
A: Recency bias: the model pays more attention to the last few examples before the query. Distributional skew: if early examples are all one class, the model may bias toward it. Consistent formatting across examples helps the model identify the pattern. Always ensure the final few examples before the query reflect the diversity you want in outputs. Randomizing example order and testing on multiple orderings gives a more robust evaluation.

**Q: When does few-shot prompting fail and what should you do instead?**
A: Fails when: the task requires knowledge the model doesn't have (no amount of examples helps); the format is so unusual that examples confuse rather than guide; or examples cover only a narrow slice of the actual input distribution. In these cases: fine-tune on domain data, use retrieval-augmented generation to provide factual context, or decompose the task into simpler subtasks that the model can handle individually.

**Q: How do you measure if your few-shot examples are actually helping?**
A: Compare: zero-shot performance vs. your few-shot configuration on a held-out eval set. If few-shot doesn't beat zero-shot by >5%, your examples may not be well-chosen. Run ablations: remove one example at a time and see which ones contribute most. Use leave-one-out cross-validation across your example pool. Track performance vs. number of examples to find the sweet spot for your context budget.

""",

    "14-zero-shot-learning.md": """## Interview Q&A

**Q: Why do larger models perform dramatically better at zero-shot tasks?**
A: Larger models trained on more data develop better world models and more generalizable representations. At sufficient scale, models can understand task descriptions and apply their general knowledge without examples. The key capability is understanding the intent of natural language instructions—this emerges with scale and instruction diversity during pretraining. GPT-3 175B showed zero-shot capabilities that didn't exist in 13B, suggesting threshold effects.

**Q: What types of tasks are well-suited for zero-shot prompting vs. requiring few-shot?**
A: Zero-shot works well for: tasks with clear natural language descriptions (translation, summarization, sentiment), tasks the model has seen many variations of during pretraining, and classification with label names that are semantically meaningful. Few-shot is needed for: tasks with unusual output formats, domain-specific terminology, nuanced classification schemes (distinguishing 5 similar sentiment levels), or tasks requiring specific reasoning patterns.

**Q: How does instruction tuning relate to zero-shot capability?**
A: Instruction tuning dramatically improves zero-shot performance by teaching the model to follow natural language task descriptions. An instruction-tuned model like InstructGPT performs far better zero-shot than a base GPT-3 model of the same size—the alignment between instruction format and model behavior is crucial. Instruction tuning is essentially "teaching the model to take instructions," which is the core requirement for zero-shot generalization.

**Q: What is the role of label verbalization in zero-shot classification?**
A: Label verbalization converts class labels into natural language descriptions that the model can reason about using its pretraining knowledge. Instead of predicting class 0 or 1, the model predicts "negative" or "positive." Good verbalizers align with how those concepts appear in the model's training data. Poor verbalizers (arbitrary codes, jargon) degrade performance significantly. For new tasks, test multiple verbalizations—they can cause 10-20% accuracy differences.

**Q: How do you evaluate zero-shot performance reliably?**
A: Use benchmark datasets with diverse task types (MMLU, BIG-Bench, SuperGLUE) to measure general zero-shot capability. For task-specific evaluation, use a held-out test set with no examples used during development. Critically: ensure your test prompts weren't present in the model's training data (data contamination check). Report variance across multiple prompt phrasings—zero-shot performance can vary 10-20% based on prompt wording alone.

**Q: When should you add a chain-of-thought instruction to zero-shot prompts?**
A: Add CoT ("let's think step by step" or similar) when: the task requires multi-step reasoning (math, logic, planning), when the model makes confident but wrong direct answers, or when you need interpretable reasoning to debug failures. CoT typically adds 3-5x more output tokens but improves accuracy 20-40% on reasoning tasks. For simple classification or extraction tasks where the answer is direct, CoT adds overhead without benefit.

""",

    "15-in-context-learning.md": """## Interview Q&A

**Q: What is the mechanistic explanation for why in-context learning works?**
A: The leading hypothesis is that transformer attention layers implement a form of implicit gradient descent in the forward pass—each example updates an implicit "task vector" in the residual stream. This explains why more examples help (more gradient steps) and why ordering matters (later examples have more influence). Alternatively, the model may be doing Bayesian inference, using examples to identify the most likely task from its prior. Both theories have empirical support; the true mechanism is likely a combination.

**Q: How does ICL differ from traditional machine learning?**
A: Traditional ML: update model parameters via gradient descent on training data—expensive but creates a persistent, reusable model. ICL: provide context examples at inference time, no parameter updates—fast to set up but examples are consumed at every inference call (memory and compute cost). ICL "learning" is temporary—reset with each new context. ICL doesn't require labeled data for training, but quality still depends on example quality and quantity.

**Q: What are the limitations of ICL for production applications?**
A: Context window cost: each example adds tokens, increasing latency and cost. Inconsistency: performance depends on example selection, ordering, and phrasing—hard to guarantee consistent behavior. Capacity limits: ICL can't inject new factual knowledge or permanently change model behavior. Context window limits the number of examples, while fine-tuning can learn from 100K+ examples. For mission-critical, consistent behavior, fine-tuning outperforms ICL.

**Q: Why does ICL performance sometimes decrease with more examples?**
A: Too many examples can: exceed the model's effective context attention window (important early examples get ignored), introduce distributional noise if examples don't represent the test query well, cause the model to pattern-match on surface features of examples rather than the underlying task, or shift the prompt distribution away from what was seen during instruction tuning. Test performance vs. example count systematically and look for the knee in the curve.

**Q: How do you make ICL more robust to prompt sensitivity?**
A: Use ensemble prompting: run the same query with 5-10 different example orderings and take majority vote. Use calibration: adjust model predictions based on baseline probabilities of output tokens with empty input. Average over multiple prompt phrasings. Use self-consistency: sample multiple reasoning chains and take the most common answer. These techniques add inference cost but reduce variance from 15-20% to 3-5%.

**Q: When is retrieval-augmented ICL (using embedding search to select examples) worth implementing?**
A: Worth it when: you have a large pool of examples (100+) and the task has diverse subtypes, baseline static few-shot performance has plateaued, you observe that model failures correlate with queries being distant from static examples, or your application has variable input types (some queries need code examples, others need text examples). Typical gain: 5-15% accuracy improvement at the cost of an embedding lookup per query.

""",

    "16-chain-of-thought.md": """## Interview Q&A

**Q: When does chain-of-thought prompting fail to improve performance?**
A: CoT fails for: tasks where the reasoning process is trivial (simple classification, information extraction where the answer is directly in the context), tasks requiring factual recall where reasoning doesn't add information, very small models (<7B) that don't have the reasoning capacity to generate correct chains, and tasks where intermediate steps are error-prone (if any step is wrong, the final answer is usually wrong). Test empirically—CoT adds tokens and sometimes hurts.

**Q: What is the difference between few-shot CoT and zero-shot CoT?**
A: Few-shot CoT: provide example (question → step-by-step reasoning → answer) triples—the model learns the reasoning format and style from examples. Zero-shot CoT: just add "Let's think step by step" to the prompt—the model generates its own reasoning chain. Few-shot CoT is more reliable and controllable (you can demonstrate the desired reasoning depth and style). Zero-shot CoT is simpler and works surprisingly well for large models—useful when you don't have reasoning examples.

**Q: How does self-consistency improve over standard CoT?**
A: Self-consistency generates multiple reasoning chains (10-40) with high temperature sampling, then takes majority vote on the final answers. It improves accuracy by 5-15% over greedy CoT because: different reasoning paths may lead to the same correct answer, majority vote cancels out chains with reasoning errors, and diverse paths explore more of the solution space. Cost: 10-40x more inference calls. Use when accuracy is critical and inference cost is secondary.

**Q: What makes a good chain-of-thought example vs a bad one?**
A: Good: shows the reasoning steps that are actually necessary (not just verbose repetition of the question), arrives at the correct answer, uses concise language (long chains introduce more error opportunities), and demonstrates how to handle the specific difficulty of your task type. Bad: shows unnecessary steps that add tokens without helping, uses idiosyncratic notation the model may not generalize, or demonstrates reasoning that's overly specific to one input type.

**Q: How do you evaluate whether CoT actually improves reasoning vs. just adding verbosity?**
A: Compare: CoT vs. direct answer on a task requiring multi-step reasoning (math, logic). Check: do models produce correct intermediate steps (not just correct final answers)? Analyze failures: when CoT gives wrong answer, is the reasoning chain plausible but incorrect, or is it clearly wrong? Run ablations with shorter, longer, and different reasoning chain lengths. If CoT improves final answer accuracy without improving intermediate step accuracy, the "reasoning" may be post-hoc rationalization.

**Q: What is tree-of-thought prompting and when is it worth the extra complexity?**
A: Tree-of-thought (ToT) extends CoT by exploring multiple reasoning branches at each step, using the model to evaluate branch quality and prune poor paths—essentially beam search over reasoning trees. Worth it for: complex planning problems with many decision points (game playing, multi-step coding), tasks where there are multiple plausible approaches to compare, or when single-chain CoT frequently fails. Not worth it for: tasks solvable with 1-3 reasoning steps, or when latency matters—ToT requires many more model calls.

""",

    "17-prompt-optimization.md": """## Interview Q&A

**Q: What is the difference between manual prompt engineering and automatic prompt optimization?**
A: Manual engineering: a human iteratively edits the prompt based on failure analysis—fast to set up, interpretable, but labor-intensive and doesn't scale. Automatic optimization: algorithms (gradient-based like AutoPrompt, reinforcement learning like PromptAgent, or LLM-based meta-prompting like APE) search the prompt space programmatically. Auto-optimization finds non-intuitive prompts that outperform manual ones by 5-20% but requires an eval dataset and compute for optimization. Use auto-optimization when the eval set is large enough and you've exhausted manual improvements.

**Q: How does the DSPy framework approach prompt optimization?**
A: DSPy replaces manual prompt strings with declarative program specifications. You write a program using typed signatures and modules; DSPy automatically generates and optimizes prompts through its teleprompter compilers, which use your few-shot examples or labeled data to find effective prompt patterns. The advantage: when you change your program logic or add modules, DSPy reoptimizes prompts automatically rather than requiring manual rewriting.

**Q: What is the role of the evaluation metric in prompt optimization?**
A: The eval metric is the optimization target—optimizing against the wrong metric finds prompts that game the metric rather than solving the task. Pitfall: optimizing for ROUGE scores finds prompts that produce verbose output with high lexical overlap but poor quality. Use task-specific metrics aligned with user goals (human evaluation, execution accuracy for code, API call success rate). If you must use proxy metrics, validate that improvements correlate with real-world quality.

**Q: When does prompt optimization overfit to the dev set?**
A: Overfitting occurs when the optimized prompt exploits specific patterns in the dev examples that don't generalize. Signs: large gap between dev set performance and fresh test set performance, the optimized prompt includes very specific instructions that only apply to dev examples, or performance drops when input style changes slightly. Mitigate: use a large diverse dev set (100+), hold out a separate test set for final evaluation, and prefer prompt structures that are semantically interpretable over those that appear arbitrary.

**Q: How do you systematically A/B test prompt variations in production?**
A: Maintain a prompt registry with version control. Deploy experiments using your traffic-splitting infrastructure (same as model A/B testing). Define primary and secondary metrics before launching. Run until statistical significance (power analysis to determine sample size). Log prompt version alongside all model inputs/outputs for retrospective analysis. Treat prompt changes with the same rigor as code changes—they affect model behavior and require validation.

**Q: What are the limits of prompt optimization and when should you fine-tune instead?**
A: Prompt optimization can't: add knowledge the model doesn't have, change fundamental capabilities (e.g., can't make a 7B model reason like a 70B model), or solve tasks that require more context than fits in the window. Switch to fine-tuning when: the best optimized prompt still underperforms your target, the task requires consistent behavior across thousands of calls (prompts have higher variance than fine-tuned models), or the same task will run millions of times (fine-tuning reduces token cost by shortening prompts).

""",

    "18-rag.md": """## Interview Q&A

**Q: When should you use RAG vs. fine-tuning for injecting knowledge into an LLM?**
A: RAG when: knowledge changes frequently (news, company data), you need source attribution, the knowledge base is large (millions of documents), or you need to control which knowledge the model uses. Fine-tuning when: knowledge is static and stable, you need consistent behavior across many calls, knowledge represents reasoning patterns or skills (not just facts), or you need lower inference latency (no retrieval step). Hybrid approaches work well: fine-tune for behavior/style, use RAG for current factual knowledge.

**Q: What are the most common failure modes of RAG systems and how do you diagnose them?**
A: (1) Retrieval failure: correct documents not in top-k—measure recall@k on a labeled query-document test set. (2) Context utilization failure: retrieved correct docs but model ignores or misinterprets them—measure answer accuracy given oracle retrieved docs. (3) Hallucination: model generates plausible but unsupported content when retrieved docs don't contain the answer—monitor with a "supported by context" classifier. Diagnose by isolating retrieval from generation in your eval pipeline.

**Q: How does chunk size affect RAG performance?**
A: Small chunks (128 tokens): high precision (retrieved chunk is relevant), low context (may miss surrounding information needed to answer). Large chunks (512+ tokens): more context per chunk, lower precision (relevant info buried in noise). Optimal chunk size depends on query type: fact lookup → small chunks, multi-hop reasoning → larger chunks. Use a two-stage approach: retrieve small chunks for precision, expand to parent chunks for context (parent document retrieval).

**Q: What is the difference between sparse retrieval (BM25) and dense retrieval (embeddings)?**
A: BM25 is a keyword-based tf-idf variant—fast, no GPU needed, works well for exact term matching. Dense retrieval embeds queries and documents into a vector space and uses approximate nearest neighbor search—slower but handles semantic similarity (paraphrases, synonyms). Dense retrieval excels when queries use different vocabulary than documents. Hybrid search (RRF combination of BM25 + dense) typically outperforms either alone by 5-10% recall.

**Q: How do you handle queries that require information from multiple retrieved chunks?**
A: Retrieve more chunks (top-10 vs top-5). Use re-ranking to select the best combination. Add a synthesis step: first retrieve, then prompt the model to extract key facts from each chunk, then synthesize. For complex multi-hop queries, use iterative retrieval: answer intermediate questions first, use those answers to retrieve additional context. Graph-based retrieval (knowledge graphs) handles multi-hop connections more efficiently than flat chunk retrieval.

**Q: What metadata should you store alongside document embeddings in a vector database?**
A: At minimum: source URL or file path, document title, creation/update timestamp, chunk position (for context expansion), and document type (used for filtering). For production: access control labels (user permissions), language, confidence/quality score of source, section headers (for context). Metadata enables filtered retrieval ("only retrieve from documents updated in last 30 days") which can significantly improve precision for time-sensitive queries.

""",

    "19-retrieval-augmented-generation.md": """## Interview Q&A

**Q: How do you measure RAG pipeline quality end-to-end?**
A: Use RAGAS metrics: faithfulness (are claims in the answer supported by retrieved context?), answer relevance (does the answer address the question?), context precision (are retrieved chunks relevant?), context recall (are all needed chunks retrieved?). Also measure end-to-end exact match or F1 on a test set with labeled answers. Separately measure retrieval quality (recall@k) to isolate retrieval vs. generation failures.

**Q: What is the "lost in the middle" problem and how does it affect RAG?**
A: Models pay less attention to information in the middle of long contexts—performance is highest when relevant information is at the beginning or end. For RAG with 10+ retrieved chunks, the most relevant chunk should be placed first (or last) rather than in the middle. Reranking helps by promoting the most relevant chunks. Alternatively, limit to 3-5 chunks and accept lower recall but better utilization.

**Q: How do you handle questions that the retrieved documents don't answer?**
A: The model should acknowledge when it cannot find the answer in the provided context, rather than hallucinating. Enforce this with explicit prompt instructions: "Only answer using the provided context. If the context doesn't contain the answer, say 'I don't have enough information.'" Additionally, train or prompt a confidence classifier to detect when retrieved context is insufficient. Never let the model silently mix retrieved context with parametric knowledge.

**Q: What is HyDE (Hypothetical Document Embeddings) and when does it help?**
A: HyDE generates a hypothetical document that would answer the query, then uses that document's embedding for retrieval instead of the query embedding. This helps when query and document are in very different linguistic styles (short sparse query vs. dense paragraph). HyDE improves recall for complex, multi-faceted queries by generating a "target document" that better matches the embedding space of real documents. Adds one LLM call overhead per query.

**Q: How do you keep the vector database synchronized with a frequently-updated document corpus?**
A: Use an incremental update strategy: hash document content, compare with stored hashes, only re-embed changed documents. Use streaming pipelines (Kafka, Kinesis) to process document changes in near-real-time. Set document TTLs for time-sensitive content. For large corpora with frequent updates, partition by recency—maintain a "recent" index updated frequently and a "stable" index updated less often, search both and merge results.

**Q: What are the latency components of a RAG pipeline and how do you optimize each?**
A: (1) Query embedding: 5-20ms—use a small embedding model (all-MiniLM-L6) or cache query embeddings. (2) Vector search: 5-50ms depending on index size—use HNSW index, quantize vectors, reduce search dimensions. (3) LLM generation: 100-2000ms—use streaming to reduce perceived latency, cache common query responses, use smaller models for simple queries. Total target: <500ms for most RAG applications. Measure each component separately to identify bottlenecks.

""",

    "20-vector-databases.md": """## Interview Q&A

**Q: What is the difference between HNSW and IVF indexing and when do you use each?**
A: HNSW (Hierarchical Navigable Small Worlds): builds a multi-layer graph, extremely fast queries (1-5ms), high recall (0.99+), but requires all vectors in memory and slow to update. IVF (Inverted File): clusters vectors, can be stored on disk with compression (IVF+PQ), slower queries (10-50ms) but handles billion-scale corpora and supports efficient updates. Use HNSW for production RAG with up to 10M vectors; use IVF+FAISS for large-scale or memory-constrained scenarios.

**Q: How does vector quantization affect retrieval quality and what trade-offs does it make?**
A: Product quantization (PQ) compresses vectors from float32 to 1-2 bytes per dimension by approximating them as products of subvector codebook entries. This reduces memory by 8-16x at the cost of 2-5% recall loss. Scalar quantization (SQ) is simpler with less compression. For most RAG applications, PQ-compressed indexes achieve sufficient recall (0.95+) while enabling much larger corpora in memory. Always measure recall@k on your dataset after quantization.

**Q: How do you handle multi-modal embeddings (text and images) in a single vector database?**
A: Option 1: Separate namespaces/collections—text embeddings in one collection, image embeddings in another, query both and merge results with reciprocal rank fusion. Option 2: Joint embedding space—use a model like CLIP that embeds both text and images into the same space, enabling cross-modal retrieval directly. Option 3: Late fusion—retrieve top-k from each modality separately, then re-rank with a cross-modal model. Joint embedding space is cleanest but requires compatible embedding models.

**Q: What metadata filtering capabilities do you need from a vector database and how do they affect performance?**
A: Common filters: by date range, document type, access level, category. Filtering implementation: pre-filter (filter before vector search, reduces recall if many vectors excluded), post-filter (vector search then filter, wasted compute), or combined index (partition vectors by filter value, search relevant partitions). Pre-filter + HNSW can degrade recall significantly if the filtered subset is small. Weaviate, Qdrant, and Pinecone implement efficient filtered HNSW that maintains recall.

**Q: How do you migrate a production vector database to a new embedding model?**
A: Zero-downtime migration: (1) deploy new embedding model alongside old; (2) start dual-writing new documents with both embeddings; (3) backfill old documents with new embeddings in background; (4) once backfill is complete, switch query routing to new index; (5) decommission old index. Validate retrieval quality before and after with a held-out query set. The backfill can take hours to days for large corpora—plan accordingly.

**Q: When should you use a dedicated vector database vs. adding vector search to PostgreSQL (pgvector)?**
A: pgvector for: <1M vectors, team already uses PostgreSQL, need transactional consistency between vectors and metadata, CRUD operations are frequent. Dedicated vector DB (Pinecone, Weaviate, Qdrant) for: >1M vectors, need multi-modal support, require advanced filtering, or need managed scaling and replication without PostgreSQL expertise. pgvector HNSW is competitive in performance up to ~10M vectors; beyond that, dedicated solutions handle sharding and replication better.

""",

    "21-semantic-search.md": """## Interview Q&A

**Q: When does semantic search underperform keyword search (BM25) and why?**
A: Semantic search underperforms for: exact match queries (product IDs, error codes, names), highly technical queries where domain vocabulary is precise, short queries with little semantic context, and queries where the user's phrasing exactly matches document phrasing. BM25's term frequency weighting is very effective for exact retrieval. Hybrid search (BM25 + semantic) nearly always outperforms either alone—don't abandon keyword search entirely.

**Q: How do you handle the vocabulary mismatch between queries and documents in semantic search?**
A: The whole point of semantic search is to handle vocabulary mismatch via embedding similarity. But for extreme domain mismatch (user asks "how to fix NullPointerException" but docs use "null reference error"), even semantic search struggles. Mitigate: fine-tune embedding models on your domain query-document pairs; use query expansion (GPT-4 generates synonyms/related terms and searches for all); use BM25 as fallback for low-confidence semantic results.

**Q: What is cross-encoder re-ranking and when is it worth the cost?**
A: Bi-encoder (standard embedding search): embed query and documents independently, use cosine similarity—fast but less accurate. Cross-encoder: feed (query, document) pair to a model that attends to both simultaneously, producing a relevance score—more accurate but O(n) complexity requiring one forward pass per document. Use cross-encoder to re-rank the top-50 bi-encoder results—you only run it 50 times, not on the full corpus. Typically improves NDCG by 5-10% at 3-10x latency cost.

**Q: How do you evaluate semantic search quality beyond simple accuracy?**
A: Use NDCG@k (normalized discounted cumulative gain)—rewards returning more relevant documents earlier. MRR (mean reciprocal rank)—measures how quickly a relevant result appears. Recall@k—what fraction of relevant docs appear in top-k. For production: track implicit feedback (click-through rates, session depth). Human evaluation for qualitative assessment. Always evaluate on a held-out test set with diverse query types including edge cases.

**Q: How do user query patterns affect semantic search system design?**
A: Short queries (2-3 words) give less semantic signal—HyDE or query expansion helps. Navigational queries (looking for a specific document) favor BM25 keyword matching. Informational queries (broad concepts) favor semantic search. Transactional queries (specific task completion) need both. Design your system to classify query intent and route to the appropriate retrieval strategy, or use a hybrid approach that works adequately for all query types.

**Q: What is the trade-off between embedding model size and search quality?**
A: Larger embedding models (e3-large: 3072 dims, 335M params) produce better embeddings but are slower to compute and require more storage. Smaller models (all-MiniLM-L6: 384 dims, 22M params) are 10-15x faster with ~5-10% quality gap on general tasks. The dimension size also affects HNSW index memory. For high-traffic production, use small models; invest in fine-tuning a small model on your domain rather than using a large generic model.

""",

    "22-semantic-caching.md": """## Interview Q&A

**Q: How do you determine the similarity threshold for cache hits in semantic caching?**
A: Set the threshold by analyzing your query distribution: compute embedding similarity for pairs of queries with the same intent (these should all be cache hits) and pairs with different intent (should be misses). Find the threshold that maximizes F1 on your labeled pairs. Typical values: 0.85-0.95 cosine similarity depending on domain. Test edge cases manually—domain-specific queries often need higher thresholds to avoid false cache hits that return wrong answers.

**Q: What types of queries should never be cached and how do you detect them?**
A: Never cache: time-sensitive queries ("what is today's date?", "latest news"), personalized queries (responses depend on user state/history), queries with side effects (actions, mutations), low-confidence cached responses. Detect with: query classifier for time-sensitive patterns, user context hash (cache separately per user), system prompt analysis (detect side-effect intents). Semantic cache hits for time-sensitive queries actively hurt quality—missing the cache is better than returning stale results.

**Q: How does semantic caching interact with RAG pipelines?**
A: Two levels of caching in RAG: (1) cache the retrieved documents for similar queries (embedding search is fast, but the LLM generation is slow—skip generation if similar query was answered before); (2) cache the final answer. Level 1 is safer but saves less compute. Level 2 saves the most compute but requires high cache hit confidence to avoid returning stale answers to slightly different queries. Implement both with different thresholds—lower threshold for doc caching, higher for answer caching.

**Q: What cache invalidation strategies work for LLM response caches?**
A: TTL (time-to-live): set expiration based on how quickly the underlying information changes—news: 1 hour, product docs: 1 week, mathematical facts: never. Event-based: invalidate when source documents change (requires tracking which documents contributed to each cached response). Version-based: invalidate entire cache when model or system prompt changes. For RAG, store which document chunks were retrieved—invalidate cached responses when those chunks are updated.

**Q: How do you measure the effectiveness of semantic caching in production?**
A: Track: cache hit rate (target 20-40% for diverse queries), latency reduction for hits vs. misses, cost per query before/after, answer quality on cache hits (sample and human-evaluate). Alert on: hit rate drops (distribution shift or threshold too high), user feedback rates higher for cached vs. non-cached responses (quality degradation). Cache analytics should be part of your LLM observability stack.

**Q: What is the memory footprint of a semantic cache and how do you manage it?**
A: Each cached entry stores: embedding vector (1536 floats × 4 bytes = 6KB for ada-002), the original query string, and the cached response (variable, 1-10KB). For 100K cached entries: ~2GB for embeddings alone. Use vector quantization to compress embeddings 8x. Implement LRU eviction policy. For production, use a dedicated vector store (Qdrant, Redis with vector extension) rather than in-memory storage. Shard by query category if cache exceeds available memory.

""",

    "23-kv-cache.md": """## Interview Q&A

**Q: Why does KV cache dramatically speed up autoregressive generation?**
A: Without KV cache, generating each token requires computing attention over all previous tokens from scratch—O(n²) operations per token, O(n³) total for n tokens. With KV cache, key and value matrices are computed once per token and stored; subsequent tokens only compute attention for the new token against stored KVs—O(n) per token. For a 100-token response, this is roughly 50x fewer attention computations.

**Q: How does KV cache memory scale with sequence length and model size, and what are the implications?**
A: KV cache memory = 2 × n_layers × n_heads × head_dim × sequence_length × batch_size × precision_bytes. For a 7B model (32 layers, 32 heads, 128 head_dim, float16): 2 × 32 × 32 × 128 × seq_len × batch × 2 bytes = 524K bytes × seq_len × batch. A batch of 32 requests at 2048 tokens: ~34GB just for KV cache. This is why long-context inference is memory-limited—the KV cache can exceed the model weights in size.

**Q: What is multi-query attention and how does it reduce KV cache memory?**
A: MQA (Multi-Query Attention) uses a single shared key/value head across all query heads, reducing KV cache by n_heads factor (32x for GPT-3-like models). GQA (Grouped-Query Attention) groups query heads to share K/V heads—a compromise between MHA (full KV, highest quality) and MQA (minimum KV, slight quality loss). Llama-2 70B uses GQA with 8 K/V heads shared by 64 query heads—8x KV cache reduction with minimal quality loss.

**Q: What is prefix caching and when does it provide the most benefit?**
A: Prefix caching stores KV values for a common prompt prefix (system prompt, few-shot examples, context documents). If the same prefix appears in many requests, compute the prefix KV once and reuse. Benefits are highest when: the shared prefix is long (few-shot examples, RAG context), traffic volume is high enough to amortize prefix compute, and the prefix doesn't change frequently. Anthropic's API offers prompt caching—significant cost savings for long system prompts repeated across many calls.

**Q: How does continuous batching interact with KV cache management?**
A: Continuous batching adds new requests to the batch as others complete, maintaining high GPU utilization. KV cache challenge: different requests have different sequence lengths and completion states—you can't preallocate a fixed KV buffer per request. PagedAttention (vLLM) solves this by managing KV cache like virtual memory—allocating fixed-size pages on demand and freeing pages when sequences complete. This increases throughput 2-4x over static KV allocation.

**Q: What are the symptoms of KV cache exhaustion in production and how do you handle it?**
A: Symptoms: sudden OOM errors, requests being rejected, latency spikes when cache is full. Handle with: streaming responses (start returning tokens before generation is complete, freeing memory sooner), request queuing with backpressure, dynamic batching (reduce batch size when memory is high), or aggressive KV cache quantization (INT8 KV cache reduces memory 2x at <1% quality loss). Monitor KV cache utilization as a primary LLM serving metric.

""",

    "24-attention-optimization.md": """## Interview Q&A

**Q: What problem does FlashAttention solve and how?**
A: Standard attention computes the full n×n attention matrix and stores it in GPU HBM (high-bandwidth memory). For n=4096, this matrix is 16GB in float16—often exceeding GPU memory. FlashAttention fuses the attention computation into a single CUDA kernel that processes attention in tiles, keeping intermediate results in fast SRAM (10x faster than HBM). It never materializes the full attention matrix, reducing memory from O(n²) to O(n) while achieving the same output.

**Q: How does multi-head attention parallelism affect inference efficiency?**
A: Each attention head computes independently and can run in parallel on separate CUDA cores. However, all heads must complete before the output projection, creating a synchronization barrier. MQA and GQA reduce this parallelism by sharing K/V heads—this trades some parallelism for KV cache savings. On modern GPUs with 1000s of CUDA cores, the reduction in data movement (smaller KV cache) outweighs the loss of head-level parallelism for most practical model sizes.

**Q: When would you use sparse attention vs. full attention?**
A: Sparse attention patterns (local windows, strided, global tokens) reduce attention from O(n²) to O(n·k) where k is the attention span. Use sparse attention for: very long documents (>8K tokens) where full attention is memory-prohibitive, structured inputs where locality matters (text has local coherence), or tasks where global context isn't necessary. Full attention is still preferred for reasoning tasks where long-range dependencies are critical—sparse patterns can miss important cross-document relationships.

**Q: What is the "needle in a haystack" problem for long-context models and how does it relate to attention?**
A: In long contexts, standard attention mechanisms (and even FlashAttention) struggle to extract information from arbitrary positions in the context. The "needle in a haystack" test measures whether a model can recall a specific fact inserted at different positions in a long document. Failure is often due to attention patterns that weight recent tokens more heavily. Positional embeddings (RoPE, ALiBi) affect this—ALiBi's linear decay helps, but isn't a complete solution. Many models have high perplexity accuracy but fail needle-in-haystack.

**Q: How does attention head pruning work and what are its trade-offs?**
A: Attention head pruning removes heads that contribute little to model predictions—measured by gradient magnitude, Taylor expansion, or probing classifiers. After pruning, the remaining heads must be fine-tuned to recover quality. Typical results: 20-30% of heads can be pruned with <1% quality loss; 50%+ pruning causes significant degradation. Trade-off: reduced inference compute and KV cache at the cost of fine-tuning effort and potential quality regression on tasks that relied on pruned heads.

**Q: What is rotary positional embedding (RoPE) and why has it become the dominant positional encoding?**
A: RoPE encodes position by rotating the query and key vectors in the complex plane based on token position, before computing dot-product attention. This makes attention scores naturally dependent on relative positions (the rotation of q-k is the rotation difference). Advantages over learned absolute positions: zero position-out-of-distribution generalization (can extend context length beyond training), no additional parameters, compatible with efficient attention implementations. Llama, Mistral, and most modern models use RoPE.

""",

    "25-context-window.md": """## Interview Q&A

**Q: Why does performance degrade for information in the middle of long contexts?**
A: The "lost in the middle" effect occurs because: (1) recency bias—attention naturally weights recent tokens more; (2) positional encoding effects—for absolute position encodings, very high position indices are rare in training; (3) long-range attention dilution—with many tokens to attend to, individual relevant tokens get lower attention weights. Mitigate by placing critical information at the beginning or end of the context, using re-ranking to promote relevant information, or fine-tuning on long-context tasks.

**Q: What is the difference between a model's trained context window and its effective context window?**
A: Trained context window: the maximum sequence length the model was trained on—the model has seen this many tokens together. Effective context window: the length at which the model can still reliably use all information in the context. In practice, effective context < trained context—performance on retrieval tasks often degrades 20-40% for content placed in the second half of a long context. Measure effective context empirically with needle-in-haystack tests at different positions.

**Q: How does context window size affect inference cost and latency?**
A: Input processing (prefill) scales O(n²) with context length due to attention computation. For a model with 128K context, processing a full-context prompt takes 128x more compute than a 1K context. KV cache memory scales O(n) with context length—a 128K context uses 128x more KV cache memory than 1K. Cost implications: long-context inputs are significantly more expensive. Design your application to use the minimum necessary context rather than always filling the window.

**Q: What techniques extend models beyond their trained context window?**
A: RoPE scaling: adjust the RoPE base frequency to allow longer relative positions. YaRN (Yet another RoPE extensioN): scales differently for different attention heads. LongLoRA: adds shift short attention during fine-tuning to efficiently train long-context adaptation. These allow inference at 2-8x training context with minimal quality loss after fine-tuning. Always validate with your specific use case—positional generalization varies significantly by task type and document structure.

**Q: How do you architect a system that processes documents longer than any context window?**
A: Hierarchical summarization: chunk into context-sized pieces, summarize each, then summarize summaries. Map-reduce: process each chunk independently, combine results. Sliding window with overlap: maintain a rolling window with overlap to preserve cross-chunk context. Retrieval-based: embed all chunks, retrieve relevant ones for each query. For most production use cases, RAG with chunk retrieval outperforms truncation or summarization because it preserves fidelity to the source.

**Q: When should you summarize context vs. retrieve context vs. extend the context window?**
A: Summarize when: you need to maintain conversational context over many turns, the full history is too long, and rough fidelity is acceptable. Retrieve when: specific factual information from long documents is needed and precision matters. Extend context window when: the task requires holistic understanding of the entire document (contract review, code analysis) and you can afford the compute. For most RAG applications, retrieval is more cost-effective than maintaining very long contexts.

""",

    "26-continuous-batching.md": """## Interview Q&A

**Q: Why does naive static batching underperform for LLM serving and how does continuous batching fix it?**
A: Static batching allocates a fixed batch at the start and waits for all sequences to complete before starting new ones. Because sequences have different lengths, GPU sits idle when shorter sequences finish but longer ones continue. Continuous batching (iteration-level scheduling) inserts new requests into the batch at every token generation step—as soon as a sequence completes, a new request takes its slot. This increases GPU utilization from 40-60% to 85-95%.

**Q: What is the difference between throughput and latency optimization in LLM serving, and how does batching affect each?**
A: Throughput: maximize tokens generated per second (total) across all users—larger batches improve throughput by amortizing fixed costs. Latency: minimize time-to-first-token and total response time for individual requests—larger batches increase queuing and per-request latency. These goals conflict. Tune by setting a target latency SLA (e.g., P95 TTFT < 2s) and maximize batch size within that constraint. During off-peak hours, use larger batches for throughput; during peak traffic, prioritize latency.

**Q: How does PagedAttention improve upon continuous batching?**
A: Continuous batching still wastes memory due to fixed KV cache pre-allocation per sequence—you allocate for the maximum possible length even if the sequence ends early. PagedAttention manages KV cache like OS virtual memory: allocates cache in fixed-size pages (e.g., 16 tokens), only allocating new pages as the sequence grows, and immediately freeing pages when sequences complete. This reduces memory waste from 20-40% to <5%, enabling higher batch sizes and better GPU utilization.

**Q: What are the key metrics to monitor for an LLM serving system?**
A: TTFT (time-to-first-token): latency of the prefill phase—affected by input length and batch size. TBT (time-between-tokens): latency of each decode step—affected by batch size and KV cache pressure. Throughput (tokens/second): overall system capacity. GPU utilization: target >80% for efficiency. Queue depth: requests waiting for GPU. KV cache utilization: when near 100%, start rejecting or queuing requests. Track P50, P95, P99 for all latency metrics.

**Q: How do you implement priority queuing for LLM serving?**
A: Assign priorities based on: request SLA (premium vs. free tier), queue age (prevent starvation), request type (interactive vs. batch). Implement a priority queue that the scheduler checks when inserting into the running batch. With continuous batching, high-priority requests can be inserted into the next available slot (within 1-2 decode steps). For batch workloads, process requests in off-peak windows rather than competing with interactive traffic.

**Q: When should you scale horizontally (more GPUs) vs. optimize serving software?**
A: Optimize software first: vLLM or TGI with PagedAttention + continuous batching can 2-4x throughput vs. naive serving. Model quantization (INT8/INT4) can further 2x throughput with acceptable quality loss. If after software optimization your P95 latency still exceeds SLA at peak traffic, scale horizontally. Monitor GPU utilization—if <70%, software optimization can help; if >90%, horizontal scaling is likely needed. Multi-GPU tensor parallelism helps latency; multi-GPU pipeline parallelism helps throughput.

""",

    "27-speculative-decoding.md": """## Interview Q&A

**Q: Why does speculative decoding reduce latency when it requires more computation?**
A: Standard decoding generates one token per forward pass—the large model is underutilized because decode batch size is 1. Speculative decoding generates multiple draft tokens with a small fast model, then verifies all of them with the large model in one forward pass (batch size = n draft tokens). The verification is cheap because it's a single forward pass over n tokens in parallel. Net result: 2-3x speedup because the large model's parallelism is better utilized.

**Q: How do you choose the draft model and what are the quality requirements?**
A: The draft model must have the same vocabulary as the target model (token alignment). It should be 5-20x smaller (e.g., Llama-7B as draft for Llama-70B). Quality requirement: acceptance rate (fraction of draft tokens accepted) should be >70% for significant speedup—lower acceptance means too much verification overhead. Use the same model family if possible (Llama-7B + Llama-70B works well; mismatched families have lower acceptance rates). Fine-tune the draft model on your domain for better acceptance rates.

**Q: What is self-speculative decoding and when does it help?**
A: Self-speculative decoding uses early exit layers of the same model as the draft model—the model generates draft tokens by running only the first N layers, then verifies with the full model. Advantage: no separate draft model to maintain, and alignment is perfect (same model weights). Works best when early layers capture most of the output distribution (true for simple/predictable tasks). Less effective for complex reasoning where final layers are critical for quality.

**Q: How does speculative decoding interact with temperature and sampling strategies?**
A: For temperature=0 (greedy decoding), acceptance rate is highest because draft and target token distributions align well for common tokens. For temperature>0 (sampling), acceptance rate decreases because even if the target model agrees a token is likely, sampling may choose a different token. Speculative decoding with sampling uses rejection sampling to maintain the exact target distribution—accepted tokens are accepted with probability min(1, p_target/p_draft). Higher temperature means lower acceptance rate and less speedup.

**Q: What are the failure modes of speculative decoding in production?**
A: Low acceptance rate (bad draft quality): benchmark acceptance rate on your workload. Out-of-distribution inputs: acceptance rate drops for queries outside draft model's training distribution. Memory overhead: running two models simultaneously doubles memory use—may not fit on a single GPU. Incorrect implementation: must use rejection sampling to maintain exact target distribution, not greedy acceptance. Latency under high load: with large batches, the target model's forward pass is already well-utilized and speculative decoding may not help.

**Q: When should you deploy speculative decoding vs. other latency reduction techniques?**
A: Use speculative decoding when: you have a memory budget for two models, latency is critical and GPU utilization is low (small batch sizes), and text is often predictable (chat responses, code completion with common patterns). Consider alternatives when: memory is constrained (quantization 2x latency gain with no memory overhead), or you're serving large batches (continuous batching + PagedAttention is more impactful). Speculative decoding is most effective for interactive chat; quantization is better for batch inference.

""",

    "28-inference-optimization.md": """## Interview Q&A

**Q: What is the optimization priority order for LLM inference and why?**
A: (1) Software stack: switch to vLLM/TGI with PagedAttention and continuous batching—often 3-5x improvement with no quality loss. (2) Quantization: INT8 weights + KV cache gives 2x memory reduction with <1% quality loss. (3) Batching: increase batch size until latency SLA is hit. (4) Hardware: upgrade GPU or add more. (5) Model distillation: smaller model for acceptable quality. This order matters because software optimization is free, quantization is cheap, and hardware is expensive.

**Q: How do you profile an LLM inference workload to identify bottlenecks?**
A: Measure separately: prefill time (scales with input length), decode time per step (scales with batch size and KV cache size), total TTFT, TBT. Use GPU profiling (nvtop, DCGM) to check GPU utilization and memory bandwidth usage. If GPU utilization is low during decode: memory-bandwidth-bound—quantization helps. If high during prefill: compute-bound—FlashAttention or better hardware helps. Profile different batch sizes and sequence lengths to understand the operating point.

**Q: What is the difference between model parallelism strategies and when do you use each?**
A: Tensor parallelism: splits each weight matrix across GPUs—reduces per-GPU memory but requires all-reduce communication each layer. Best for single large model with real-time latency requirements. Pipeline parallelism: assigns different layers to different GPUs—lower communication overhead but causes pipeline bubbles (GPUs idle waiting). Best for throughput-oriented workloads. Expert parallelism (for MoE models): routes different tokens to different expert GPUs. Combine based on model size and traffic pattern.

**Q: How does int8 quantization affect model quality and which layers are most sensitive?**
A: INT8 weight quantization typically causes <0.5% accuracy drop for most NLP tasks. Most sensitive layers: the first and last layers (embedding and output projection), attention output projections, and layers with high dynamic range. LLM.int8() handles outlier activations that cause large quantization errors by keeping a small fraction of dimensions in float16. Activation quantization (for KV cache) is harder—KV cache INT8 works well but INT4 requires careful calibration.

**Q: What is weight sharing and how does it reduce inference cost without quality loss?**
A: Weight sharing reuses the same weight matrix for multiple layers (cross-layer weight sharing). ALBERT shares weights across all transformer layers, reducing parameters 18x with ~5% quality loss. For inference, parameter sharing reduces GPU memory load and can improve cache efficiency. Most production LLMs don't use weight sharing as it trades quality for size—but for edge deployment where model size is critical, it's a viable option alongside distillation and quantization.

**Q: How do you benchmark inference optimization and ensure improvements are real?**
A: Measure on realistic workloads: use your actual production query distribution (length distribution, batch size distribution). Report: TTFT (P50, P95), TBT, throughput (tokens/sec), GPU utilization, cost per 1K tokens. Pitfalls: measuring only peak throughput misses latency SLA violations; measuring only one batch size misses variability. Run for 30+ minutes to catch memory fragmentation issues. Validate quality (not just speed): sample responses before and after optimization.

""",

    "29-token-optimization.md": """## Interview Q&A

**Q: What is the most impactful way to reduce token usage in a production LLM application?**
A: Prompt compression is typically highest impact: remove redundant examples, use concise phrasing, remove explanatory scaffolding that doesn't improve output quality. Measure: compress each prompt component independently and measure output quality impact. For RAG applications, retrieved context is usually the largest token consumer—better retrieval precision (fewer, more relevant chunks) reduces tokens more than prompt rewording. A 30-50% token reduction with no quality loss is achievable in most applications.

**Q: How do you measure the relationship between prompt length and output quality?**
A: A/B test: for each prompt component (system instructions, examples, context), create a compressed version and measure output quality on a held-out eval set. Track: quality metric (task-specific), input tokens, output tokens, cost per query. Plot quality vs. token count curves for each component. Some components have high token-to-quality ratios (critical); others can be cut by 80% with <2% quality impact. Focus optimization on large components with flat quality curves.

**Q: When does reducing output length hurt quality and how do you avoid it?**
A: Constraining output length hurts when: the task naturally requires verbose output (detailed explanations, code with comments), the model needs reasoning steps to arrive at correct answers (cutting CoT), or when the user expects comprehensive coverage. Avoid by: setting minimum output length for complex tasks, using structured output formats that are concise by design, measuring both quality and length jointly. Adding "be concise" instructions often works but can cause models to omit critical information.

**Q: What is streaming and how does it affect perceived token cost?**
A: Streaming sends tokens to the user as they're generated rather than buffering the complete response. This doesn't reduce token count or compute cost—you still generate all tokens. But streaming dramatically improves user experience: TTFT is the latency to first visible text, which is seconds vs. 10-60s for non-streamed responses. For long responses, streaming makes the application feel 5-10x faster. Always implement streaming for interactive applications.

**Q: How do system prompt caching and batching interact with token optimization?**
A: Prompt caching (Anthropic, OpenAI) reuses KV computations for repeated prefix content—you pay full price for first computation but 10-15% of cost for subsequent requests with the same prefix. Effective when: long system prompts are reused across many requests (RAG context, few-shot examples, persona instructions). Pair with token optimization: reduce prompt tokens AND cache the remaining tokens. For batch workloads, group requests with the same system prompt to maximize cache hit rates.

**Q: What are the cost implications of different LLM context lengths and how do you manage them?**
A: Cost scales linearly with input+output tokens for most APIs. Input tokens are cheaper than output tokens (typically 3-5x). Long system prompts repeated across many calls are expensive—cache them. For applications with 100K+ token contexts, cost can be 100x higher than short-context equivalents. Implement token budgets: track usage per user/session, implement prompt compression when approaching limits, alert on unexpected token usage spikes that may indicate prompt injection or runaway loops.

""",

    "30-quantization.md": """## Interview Q&A

**Q: What is the difference between post-training quantization (PTQ) and quantization-aware training (QAT)?**
A: PTQ: quantize a pretrained model without retraining—fast (minutes to hours), minimal quality loss for INT8, significant loss for INT4. QAT: simulate quantization during training so the model learns to be robust to quantization noise—slower (requires full training run), better quality, especially for INT4 and INT2. For LLMs, PTQ with GPTQ or AWQ achieves near-QAT quality by finding optimal quantization points using calibration data, making QAT rarely necessary at int8/int4.

**Q: Why does GPTQ outperform naive round-to-nearest quantization?**
A: GPTQ uses second-order information (Hessian of the loss) to quantize weights in a way that minimizes the output error. For each layer, it quantizes one column at a time and updates remaining unquantized weights to compensate for the quantization error—a form of error propagation compensation. This reduces quantization-induced output error by 5-10x compared to naive rounding, enabling INT4 quality close to FP16 at 4x memory reduction.

**Q: When would you choose INT8 vs. INT4 quantization?**
A: INT8: highest quality (perplexity increase <0.3), 2x memory reduction, hardware-efficient (NVIDIA Ampere+ supports INT8 tensor cores). Use when quality is critical and 2x reduction is sufficient. INT4: significant memory reduction (4x), perplexity increase 0.5-2.0 depending on method (GPTQ, AWQ), some quality loss on complex reasoning. Use when fitting the model on available hardware requires the extra compression. INT4 enables running 70B models on a single 48GB GPU; INT8 requires 80GB.

**Q: How does activation quantization differ from weight quantization?**
A: Weight quantization: static—weights are quantized offline and don't change at inference. Activation quantization: dynamic—activations are computed at runtime and must be quantized on-the-fly. Activations have higher dynamic range than weights (due to outlier activations in LLMs), making INT8 activation quantization harder. SmoothQuant migrates quantization difficulty from activations to weights by rescaling, making both INT8-quantizable. W8A8 (INT8 weights and activations) achieves 2x compute and 2x memory reduction.

**Q: What is the impact of quantization on different model tasks (reasoning, coding, generation)?**
A: Quantization affects tasks differently. Factual recall and classification: minimal degradation even at INT4. Mathematical reasoning: more sensitive—INT4 can cause 5-10% accuracy drop on GSM8K. Code generation: sensitive to syntax—INT4 can produce syntactically invalid code more frequently. Creative generation: minimal perceptible difference. Always benchmark your specific use case at your target quantization level—don't assume INT4 is acceptable based on perplexity alone.

**Q: How do you deploy a quantized model in production and what serving infrastructure considerations apply?**
A: Use frameworks that support quantized inference natively: vLLM (supports GPTQ, AWQ, INT8), TGI (similar support), llama.cpp for CPU/edge deployment with GGUF format. Key considerations: quantized models require specific CUDA kernels (not all hardware supports INT4 efficiently), kernel warmup time on first inference, batch size limitations for some quantized kernels. Benchmark end-to-end throughput with quantized models—sometimes the specialized kernels have overhead that reduces gains at small batch sizes.

""",

    "31-multimodal.md": """## Interview Q&A

**Q: How do vision-language models (VLMs) align visual and textual representations?**
A: Most VLMs use a visual encoder (ViT or CLIP) to extract patch embeddings, then a projection layer to map visual embeddings into the LLM's token embedding space. The projection layer bridges the visual and language representation spaces. Training involves: first training the projection layer while freezing both encoder and LLM, then jointly fine-tuning with vision-language data. Newer approaches (LLaVA-1.5) use a simple MLP projection and find that instruction-following data quality matters more than architecture complexity.

**Q: What are the limitations of current VLMs for tasks requiring fine-grained spatial understanding?**
A: VLMs struggle with: counting objects accurately (often off by 1-2), precise coordinate localization (bounding box prediction), reading dense text at very high resolution (character-level detail), and understanding spatial relationships for unusual viewpoints. ViT patches (16×16 or 14×14 pixels) create a resolution bottleneck. Specialized models (document understanding models, OCR-specialized VLMs) outperform general VLMs for fine-grained spatial tasks by using higher resolution inputs or specialized position encodings.

**Q: How do you handle documents with mixed text and images in a RAG pipeline?**
A: Three approaches: (1) extract text with OCR, discard images—loses visual information; (2) describe images with a VLM and include descriptions in the text index—preserves semantic content but loses visual fidelity; (3) multi-modal retrieval—embed both text blocks and images separately, retrieve both modalities. ColPali takes approach 3 with direct page embedding from a VLM, enabling retrieval on PDF pages as images without OCR. Choose based on how much visual content (charts, diagrams) matters for your use case.

**Q: What is the role of image resolution in VLM performance and how do you handle high-resolution images?**
A: Higher resolution preserves more detail but increases token count quadratically (2× resolution = 4× patches). Most VLMs use 224×224 or 336×336 inputs—adequate for scene understanding, poor for document/OCR tasks. High-resolution strategies: dynamic tiling (split high-res image into overlapping tiles, process each, combine), mixture of resolutions (process at multiple resolutions and fuse features), or specialized high-res encoders. For document tasks, use models specifically designed for high-resolution like LLaVA-HD or InternVL.

**Q: How do you evaluate the quality of a VLM for a specific use case?**
A: General benchmarks: VQAv2 (visual question answering), TextVQA (text in images), MMBench (comprehensive), POPE (hallucination). Task-specific: for document understanding use DocVQA, for chart reading use ChartQA. Build your own eval dataset from representative examples of your actual use case—general benchmarks often don't capture domain-specific requirements. Measure: accuracy on structured tasks, hallucination rate (does the model invent visual details?), and latency (VLMs are slower than text-only LLMs).

**Q: What are the key considerations for deploying VLMs in production vs. text-only LLMs?**
A: Image preprocessing: standardize image size, handle JPEG/PNG/PDF inputs, implement malicious image detection. Latency: image encoding adds 100-500ms; use caching for repeated images. Cost: VLM inference is 2-5x more expensive per request than text-only due to image tokens. Context limits: images consume significant context (256-1024 tokens per image); limit simultaneous images. Rate limiting: separate rate limits for visual vs. text-only requests. Monitoring: track image token usage separately from text token usage.

""",

    "32-evaluation.md": """## Interview Q&A

**Q: Why is perplexity a poor metric for evaluating LLM quality for downstream tasks?**
A: Perplexity measures average per-token log-probability—it captures how well the model fits the text distribution but doesn't measure usefulness for tasks. A model with lower perplexity may perform worse on instruction following, reasoning, or safety. Models can achieve low perplexity by being verbose and hedged in ways that annoy users. Always pair perplexity with task-specific metrics (exact match, ROUGE, human preference win rate) relevant to your actual use case.

**Q: How do you design an LLM evaluation suite that detects regressions before deployment?**
A: Include: task accuracy benchmarks (MMLU, HumanEval for code), safety benchmarks (ToxiGen, TruthfulQA), capability regression tests (tasks your users care most about), and golden set comparisons (sample outputs from prod that were rated highly—check new model maintains quality). Run automated evals in CI/CD. Flag regressions >2% on any benchmark. Include adversarial examples: jailbreaks, edge cases that previously caused failures. Human eval is the gold standard but too slow for CI—use it for final pre-release validation.

**Q: What are the limitations of LLM-as-judge evaluation?**
A: LLM judges have systematic biases: position bias (prefers responses listed first), verbosity bias (prefers longer responses), self-enhancement bias (GPT-4 rates GPT-4 outputs higher). These biases can mislead optimization—models fine-tuned to score higher on LLM-judge may not actually be better for users. Mitigate: randomize response order, use multiple judges, calibrate against human preferences, use structured rubrics rather than free-form scoring, and regularly validate LLM judge scores against human agreement.

**Q: How do you measure hallucination rate and what constitutes a reliable evaluation?**
A: Hallucination types: (1) factual hallucination—claims contradicted by external knowledge; (2) faithfulness hallucination—claims not supported by provided context (RAG). Measure faithfulness with: FActScore (decompose response into atomic facts, verify each against source), NLI-based entailment classifiers, or LLM-as-judge with explicit attribution criteria. Factual hallucination is harder—requires knowledge base queries or search verification. Sample 100-500 responses from production and measure; automate with a classifier validated on human labels.

**Q: What is the difference between static benchmark evaluation and online evaluation?**
A: Static benchmarks (MMLU, HumanEval): fixed test sets, reproducible, good for regression testing, but may become saturated (models trained on benchmark-like data) or not reflect real user needs. Online evaluation: uses actual user queries and interactions, reflects real distribution, includes implicit feedback (thumbs up/down, re-asks, session abandonment). Online evaluation is more valid but harder to instrument and interpret. Best practice: use static benchmarks for development decisions; use online metrics for production quality monitoring.

**Q: How do you handle evaluation for subjective tasks where there is no single correct answer?**
A: For creative writing, explanation quality, or tone—use pairwise comparison (A vs. B) rather than absolute scoring, which is more reliable and less affected by scale calibration issues. Collect multiple human ratings per response and report inter-annotator agreement (Cohen's kappa). Use diverse annotators to capture different preferences. For production, use implicit signals (user satisfaction, follow-up rate, session length) as proxies. Separate "factually correct" from "stylistically preferred" in your rubrics.

""",
}


def insert_qa_section(filepath, qa_content):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already has Q&A section
    if '## Interview Q&A' in content:
        print(f"SKIP (already has Q&A): {os.path.basename(filepath)}")
        return False

    # Find insertion point: first occurrence of ## Related Topics or ## Resources
    lines = content.split('\n')
    insert_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == '## Related Topics' or stripped == '## Resources':
            insert_idx = i
            break

    if insert_idx is None:
        print(f"WARNING: No insertion point found in {os.path.basename(filepath)}")
        return False

    # Insert the Q&A section before the found line
    new_lines = lines[:insert_idx] + qa_content.split('\n') + lines[insert_idx:]
    new_content = '\n'.join(new_lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"OK: {os.path.basename(filepath)}")
    return True


def main():
    processed = 0
    skipped = 0
    warnings = 0

    for filename, qa_content in QA_SECTIONS.items():
        filepath = os.path.join(CONCEPTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"ERROR: File not found: {filepath}")
            warnings += 1
            continue

        result = insert_qa_section(filepath, qa_content)
        if result:
            processed += 1
        else:
            skipped += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Warnings: {warnings}")


if __name__ == '__main__':
    main()
