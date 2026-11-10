# LLM Curriculum Practice

Use each scenario as a short interview drill. State assumptions, propose a concrete approach, define how you would evaluate it, and identify a failure mode.

## Tokenization

### Prompt
Design or diagnose a production workflow that depends on Tokenization. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
Tokenization maps text through a tokenizer's normalization and segmentation rules into IDs that index a particular model's embedding table. I would pin the tokenizer with the checkpoint and measure token counts on representative production inputs, including code, numbers, and each supported language, because fragmentation affects truncation, context use, and token-based cost. For an input that becomes an unknown token, I would inspect its encoded IDs and compare them with `unk_token_id`; byte fallback instead represents the UTF-8 bytes with ordinary vocabulary IDs and should decode back to the original string. I would test round trips and monitor token counts and truncation by input type. If a language or domain is unusually fragmented, I would benchmark a compatible multilingual/domain tokenizer and model together rather than swap tokenizers under the existing checkpoint.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Embeddings

### Prompt
Design or diagnose a production workflow that depends on Embeddings. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
An embedding is a vector whose useful geometry depends on the encoder, pooling method, training objective, and task. For semantic search, I would encode queries and passages with compatible versions, choose cosine or dot product consistently with normalization and the encoder's intended use, and evaluate on judged query-document pairs with Recall@k or nDCG@k. I would compare against lexical search and a hybrid baseline, especially for product IDs and exact terms. I would version the model, preprocessing, pooling, normalization, and vector dimension, and re-embed the corpus after incompatible changes. A concrete input check must reject unequal vector dimensions; for instance, cosine([1, 0], [1]) is undefined and should raise an error rather than silently truncating via `zip`.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Pretraining

### Prompt
Design or diagnose a production workflow that depends on Pretraining. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Fine-tuning

### Prompt
Design or diagnose a production workflow that depends on Fine-tuning. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
Fine-tuning updates selected model parameters by minimizing a loss on serialized input-target examples; it is appropriate when a repeated behavior gap cannot be met reliably through prompting or retrieval. I would first define the target behavior, pin the base checkpoint and tokenizer, remove duplicates, split examples by source or group to prevent leakage, and specify which tokens receive loss. I would compare the tuned checkpoint with the base model on a frozen held-out task set and a capability-regression suite, reporting task metrics and uncertainty rather than training loss alone. Training costs include compute, optimizer/activation memory, data review, and checkpoint operations; test contamination, narrow data, or forgetting are key risks. If the need is changing factual knowledge, retrieval is usually easier to update and audit. The paired notebook and implementation in this repository show inference, not a fine-tuning run.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Instruction Tuning

### Prompt
Design or diagnose a production workflow that depends on Instruction Tuning. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
Instruction tuning is supervised fine-tuning on task/request and desired-response demonstrations. I would define the supported task mixture, quality-review examples, apply the base model's actual chat template, and mask prompt tokens when the objective is to train only on assistant responses. To test transfer, I would hold out task families and paraphrases as well as source documents, then compare the tuned model with the base under identical prompts and decoding; report per-task accuracy or rubric scores plus factuality, safety, and format regressions. A narrow or imbalanced mixture can cause phrasing overfit or degrade other skills, and synthetic teacher errors can be learned. If the model already follows the task with a prompt, prompting is cheaper; if answers need frequently changing facts, use retrieval. This repository's paired notebook demonstrates inference and does not perform instruction-tuning training.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## RLHF (Reinforcement Learning from Human Feedback)

### Prompt
Design or diagnose a production workflow that depends on RLHF (Reinforcement Learning from Human Feedback). Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## DPO (Direct Preference Optimization)

### Prompt
Design or diagnose a production workflow that depends on DPO (Direct Preference Optimization). Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## LoRA (Low-Rank Adaptation)

### Prompt
Design or diagnose a production workflow that depends on LoRA (Low-Rank Adaptation). Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
LoRA freezes a pretrained weight matrix `W` and learns a low-rank update `sBA`, with `A` shaped `r × d_in` and `B` shaped `d_out × r`; the adapter therefore has `r(d_in + d_out)` trainable parameters instead of `d_in d_out` for that matrix. I would compare a small set of ranks and target-module choices on the same data and recipe, measuring held-out task quality, regression scores, peak memory, throughput, and adapter size. Low rank can underfit, while larger rank costs more and can overfit; trainable-parameter savings do not remove base-weight or activation memory. I would verify that base weights are frozen and intended modules matched, then compare merged and separate-adapter outputs. Full fine-tuning is an alternative when its added capacity is justified; prompting is preferable when no learned weight change is needed.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Adapters

### Prompt
Your team needs to adapt a frozen encoder for three classification tasks. Propose a bottleneck-adapter design and a validation plan. Explain where the adapter is inserted, how its shape affects parameter count, and how you would compare it with full fine-tuning and LoRA.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A bottleneck adapter typically applies `h + W_up σ(W_down h)`, where `W_down` maps hidden width `d` to bottleneck width `r`, and `W_up` maps back to `d`. The backbone remains frozen while adapter and usually task-head parameters are trained. The two projection matrices add `2dr` weights per adapted location, excluding biases and normalization parameters; at `d=768, r=32`, that is 49,152 weights. I would select insertion points and `r` on a development set, verify frozen backbone parameters and gradient flow into adapters, and compare per-task held-out quality, latency, peak memory, and storage against full fine-tuning and LoRA under the same data and compute budget. A narrow bottleneck may underfit, while adding adapters at every layer can increase runtime and task interference. Separate adapters can be stored per task, but all still require the base model at inference.

### Evaluation
Score: mechanism and residual path (0–2), correct parameter arithmetic (0–2), fair alternatives and resource trade-offs (0–2), controlled validation plan (0–2), and a concrete failure diagnostic (0–2).

### Follow-up
If adapter training loss falls but held-out task quality does not improve, what would you check in the bottleneck width, insertion points, labels, and gradient flow?

### Common Mistake
Calling LoRA a bottleneck adapter: LoRA adds a low-rank update to selected weight matrices, while a bottleneck adapter is a nonlinear residual module inserted into the network.

## Prefix Tuning

### Prompt
Design or diagnose a production workflow that depends on Prefix Tuning. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Parameter-Efficient Fine-Tuning (PEFT)

### Prompt
Design or diagnose a production workflow that depends on Parameter-Efficient Fine-Tuning (PEFT). Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Prompting

### Prompt
Design or diagnose a production workflow that depends on Prompting. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Few-Shot Learning

### Prompt
You have 20 labeled support examples for a new intent and 200 held-out messages. How would you build and evaluate a few-shot classifier when no task-specific model training is allowed? Explain support-set selection, leakage controls, a baseline, and how you would report uncertainty.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
I would define the label set and split the 200 messages before selecting demonstrations, then choose representative, non-duplicate examples from the 20 labeled support items. With a language model, I would place a balanced set of labeled examples and a constrained output schema in the prompt; a simple lexical or nearest-neighbor baseline gives a useful comparison. I would evaluate once on the held-out messages using macro-F1, per-class recall, and a confusion matrix, and compare zero-shot with few-shot on the identical cases. I would inspect errors and vary support examples and ordering to measure sensitivity. If the 20 examples are imbalanced or unrepresentative, reported performance may not transfer; I would state the small-sample uncertainty and avoid tuning against the held-out set. This is inference conditioned on examples, not parameter updates.

### Evaluation
Score: correct distinction from model training (0–2), representative demonstrations (0–2), leakage-safe split (0–2), suitable metrics and baseline (0–2), and sensitivity/uncertainty analysis (0–2).

### Follow-up
The examples contain no instances of one rare intent. How would you detect and address that coverage gap without selecting examples based on held-out labels?

### Common Mistake
Calling a result “few-shot” while training on the evaluation set, or reporting accuracy alone when rare classes may be missed.

## Zero-Shot Learning

### Prompt
Design or diagnose a production workflow that depends on Zero-Shot Learning. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## In-Context Learning

### Prompt
An inference prompt contains eight examples, but accuracy drops when their order changes and some prompts exceed the context budget. Diagnose the issue and propose an evaluation and prompt-construction strategy. Distinguish this from updating model weights.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
In-context learning conditions a pretrained model on instructions and demonstrations in the input; it does not update the weights. I would first freeze a representative evaluation set and compare zero-shot with few-shot prompts using the same model settings. Then I would test example selection, class balance, ordering, label wording, and formatting as controlled factors, measuring task quality, prompt tokens, latency, and cost. To stay within budget, I would retrieve or select a smaller, representative set and reserve space for the user query and response. Order sensitivity is a real prompt robustness issue, so I would evaluate several permutations or use a validated ordering policy rather than selecting the best order on the test set. If performance is unstable across examples or far below a trained model, supervised fine-tuning may be more appropriate.

### Evaluation
Score: weight-update distinction (0–2), controlled order/selection experiments (0–2), context budgeting (0–2), quality and operational metrics (0–2), and leakage-aware alternative choice (0–2).

### Follow-up
How would you test whether the observed order effect is larger than ordinary sampling variation in a stochastic model?

### Common Mistake
Treating in-context learning as fine-tuning, or changing example choice and model settings together so the cause of a quality change is unknown.

## Chain-of-Thought (CoT)

### Prompt
An assistant answers multi-step invoice questions and sometimes gives a plausible but incorrect total. Design a safe decomposition and verification approach for this workflow. Explain what intermediate artifacts you would expose and what they cannot establish about a language model’s internal reasoning.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
I would parse line items into a typed structure, compute subtotals and totals with deterministic code, and validate inputs such as quantities, currency, and missing fields. The user-facing response can show a concise itemized calculation and source values, while an independent verifier recomputes the result from the parsed data. I would test arithmetic errors, malformed fields, rounding rules, and adversarial text, and measure exact-answer accuracy and abstention on a held-out set. A written explanation can help users audit the calculation, but it is not proof that a language model’s text reveals its private internal reasoning. For reliable arithmetic, a deterministic calculator is preferable to asking the model to reason unaided; the model can assist with extraction while the calculator owns computation.

### Evaluation
Score: safe task decomposition (0–2), independent verification (0–2), edge-case tests and metric (0–2), clarity about explanation limits (0–2), and appropriate automation boundary (0–2).

### Follow-up
The displayed steps look correct, but the final result disagrees with the source amounts. How would you localize whether the failure came from extraction, computation, or rendering?

### Common Mistake
Claiming that a generated step-by-step explanation exposes hidden chain-of-thought or guarantees correctness without an independent check.

## Prompt Optimization

### Prompt
Design or diagnose a production workflow that depends on Prompt Optimization. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## RAG (Retrieval-Augmented Generation)

### Prompt
An internal policy assistant gives answers unsupported by its document collection. Design a retrieval-augmented workflow from ingestion through answer citation. Explain how you would distinguish retrieval failures from generation failures and handle a query with no adequate evidence.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
I would parse and version the policy corpus, split documents into traceable chunks with metadata and access controls, and retrieve candidate chunks using a lexical, vector, or hybrid search appropriate to the query. A reranker can improve ordering at added latency. The generator should receive only authorized, relevant evidence with stable source IDs and instructions to abstain when evidence is insufficient; the answer should cite those IDs. I would maintain labeled query-to-evidence cases and evaluate Recall@k or nDCG for retrieval separately from answer correctness, citation support, and abstention quality. For an unsupported answer, inspect the retrieved chunks first: missing or irrelevant evidence points to ingestion, chunking, filters, or ranking; adequate evidence with an unsupported claim points to generation or citation behavior. No-match queries should return a clear limitation or route to a human, never a fabricated citation. I would compare against a lexical baseline and monitor freshness, access-control filtering, latency, and cost.

### Evaluation
Score: end-to-end pipeline and source traceability (0–2), separated retrieval/generation metrics (0–2), evidence-based abstention (0–2), access/freshness controls and trade-offs (0–2), and actionable failure diagnosis (0–2).

### Follow-up
The answer is wrong although a relevant chunk exists in the corpus. Which logs and intermediate artifacts would you inspect before changing the generator?

### Common Mistake
Treating a successful document lookup as proof the answer is grounded, or evaluating only end-to-end answers without checking retrieval recall and citation support.

## Retrieval-Augmented Generation (RAG)

### Prompt
Design or diagnose a production workflow that depends on Retrieval-Augmented Generation (RAG). Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Vector Databases

### Prompt
Design or diagnose a production workflow that depends on Vector Databases. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Semantic Search

### Prompt
Design or diagnose a production workflow that depends on Semantic Search. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Semantic Caching

### Prompt
Design or diagnose a production workflow that depends on Semantic Caching. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## KV Cache

### Prompt
For a decoder with 32 layers, 8 KV heads, head dimension 128, batch size 1, 4,096 cached tokens, and 2-byte K/V values, estimate cache memory. Explain what work is reused on the next generated token, what still grows with context, and how you would validate safe cache reuse.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
The estimate is `2 * B * L * T * H_kv * d * bytes = 536,870,912` bytes, or 512 MiB, excluding allocator and runtime overhead. The factor two stores keys and values. With a KV cache, prompt-token K/V projections are reused and only the new token's K/V projections are appended; the new query still attends to the available history, so attention work and cache memory still rise with context length. I would compare cached decoding against full-prefix recomputation on fixed prompts by checking token logits or outputs, and measure per-token latency across context lengths. Cache identity must bind the exact prefix, model revision, adapters, positions, and tenant boundary. At concurrency, estimate bytes from live tokens and reserve memory for weights and workspaces. Lower-precision or paged cache allocation is an alternative, but each has numerical or allocator trade-offs.

### Evaluation
Score 4 points: 1 for the correct 512 MiB estimate and assumptions; 1 for distinguishing reused projections from remaining attention work; 1 for a valid cached-versus-reference test; 1 for identifying cache identity and concurrency memory risks.

### Follow-up
If latency still rises with context despite caching, which measurements distinguish attention cost from memory pressure or kernel effects?

### Common Mistake
Claiming that caching makes each decode step constant-time or consumes no additional memory.

### Concept
[KV Cache](../concepts/23-kv-cache.md)

## Attention Optimization

### Prompt
You serve long-context requests and can choose either an exact tiled attention kernel or local attention with a window of 256 tokens. Compare their semantic guarantees, memory/computation effects, and the evaluation evidence needed before choosing local attention.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
An exact tiled/fused kernel computes the same full attention result up to floating-point ordering differences while avoiding materializing the full sequence-by-sequence score matrix; it can reduce peak intermediate memory and memory traffic, though speed depends on hardware and kernel quality. Local attention limits each token to a neighborhood, reducing score work from quadratic in sequence length toward `O(nw)` for window width `w`, but it changes which context can influence the output and may lose long-range evidence. I would first compare outputs from the exact kernel to a trusted implementation within a numeric tolerance, then measure peak memory and latency on the target hardware and sequence lengths. For local attention, I would use a fixed long-context task set with distant dependencies, compare quality by length and category, and monitor tail latency. A hybrid/global-token design or exact fused attention is an alternative when distant context matters.

### Evaluation
Score 4 points: 1 for explaining exact kernel equivalence; 1 for identifying local attention's changed receptive field and `O(nw)` work; 1 for proposing target-hardware resource measurements; 1 for a long-range quality test and a suitable fallback.

### Follow-up
What test would detect a failure that appears only when relevant evidence is farther away than the local window?

### Common Mistake
Treating sparse/local attention and an exact fused kernel as interchangeable optimizations with the same output semantics.

### Concept
[Attention Optimization](../concepts/24-attention-optimization.md)

## Context Window

### Prompt
Design or diagnose a production workflow that depends on Context Window. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Continuous Batching

### Prompt
A server has two decode slots. Requests A, B, and C arrive together with output lengths 3, 1, and 2 tokens. Show how iteration-level continuous batching admits C while A is still active, compare it with static batches, and identify production constraints the toy schedule omits.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
With static groups [A,B] then [C], the first group runs three ticks and the second two, so completion takes five ticks. Six request-iterations are useful and ten slot-iterations are allocated, for 60% slot utilization. With continuous scheduling, ticks 0, 1, and 2 run [A,B], [A,C], and [A,C]; C is admitted at tick 1 when B completes. The schedule takes three ticks, with six useful and six allocated slot-iterations (100% in this simplified trace). Real admission also depends on KV-cache blocks, prefill work, request compatibility, cancellation, and fairness. I would compare throughput, TTFT, inter-token latency and p95/p99 by request length and tenant, while tracking active sequences, KV occupancy, preemption, and queue age. An alternative is static batching, which is simpler and can work well for uniform short outputs; continuous scheduling adds bookkeeping and may raise queueing or memory pressure.

### Evaluation
Score 4 points: 1 for admitting C after B completes at fixed capacity; 1 for correct static and continuous slot calculations; 1 for naming KV and queue/fairness constraints; 1 for latency and throughput metrics broken down by workload slice.

### Follow-up
How would an admission policy change if C's prompt required more KV blocks than were free after B completed?

### Common Mistake
Comparing useful slot work in one schedule with allocated slot work in another, or assuming capacity alone guarantees admission.

### Concept
[Continuous Batching](../concepts/26-continuous-batching.md)

## Speculative Decoding

### Prompt
Design or diagnose a production workflow that depends on Speculative Decoding. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Inference Optimization

### Prompt
Design or diagnose a production workflow that depends on Inference Optimization. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Token Optimization

### Prompt
Design or diagnose a production workflow that depends on Token Optimization. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Quantization (for LLMs)

### Prompt
Quantize weights `[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]` using a fixed symmetric scale of 0.5 and a signed three-bit code range from -2 to 2. Show codes, clipping, mean squared reconstruction error, and what additional evidence is required before deploying a low-bit model.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
Using `q = clip(round(x / 0.5), -2, 2)`, the codes are `[-2,-1,0,1,2,2]` and dequantized values are `[-1,-0.5,0,0.5,1,1]`. The value 1.5 is clipped. Squared error is 0.25 for that value and zero for the others, so MSE is `0.25 / 6 = 1/24 ≈ 0.04167`. Five distinct codes require at least three bits per value. Six values need 24 bytes as fp32; the ideal three-bit bit count is 18 bits, or 2.25 byte-equivalents (10.67x smaller). A byte-aligned packed buffer needs at least 3 bytes, before scale metadata and runtime state. This toy says nothing about task quality or speed. Deployment needs a fixed calibration set, comparison to the same full-precision checkpoint on held-out tasks and sensitive slices, plus target-hardware latency and peak memory. If clipping hurts outliers, per-channel/group scales or a higher bit width are alternatives; specialized kernels may also make a nominally smaller model slower.

### Evaluation
Score 4 points: 1 for correct codes and clipped value; 1 for MSE and ideal storage calculation; 1 for distinguishing toy reconstruction from task quality; 1 for a fixed held-out and hardware validation plan.

### Follow-up
How would you determine whether to change the fixed calibration scale or use per-group scales when a rare outlier causes regressions?

### Common Mistake
Deriving the scale from the evaluation weights and then claiming the same test demonstrates clipping, or assuming fewer bits guarantees lower latency.

### Concept
[Quantization (for LLMs)](../concepts/30-quantization.md)

## Multimodal

### Prompt
Design or diagnose a production workflow that depends on Multimodal. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Evaluation (for LLMs)

### Prompt
On six moderation examples, the gold labels are `[1,1,1,1,0,0]`; System A predicts `[1,1,1,0,1,0]`, and System B predicts `[1,1,0,0,0,0]`. Calculate confusion counts, accuracy, precision, recall, and F1 for both. Explain why accuracy does not select a winner and propose a stronger LLM evaluation plan.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
System A has TP=3, FP=1, FN=1, TN=1; accuracy=4/6, precision=3/4, recall=3/4, and F1=0.75. System B has TP=2, FP=0, FN=2, TN=2; accuracy=4/6, precision=1, recall=1/2, and F1=2/3. Equal accuracy hides different false-positive/false-negative trade-offs, so the operating choice depends on error costs. For an LLM, freeze a representative held-out set and compare the candidate with a baseline under identical prompt, model, and decoding conditions. Report task metrics and slice results, inspect paired disagreements, and include human rubric or blinded preference ratings where outputs are open-ended. Check label quality, contamination, uncertainty, safety, latency, and cost; choose thresholds on validation data rather than the final test set.

### Evaluation
Score 4 points: 1 for System A counts/metrics; 1 for System B counts/metrics; 1 for explaining the equal-accuracy trade-off; 1 for a frozen paired evaluation with slices and human review where appropriate.

### Follow-up
If false negatives cost five times as much as false positives, what additional validation and threshold-selection evidence do you need before choosing a system?

### Common Mistake
Reporting one aggregate score without examining error types, subgroup behavior, or uncertainty.

### Concept
[Evaluation (for LLMs)](../concepts/32-evaluation.md)

## Vision Transformers (ViT)

### Prompt
Design or diagnose a production workflow that depends on Vision Transformers (ViT). Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Multimodal Fusion

### Prompt
Design or diagnose a production workflow that depends on Multimodal Fusion. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Model Interpretability

### Prompt
Design or diagnose a production workflow that depends on Model Interpretability. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Adversarial Robustness

### Prompt
Design or diagnose a production workflow that depends on Adversarial Robustness. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Knowledge Distillation

### Prompt
Design or diagnose a production workflow that depends on Knowledge Distillation. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Neural Architecture Search

### Prompt
Design or diagnose a production workflow that depends on Neural Architecture Search. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Long-Context Handling

### Prompt
Design or diagnose a production workflow that depends on Long-Context Handling. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Retrieval Systems

### Prompt
Design or diagnose a production workflow that depends on Retrieval Systems. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Prompt Injection Security

### Prompt
Design or diagnose a production workflow that depends on Prompt Injection Security. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Model Editing

### Prompt
Design or diagnose a production workflow that depends on Model Editing. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Mixture of Experts (MoE)

### Prompt
Design or diagnose a production workflow that depends on Mixture of Experts (MoE). Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.

## Efficient Attention

### Prompt
Design or diagnose a production workflow that depends on Efficient Attention. Explain its key assumptions, resource costs, and one alternative you would consider.

### Role/Domain
LLM engineering; explain the design to a senior interviewer.

### Difficulty
Intermediate

### Expected Answer
A strong answer names the core mechanism, states assumptions, and connects the design choice to measurable quality, latency, cost, or safety requirements.

### Evaluation
Score correctness, explicit trade-offs, a measurable validation plan, and treatment of the stated failure mode.

### Follow-up
What evidence would make you choose the alternative, and what would you monitor after launch?

### Common Mistake
Describing the technique without identifying its assumptions, evaluation method, or operational failure mode.
