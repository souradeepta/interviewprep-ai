# Repo Comprehensive Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform every stub, placeholder, and thin section in the interviewprep-ml repo into production-quality, interview-ready content with real code, real diagrams, and SDE3-depth analysis.

**Architecture:** Five independent sub-projects each targeting a distinct gap: (A) 60 stub files in modern-ai/36-55, (B) 18 system-design patterns needing SDE3 depth, (C) 27 arch-review systems missing Mermaid diagrams, (D) coding section with only 1 file, (E) INDEX and navigation improvements that tie everything together.

**Tech Stack:** Python, numpy, torch, matplotlib (notebooks); Mermaid (diagrams); Markdown

---

## Gap Inventory

| Sub-Project | Files Affected | Current State | Target State |
|-------------|---------------|--------------|-------------|
| A — modern-ai stubs | 60 files (20 concepts + 20 notebooks + 20 implementations) | Placeholder text, 11 code lines per notebook | 800-1200 word concepts, 600-900 line notebooks, 150-250 line implementations |
| B — system-design SDE3 | 18 patterns (14-31) | Basic TL;DR with tables | 1800-2400 words with failure scenarios, cost models, 8-10 Q&A |
| C — arch-review diagrams | 81 diagram files for 27 systems | Missing entirely | 3 Mermaid diagram files per system (architecture, app flow, process flow) |
| D — coding expansion | 20 new files | 1 file (arrays-strings.md) | 21 algorithm/DS topic files with problems, patterns, code |
| E — navigation | README, INDEX, roadmaps | Incomplete cross-links | Full cross-links, concept count, difficulty labels |

---

## Sub-Project A: Modern-AI Stubs (36-55)

**Priority: CRITICAL** — 20 concepts, 20 notebooks, 20 implementations are all placeholder shells.

### Concept Content Specs (per topic)

Each concept `.md` must have these 8 sections:
1. **Detailed Explanation** (150-250 words): what it is, why it matters, production context
2. **Core Intuition** (2-3 sentences max): memorable analogy
3. **How It Works** (4-6 numbered steps + Mermaid graph)
4. **Architecture / Trade-offs** (2+ comparison tables)
5. **Interview Q&A** (5-8 questions, judgment-focused)
6. **Best Practices** (5-8 bullets with numbers/ranges)
7. **Common Pitfalls** (3-5 with symptoms and fixes)
8. **Related Concepts** (3-5 links)

### Notebook Specs (per topic)

16 cells: title+objectives | imports+device | Level1 markdown | Level1 code (50-80 lines numpy) | Level2 markdown | Level2 torch (100-140 lines) | RW1 markdown | RW1 code (60-90 lines) | RW2 markdown | RW2 code | RW3 markdown | RW3 code | Comparison markdown | Comparison code+matplotlib | Takeaways markdown | Exercises markdown

### Topic-Specific Content Guide

#### 36 — Token Pruning and Merging
**Algorithm**: Compute importance score per token: `s_i = ||Δh_i||` (gradient norm or attention rollout). Prune tokens where `s_i < threshold`. ToMe merges similar tokens via bipartite matching: `merge(t_i, t_j)` if `cosine_sim(t_i, t_j) > θ`. Result: 2-5x speedup, <2% accuracy drop.
- Level 1 (numpy): importance scoring on synthetic token sequence, threshold pruning
- Level 2 (torch): ToMe bipartite soft matching, speed/accuracy tradeoff benchmark
- RW1: ViT image token pruning — reduce 196 patches → 64 via attention rollout
- RW2: LLM prompt compression — prune low-importance input tokens before generation
- RW3: Adaptive threshold — auto-tune threshold targeting latency SLA
- Comparison: Full tokens vs ToMe-r4 vs ToMe-r8 vs hard pruning throughput/quality
- Key pitfall: Position embeddings break after merging — must use relative positions

#### 37 — Adaptive Layer Selection
**Algorithm**: Early-exit classifiers at every N layers. Exit if `max(softmax(h_l·W_exit)) > confidence_threshold`. DeeBERT: add exit heads at layers 4,6,8,10 of 12-layer BERT. Average exit layer = 6 → 2x speedup. Hard inputs use all 12 layers.
- Level 1: Confidence-based exit simulation in numpy
- Level 2: Multi-exit BERT-style model in torch with exit heads, per-layer tracking
- RW1: Text classification with early exit — calibration across 4 exit points
- RW2: Adaptive compute budget — set compute target, let model auto-select exit layers
- RW3: Anytime inference — stream partial predictions as layers complete
- Key pitfall: Exit classifiers need separate training; naive joint training overconfident at early exits

#### 38 — Layer Skipping
**Algorithm**: Router predicts skip mask `m_l ∈ {0,1}` per layer. If `m_l=0`, skip layer `l` and pass residual unchanged: `h_{l+1} = h_l`. Skippable layers identified by: low gradient norm during training, high cosine similarity `cos(h_{l-1}, h_l) > 0.95`. Speed: O(1-skip_rate) computation.
- Level 1: Residual bypass simulation — measure output delta from skipping each layer
- Level 2: Learned skip router in torch — binary skip decision per layer, gradient analysis
- RW1: Latency-constrained inference — skip layers until budget met
- RW2: Layer importance analysis — rank layers by sensitivity, identify safe-to-skip
- RW3: Progressive skipping during generation — skip more on later tokens (less uncertainty)
- Key pitfall: Skipping attention layers loses positional context; safer to skip FFN sublayers

#### 39 — Router Learning
**Algorithm**: Router is a lightweight 2-layer MLP trained end-to-end with auxiliary load-balancing loss. Input: token hidden state `h`. Output: expert logits. Training: `L_total = L_task + α·L_balance` where `L_balance = Σ_e (f_e - 1/N_experts)²`. Router regularization prevents all tokens choosing expert 0.
- Level 1: Gating network + expert dispatch in numpy
- Level 2: Differentiable router with load-balancing loss in torch, expert utilization tracking
- RW1: Router collapse analysis — detect when all tokens go to 1 expert, fix with α tuning
- RW2: Heterogeneous experts — route by token type (numbers→math expert, syntax→language expert)
- RW3: Continual router updating — fine-tune router for new domain without retraining experts
- Key pitfall: Router must be on same device as first expert; multi-GPU routing needs careful placement

#### 40 — Beam Search Optimization
**Algorithm**: Maintain top-K sequences (beam width K). At each step: expand each beam by vocab, score with `log p(y_t|y_{<t}, x)`, keep top-K. Diverse beam search adds diversity penalty: `score(y_t) = log_p - λ·sim(y_t, prev_beams)`. Batch beam search: process K×batch simultaneously.
- Level 1: Basic beam search in numpy on toy vocabulary
- Level 2: Vectorized batch beam search in torch, length normalization, early stopping
- RW1: Code generation beam search — k=5, penalize duplicate tokens
- RW2: Diverse beam search — generate K diverse summaries from same document
- RW3: Constrained beam search — force output to contain required keywords via score bonus
- Key pitfall: Beam search with large vocab + large batch → OOM; use token-by-token generation with pruning

#### 41 — Grammar-Constrained Generation
**Algorithm**: Build token mask `M_t ∈ {0,1}^|V|` from grammar state after seeing tokens `y_{<t}`. At sampling step: `logits_t = logits_t + log(M_t)` (mask invalid tokens with -inf). Grammar state machine tracks valid next tokens via FSM transition table. Supports JSON schema, regex, context-free grammars (Earley parser).
- Level 1: Regex-constrained sampling with finite state machine in numpy
- Level 2: JSON schema grammar compilation to token mask, torch integration
- RW1: SQL generation with grammar constraint — force valid SQL syntax
- RW2: Structured extraction — generate JSON with required fields guaranteed
- RW3: Code generation — enforce Python indent/syntax via token masking
- Key pitfall: Grammar compilation to token mask is O(|grammar| × |vocab|); precompile and cache per schema

#### 42 — Mixed-Bit Quantization
**Algorithm**: Sensitivity analysis: compute `Δ_l = ||ΔL / ΔW_l||` for each layer. Assign INT4 to insensitive layers (`Δ_l < τ`), INT8 to sensitive ones (`Δ_l ≥ τ`). GPTQ-style layer-wise: minimize `||WX - W_q·X||²` by solving optimal rounding. Memory: INT4 = 4GB for 7B, INT8 = 7GB for 7B (vs FP16 = 14GB).
- Level 1: Sensitivity-based bit assignment in numpy, layer-by-layer error analysis
- Level 2: Mixed 4/8-bit quantization in torch with per-layer calibration, memory estimation
- RW1: 7B model bit assignment — identify which layers tolerate 4-bit vs need 8-bit
- RW2: Quality-memory Pareto frontier — sweep τ threshold, plot accuracy vs memory
- RW3: Dynamic bit allocation at inference — assign bits based on activation variance
- Key pitfall: First and last transformer layers are most sensitive; always keep them INT8 or FP16

#### 43 — Embedding Quantization
**Algorithm**: Embeddings stored as INT8: `e_q = clamp(round(e / scale), -128, 127)`. Scale = `max(|e|) / 127`. Lookup: dequantize during forward pass. For retrieval embeddings: binary quantization `b_i = sign(e_i)` → Hamming distance. 32x compression, 50x faster similarity. Fine-grain: per-dimension scale improves quality.
- Level 1: INT8 embedding table with scale factors in numpy, reconstruction error
- Level 2: Binary embedding quantization in torch, Hamming vs cosine retrieval benchmark
- RW1: RAG vector store compression — INT8 embeddings for 1M documents, memory savings
- RW2: Product quantization — split embedding into 8 sub-spaces, quantize each
- RW3: Online quantization calibration — update scales during inference with EMA
- Key pitfall: Quantizing query embeddings during retrieval loses more than doc embeddings; quantize docs only

#### 44 — Token Decompression
**Algorithm**: Compressed representation: encode frequent token spans as single "mega-tokens". Frequency table: `freq[span] = count(span) / total_tokens`. Compress span if `len(span)/freq[span] > threshold`. Decompress: lookup table expansion at inference. 20-40% token reduction on code/structured text.
- Level 1: Byte-pair encoding-style compression in numpy on text corpus
- Level 2: Span-level tokenization with compression ratio analysis in torch
- RW1: Code tokenization — compress common Python patterns (def, for x in range)
- RW2: JSON compression — mega-tokens for schema keys and fixed patterns
- RW3: Adaptive compression — learn token spans per domain with different threshold
- Key pitfall: Decompression overhead at inference can negate savings; must pre-decompress KV cache entries

#### 45 — Attention Pattern Learning
**Algorithm**: Train attention to focus on relevant positions by adding auxiliary loss: `L_att = λ·KL(A||A_target)` where `A_target` is label-informed attention (from entity spans or dependency parse). Sparse attention learning: add differentiable top-k mask via straight-through estimator. Produces interpretable, focused attention heads.
- Level 1: Attention visualization and entropy analysis in numpy
- Level 2: Guided attention training with auxiliary loss in torch, head specialization tracking
- RW1: Named entity recognition — train heads to attend entity tokens
- RW2: Summarization — train global-token heads to accumulate document semantics
- RW3: QA — train span-extraction heads to focus on answer context
- Key pitfall: Auxiliary attention loss interferes with main task if λ > 0.1; anneal from 0.01 → 0.001

#### 46 — Neuron Importance Scoring
**Algorithm**: Score neuron `i` in layer `l`: `s_i = Σ_{x∈D} |W_i · h_i(x)|` (magnitude × activation). Or gradient-based: `s_i = |∂L/∂h_i|`. Wanda pruning: `score = |W| × ||X||`. Prune bottom-k% neurons. Structured pruning: prune whole heads/rows for actual speedup vs unstructured (theoretical only).
- Level 1: Magnitude-based neuron scoring in numpy, threshold pruning
- Level 2: Wanda-style structured pruning in torch, head-level pruning, latency measurement
- RW1: One-shot pruning — prune 30% neurons in single pass using calibration data
- RW2: Iterative magnitude pruning — prune 5% per round, fine-tune, repeat × 6
- RW3: Task-specific pruning — prune different neurons for classification vs generation
- Key pitfall: Unstructured pruning shows no speedup on GPU without sparse kernels; always prune structured (rows/heads)

#### 47 — Dynamic Batching
**Algorithm**: Accumulate requests in queue. Batch when `queue.size ≥ min_batch OR wait_time ≥ max_wait_ms`. Variable-length: pad to `max(len(reqs_in_batch))` or use packing (FlashAttention). Continuous batching: add/remove sequences mid-forward-pass at token boundaries. Optimal batch size: `B* = argmax(throughput(B))` subject to `latency(B) ≤ SLA`.
- Level 1: Request queue with timeout-based batching in numpy/asyncio simulation
- Level 2: Continuous batching simulator in Python — arriving requests, mid-batch insertion
- RW1: Throughput optimization — find optimal batch size for different sequence lengths
- RW2: SLA-aware batching — maximize batch while keeping p99 < 100ms
- RW3: Predictive batching — use predicted decode length to group similar-length requests
- Key pitfall: Variable-length padding wastes GPU; use FlashAttention with packed sequences or sort by length

#### 48 — Model Cascading
**Algorithm**: Route requests: small model first (0.5B), if `confidence > θ_high` return result. If `confidence < θ_low` escalate to large model (70B). Else defer (send both). Routing calibration: `θ_low, θ_high = calibrate(val_set, target_accuracy=0.95)`. Cost savings: `C = p_small × C_small + p_large × C_large` where `p_small ≈ 0.7-0.85`.
- Level 1: Threshold-based cascade router in numpy, confidence estimation
- Level 2: Two-model cascade with calibration in torch, accuracy/cost tradeoff analysis
- RW1: Customer support routing — 80% handled by 7B, 20% escalate to 70B
- RW2: Code generation cascade — simple functions → small, complex APIs → large
- RW3: Adaptive threshold tuning — adjust θ based on real-time accuracy feedback
- Key pitfall: Cascade latency = small_model_time + routing_overhead + (escalation_rate × large_model_time); measure full pipeline

#### 49 — Latency SLA Prediction
**Algorithm**: Features: `[prompt_len, log_perplexity, token_types, time_of_day]`. Predict percentile bins (20 bins). Soft labels: Gaussian smoothing around true bin. Calibration: isotonic regression on validation. P75 reservation (not mean) prevents OOM. Use predicted latency to: set timeouts, batch by predicted length, reject if would miss SLA.
- Level 1: Linear regression for output length prediction from prompt features
- Level 2: 20-bin classifier with Gaussian soft labels, P75 reservation, calibration curve
- RW1: Dynamic timeout management — per-request timeout from predicted output length
- RW2: Batch grouping by prediction — reduces padding waste by 30-40%
- RW3: SLA admission control — reject requests predicted to miss SLA before queuing
- Key pitfall: Heavy-tailed distribution; mean prediction leads to OOM on long outputs; always use P75 for KV reservation

#### 50 — Cache-Aware Scheduling
**Algorithm**: Prefix hash: `h = hash(tokens[:k])` for system prompt. Group requests by shared prefix into same batch. Schedule same-prefix requests consecutively to maximize KV cache reuse. Cache eviction: LRU with ref counting. Hit rate target: >60% on typical RAG workloads (same system prompt shared by all requests).
- Level 1: LRU cache with prefix hashing in Python
- Level 2: Prefix-aware batch scheduler in Python — greedy grouping by longest shared prefix
- RW1: RAG workload caching — 100K requests sharing 10 system prompts, cache hit analysis
- RW2: Multi-tenant scheduling — balance cache reuse vs fairness across tenants
- RW3: Cache warming — pre-populate system prompt prefixes at startup
- Key pitfall: Hash collisions cause false cache hits with corrupt KV; use SHA-256 not CRC32

#### 51 — Distributed Inference
**Algorithm**: Tensor parallelism (TP): split attention heads across GPUs, `h_i → GPU_{i mod P}`. Pipeline parallelism (PP): assign layers to GPUs, `layers[0:L/P] → GPU_0`. Hybrid: TP within node (NVLink), PP across nodes (IB). Optimal: TP-degree = NVLink bandwidth / attention compute. PP-degree = num_nodes. Bubble overhead in PP: `B = (P-1)/(M+P-1)` where M = micro-batches.
- Level 1: Tensor split simulation in numpy — attention head partitioning
- Level 2: Pipeline parallel forward pass in torch with micro-batching, bubble analysis
- RW1: 70B model on 8×A100 — TP=8 within node, latency vs throughput
- RW2: Cross-node PP — 2 nodes × 4 GPUs, PP=2 TP=4, bubble mitigation
- RW3: Expert parallelism for MoE — route tokens to expert GPUs, AllReduce pattern
- Key pitfall: PP bubble overhead dominates at small batch; use micro-batching (M≥4) to amortize bubble

#### 52 — Heterogeneous Ensemble
**Algorithm**: Combine diverse models: `output = Σ_m w_m · f_m(x)`. Weights `w_m` learned via Bayesian model combination or simple stack (LR on held-out set). Diversity maximization: maximize `Σ_{i,j} cos_dist(f_i, f_j)`. Routing ensemble: classify input → route to specialist model.
- Level 1: Weighted ensemble of numpy classifiers, diversity metric
- Level 2: Stacking ensemble in torch — meta-learner trained on base model outputs
- RW1: Multi-modal ensemble — vision + text models combined via learned routing
- RW2: Online weight update — adjust ensemble weights based on rolling accuracy
- RW3: Cost-aware ensemble — weight by accuracy/cost Pareto
- Key pitfall: Correlation kills ensemble benefit; always measure pairwise prediction disagreement before adding a new member

#### 53 — Conditional Computation
**Algorithm**: Compute features `c = f_cond(x)`. Route to branch based on condition: token type, domain, input complexity. Sparse gating: `gate = softmax(W·c)`, compute only top-k branches. SkipNet: per-sample bypass via reinforcement learning gating. Savings: O(skip_rate × compute_per_layer).
- Level 1: Decision-tree style conditional routing in numpy
- Level 2: Soft conditional gating with Gumbel-softmax in torch, branch utilization tracking
- RW1: Domain routing — technical vs casual text → different processing paths
- RW2: Complexity routing — simple queries → shallow path, complex → deep
- RW3: Hardware-adaptive routing — route to branches sized for available compute budget
- Key pitfall: Training conditional computation requires careful gradient flow; use REINFORCE or Gumbel-softmax for binary gates

#### 54 — Context Compression
**Algorithm**: Extractive: select top-K sentences by attention weight or BM25 score. Abstractive: use small compressor model: `summary = compressor(context)`. LLMLingua token pruning: iterative token deletion minimizing `Δperplexity`. Compression ratio: 4-10x. RALMs: compress retrieved docs to 10% size before generation.
- Level 1: BM25-based extractive compression in numpy
- Level 2: Attention-based token selection in torch, perplexity-preserving pruning
- RW1: RAG compression — compress 10 retrieved 1000-token docs to 200 tokens before generation
- RW2: Long conversation compression — compress chat history while preserving key facts
- RW3: Adaptive compression — compress more aggressively when context window is nearly full
- Key pitfall: Abstractive compression loses exact quotes/numbers; use extractive for factual domains, abstractive for summaries

#### 55 — Online Knowledge Distillation
**Algorithm**: Teacher and student train together: `L = α·CE(student, labels) + (1-α)·T²·KL(student_T || teacher_T)` where T=temperature, α=0.1-0.5. Online: teacher updates alongside student (no frozen teacher). Deep Mutual Learning (DML): two peers distill from each other simultaneously. Born-Again Networks: student becomes teacher for next generation.
- Level 1: Basic KD loss computation in numpy, temperature sweep
- Level 2: Online mutual learning in torch — two models training together, peer distillation
- RW1: Progressive model compression — 3B student learns from 7B peer during training
- RW2: Born-again networks — student trained, then used as teacher for smaller student
- RW3: Ensemble distillation — multiple teachers distilled into single student simultaneously
- Key pitfall: Online teacher is not better-calibrated than student initially; warm up teacher for 1-2 epochs before turning on distillation loss

---

### Task A1: modern-ai concepts 36-45 (batch 1)

**Files:**
- Modify: `modern-ai/concepts/36-token-pruning-merging.md`
- Modify: `modern-ai/concepts/37-adaptive-layer-selection.md`
- Modify: `modern-ai/concepts/38-layer-skipping.md`
- Modify: `modern-ai/concepts/39-router-learning.md`
- Modify: `modern-ai/concepts/40-beam-search-optimization.md`
- Modify: `modern-ai/concepts/41-grammar-constrained-generation.md`
- Modify: `modern-ai/concepts/42-mixed-bit-quantization.md`
- Modify: `modern-ai/concepts/43-embedding-quantization.md`
- Modify: `modern-ai/concepts/44-token-decompression.md`
- Modify: `modern-ai/concepts/45-attention-pattern-learning.md`

- [ ] **Step 1: Replace concept 36 (token-pruning-merging)** — overwrite with 8-section content. Core content:
  ```
  ## Detailed Explanation (150-200 words)
  Token pruning removes low-importance tokens during inference; token merging fuses
  similar tokens into shared representations. Both reduce sequence length before or 
  during attention computation, cutting quadratic O(n²) attention cost. ToMe (Token 
  Merging) achieves 2-5x throughput improvement with <2% accuracy degradation on ViTs 
  and LLMs by merging redundant patch/token representations...

  ## Core Intuition
  Imagine a meeting where 5 people keep repeating the same point. Merging them into 
  one voice doesn't change the meeting's outcome, but cuts the runtime. Token merging 
  does the same: redundant tokens are fused into one before attention runs.

  ## How It Works
  1. Compute token importance scores (gradient norm, attention rollout, activation magnitude)
  2. Identify redundant token pairs via cosine similarity > threshold
  3. Merge similar tokens via bipartite matching: average (k_i+k_j)/2, sum values
  4. Run attention on compressed sequence (2-5x fewer tokens)
  5. Unmerge at output layer if spatial resolution needed (e.g., detection)
  
  ## Architecture / Trade-offs
  | Method | Speed | Quality Drop | Reversible | Use Case |
  ...
  ```

- [ ] **Step 2: Replace concepts 37-45** in sequence using the algorithm specs in this plan

- [ ] **Step 3: Verify all 10 files have 8 sections and >800 words each**
  ```bash
  for f in modern-ai/concepts/3[6-9]-*.md modern-ai/concepts/4[0-5]-*.md; do
    words=$(wc -w < "$f")
    echo "$words $f"
  done
  ```

- [ ] **Step 4: Commit**
  ```bash
  git add modern-ai/concepts/3[6-9]-*.md modern-ai/concepts/4[0-5]-*.md
  git commit -m "feat: replace stub concepts 36-45 with full 8-section content"
  ```

---

### Task A2: modern-ai concepts 46-55 (batch 2)

**Files:** `modern-ai/concepts/46-*.md` through `modern-ai/concepts/55-*.md`

- [ ] **Step 1: Replace concept 46-55** using algorithm specs from this plan (same 8-section format as A1)

- [ ] **Step 2: Verify word counts**
  ```bash
  for f in modern-ai/concepts/4[6-9]-*.md modern-ai/concepts/5[0-5]-*.md; do
    wc -w "$f"
  done
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add modern-ai/concepts/4[6-9]-*.md modern-ai/concepts/5[0-5]-*.md
  git commit -m "feat: replace stub concepts 46-55 with full 8-section content"
  ```

---

### Task A3: modern-ai notebooks 36-45 (batch 1)

**Files:** `modern-ai/notebooks/36-*.ipynb` through `modern-ai/notebooks/45-*.ipynb`

Each notebook needs 16 cells reaching 600-900 code lines total. Current state: 11 code lines.

- [ ] **Step 1: Write notebook 36 (token-pruning-merging)**

  Cell 1 (markdown): Title + 4 objectives
  Cell 2 (code, ~20 lines):
  ```python
  import numpy as np
  import torch
  import torch.nn as nn
  import matplotlib.pyplot as plt
  import time
  np.random.seed(42); torch.manual_seed(42)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")
  ```
  Cell 3 (markdown): `## Level 1: Basic Token Importance Scoring`
  Cell 4 (code, ~70 lines): numpy implementation of importance scoring + threshold pruning
  ```python
  def compute_token_importance(attention_weights, method='rollout'):
      """attention_weights: [layers, heads, seq_len, seq_len]"""
      if method == 'rollout':
          rollout = np.eye(attention_weights.shape[-1])
          for layer in attention_weights:
              avg_heads = layer.mean(0)  # [seq, seq]
              rollout = avg_heads @ rollout
          return rollout[0]  # CLS token attention
      elif method == 'magnitude':
          return attention_weights.mean(axis=(0,1)).sum(axis=-1)
  
  # Simulate 12-layer, 12-head attention on 128 tokens
  attn = np.random.dirichlet(np.ones(128), size=(12, 12, 128))
  scores = compute_token_importance(attn)
  
  def prune_tokens(tokens, scores, keep_ratio=0.5):
      k = int(len(tokens) * keep_ratio)
      top_idx = np.argsort(scores)[-k:]
      return tokens[np.sort(top_idx)], top_idx
  
  tokens = np.random.randn(128, 64)  # 128 tokens, dim=64
  kept, idx = prune_tokens(tokens, scores, keep_ratio=0.5)
  print(f"Original: {tokens.shape}, Pruned: {kept.shape}")
  print(f"Compression: {1 - kept.shape[0]/tokens.shape[0]:.1%}")
  ```
  
  (Continue building cells 5-15 with Level 2 torch ToMe implementation, 3 RW examples, comparison plot, takeaways, exercises)

- [ ] **Step 2: Write notebooks 37-45** using topic-specific content from this plan

- [ ] **Step 3: Verify code line counts**
  ```python
  import json, glob
  for f in sorted(glob.glob('modern-ai/notebooks/3[6-9]-*.ipynb') + glob.glob('modern-ai/notebooks/4[0-5]-*.ipynb')):
      nb = json.load(open(f))
      code_lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type'] == 'code')
      status = '✅' if code_lines >= 600 else '⚠️ '
      print(f"{status} {f.split('/')[-1]}: {code_lines} lines")
  ```
  Expected: all ≥ 600 lines

- [ ] **Step 4: Commit**
  ```bash
  git add modern-ai/notebooks/3[6-9]-*.ipynb modern-ai/notebooks/4[0-5]-*.ipynb
  git commit -m "feat: replace stub notebooks 36-45 with real 600-900 line implementations"
  ```

---

### Task A4: modern-ai notebooks 46-55 (batch 2)

**Files:** `modern-ai/notebooks/46-*.ipynb` through `modern-ai/notebooks/55-*.ipynb`

- [ ] **Step 1: Write notebooks 46-55** (same 16-cell structure, topic content from this plan)
- [ ] **Step 2: Verify code line counts ≥ 600 each**
- [ ] **Step 3: Commit**
  ```bash
  git add modern-ai/notebooks/4[6-9]-*.ipynb modern-ai/notebooks/5[0-5]-*.ipynb
  git commit -m "feat: replace stub notebooks 46-55 with real implementations"
  ```

---

### Task A5: modern-ai implementations 36-55

**Files:** `modern-ai/implementations/36-*.py` through `modern-ai/implementations/55-*.py`
Current: 30 lines (stubs). Target: 150-250 lines standalone runnable scripts.

Each implementation should: import only numpy/torch, define 2-3 classes/functions, include `if __name__ == "__main__"` demo with timing, add `# Example output:` comment showing expected output.

- [ ] **Step 1: Write implementations 36-45** (150-250 lines each, self-contained)

  Example skeleton for 36-token-pruning-merging.py:
  ```python
  """Token Pruning and Merging — standalone implementation."""
  import numpy as np
  import torch
  import torch.nn as nn
  import time
  from typing import Tuple
  
  class ToMeLayer(nn.Module):
      """Token Merging layer. Reduces seq_len by r tokens per layer."""
      def __init__(self, dim: int, r: int = 8):
          super().__init__()
          self.r = r
          self.dim = dim
      
      def bipartite_soft_matching(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
          """Split tokens into src/dst, find best merge pairs."""
          B, N, C = x.shape
          src, dst = x[:, ::2], x[:, 1::2]
          # Cosine similarity between src and dst tokens
          sim = torch.bmm(
              nn.functional.normalize(src, dim=-1),
              nn.functional.normalize(dst, dim=-1).transpose(1, 2)
          )  # [B, N//2, N//2]
          # Greedy matching: each src token merges with most similar dst
          scores, indices = sim.max(dim=-1)  # [B, N//2]
          # Keep only top-r merges
          _, top_r = scores.topk(self.r, dim=-1)
          return src, dst, indices, top_r
      
      def forward(self, x: torch.Tensor) -> torch.Tensor:
          B, N, C = x.shape
          src, dst, merge_idx, top_r = self.bipartite_soft_matching(x)
          # Merge: average src tokens into their matched dst tokens
          merged = dst.clone()
          for b in range(B):
              for i in top_r[b]:
                  merged[b, merge_idx[b, i]] = (merged[b, merge_idx[b, i]] + src[b, i]) / 2
          # Unmerged src tokens stay
          mask = torch.ones(B, src.shape[1], dtype=torch.bool)
          for b in range(B):
              mask[b, top_r[b]] = False
          kept_src = src[mask.unsqueeze(-1).expand_as(src)].view(B, -1, C)
          return torch.cat([kept_src, merged], dim=1)
  
  def benchmark_token_merging(seq_len=256, dim=512, r=32, n_layers=4):
      """Compare standard attention vs ToMe attention latency."""
      ...
  
  if __name__ == "__main__":
      print("Token Pruning and Merging Demo")
      ...
  ```

- [ ] **Step 2: Write implementations 46-55**
- [ ] **Step 3: Verify line counts**
  ```bash
  for f in modern-ai/implementations/3[6-9]-*.py modern-ai/implementations/[45][0-9]-*.py; do
    wc -l "$f"
  done
  ```
- [ ] **Step 4: Commit**
  ```bash
  git add modern-ai/implementations/3[6-9]-*.py modern-ai/implementations/[45][0-9]-*.py
  git commit -m "feat: replace stub implementations 36-55 with 150-250 line standalone scripts"
  ```

---

## Sub-Project B: System-Design Patterns SDE3 Enhancement (14-31)

**Priority: HIGH** — Patterns 1-13 were enhanced to SDE3 level (1800-2400 words). Patterns 14-31 remain at basic depth (400-900 words).

### SDE3 Enhancement Format (per pattern)

Add these 4 sections to each existing file (do not replace — extend):

**Section: Detailed Trade-off Analysis**
- Quantitative comparison table (4-5 rows × 5-6 columns with actual numbers)
- Decision matrix (scenario → recommendation with reasoning + cost)

**Section: Production Failure Scenarios (3-4 scenarios)**
Format per scenario:
```
**Scenario N: [Title]**
**What breaks:** [specific system behavior]
**Why it happens:** [root cause]
**Detection:** [metrics + alert thresholds]
**Recovery:** [numbered steps with timelines]
**Prevention:** [3-4 specific mitigations]
```

**Section: Sophisticated Interview Q&A (6-8 questions)**
Format: `**QN: [judgment/decision question]?**`
`A: [context + reasoning + specific numbers]`

**Section: Cost & ROI Analysis**
- Specific cost model with real dollar numbers
- Break-even calculation

### Pattern-Specific Content

#### 14 — A/B Testing

**Trade-off table additions:**
```
| Configuration | Min Sample | Duration | Revenue Risk | Infra Cost |
|--------------|-----------|----------|-------------|-----------|
| 50/50 split | 1,000/var | 7-14 days | 50% × loss | Low |
| 90/10 split | 5,000/var | 14-30 days | 10% × loss | Low |
| Multi-armed bandit | 200-500/var | 1-3 days | 5-15% × loss | Medium |
| Canary (no stats) | N/A | 4-8h | 5% × loss | Low |
```

**Failure Scenarios:**
1. Novelty effect: Week 1 CTR +15%, Week 3 CTR -2%. Declare winner too early.
   - Detection: `if (test_duration < 7d) → warn("novelty bias risk")`
   - Fix: Minimum 14-day test, monitor week-over-week stability
2. Peeking problem: Check p-value daily, stop when p<0.05. False positive rate 22% (expected 5%).
   - Fix: Sequential probability ratio test (SPRT) or Bonferroni correction
3. Network effects: Social features where treatment users interact with control users. Contamination.
   - Fix: Cluster randomization (assign by user cluster, not individual)
4. Metric definition drift: CTR measured differently in A vs B (bug in logging). 8% "improvement" is measurement artifact.
   - Fix: Unified logging validation before test launch; log both model and metric pipeline version

**Q&A additions:**
- Q: Sample size calculator says 10,000 users. Traffic is 50/day. Run for 200 days?
  A: No — time decay is real. User behavior in month 1 ≠ month 7 (seasonal effects, product changes). Cap at 30 days. If traffic too low, accept higher MDE or test on more engaged segment.
- Q: A/B test shows +5% CTR but -2% conversion. Do you ship?
  A: Depends on business objective. If CTR is top metric, ship if delta>MDE. But regression in conversion needs explanation first — likely quality/relevance trade-off. Always define primary + guardrail metrics upfront.

#### 15 — Drift Detection

**Trade-off table:**
```
| Method | Sensitivity | Lag | Cost | Best For |
|--------|-----------|-----|------|---------|
| PSI (Population Stability Index) | Low | 1 day | Low | Feature drift |
| KS test | Medium | 1h | Low | Distribution shift |
| ADWIN (adaptive windowing) | High | Minutes | Medium | Concept drift |
| DDM (drift detection method) | High | 100 samples | Low | Accuracy drop |
| Evidently AI | Turnkey | 1 day | SaaS | Full monitoring |
```

**Failure Scenarios:**
1. Feature drift not caught: External API changes ZIP→ZIPPLUS4. Old feature = 0 for all new records. Model accuracy drops 15%. Drift detector never triggered.
   - Detection: Monitor `null_rate_per_feature`. Alert if null_rate[f] changes >5%.
   - Fix: Schema validation on every pipeline run
2. Gradual drift missed: KS threshold too high (0.2). Real drift score of 0.15 — just below threshold. Model degrades over 3 months undetected.
   - Fix: Use PSI + KS + accuracy metrics together. Track trend, not just threshold.
3. Spurious alerts on seasonal data: Christmas traffic triggers drift alert (different user mix). Deploy new model unnecessarily.
   - Fix: Seasonal baselines. Compare Nov 30 to last Nov 30, not yesterday.

#### 16 — Monitoring and Observability

**Failure Scenarios:**
1. Dashboard lag: Metrics aggregated hourly. Model breaks at 9am. Detected at 10am by an angry user.
   - Fix: Real-time metrics for error_rate and latency (1-minute granularity). Daily for accuracy.
2. Wrong SLO: SLO is "p50 latency < 100ms". p99 is 2000ms. 1% of users get terrible experience.
   - Fix: Always include p99/p999 in SLOs. p50 hides tail latency.
3. Alert fatigue: 50 alerts/day, team ignores them. Critical failure missed in noise.
   - Fix: Tiered alerts (P1/P2/P3). Only 1-3 P1 alerts should fire per month.

#### 17 — Model Debugging

**Failure Scenarios:**
1. Silent bias: Model performs well overall (AUC=0.92) but accuracy for Spanish-language input is 0.71 vs 0.94 overall. Not caught until regulatory audit.
   - Fix: Slice-based evaluation across language, age, geography before launch.
2. Training-serving skew: Train on normalized features (Z-score). Production normalizes with wrong mean/std (recalculated on 1000 samples vs full training set). Silent degradation.
   - Fix: Save normalization stats with model artifact. Serve-side validation.

#### 18-31 — (A/B, Drift, Monitoring, Debugging, Explainability, Interpretability, Feature Importance, Reproducibility, Cost Optimization, Production Readiness, Bias Detection, Fairness Metrics, Data Governance, ML Governance, Privacy-Preserving ML, Differential Privacy, Federated Learning, Disaster Recovery)

Each needs the same 4-section SDE3 treatment as above. See template in docs/superpowers/templates/sde3-pattern-enhancement-template.md.

---

### Task B1: SDE3-enhance patterns 14-17

**Files:**
- Modify: `system-design/patterns/14-ab-testing.md`
- Modify: `system-design/patterns/15-drift-detection.md`
- Modify: `system-design/patterns/16-monitoring-and-observability.md`
- Modify: `system-design/patterns/17-model-debugging.md`

- [ ] **Step 1: Append SDE3 sections to pattern 14 (A/B Testing)**
  Append after existing content:
  ```
  ## Detailed Trade-off Analysis
  [quantitative table from this plan]
  
  ## Production Failure Scenarios
  [4 scenarios from this plan]
  
  ## Sophisticated Interview Q&A
  [6-8 questions from this plan]
  
  ## Cost & ROI Analysis
  [cost model]
  ```

- [ ] **Step 2: Append SDE3 sections to patterns 15, 16, 17** (same format)

- [ ] **Step 3: Verify word counts ≥ 1800**
  ```bash
  for f in system-design/patterns/1[4-7]-*.md; do wc -w "$f"; done
  ```

- [ ] **Step 4: Commit**
  ```bash
  git add system-design/patterns/1[4-7]-*.md
  git commit -m "feat: SDE3-enhance system-design patterns 14-17 with failure scenarios, cost models, Q&A"
  ```

---

### Task B2: SDE3-enhance patterns 18-23

**Files:** `system-design/patterns/18-*.md` through `system-design/patterns/23-*.md`

- [ ] **Step 1: Enhance model-explainability (18)** — add SHAP/LIME comparison table, failure scenarios (SHAP explanations contradict each other due to correlated features), Q&A (how do you explain a model decision in <1 second at inference time?)
- [ ] **Step 2: Enhance interpretability (19)** — add interpretability vs explainability distinction, failure scenario (LIME neighborhood too large → misleading local explanation), Q&A
- [ ] **Step 3: Enhance feature-importance-tracking (20)** — add permutation vs SHAP vs gradient importance comparison, failure scenario (importance rankings flip with train/test split), cost model
- [ ] **Step 4: Enhance reproducibility (21)** — add reproducibility checklist table, failure scenario (different CUDA versions produce different results), Q&A on seeding distributed training
- [ ] **Step 5: Enhance cost-optimization (22)** — add 5-dimension cost lever table with ROI, failure scenario (distillation causes accuracy regression in production), Q&A on quantization trade-offs
- [ ] **Step 6: Enhance production-readiness (23)** — add 10-dimension readiness checklist, failure scenario (model passes offline tests, fails production load), Q&A on launch criteria
- [ ] **Step 7: Verify word counts ≥ 1800 each**
- [ ] **Step 8: Commit**
  ```bash
  git add system-design/patterns/1[8-9]-*.md system-design/patterns/2[0-3]-*.md
  git commit -m "feat: SDE3-enhance system-design patterns 18-23"
  ```

---

### Task B3: SDE3-enhance patterns 24-31

**Files:** `system-design/patterns/24-*.md` through `system-design/patterns/31-*.md`

Key content per pattern:

- **24 Bias Detection**: Demographic parity vs equalized odds vs calibration comparison table; failure scenario (bias metric passes at launch, breaks after demographic shift in user base); Q&A on fairness-accuracy trade-off quantification
- **25 Fairness Metrics**: Impossibility theorem table (3 metrics mutually exclusive); failure scenario (optimizing demographic parity hurts smaller groups); Q&A on which metric to use in regulated industries
- **26 Data Governance**: Data contract enforcement failure scenario (upstream schema change breaks pipeline silently); Q&A on data lineage vs catalog tools
- **27 ML Governance**: Model card enforcement failure scenario; audit trail gaps; Q&A on governance vs velocity trade-off
- **28 Privacy-Preserving ML**: k-anonymity vs differential privacy vs federated comparison table; failure scenario (k-anon fails with quasi-identifier combination); Q&A on privacy budget
- **29 Differential Privacy**: ε budget exhaustion failure scenario; noise calibration table; Q&A on ε value selection per use case
- **30 Federated Learning**: Data heterogeneity (non-IID) failure scenario (client drift with FedAvg); comparison: FedAvg vs FedProx vs Scaffold; Q&A on Byzantine fault tolerance
- **31 Disaster Recovery**: RTO/RPO target table per system tier; failure scenario (model registry backup fails silently); Q&A on rolling back ML model vs code together

- [ ] **Step 1-8:** Write and append SDE3 sections for each of patterns 24-31
- [ ] **Step 9: Verify word counts**
  ```bash
  for f in system-design/patterns/2[4-9]-*.md system-design/patterns/3[01]-*.md; do
    wc -w "$f"
  done
  ```
- [ ] **Step 10: Commit**
  ```bash
  git add system-design/patterns/2[4-9]-*.md system-design/patterns/3[01]-*.md
  git commit -m "feat: SDE3-enhance system-design patterns 24-31 with failure scenarios, cost models, Q&A"
  ```

---

## Sub-Project C: Arch-Review Diagram Completion

**Priority: MEDIUM** — 27 systems have no Mermaid diagram files (only 01, 02, 05 have diagrams).

### Diagram File Format (3 files per system)

**File 1: `NN-system-name-01-system-architecture.md`**
```markdown
# [System Name] — System Architecture

## Infrastructure and Deployment Architecture

\```mermaid
graph TD
  subgraph "Ingress"
    LB[Load Balancer]
  end
  subgraph "Application Layer"
    API[API Servers]
  end
  ...
\```
```

**File 2: `NN-system-name-02-application-architecture.md`**
```markdown
# [System Name] — Application Architecture

## Component Interaction and Data Flow

\```mermaid
graph LR
  Client --> APIGateway
  APIGateway --> ModelService
  ModelService --> VectorDB
  ...
\```
```

**File 3: `NN-system-name-03-process-flow.md`**
```markdown
# [System Name] — Process Flow

## Request Lifecycle

\```mermaid
sequenceDiagram
  User->>API: POST /predict
  API->>Cache: GET cached result
  Cache-->>API: miss
  API->>Model: run inference
  ...
\```
```

### Task C1: Diagrams for systems 03-10

**Files to Create (24 new files):**
- `arch-review/diagrams/03-llm-api-gateway-01-system-architecture.md`
- `arch-review/diagrams/03-llm-api-gateway-02-application-architecture.md`
- `arch-review/diagrams/03-llm-api-gateway-03-process-flow.md`
- (same pattern for systems 04, 06, 07, 08, 09, 10)

- [ ] **Step 1: Create system 03 (LLM API Gateway) diagrams**

  File 1 system-architecture:
  ```mermaid
  graph TD
    subgraph "Edge"
      CDN[CDN / WAF]
    end
    subgraph "Gateway Layer"
      LB[Load Balancer]
      GW1[API Gateway Pod 1]
      GW2[API Gateway Pod 2]
    end
    subgraph "Auth & Rate Limiting"
      Auth[Auth Service]
      RL[Rate Limiter - Redis]
    end
    subgraph "Model Routing"
      Router[Model Router]
      Cache[Semantic Cache - Redis]
    end
    subgraph "Model Backends"
      GPT4[OpenAI GPT-4]
      Claude[Anthropic Claude]
      OSS[Self-hosted Llama]
    end
    CDN --> LB
    LB --> GW1 & GW2
    GW1 & GW2 --> Auth --> RL --> Router
    Router --> Cache
    Cache -->|miss| GPT4 & Claude & OSS
  ```

  File 3 process-flow (sequenceDiagram):
  ```mermaid
  sequenceDiagram
    Client->>Gateway: POST /v1/chat {model, messages}
    Gateway->>Auth: validate API key
    Auth-->>Gateway: OK, user_id, tier
    Gateway->>RateLimiter: check(user_id, tier)
    RateLimiter-->>Gateway: tokens_remaining=95
    Gateway->>SemanticCache: lookup(hash(messages))
    SemanticCache-->>Gateway: MISS
    Gateway->>Router: route(model_preference, cost_budget)
    Router-->>Gateway: selected=gpt-4-turbo
    Gateway->>OpenAI: stream completion
    OpenAI-->>Gateway: SSE stream tokens
    Gateway-->>Client: SSE stream tokens
    Gateway->>Logging: log(latency, tokens, cost)
  ```

- [ ] **Step 2: Create diagrams for systems 04, 06, 07, 08, 09, 10** (3 files each)

- [ ] **Step 3: Verify all 24 files created**
  ```bash
  ls arch-review/diagrams/ | grep -E "^0[3-9]|^10" | wc -l
  ```
  Expected: 24

- [ ] **Step 4: Commit**
  ```bash
  git add arch-review/diagrams/0[3-9]-* arch-review/diagrams/10-*
  git commit -m "feat: add Mermaid architecture diagrams for arch-review systems 03-10"
  ```

---

### Task C2: Diagrams for systems 11-20

- [ ] **Step 1-2: Create 3 diagram files each for systems 11-20**
  Key diagrams per system:
  - 11 (autonomous research agent): agent loop sequence diagram, tool call orchestration
  - 12 (multi-agent software dev): multi-agent coordination with planner/executor/reviewer
  - 13 (customer automation): multi-channel ingress + intent router + escalation flow
  - 14 (data analysis agent): code interpreter sandbox + data pipeline
  - 15 (supply chain): multi-horizon forecasting pipeline + constraint optimization
  - 16 (financial analysis): market data ingestion + signal aggregation + risk guard
  - 17 (security threat): real-time event stream + threat classifier + auto-response
  - 18 (document processing): OCR + extraction pipeline + validation
  - 19 (content creation): multi-agent creative pipeline with editor/reviewer
  - 20 (db query): NL → SQL agent with schema retrieval + execution sandbox

- [ ] **Step 3: Verify 30 files created**
- [ ] **Step 4: Commit**
  ```bash
  git add arch-review/diagrams/1[1-9]-* arch-review/diagrams/20-*
  git commit -m "feat: add Mermaid architecture diagrams for arch-review systems 11-20"
  ```

---

### Task C3: Diagrams for systems 21-30

- [ ] **Step 1-2: Create 3 diagram files each for systems 21-30**
  Key diagrams:
  - 21 (fraud detection): feature pipeline + dual-model ensemble + decisioning
  - 22 (recommendation): two-tower candidate gen + reranking + A/B layer
  - 23 (video understanding): frame sampling + multimodal fusion + temporal reasoning
  - 24 (multimodal platform): vision encoder + LLM fusion + modality routing
  - 25 (observability platform): trace collection + anomaly detection + dashboarding
  - 26 (ecommerce): personalization pipeline + inventory + pricing agent
  - 27 (anomaly detection): statistical + ML hybrid + alert routing
  - 28 (news feed): NLP pipeline + engagement prediction + diversity sampling
  - 29 (speech-nlp): ASR + entity extraction + downstream NLP
  - 30 (llmops): model lifecycle management + evaluation + deployment orchestration

- [ ] **Step 3: Verify 30 files created**
- [ ] **Step 4: Commit**
  ```bash
  git add arch-review/diagrams/2[1-9]-* arch-review/diagrams/30-*
  git commit -m "feat: add Mermaid architecture diagrams for arch-review systems 21-30"
  ```

---

## Sub-Project D: Coding Section Expansion

**Priority: HIGH** — Only 1 file exists (`arrays-strings.md`). Missing 20+ critical interview topics.

### File Format (per topic)

```markdown
# [Topic]

## TL;DR
[2-3 sentences on what to master and why]

## Core Concepts
[data structure properties, time/space complexity table]

## Key Patterns
[2-5 patterns with brief description]

## Essential Problems

| Problem | Approach | Time | Space |
|---------|----------|------|-------|
| ... | ... | ... | ... |

## Implementation Templates

### Template 1: [Pattern Name]
\```python
# Explanation of when to use
def template(arr):
    ...
\```

## Common Mistakes / Gotchas
[3-5 bullets with specific errors and fixes]

## Interview Quick-Reference
| Question | What to say |
|---------|------------|
| ... | ... |

## Related Topics
[links to related files]
```

### Task D1: Core data structures (5 files)

**Files to Create:**
- `coding/data-structures/linked-lists.md`
- `coding/data-structures/trees-and-bst.md`
- `coding/data-structures/graphs.md`
- `coding/data-structures/heaps-priority-queues.md`
- `coding/data-structures/hash-maps-sets.md`

- [ ] **Step 1: Create `linked-lists.md`**

  ```markdown
  # Linked Lists
  
  ## TL;DR
  Nodes with pointers. O(1) insert/delete if pointer given; O(n) random access.
  Master: reversal (iterative and recursive), cycle detection (Floyd's), merge operations.
  
  ## Core Concepts
  | Operation | Singly | Doubly |
  |-----------|--------|--------|
  | Access by index | O(n) | O(n) |
  | Prepend | O(1) | O(1) |
  | Append (with tail ptr) | O(1) | O(1) |
  | Delete (with node ptr) | O(n) — need prev | O(1) |
  | Search | O(n) | O(n) |
  
  ## Key Patterns
  - **Two pointers (slow/fast):** cycle detection, find middle, kth from end
  - **Dummy head node:** simplifies insert/delete at head (avoids null checks)
  - **Reverse in groups:** recurse/iterate in k-sized windows
  - **Merge pattern:** compare heads, advance smaller, link
  
  ## Implementation Templates
  
  ### Iterative Reversal
  \```python
  def reverse(head):
      prev, curr = None, head
      while curr:
          nxt = curr.next
          curr.next = prev
          prev = curr
          curr = nxt
      return prev
  \```
  
  ### Floyd's Cycle Detection
  \```python
  def has_cycle(head):
      slow = fast = head
      while fast and fast.next:
          slow = slow.next
          fast = fast.next.next
          if slow is fast:
              return True
      return False
  \```
  
  ### Find Cycle Entry (Floyd's phase 2)
  \```python
  def cycle_entry(head):
      slow = fast = head
      while fast and fast.next:
          slow, fast = slow.next, fast.next.next
          if slow is fast:
              break
      else:
          return None  # no cycle
      slow = head
      while slow is not fast:
          slow, fast = slow.next, fast.next
      return slow
  \```
  
  ## Essential Problems
  | Problem | Approach | Time | Space |
  |---------|----------|------|-------|
  | Reverse linked list | Two pointers (iterative) | O(n) | O(1) |
  | Detect cycle | Floyd's slow/fast | O(n) | O(1) |
  | Find cycle entry | Floyd's phase 2 | O(n) | O(1) |
  | Middle of list | Slow/fast pointers | O(n) | O(1) |
  | Merge two sorted lists | Compare heads iteratively | O(n+m) | O(1) |
  | Merge K sorted lists | Heap of K heads | O(n log k) | O(k) |
  | Reverse in K groups | Recurse in groups | O(n) | O(n/k) stack |
  | LRU Cache | HashMap + doubly linked list | O(1) get/put | O(capacity) |
  
  ## Common Mistakes / Gotchas
  - **Losing next pointer:** save `nxt = curr.next` before relinking
  - **Null termination:** after reversal, old head's `.next` must be None
  - **Dummy head:** use `dummy = ListNode(0); dummy.next = head` to avoid edge cases on head deletion
  - **K-group reversal:** don't reverse if fewer than K nodes remain — check count first
  - **Off-by-one in slow/fast:** initialize `fast = head.next` for finding middle before split
  
  ## Interview Quick-Reference
  | Question | What to say |
  |---------|-------------|
  | "When linked list over array?" | Insert/delete at known position O(1) vs O(n). Never for random access. |
  | "How detect cycle in O(1) space?" | Floyd's two pointers — slow moves 1, fast moves 2. They meet iff cycle exists. |
  | "LRU cache design?" | HashMap (key→node) + doubly linked list (order). O(1) get and put. |
  ```

- [ ] **Step 2: Create `trees-and-bst.md`** — covering: DFS (pre/in/post-order), BFS level-order, BST properties, LCA, path sum problems, balanced/height checks, Morris traversal (O(1) space inorder), Red-Black/AVL brief overview

- [ ] **Step 3: Create `graphs.md`** — covering: BFS/DFS templates, topological sort (Kahn + DFS), union-find, Dijkstra, Bellman-Ford, cycle detection (directed/undirected), connected components, bipartite check

- [ ] **Step 4: Create `heaps-priority-queues.md`** — covering: heapify, heap sort, top-K pattern, k-way merge, median maintenance (dual heap), priority queue in Python (`heapq`)

- [ ] **Step 5: Create `hash-maps-sets.md`** — covering: collision handling, load factor, two-sum pattern, frequency counting, anagram grouping, sliding window with map, LRU pattern

- [ ] **Step 6: Commit**
  ```bash
  git add coding/data-structures/linked-lists.md coding/data-structures/trees-and-bst.md \
          coding/data-structures/graphs.md coding/data-structures/heaps-priority-queues.md \
          coding/data-structures/hash-maps-sets.md
  git commit -m "feat: add 5 core data structure guides to coding section"
  ```

---

### Task D2: Algorithm patterns (5 files)

**Files to Create:**
- `coding/data-structures/dynamic-programming.md`
- `coding/data-structures/binary-search.md`
- `coding/data-structures/sliding-window.md`
- `coding/data-structures/two-pointers.md`
- `coding/data-structures/recursion-backtracking.md`

- [ ] **Step 1: Create `dynamic-programming.md`**

  Must include:
  - Top-down (memoization) vs bottom-up (tabulation) templates
  - Problem patterns: 0/1 knapsack, unbounded knapsack, LCS, LIS, edit distance, coin change, matrix path, partition problems
  - State identification framework: `dp[i]` = "best value considering first i items"
  - Transition function template: `dp[i] = max(dp[i-1], dp[i-w] + v[i])`
  - Space optimization: rolling array from O(n²) → O(n)

  Template to include:
  ```python
  # 0/1 Knapsack template
  def knapsack(weights, values, capacity):
      n = len(weights)
      dp = [0] * (capacity + 1)
      for i in range(n):
          for w in range(capacity, weights[i] - 1, -1):  # reverse to avoid reuse
              dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
      return dp[capacity]
  
  # LCS template
  def lcs(s1, s2):
      m, n = len(s1), len(s2)
      dp = [[0] * (n + 1) for _ in range(m + 1)]
      for i in range(1, m + 1):
          for j in range(1, n + 1):
              if s1[i-1] == s2[j-1]:
                  dp[i][j] = dp[i-1][j-1] + 1
              else:
                  dp[i][j] = max(dp[i-1][j], dp[i][j-1])
      return dp[m][n]
  ```

- [ ] **Step 2: Create `binary-search.md`** — templates for: exact match, lower bound, upper bound, search on answer space, rotated array, fractional binary search; common pitfall: `mid = lo + (hi - lo) // 2` prevents overflow

- [ ] **Step 3: Create `sliding-window.md`** — fixed-size vs variable-size templates, when to shrink window, two-pointer implementation, problems: max sum subarray, longest without repeat, min window substring

- [ ] **Step 4: Create `two-pointers.md`** — opposite ends, same direction, merge, cycle detection patterns with problem table

- [ ] **Step 5: Create `recursion-backtracking.md`** — decision tree mental model, permutations/combinations/subsets templates, N-queens, Sudoku solver, pruning strategies

- [ ] **Step 6: Commit**
  ```bash
  git add coding/data-structures/dynamic-programming.md coding/data-structures/binary-search.md \
          coding/data-structures/sliding-window.md coding/data-structures/two-pointers.md \
          coding/data-structures/recursion-backtracking.md
  git commit -m "feat: add 5 algorithm pattern guides to coding section"
  ```

---

### Task D3: Advanced topics (5 files)

**Files to Create:**
- `coding/data-structures/monotonic-stack.md`
- `coding/data-structures/union-find.md`
- `coding/data-structures/trie.md`
- `coding/data-structures/intervals.md`
- `coding/data-structures/bit-manipulation.md`

- [ ] **Step 1: Create `monotonic-stack.md`** — decreasing vs increasing stack, problems: next greater element, daily temperatures, largest rectangle in histogram, trapping rain water. Template:
  ```python
  # Next greater element (monotonic decreasing stack)
  def next_greater(nums):
      result = [-1] * len(nums)
      stack = []  # stores indices, values are decreasing
      for i, num in enumerate(nums):
          while stack and nums[stack[-1]] < num:
              result[stack.pop()] = num
          stack.append(i)
      return result
  ```

- [ ] **Step 2: Create `union-find.md`** — union by rank + path compression, cycle detection, connected components, problem table (number of islands, friend circles, accounts merge)

- [ ] **Step 3: Create `trie.md`** — insert/search/startsWith, word search II, autocomplete, implementation with dict vs array

- [ ] **Step 4: Create `intervals.md`** — sort by start, merge overlapping, insert interval, meeting rooms, event scheduling patterns

- [ ] **Step 5: Create `bit-manipulation.md`** — AND/OR/XOR masks, Brian Kernighan's bit count, XOR tricks (find single number, missing number), bit DP

- [ ] **Step 6: Commit**
  ```bash
  git add coding/data-structures/monotonic-stack.md coding/data-structures/union-find.md \
          coding/data-structures/trie.md coding/data-structures/intervals.md \
          coding/data-structures/bit-manipulation.md
  git commit -m "feat: add 5 advanced algorithm guides to coding section"
  ```

---

### Task D4: ML-specific coding patterns (5 files)

**Files to Create:**
- `coding/data-structures/matrix-operations.md`
- `coding/data-structures/sorting-algorithms.md`
- `coding/data-structures/prefix-sums.md`
- `coding/data-structures/segment-trees.md`
- `coding/data-structures/ml-coding-patterns.md`

- [ ] **Step 1: Create `ml-coding-patterns.md`** — ML-interview-specific coding patterns:
  - Implement softmax without overflow: `e^(x - max(x)) / sum(e^(x - max(x)))`
  - K-nearest neighbors from scratch (brute force + KD-tree)
  - Moving average / exponential moving average
  - Batch matrix multiplication
  - Cosine similarity at scale
  - Feature scaling (min-max, Z-score) without sklearn
  - Confusion matrix from predictions
  - F1 score calculation edge cases (division by zero)

- [ ] **Step 2: Create remaining 4 files** (matrix-operations, sorting-algorithms, prefix-sums, segment-trees)

- [ ] **Step 3: Update coding section README**
  Update `coding/README.md` with full table of contents linking to all 21 files

- [ ] **Step 4: Commit**
  ```bash
  git add coding/data-structures/ coding/README.md
  git commit -m "feat: complete coding section with 21 algorithm/DS topic files"
  ```

---

## Sub-Project E: Navigation and INDEX Improvements

**Priority: MEDIUM** — Current INDEX.md and README.md don't fully reflect the 500+ files in the repo.

### Task E1: Update INDEX.md

**File:** `INDEX.md`

- [ ] **Step 1: Read current INDEX.md**
  ```bash
  cat INDEX.md
  ```

- [ ] **Step 2: Rewrite INDEX.md with complete section counts and links**

  The index should show:
  - Total concept count per section
  - Difficulty labels (Beginner / Intermediate / Advanced / Expert)
  - Quick-links to the most frequently needed topics
  - Cross-section learning paths (e.g., "ML Interview Track", "LLM Engineer Track", "MLOps Track")

  Learning paths to add:
  ```markdown
  ## Learning Paths
  
  ### ML Engineer Interview Track (4-6 weeks)
  1. [AI Fundamentals](ai/) — core ML concepts (40 topics)
  2. [Coding Patterns](coding/) — algorithms and DS (21 topics)
  3. [System Design](system-design/) — MLOps patterns (31 patterns)
  4. [Real Systems](arch-review/) — 30 real AI system architectures
  
  ### LLM Engineer Track (3-4 weeks)
  1. [LLM Concepts](llm/) — transformers to RAG (44 topics)
  2. [Modern AI](modern-ai/) — cutting-edge techniques (55 topics)
  3. [Arch Review](arch-review/) — LLM system designs (systems 1-10)
  
  ### MLOps/Infra Track (3-4 weeks)
  1. [MLOps Concepts](mlops/) — pipelines to deployment (16 topics)
  2. [System Design Patterns](system-design/patterns/) — 31 production patterns
  3. [Arch Review](arch-review/) — MLOps systems (systems 25-30)
  
  ### Agentic AI Track (2-3 weeks)
  1. [Agentic AI](agentic-ai/) — agents to multi-agent systems (64 topics)
  2. [Modern AI](modern-ai/) — agent optimization (25-55)
  3. [Arch Review](arch-review/) — agentic system designs (systems 11-20)
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add INDEX.md
  git commit -m "docs: update INDEX.md with complete section counts, learning paths, difficulty labels"
  ```

---

### Task E2: Update README.md with comprehensive overview

**File:** `README.md`

- [ ] **Step 1: Expand README with stats and visual directory**

  Add a section with:
  - Total file counts per section
  - Progress table (concepts vs notebooks vs implementations)
  - "Start here" section for different user personas
  - Quick-start commands

- [ ] **Step 2: Commit**
  ```bash
  git add README.md
  git commit -m "docs: expand README with stats, learning paths, start-here guide"
  ```

---

---

## Sub-Project F: Reinforcement Learning Section (New)

**Priority: HIGH** — Zero RL coverage anywhere in the repo. Critical for RLHF (used in every modern LLM), game AI, robotics, and direct interview questions at FAANG.

**New directory:** `rl/` with `concepts/`, `notebooks/`, `implementations/`

### 20 RL Concepts to Create

| # | Slug | Category | Key Algorithm |
|---|------|----------|--------------|
| 01 | markov-decision-processes | Foundations | MDP tuple (S, A, P, R, γ) |
| 02 | bellman-equations | Foundations | V(s) = max_a [R + γ·V(s')] |
| 03 | dynamic-programming-rl | Planning | Value iteration, policy iteration |
| 04 | monte-carlo-methods | Model-Free | First-visit MC, every-visit MC |
| 05 | temporal-difference-learning | Model-Free | TD(0), TD(λ), eligibility traces |
| 06 | q-learning | Value-Based | Off-policy TD, Q(s,a) update |
| 07 | sarsa | Value-Based | On-policy TD, ε-greedy |
| 08 | deep-q-networks | Deep RL | DQN, experience replay, target net |
| 09 | policy-gradient | Policy-Based | REINFORCE, log-gradient trick |
| 10 | actor-critic | Policy-Based | A2C, advantage function |
| 11 | proximal-policy-optimization | Advanced | PPO clip, KL penalty, RLHF use |
| 12 | soft-actor-critic | Advanced | SAC, entropy regularization, continuous action |
| 13 | multi-armed-bandit | Exploration | ε-greedy, UCB, Thompson sampling |
| 14 | exploration-exploitation | Core Concept | ε-greedy, UCB, intrinsic motivation |
| 15 | reward-shaping | Engineering | Potential-based, sparse→dense reward |
| 16 | model-based-rl | Model-Based | Dyna-Q, world models, MBPO |
| 17 | rlhf | LLM Alignment | Reward model, PPO fine-tuning, KL divergence penalty |
| 18 | inverse-rl | Advanced | Learning reward from demonstrations |
| 19 | multi-agent-rl | Advanced | Nash equilibrium, CTDE, MADDPG |
| 20 | offline-rl | Advanced | Conservative Q-learning, BCQ, dataset constraints |

### Algorithm Details for Key Concepts

#### 01 — Markov Decision Processes
**Formal definition**: Tuple (S, A, P, R, γ). `P(s'|s,a)` = transition probability. `R(s,a,s')` = reward. `γ` = discount. Policy `π(a|s)` maps state to action distribution. Goal: find `π*` that maximizes `E[Σ_t γ^t·R_t]`.
- Level 1: GridWorld MDP in numpy — states, actions, transitions, rewards, policy evaluation
- Level 2: Value iteration on stochastic MDP with discount sweep, convergence analysis
- RW1: Inventory management MDP — order/hold/sell decision with stochastic demand
- RW2: Routing MDP — shortest path with stochastic edge costs (vs Dijkstra)
- RW3: Portfolio optimization MDP — discrete allocation with market state transitions

#### 06 — Q-Learning
**Algorithm**: `Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]`. Off-policy: learn optimal Q regardless of behavior policy. Convergence guaranteed if (1) all state-action pairs visited infinitely often, (2) learning rate satisfies Robbins-Monro: `Σα=∞, Σα²<∞`.
- Level 1: Tabular Q-learning on GridWorld in numpy with ε-greedy, convergence plot
- Level 2: Q-table with eligibility traces (Q(λ)), comparison across λ values
- RW1: Taxi problem (gym-compatible) — 500 states, 6 actions, trained to convergence
- RW2: Cliff walking — compare Q-learning (optimal, risky path) vs SARSA (safe path)
- RW3: Multi-step Q-learning — n-step returns, bias-variance tradeoff across n

#### 11 — PPO (used in RLHF)
**Algorithm**: Clip objective: `L_CLIP(θ) = E[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]` where `r_t = π_θ(a|s)/π_θ_old(a|s)`. KL penalty variant: `L_KL = L_CLIP - β·KL[π_old||π_new]`. RLHF use: `π_θ` is LLM, reward from reward model, `β·KL` prevents reward hacking.
- Level 1: Policy gradient (REINFORCE) from scratch in numpy on bandit problem
- Level 2: PPO with clipped objective in torch — actor/critic, GAE advantage, multiple epochs
- RW1: CartPole benchmark — compare vanilla PG vs PPO sample efficiency
- RW2: RLHF simulation — synthetic reward model, PPO fine-tuning of toy language model
- RW3: PPO hyperparameter analysis — clip_ε, GAE λ, KL β sweep on continuous control

#### 17 — RLHF
**Algorithm**: (1) SFT on demonstrations; (2) Train reward model: `R(x,y) ← σ(r_θ(x,y_w) - r_θ(x,y_l))` on preference pairs; (3) PPO: `L = E[r_θ(x,y)] - β·KL[π_θ(x)||π_ref(x)]`. Bradley-Terry model for pairwise preferences. KL penalty prevents policy from diverging to reward hack. `β=0.1-0.2` typical.
- Level 1: Preference learning — Bradley-Terry model, pairwise comparison in numpy
- Level 2: Full RLHF pipeline in torch — toy LM + reward model + PPO fine-tuning + KL tracking
- RW1: Reward hacking detection — generate examples where high reward ≠ high quality
- RW2: DPO as RLHF alternative — compare PPO vs DPO compute/quality/stability
- RW3: Constitutional AI simulation — AI feedback replacing human preference labels

### Task F1: RL concepts 01-10

**Files to Create:**
- `rl/concepts/01-markov-decision-processes.md` through `rl/concepts/10-actor-critic.md`
- `rl/notebooks/01-markov-decision-processes.ipynb` through `rl/notebooks/10-actor-critic.ipynb`
- `rl/implementations/01-markov-decision-processes.py` through `rl/implementations/10-actor-critic.py`
- `rl/README.md`

- [ ] **Step 1: Create `rl/README.md`**
  ```markdown
  # Reinforcement Learning
  
  20 concepts from MDP foundations to RLHF. Each concept has a markdown explanation,
  runnable notebook, and standalone implementation.
  
  ## Prerequisites
  - Python, numpy, torch
  - Basic probability and linear algebra
  
  ## Concepts
  | # | Concept | Category | Notebook |
  |---|---------|----------|---------|
  | 01 | Markov Decision Processes | Foundations | [notebook](notebooks/01-markov-decision-processes.ipynb) |
  ...
  ```

- [ ] **Step 2: Create concepts 01-10** (same 8-section format as modern-ai)
- [ ] **Step 3: Create notebooks 01-10** (same 16-cell format)
- [ ] **Step 4: Create implementations 01-10** (150-250 lines each)
- [ ] **Step 5: Verify**
  ```bash
  ls rl/concepts/ | wc -l  # 10
  python3 -c "
  import json, glob
  for f in sorted(glob.glob('rl/notebooks/0[1-9]-*.ipynb')):
      nb = json.load(open(f))
      code_lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type'] == 'code')
      print(f'{\"✅\" if code_lines >= 400 else \"⚠️ \"} {f.split(\"/\")[-1]}: {code_lines} lines')
  "
  ```
- [ ] **Step 6: Commit**
  ```bash
  git add rl/
  git commit -m "feat: add RL concepts 01-10 (MDP through Actor-Critic) with notebooks and implementations"
  ```

---

### Task F2: RL concepts 11-20

**Files:** `rl/concepts/11-*.md` through `rl/concepts/20-*.md` + notebooks + implementations

- [ ] **Step 1: Create concepts 11-20** (PPO, SAC, MAB, exploration, reward shaping, model-based, RLHF, inverse RL, multi-agent RL, offline RL)
- [ ] **Step 2: Create notebooks 11-20**
  Special attention for notebook 17 (RLHF): simulate full pipeline with toy 10-vocab LM + reward model + PPO, show KL divergence tracking, reward hacking example
- [ ] **Step 3: Create implementations 11-20**
- [ ] **Step 4: Add RL roadmap `roadmaps/rl-roadmap.md`** with 4-phase progression: (1) Foundations MDP/Bellman, (2) Model-Free Q/TD, (3) Deep RL DQN/PPO, (4) Modern RLHF/Offline
- [ ] **Step 5: Commit**
  ```bash
  git add rl/ roadmaps/rl-roadmap.md
  git commit -m "feat: add RL concepts 11-20 (PPO through Offline RL) with notebooks, implementations, roadmap"
  ```

---

## Sub-Project G: Statistics & Probability Deep-Dive (New)

**Priority: HIGH** — Only 1 file (`ml/concepts/probability-statistics.md`). Statistics underpins every ML interview: hypothesis testing, experimental design, Bayesian reasoning, information theory.

**New directory:** `stats/` with `concepts/` and `notebooks/`

### 15 Stats Concepts

| # | Slug | Category |
|---|------|----------|
| 01 | probability-fundamentals | Foundations |
| 02 | distributions-reference | Foundations |
| 03 | bayesian-inference | Bayesian |
| 04 | maximum-likelihood-estimation | Estimation |
| 05 | hypothesis-testing | Inference |
| 06 | confidence-intervals | Inference |
| 07 | statistical-power-sample-size | Experimental Design |
| 08 | ab-testing-statistics | Experimental Design |
| 09 | causal-inference | Causal |
| 10 | information-theory | Information |
| 11 | monte-carlo-sampling | Computational |
| 12 | markov-chains-mcmc | Computational |
| 13 | multivariate-statistics | Advanced |
| 14 | time-series-statistics | Time Series |
| 15 | statistical-ml-connections | Synthesis |

### Key Content Per Concept

#### 05 — Hypothesis Testing
Core content must include:
- H0/H1 setup, Type I/II error, p-value definition (NOT "probability H0 is true")
- One-tailed vs two-tailed tests
- t-test, chi-square, Mann-Whitney U (non-parametric)
- Multiple comparison correction: Bonferroni, FDR (Benjamini-Hochberg)
- Common mistake: p<0.05 ≠ effect is large (need effect size: Cohen's d)
- Interview Q: "Your A/B test shows p=0.03 after 3 days. Do you ship?" → No: peeking, novelty effect, insufficient power

#### 08 — A/B Testing Statistics
Beyond the system-design pattern (which covers infrastructure), this covers the math:
- Sample size formula: `n = (z_{α/2} + z_β)² × 2σ² / Δ²`
- Sequential testing (SPRT) — when to stop early
- Bayesian A/B testing — posterior distributions, credible intervals
- Multi-metric testing — False Discovery Rate across 20 metrics
- Variance reduction: CUPED (`Y_cuped = Y - θ(X - E[X])`)

#### 10 — Information Theory
Essential for LLM interviews:
- Entropy: `H(X) = -Σ p(x)·log p(x)`
- Cross-entropy loss: `H(p,q) = -Σ p(x)·log q(x)` (what models minimize)
- KL divergence: `KL(P||Q) = Σ P·log(P/Q)` (asymmetric, used in KD, RLHF)
- Mutual information: `I(X;Y) = H(X) - H(X|Y)` (feature selection)
- Perplexity: `PP = 2^H` (language model evaluation)
- Coding: implement from-scratch entropy, cross-entropy, KL in numpy

### Task G1: Stats concepts 01-08 with notebooks

**Files to Create:**
- `stats/concepts/01-probability-fundamentals.md` through `stats/concepts/08-ab-testing-statistics.md`
- `stats/notebooks/01-probability-fundamentals.ipynb` through `stats/notebooks/08-ab-testing-statistics.ipynb`
- `stats/README.md`

- [ ] **Step 1: Create `stats/README.md`** with concept table and prerequisites
- [ ] **Step 2: Create concepts 01-08** (8-section format, 800-1200 words, with Mermaid diagrams and comparison tables)
- [ ] **Step 3: Create notebooks 01-08** (12-cell format: numpy simulations, matplotlib visualizations)

  Special requirement for notebook 08 (A/B testing statistics):
  - Cell 5: CUPED variance reduction implementation
    ```python
    def cuped_adjustment(y_treatment, y_control, x_treatment, x_control):
        """CUPED: Controlled-experiment Using Pre-Experiment Data."""
        # Estimate theta: correlation between pre-experiment and post-experiment metric
        x_pooled = np.concatenate([x_treatment, x_control])
        y_pooled = np.concatenate([y_treatment, y_control])
        theta = np.cov(y_pooled, x_pooled)[0,1] / np.var(x_pooled)
        
        # Adjust metrics
        y_t_adj = y_treatment - theta * (x_treatment - x_pooled.mean())
        y_c_adj = y_control - theta * (x_control - x_pooled.mean())
        return y_t_adj, y_c_adj, theta
    ```
  - Cell 7: Sequential SPRT implementation
  - Cell 9: Bayesian A/B with Beta-Binomial conjugate prior

- [ ] **Step 4: Commit**
  ```bash
  git add stats/
  git commit -m "feat: add stats section with 8 core probability and inference concepts"
  ```

---

### Task G2: Stats concepts 09-15 with notebooks

- [ ] **Step 1: Create concepts 09-15** (causal inference, information theory, Monte Carlo, MCMC, multivariate, time-series stats, connections)
- [ ] **Step 2: Create notebooks 09-15**

  Special requirement for notebook 10 (information theory):
  ```python
  import numpy as np
  
  def entropy(p: np.ndarray) -> float:
      """Shannon entropy H(X) = -Σ p·log2(p)."""
      p = p[p > 0]  # avoid log(0)
      return -np.sum(p * np.log2(p))
  
  def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
      """H(p,q) = -Σ p·log q. Cross-entropy loss in neural networks."""
      q = np.clip(q, 1e-10, 1)  # numerical stability
      return -np.sum(p * np.log(q))
  
  def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
      """KL(P||Q) = Σ P·log(P/Q). Asymmetric. Used in KD, RLHF."""
      mask = p > 0
      return np.sum(p[mask] * np.log(p[mask] / q[mask]))
  
  def mutual_information(joint: np.ndarray) -> float:
      """I(X;Y) = Σ p(x,y)·log[p(x,y)/(p(x)·p(y))]."""
      p_x = joint.sum(axis=1, keepdims=True)
      p_y = joint.sum(axis=0, keepdims=True)
      product = p_x * p_y
      mask = joint > 0
      return np.sum(joint[mask] * np.log(joint[mask] / product[mask]))
  
  def perplexity(log_probs: np.ndarray) -> float:
      """Perplexity = 2^H. Lower = better language model."""
      avg_neg_log2 = -np.mean(log_probs) / np.log(2)
      return 2 ** avg_neg_log2
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add stats/concepts/0[9]-*.md stats/concepts/1[0-5]-*.md \
          stats/notebooks/0[9]-*.ipynb stats/notebooks/1[0-5]-*.ipynb
  git commit -m "feat: add stats concepts 09-15 including information theory and causal inference"
  ```

---

## Sub-Project H: ML Interview Deep-Dive (New)

**Priority: HIGH** — Currently only 10 theory Q, 2 case studies. A complete ML interview prep hub needs 50+ theory Q, 15+ case studies, company-specific guides, behavioral prep.

**New/expanded files:**

### Task H1: Expand ML theory questions to 50

**File:** `ml/interview-prep/ml-theory-questions.md`

Current: 10 questions. Target: 50 questions across 6 domains.

- [ ] **Step 1: Add questions 11-20: Supervised Learning Deep Cuts**
  Questions to add:
  - Q11: Why does gradient boosting work better than random forests on tabular data?
  - Q12: How would you handle 1000 features with 500 samples? (regularization, dimensionality reduction, feature selection order)
  - Q13: Your model AUC=0.95 in offline eval, but only 0.72 in production. What's wrong? (training-serving skew checklist)
  - Q14: When would you NOT use cross-entropy loss? (imbalanced data → focal loss; ordinal targets → ordinal loss)
  - Q15: Design a model that must be explainable to regulators. (SHAP, LIME, decision tree constraint, documentation)
  - Q16: A feature has 40% missing values. Drop it or impute? (framework: missing at random vs MCAR vs MNAR)
  - Q17: How do you prevent target leakage in time-series data? (time-based splits, causality check)
  - Q18: Your learning curves show high train accuracy but high test loss. 3 things to try? (more data, regularization, simpler model)
  - Q19: How do you pick k in k-means when you don't have labels? (elbow method, silhouette score, BIC for GMM)
  - Q20: Explain precision-recall tradeoff. When do you care more about precision? About recall? (fraud: recall. spam: precision. explain why)

- [ ] **Step 2: Add questions 21-35: Neural Networks and Deep Learning**
  Topics: BatchNorm vs LayerNorm in transformers, why ReLU over sigmoid, vanishing gradient symptoms and fixes, when to use attention vs conv, transformer without positional encoding, weight decay vs dropout differences, learning rate warmup purpose, gradient clipping when and why, batch size effect on generalization, knowledge distillation loss temperature intuition

- [ ] **Step 3: Add questions 36-50: LLM and Production**
  Topics: why fine-tuning fails (catastrophic forgetting, insufficient data, wrong hyperparams), RAG vs fine-tuning decision framework, how to evaluate LLM outputs at scale (LLM-as-judge, human eval, task-specific metrics), prompt injection defense, quantization quality drop causes, RLHF reward hacking examples, serving latency optimization hierarchy (model size → batching → quantization → caching), context window scaling techniques, multi-modal training challenges, model collapse in RLHF

- [ ] **Step 4: Commit**
  ```bash
  git add ml/interview-prep/ml-theory-questions.md
  git commit -m "feat: expand ML theory questions from 10 to 50 across 6 domains"
  ```

---

### Task H2: Expand ML case studies to 15

**File:** `ml/interview-prep/case-studies.md`

Current: 2 case studies (recommendation, content). Target: 15 covering major ML system types.

Format per case study (existing 6-step format):
1. Clarifying Questions to Ask
2. ML Problem Formulation
3. System Design (data, features, model, serving)
4. Offline Evaluation
5. Online Evaluation and A/B Test
6. Production Monitoring

- [ ] **Step 1: Add case studies 03-08 (Core ML systems)**
  - **03: Ad Click Prediction** — feature engineering for user×ad, sparse ID features, embedding tables, calibration, real-time scoring at 1M QPS
  - **04: Fraud Detection** — imbalanced (0.1% positive rate), real-time scoring <50ms, rule-based+ML ensemble, concept drift from fraud pattern evolution
  - **05: Search Ranking** — query understanding, document features, learning-to-rank (pointwise/pairwise/listwise), position bias correction
  - **06: Customer Lifetime Value Prediction** — censored survival analysis (customers still alive), business target vs model metric alignment
  - **07: Spam Detection** — text features, adversarial robustness, feedback loop from user reports, latency requirements for email
  - **08: Price Optimization** — demand elasticity modeling, explore-exploit in dynamic pricing, business constraints (min margin)

- [ ] **Step 2: Add case studies 09-12 (LLM systems)**
  - **09: Enterprise Semantic Search** — embedding models, vector DB at scale, hybrid BM25+dense, reranking, evaluation (MRR@10, NDCG)
  - **10: Code Completion** — latency SLA (50ms prefix, 200ms full), context window management, model distillation for edge
  - **11: Document Q&A System** — multi-document RAG, hallucination detection, citation accuracy, knowledge freshness
  - **12: Multimodal Product Search** — image+text queries, CLIP-style retrieval, catalog with 100M items

- [ ] **Step 3: Add case studies 13-15 (Advanced/Infra)**
  - **13: Real-Time Personalization** — feature freshness, feature store design, online learning vs batch, cold start
  - **14: Model Monitoring at Scale** — drift detection pipeline, automated retraining triggers, shadow models, canary evaluation
  - **15: Cost Optimization for ML Platform** — GPU utilization, auto-scaling, spot instances, model distillation ROI

- [ ] **Step 4: Commit**
  ```bash
  git add ml/interview-prep/case-studies.md
  git commit -m "feat: expand ML case studies from 2 to 15 covering ads, fraud, search, LLM systems"
  ```

---

### Task H3: Company-specific interview guides

**New files:**
- `ml/interview-prep/company-guides/meta.md`
- `ml/interview-prep/company-guides/google.md`
- `ml/interview-prep/company-guides/amazon.md`
- `ml/interview-prep/company-guides/microsoft.md`
- `ml/interview-prep/company-guides/openai.md`

Format per company:
```markdown
# ML Interview at [Company]

## What They're Looking For
[3-5 sentences on their specific values and ML philosophy]

## Interview Rounds
| Round | Type | Duration | What's Tested |
|-------|------|----------|--------------|

## Most Common Question Topics
[Ranked list of most frequent topics based on public reports]

## Their Tech Stack (Known)
[ML frameworks, infrastructure, model types they use]

## Sample Questions from This Company
[5-10 questions with format: Q / What they're testing / Red flags / Green flags]

## Preparation Strategy
[2-3 week prep plan specific to this company]
```

- [ ] **Step 1: Create `ml/interview-prep/company-guides/meta.md`**

  Key content for Meta:
  - Ranking systems (news feed, marketplace, ads) — their core ML domain
  - Extreme classification (100M+ classes in content recommendation)
  - Real-time feature engineering at 3B+ DAU scale
  - PyTorch (they built it), internal stack: FBLearner, Jarvis
  - Sample Q: "How would you redesign news feed ranking to reduce misinformation without hurting engagement?" (tests values + ML tradeoffs)

- [ ] **Step 2: Create company guides for Google, Amazon, Microsoft, OpenAI**

  Google: search ranking, YouTube recommendation, Gemini/TPU, Vertex AI; emphasis on scale (billions of users), ML theory depth, whiteboard coding
  
  Amazon: demand forecasting (core business), recommendation (every product page), Alexa NLP, AWS SageMaker; behavioral heavily weighted (Leadership Principles × ML decisions)
  
  Microsoft: Azure AI, Copilot/LLMs, Teams NLP, Xbox recommendation; emphasis on product sense + ML integration, less hardcore theory than Google
  
  OpenAI: LLM training pipeline, RLHF deep knowledge, safety/alignment, scaling laws; expects ability to discuss frontier research papers

- [ ] **Step 3: Commit**
  ```bash
  git add ml/interview-prep/company-guides/
  git commit -m "feat: add company-specific ML interview guides for Meta, Google, Amazon, Microsoft, OpenAI"
  ```

---

### Task H4: Behavioral interview prep for ML engineers

**New file:** `ml/interview-prep/behavioral.md`

ML engineers face behavioral questions that mix technical judgment with soft skills.

- [ ] **Step 1: Create `ml/interview-prep/behavioral.md`**

  Structure:
  ```markdown
  # ML Engineer Behavioral Interview Prep
  
  ## The STAR Framework for ML Stories
  [Situation, Task, Action, Result — adapted for ML context]
  
  ## 20 Most Common Behavioral Questions for ML Roles
  
  ### Q: Tell me about a time your model failed in production.
  **What they're really testing:** ability to detect, diagnose, and learn from ML failures
  **Green flags:** metric-driven detection, root cause analysis, systematic fix, process improvement
  **Red flags:** "it never failed" or blaming data quality without owning it
  **Story structure:**
  - Situation: [describe the model and business context]
  - What failed: [specific metric, specific user impact]
  - How you detected it: [monitoring, user reports, periodic evaluation]
  - Root cause: [data drift / training-serving skew / label shift / bug]
  - Fix: [what you changed and how you validated the fix]
  - Process improvement: [what monitoring/testing you added after]
  
  ### Q: Tell me about a time you had to make a trade-off between model accuracy and latency.
  ...
  
  ### Q: Describe a project where you had to explain a complex ML decision to non-technical stakeholders.
  ...
  [18 more questions with full STAR templates]
  
  ## Story Bank Template
  [Blank STAR template for candidates to fill in their own stories]
  ```

- [ ] **Step 2: Commit**
  ```bash
  git add ml/interview-prep/behavioral.md
  git commit -m "feat: add ML engineer behavioral interview prep with 20 STAR templates"
  ```

---

## Sub-Project I: Cheat Sheets (New)

**Priority: MEDIUM** — Zero quick-reference materials. Cheat sheets are high-value for last-minute review before interviews.

**New directory:** `cheatsheets/`

### 10 Cheat Sheets to Create

| File | Content |
|------|---------|
| `ml-algorithms.md` | Algorithm comparison: time/space complexity, when to use, hyperparams |
| `neural-network-architectures.md` | CNN, RNN, Transformer, ViT, Diffusion — diagram + use case |
| `optimizers.md` | SGD, Adam, AdaW, LAMB — update rules, best use cases |
| `evaluation-metrics.md` | Classification, regression, ranking, generation metrics |
| `llm-models.md` | Model comparison table: GPT-4, Claude, Llama, Mistral, Gemini — context, cost, capability |
| `regularization.md` | L1/L2/dropout/batch-norm — formula, effect on weights, when to use |
| `deployment-strategies.md` | Blue-green, canary, shadow, rolling — decision matrix |
| `system-design-quick-ref.md` | ML system components, capacity estimates, latency budgets |
| `python-ml-snippets.md` | Copy-paste sklearn, torch, numpy snippets for common tasks |
| `interview-formulas.md` | All formulas you need to know: gradient descent, attention, loss functions, evaluation |

### Task I1: Create all 10 cheat sheets

**Files to Create:** `cheatsheets/` directory + 10 `.md` files

- [ ] **Step 1: Create `cheatsheets/ml-algorithms.md`**

  Must include a master comparison table:
  ```markdown
  | Algorithm | Type | Train Time | Inference | Memory | Best For | Avoid When |
  |-----------|------|-----------|-----------|--------|---------|-----------|
  | Linear Regression | Regression | O(nd²) | O(d) | O(d) | Interpretable baseline | Non-linear patterns |
  | Logistic Regression | Classification | O(ndi) | O(d) | O(d) | Probability calibration | Complex boundaries |
  | Decision Tree | Both | O(nd log n) | O(depth) | O(nodes) | Interpretable, mixed features | Noisy data |
  | Random Forest | Both | O(T·nd log n) | O(T·depth) | O(T·nodes) | Robust, out-of-box good | Ultra-low latency |
  | Gradient Boosting | Both | O(T·nd) | O(T·depth) | O(T·nodes) | Best tabular accuracy | Real-time serving |
  | SVM | Both | O(n²d)-O(n³d) | O(sv·d) | O(sv) | Small data, high dim | n > 100K |
  | KNN | Both | O(1) fit | O(nd) | O(nd) | Non-parametric baseline | Large datasets |
  | K-Means | Clustering | O(nkdi) | O(kd) | O((n+k)d) | Fast segmentation | Non-spherical clusters |
  | Neural Network | Both | O(layers·nd) | O(layers·d) | O(params) | Unstructured data, scale | Small data, interpretability |
  | Transformer | Both | O(n²d) attention | O(n²d) | O(n²+params) | Sequences, language, vision | Long sequences at edge |
  ```

- [ ] **Step 2: Create `cheatsheets/interview-formulas.md`**

  Must include all formulas a candidate might need to write on a whiteboard:
  ```markdown
  ## Loss Functions
  MSE: L = (1/n)·Σ(y - ŷ)²
  Cross-entropy: L = -Σ y·log(ŷ)
  Focal loss: L = -(1-ŷ)^γ · y·log(ŷ)
  
  ## Gradient Descent
  θ ← θ - α·∇_θL
  Adam: m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g²; θ ← θ - α·m̂/√v̂+ε
  
  ## Attention
  Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V
  
  ## Evaluation
  Precision = TP/(TP+FP)
  Recall = TP/(TP+FN)
  F1 = 2·P·R/(P+R)
  AUC = P(score_pos > score_neg)
  NDCG@K = DCG@K / IDCG@K
  
  ## Statistics
  Confidence interval: x̄ ± z_{α/2}·σ/√n
  Sample size: n = (z_{α/2} + z_β)² · 2σ² / Δ²
  Cohen's d: d = (μ₁ - μ₂) / σ_pooled
  ```

- [ ] **Step 3: Create remaining 8 cheat sheets** (neural-network-architectures, optimizers, evaluation-metrics, llm-models, regularization, deployment-strategies, system-design-quick-ref, python-ml-snippets)

- [ ] **Step 4: Create `cheatsheets/README.md`** with table of contents and "how to use before interview"

- [ ] **Step 5: Commit**
  ```bash
  git add cheatsheets/
  git commit -m "feat: add 10 cheat sheets for quick interview reference"
  ```

---

## Sub-Project J: Production ML Post-Mortems (New)

**Priority: MEDIUM** — No failure case studies exist. Senior engineers are expected to know how real ML systems fail.

**New directory:** `arch-review/post-mortems/`

### 8 Post-Mortem Case Studies

| # | Title | Failure Type | Systems Involved |
|---|-------|-------------|-----------------|
| 01 | Recommendation Model Silent Drift | Feature drift undetected for 3 months | Feature store, monitoring |
| 02 | Training-Serving Skew at Launch | Preprocessing mismatch | Feature pipeline, model serving |
| 03 | Reward Hacking in RLHF Fine-Tune | Reward model exploited | RLHF pipeline, safety filters |
| 04 | Fraud Model Bias Discovered Post-Launch | Demographic performance gap | Bias detection, model evaluation |
| 05 | Vector DB Cold Start After Migration | Cache invalidation, index rebuild | RAG pipeline, vector DB |
| 06 | Cascade Failure from Model Confidence | Overconfident model fed wrong inputs downstream | Model cascading, error handling |
| 07 | Data Pipeline Poisoning | Upstream schema change corrupts features | Data validation, schema contracts |
| 08 | LLM Prompt Injection in Production | User-supplied prompt escapes system context | LLM API gateway, output validation |

### Post-Mortem Format

```markdown
# Post-Mortem: [Title]

## Incident Summary
**Date:** [simulated date]
**Duration:** [time to detect + time to resolve]
**Business Impact:** [users affected, revenue lost, accuracy degradation]
**Severity:** P1 / P2 / P3

## Timeline
| Time | Event |
|------|-------|
| T+0 | Incident begins |
| T+2h | First alert fires |
| T+4h | Root cause identified |
| T+6h | Mitigation deployed |
| T+24h | Full resolution |

## What Happened (Technical)
[3-4 paragraphs: normal state, what changed, why it wasn't caught, blast radius]

## Root Cause Analysis
**Contributing factors:**
1. [Primary cause]
2. [Contributing factor]
3. [Process gap that allowed it]

**5 Whys:**
- Why did the model degrade? → [answer]
- Why wasn't it caught earlier? → [answer]
- Why didn't the alert fire? → [answer]
- Why was the alert threshold wrong? → [answer]
- Why wasn't it reviewed after last incident? → [answer]

## What Went Well
- [Thing 1 that helped or limited damage]

## Action Items
| Item | Owner | Due | Status |
|------|-------|-----|--------|
| Add feature drift alert on top-20 features | ML Platform | 1 week | Done |
...

## Interview Discussion Points
- What would you have done differently?
- How would you prevent this category of failure?
- What monitoring gaps does this reveal?
```

### Task J1: Create 8 production post-mortems

**Files to Create:** `arch-review/post-mortems/01-*.md` through `arch-review/post-mortems/08-*.md`

- [ ] **Step 1: Create post-mortem 01 (Recommendation Model Silent Drift)**

  Key narrative: Model trained on pre-COVID data. During COVID, user behavior shifts (home content vs commute content). Feature "hour_of_day" distribution drifts. Model continues to predict "commute shows" at prime time. Revenue dips 8% over 3 months. Finally caught when new model A/B test shows "no improvement" vs baseline that is already degraded.
  
  Root cause: PSI only monitored on 5 features (not the drifting 12), PSI threshold too high (0.2 vs standard 0.1), no accuracy monitoring on held-out sample.
  
  Action items: PSI monitoring on all top-30 features, weekly accuracy eval on fresh labeled data, shadow model always running for comparison.

- [ ] **Step 2: Create post-mortems 02-08** using the same format with specific technical narratives

- [ ] **Step 3: Create `arch-review/post-mortems/README.md`** summarizing all 8 post-mortems and indexing by failure type

- [ ] **Step 4: Commit**
  ```bash
  git add arch-review/post-mortems/
  git commit -m "feat: add 8 production ML post-mortems covering drift, skew, bias, cascades, and prompt injection"
  ```

---

## Sub-Project K: Computer Vision and NLP Pre-LLM Sections (New)

**Priority: MEDIUM** — No dedicated CV section. No pre-LLM NLP (Word2Vec, BERT, seq2seq). These appear in ~30% of ML interviews.

### K1: Computer Vision Deep-Dive (8 concepts)

**New directory:** `cv/concepts/`

| # | Topic | Key Content |
|---|-------|-------------|
| 01 | image-classification | CNN pipeline, ImageNet, data augmentation, transfer learning |
| 02 | object-detection | YOLO v8, Faster R-CNN, anchor boxes, NMS, mAP@50 |
| 03 | image-segmentation | Semantic vs instance vs panoptic, U-Net, SAM, Mask R-CNN |
| 04 | vision-transformers | ViT architecture, patch embeddings, CLS token, comparison to CNN |
| 05 | contrastive-learning-vision | CLIP, SimCLR, self-supervised pretraining, downstream tasks |
| 06 | diffusion-models | DDPM forward/reverse process, U-Net denoiser, DDIM sampling, guidance |
| 07 | video-understanding | Optical flow, 3D convolutions, temporal attention, action recognition |
| 08 | multimodal-vision-llm | LLaVA architecture, visual instruction tuning, VQA |

### Task K1: CV concepts and notebooks

**Files to Create:**
- `cv/concepts/01-image-classification.md` through `cv/concepts/08-multimodal-vision-llm.md`
- `cv/notebooks/01-image-classification.ipynb` through `cv/notebooks/08-multimodal-vision-llm.ipynb`
- `cv/README.md`

- [ ] **Step 1: Create `cv/README.md`** and all 8 concept files (8-section format)
- [ ] **Step 2: Create all 8 notebooks** (12-cell format, torch + torchvision)

  Special requirement for notebook 06 (diffusion models):
  - Level 1: DDPM forward process in numpy — add noise schedule, visualize
  - Level 2: Simplified DDPM with U-Net in torch — train on MNIST, visualize denoising
  - RW1: DDIM accelerated sampling — 50-step vs 1000-step quality comparison
  - RW2: Classifier-free guidance — unconditional vs guided generation
  
- [ ] **Step 3: Commit**
  ```bash
  git add cv/
  git commit -m "feat: add CV section with 8 concepts from image classification to multimodal vision LLMs"
  ```

---

### K2: NLP Pre-LLM Foundations (8 concepts)

**New directory:** `nlp/concepts/`

| # | Topic | Key Content |
|---|-------|-------------|
| 01 | text-preprocessing | Tokenization, stemming, lemmatization, TF-IDF |
| 02 | word-embeddings | Word2Vec (CBOW/Skip-gram), GloVe, FastText, analogy tasks |
| 03 | recurrent-neural-networks | LSTM, GRU, vanishing gradients, sequence modeling |
| 04 | seq2seq-attention | Encoder-decoder, Bahdanau attention, alignment |
| 05 | bert-and-pretraining | MLM, NSP, fine-tuning, sentence embeddings |
| 06 | text-classification | BERT fine-tuning, few-shot, zero-shot |
| 07 | named-entity-recognition | Sequence labeling, BIO tags, CRF, spaCy |
| 08 | information-retrieval | BM25, dense retrieval, re-ranking, BEIR benchmark |

### Task K2: NLP concepts and notebooks

**Files to Create:** `nlp/` directory structure + 8 concepts + 8 notebooks + `nlp/README.md`

- [ ] **Step 1: Create all 8 NLP concept files** (8-section format)
- [ ] **Step 2: Create all 8 NLP notebooks** (12-cell format)

  Special requirement for notebook 02 (word embeddings):
  - Level 1: Skip-gram from scratch in numpy — training on toy corpus, word vector arithmetic
  - Level 2: Full Word2Vec with negative sampling in torch, nearest neighbors evaluation
  - RW1: Sentiment with GloVe vectors — fine-tune vs freeze comparison
  - RW2: Word analogies — "king - man + woman = ?" evaluation on standard datasets

- [ ] **Step 3: Commit**
  ```bash
  git add nlp/
  git commit -m "feat: add NLP pre-LLM section with 8 concepts from text preprocessing to information retrieval"
  ```

---

## Updated Execution Order

```
Week 1 (Fix existing gaps):
  A1 + A2 (modern-ai concepts 36-55) | B1 + B2 (system-design patterns 14-23) | D1 (core DS)

Week 2 (Continue fixes + start new content):
  A3 + A4 (modern-ai notebooks) | B3 (patterns 24-31) | D2 + D3 (algo patterns)
  F1 (RL concepts 01-10) | G1 (stats 01-08) | I1 start (cheatsheets)

Week 3 (New content heavy):
  A5 (implementations 36-55) | C1 + C2 (arch diagrams 03-20)
  F2 (RL concepts 11-20) | G2 (stats 09-15) | H1 (theory questions)

Week 4 (Interview prep + completion):
  C3 (arch diagrams 21-30) | H2 + H3 + H4 (case studies, company guides, behavioral)
  J1 (post-mortems) | K1 + K2 (CV + NLP sections) | E1 + E2 (navigation)
```

**Optimal parallel agent dispatch (10 agents):**
- Agent 1: Sub-Project A — 60 modern-ai stub files
- Agent 2: Sub-Project B — 18 system-design patterns
- Agent 3: Sub-Project C — 81 arch diagram files
- Agent 4: Sub-Project D — 21 coding algorithm files
- Agent 5: Sub-Project E — INDEX + README navigation
- Agent 6: Sub-Project F — 20 RL concepts + notebooks + implementations
- Agent 7: Sub-Project G — 15 stats concepts + notebooks
- Agent 8: Sub-Project H — interview deep-dive (50 Q, 15 case studies, company guides, behavioral)
- Agent 9: Sub-Project I + J — 10 cheat sheets + 8 post-mortems
- Agent 10: Sub-Project K — 8 CV + 8 NLP pre-LLM concepts

---

## Verification Checklist

After completion, run:

```bash
# modern-ai stub check
python3 -c "
import json, glob
for f in sorted(glob.glob('modern-ai/notebooks/3[6-9]-*.ipynb') + glob.glob('modern-ai/notebooks/[45][0-9]-*.ipynb')):
    nb = json.load(open(f))
    code_lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type'] == 'code')
    status = '✅' if code_lines >= 600 else '⚠️ '
    print(f'{status} {f.split(\"/\")[-1]}: {code_lines} lines')
"

# system-design depth check
for f in system-design/patterns/1[4-9]-*.md system-design/patterns/2[0-9]-*.md system-design/patterns/3[01]-*.md; do
    words=$(wc -w < "$f")
    status="⚠️ "
    [ "$words" -ge 1800 ] && status="✅"
    echo "$status $words words: $f"
done

# coding coverage check
ls coding/data-structures/ | wc -l  # should be 21+

# diagram coverage check
ls arch-review/diagrams/ | grep -v README | wc -l  # should be 90+

# concept stubs check
for f in modern-ai/concepts/3[6-9]-*.md modern-ai/concepts/[45][0-9]-*.md; do
    words=$(wc -w < "$f")
    status="⚠️ "
    [ "$words" -ge 800 ] && status="✅"
    echo "$status $words words: $f"
done
```
