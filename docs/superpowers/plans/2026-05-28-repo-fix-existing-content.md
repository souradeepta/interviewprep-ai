# Repo Fix Existing Content Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all five existing content gaps — modern-ai notebooks 46/48-55, system-design patterns 14-31, coding section, ml/ notebooks, and arch-review diagrams — so every section is complete and interview-ready.

**Architecture:** Two-phase parallel batch. Phase 1 runs Agents 1-3 in parallel (sub-projects A, B, C — medium effort). Phase 2 runs Agents 4-5 in parallel (sub-projects D, E — large effort) after Phase 1 validation passes.

**Tech Stack:** Python, numpy, torch, matplotlib, sklearn (notebooks); Mermaid (diagrams); Markdown; nbformat; pytest

**Spec:** `docs/superpowers/specs/2026-05-28-repo-fix-existing-content-design.md`

---

## File Map

### Phase 1

**Agent 1 — Sub-project A (modern-ai notebooks 46, 48-55):**
- Modify: `modern-ai/notebooks/46-neuron-importance-scoring.ipynb`
- Modify: `modern-ai/notebooks/48-model-cascading.ipynb`
- Modify: `modern-ai/notebooks/49-latency-sla-prediction.ipynb`
- Modify: `modern-ai/notebooks/50-cache-aware-scheduling.ipynb`
- Modify: `modern-ai/notebooks/51-distributed-inference.ipynb`
- Modify: `modern-ai/notebooks/52-heterogeneous-ensemble.ipynb`
- Modify: `modern-ai/notebooks/53-conditional-computation.ipynb`
- Modify: `modern-ai/notebooks/54-context-compression.ipynb`
- Modify: `modern-ai/notebooks/55-online-knowledge-distillation.ipynb`
- Reference: `modern-ai/concepts/` (matching concept file for each notebook)
- Reference: `modern-ai/notebooks/36-token-pruning-merging.ipynb` (gold standard, 621 lines)

**Agent 2 — Sub-project B (system-design patterns 14-31):**
- Modify: `system-design/patterns/14-ab-testing.md` through `system-design/patterns/31-disaster-recovery.md` (18 files)
- Reference: `mlops/concepts/01-data-pipelines.md` (gold standard, 4167 words, 8-section format)

**Agent 3 — Sub-project C (coding section):**
- Create: `coding/data-structures/linked-lists.md`
- Create: `coding/data-structures/stacks-queues.md`
- Create: `coding/data-structures/trees.md`
- Create: `coding/data-structures/graphs.md`
- Create: `coding/data-structures/heaps.md`
- Create: `coding/data-structures/hash-tables.md`
- Create: `coding/data-structures/tries.md`
- Create: `coding/data-structures/union-find.md`
- Create: `coding/data-structures/segment-trees.md`
- Create: `coding/algorithms/sorting.md`
- Create: `coding/algorithms/binary-search.md`
- Create: `coding/algorithms/sliding-window.md`
- Create: `coding/algorithms/two-pointers.md`
- Create: `coding/algorithms/dynamic-programming.md`
- Create: `coding/algorithms/backtracking.md`
- Create: `coding/algorithms/greedy.md`
- Create: `coding/algorithms/recursion.md`
- Create: `coding/algorithms/graph-traversal.md`
- Create: `coding/algorithms/divide-conquer.md`
- Create: `coding/algorithms/bit-manipulation.md`
- Create: `coding/algorithms/string-patterns.md`
- Modify: `coding/README.md`
- Reference: `coding/data-structures/arrays-strings.md` (existing file, follow its format)

### Phase 2

**Agent 4 — Sub-project D (ml/ notebooks):**
- Create: `ml/notebooks/` directory (40 notebooks total)
  - `01-activation-functions.ipynb` through `40-weight-initialization.ipynb`
  - Full mapping: see Task D1
- Reference: `ai/notebooks/01-gradient-descent.ipynb` (gold standard 12-cell format)
- Reference: `ml/concepts/` (one concept file per notebook for content)

**Agent 5 — Sub-project E (arch-review diagrams):**
- Create 81 files in `arch-review/diagrams/`:
  - 3 files × 27 systems (systems 03, 04, 06-30)
  - Pattern: `{NN}-{system-name}-01-system-architecture.md`
  - Pattern: `{NN}-{system-name}-02-application-architecture.md`
  - Pattern: `{NN}-{system-name}-03-process-flow.md`
- Reference: `arch-review/diagrams/01-customer-service-01-system-architecture.md` (existing, follow format)
- Reference: `arch-review/systems/` (read each system file for diagram content)

---

## PHASE 1 — AGENT 1: Sub-project A

> **Agent 1 scope:** Expand modern-ai notebooks 46 and 48-55 from 320-480 lines to 600+ lines each using the 16-cell format. Notebook 47 is already complete (619 lines) — skip it.

### Task A1: Validate current state and set up reference

- [ ] **Step 1: Check current line counts**

  ```bash
  cd /home/sbisw/github/interviewprep-ml
  python3 -c "
  import json, glob
  targets = ['46','48','49','50','51','52','53','54','55']
  for f in sorted(glob.glob('modern-ai/notebooks/[45][0-9]-*.ipynb')):
      num = f.split('/')[-1].split('-')[0]
      if num in targets:
          nb = json.load(open(f))
          lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type']=='code')
          cells = len(nb['cells'])
          print(f'{num}: {lines} code lines, {cells} cells — {f.split(\"/\")[-1]}')
  "
  ```

  Expected: each shows current line count (320-480 range) and cell count.

- [ ] **Step 2: Read the gold standard notebook**

  Read `modern-ai/notebooks/36-token-pruning-merging.ipynb` to understand the 16-cell structure, code depth, and quality bar.

  The 16-cell structure must be:
  1. Title + objectives (Markdown)
  2. Imports + device + seeds (Code)
  3. Level 1 header (Markdown)
  4. Level 1 numpy code, 50-80 lines (Code)
  5. Level 2 header (Markdown)
  6. Level 2 torch code, 100-140 lines (Code)
  7. RW Example 1 header (Markdown)
  8. RW Example 1 code, 60-90 lines (Code)
  9. RW Example 2 header (Markdown)
  10. RW Example 2 code, 60-90 lines (Code)
  11. RW Example 3 header (Markdown)
  12. RW Example 3 code, 60-90 lines (Code)
  13. Comparison header (Markdown)
  14. Comparison code + matplotlib, 40-60 lines (Code)
  15. Key Takeaways with table (Markdown)
  16. Exercises (Markdown)

### Task A2: Expand notebook 46 (neuron-importance-scoring)

- [ ] **Step 1: Read current notebook and concept file**

  Read `modern-ai/notebooks/46-neuron-importance-scoring.ipynb` and `modern-ai/concepts/46-neuron-importance-scoring.md` to understand current state and content requirements.

- [ ] **Step 2: Rewrite notebook to 16-cell 600+ line format**

  Use nbformat to rewrite. Required content for this topic:
  - **Level 1 (numpy):** Compute gradient-based importance scores on a toy linear network. Score each neuron: `importance[i] = |weight[i] * gradient[i]|`. Print top-K neurons.
  - **Level 2 (torch):** Full MLP with Taylor expansion importance scoring. Track scores across layers. Include OOM-safe batch gradient accumulation.
  - **RW1:** BERT layer importance — score attention heads by gradient magnitude on SST-2 task, prune bottom 20%.
  - **RW2:** Structured pruning pipeline — score → sort → prune → fine-tune → evaluate accuracy vs sparsity.
  - **RW3:** Incremental importance updates — update scores online during training without full recomputation.
  - **Comparison:** Plot accuracy vs sparsity for random pruning vs importance-guided pruning.

  ```python
  import nbformat

  nb = nbformat.read('modern-ai/notebooks/46-neuron-importance-scoring.ipynb', as_version=4)
  # Build new cells list following 16-cell spec
  # Write with nbformat.write()
  nbformat.write(nb, 'modern-ai/notebooks/46-neuron-importance-scoring.ipynb')
  ```

- [ ] **Step 3: Validate**

  ```bash
  python3 -c "
  import json
  f = 'modern-ai/notebooks/46-neuron-importance-scoring.ipynb'
  nb = json.load(open(f))
  lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type']=='code')
  cells = len(nb['cells'])
  print(f'Cells: {cells} (need 16), Code lines: {lines} (need 600+)')
  assert cells == 16, f'Need 16 cells, got {cells}'
  assert lines >= 600, f'Need 600+ lines, got {lines}'
  print('PASS')
  "
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add modern-ai/notebooks/46-neuron-importance-scoring.ipynb
  git commit -m "feat: expand notebook 46 neuron-importance-scoring to 16-cell 600+ line format"
  ```

### Task A3: Expand notebooks 48-50

Repeat Task A2 pattern for each. Required content per topic:

**48 — model-cascading:**
- L1: Simple 2-model cascade: fast model → confidence check → slow model fallback. Numpy simulation with synthetic accuracy/latency data.
- L2: Torch cascade with learned confidence threshold. Benchmark fast-only vs cascade vs slow-only throughput/accuracy.
- RW1: BERT-tiny → BERT-base cascade for classification. Route by softmax entropy.
- RW2: Cost-aware cascade — optimize cascade for latency budget $T$ ms.
- RW3: N-model cascade with adaptive threshold per stage.
- Comparison: latency vs accuracy across cascade configurations.

**49 — latency-sla-prediction:**
- L1: Linear regression on request features to predict latency. Numpy implementation with toy data.
- L2: Torch MLP latency predictor. Features: batch_size, seq_len, model_params. Calibration curve.
- RW1: Online latency prediction with exponential moving average updates.
- RW2: SLA violation detection — predict whether request will exceed 200ms p99 threshold.
- RW3: Feedback loop — use real latency measurements to retrain predictor.
- Comparison: MAE of linear vs tree vs MLP predictor on held-out requests.

**50 — cache-aware-scheduling:**
- L1: Request scheduler that groups requests by KV-cache prefix to maximize reuse. Numpy simulation.
- L2: Torch-compatible scheduler with priority queue, cache hit rate tracking, batch formation.
- RW1: vLLM-style prefix caching — hash request prefix, lookup in cache, skip computation on hit.
- RW2: Speculative scheduling — pre-load likely next requests into cache based on session history.
- RW3: Multi-tenant cache partitioning — allocate cache slots per tenant by SLA priority.
- Comparison: Throughput and cache hit rate for FIFO vs cache-aware vs speculative scheduling.

- [ ] **Step 1: Expand notebook 48 using content spec above and save**
- [ ] **Step 2: Validate notebook 48 (16 cells, 600+ lines)**
- [ ] **Step 3: Expand notebook 49 using content spec above and save**
- [ ] **Step 4: Validate notebook 49 (16 cells, 600+ lines)**
- [ ] **Step 5: Expand notebook 50 using content spec above and save**
- [ ] **Step 6: Validate notebook 50 (16 cells, 600+ lines)**

  ```bash
  python3 -c "
  import json, glob
  for n in ['48','49','50']:
      f = next(f for f in glob.glob(f'modern-ai/notebooks/{n}-*.ipynb'))
      nb = json.load(open(f))
      lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type']=='code')
      cells = len(nb['cells'])
      status = 'PASS' if cells==16 and lines>=600 else 'FAIL'
      print(f'{status}: {f.split(\"/\")[-1]} — {cells} cells, {lines} lines')
  "
  ```

- [ ] **Step 7: Commit**

  ```bash
  git add modern-ai/notebooks/48-*.ipynb modern-ai/notebooks/49-*.ipynb modern-ai/notebooks/50-*.ipynb
  git commit -m "feat: expand modern-ai notebooks 48-50 to 16-cell 600+ line format"
  ```

### Task A4: Expand notebooks 51-55

Required content per topic:

**51 — distributed-inference:**
- L1: Tensor parallelism simulation — split weight matrix across 2 "GPUs" (numpy arrays), compute matmul, gather.
- L2: Torch DistributedDataParallel simulation. Model sharding across devices, benchmark allreduce overhead.
- RW1: Pipeline parallelism — split transformer layers across 4 stages, measure inter-stage latency.
- RW2: Continuous batching across distributed nodes — load balance requests by current queue depth.
- RW3: Fault tolerance — detect node failure via heartbeat, reroute requests, recover state.
- Comparison: Single-GPU vs tensor-parallel vs pipeline-parallel throughput and latency.

**52 — heterogeneous-ensemble:**
- L1: Ensemble of 3 numpy classifiers (logistic, decision stump, naive bayes). Majority vote + weighted vote.
- L2: Torch heterogeneous ensemble (CNN + LSTM + MLP on same task). Learnable mixing weights.
- RW1: Quality-cost routing ensemble — route cheap inputs to small model, hard inputs to large model.
- RW2: Stacking ensemble with meta-learner trained on validation predictions.
- RW3: Online ensemble weight adaptation — update weights based on recent prediction accuracy.
- Comparison: Individual model accuracy vs fixed ensemble vs adaptive ensemble.

**53 — conditional-computation:**
- L1: Gating function that skips computation below confidence threshold. Numpy simulation.
- L2: Torch mixture-of-experts with top-k routing. Load balancing loss. Expert utilization tracking.
- RW1: Conditional FFN layers — skip expensive FFN if attention output is already high confidence.
- RW2: Adaptive depth — exit early at layer L if confidence exceeds threshold.
- RW3: Input-conditioned compute — route math tokens to specialized expert, language to general.
- Comparison: FLOPs vs accuracy for always-compute vs conditional compute at different thresholds.

**54 — context-compression:**
- L1: Simple truncation vs sliding window compression. Compare information retention on toy sequences.
- L2: Torch attention-based context compression — learn which tokens to keep via learned scoring.
- RW1: RAG context compression — compress retrieved documents before feeding to generator.
- RW2: KV-cache compression — evict low-attention-weight tokens from KV cache during generation.
- RW3: Hierarchical compression — compress old context into summary embeddings, keep recent tokens verbatim.
- Comparison: Perplexity vs compression ratio for truncation vs sliding window vs learned compression.

**55 — online-knowledge-distillation:**
- L1: Offline KD with fixed teacher. Student learns from teacher soft labels. Numpy toy example.
- L2: Torch online KD — teacher and student train simultaneously. Mutual learning between peers.
- RW1: Self-distillation — deeper layers teach shallower layers within same model.
- RW2: Data-free online KD — generate pseudo-data via GAN, distill without original training data.
- RW3: Continual distillation — distill new task into student without forgetting old tasks.
- Comparison: Accuracy of baseline student vs offline KD vs online KD vs self-distillation.

- [ ] **Step 1: Expand notebooks 51-55 one at a time using specs above**
- [ ] **Step 2: Validate all 5**

  ```bash
  python3 -c "
  import json, glob
  for n in ['51','52','53','54','55']:
      f = next(f for f in glob.glob(f'modern-ai/notebooks/{n}-*.ipynb'))
      nb = json.load(open(f))
      lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type']=='code')
      cells = len(nb['cells'])
      status = 'PASS' if cells==16 and lines>=600 else 'FAIL'
      print(f'{status}: {f.split(\"/\")[-1]} — {cells} cells, {lines} lines')
  "
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add modern-ai/notebooks/51-*.ipynb modern-ai/notebooks/52-*.ipynb modern-ai/notebooks/53-*.ipynb modern-ai/notebooks/54-*.ipynb modern-ai/notebooks/55-*.ipynb
  git commit -m "feat: expand modern-ai notebooks 51-55 to 16-cell 600+ line format"
  ```

---

## PHASE 1 — AGENT 2: Sub-project B

> **Agent 2 scope:** Expand 18 system-design patterns (14-31) from ~1000 words to 1800-2400 words each. Add: Failure Scenarios section, Cost Model section, expand Q&A to 8-10 judgment questions.

### Task B1: Set up reference and validate current state

- [ ] **Step 1: Check current word counts**

  ```bash
  cd /home/sbisw/github/interviewprep-ml
  for f in system-design/patterns/1[4-9]-*.md system-design/patterns/2[0-9]-*.md system-design/patterns/3[01]-*.md; do
      words=$(wc -w < "$f")
      echo "$words words: $(basename $f)"
  done
  ```

- [ ] **Step 2: Read the gold standard reference**

  Read `mlops/concepts/01-data-pipelines.md` to understand the 8-section format and depth expected. Pay attention to:
  - "Failure Scenarios" section structure (symptom → root cause → detection → fix)
  - "Cost Model" section (use real numbers: $/request, requests/day, total $/month)
  - Interview Q&A style (judgment questions: "when would you NOT...", "how would you debug...", "what's the trade-off between...")

### Task B2: Expand patterns 14-19

For each pattern, READ the existing file first, then ADD the missing sections. Do not remove existing content — extend it.

**Sections to ADD to each file (if missing or thin):**
1. Expand "How It Works" to include a Mermaid `flowchart TD` diagram
2. Add **Failure Scenarios** section with 3-5 entries in this format:
   ```
   ### Failure: [Name]
   **Symptom:** [Observable sign something is wrong]
   **Root Cause:** [Why it happens]
   **Detection:** [How to catch it — metric, alert, log pattern]
   **Fix:** [Specific remediation steps]
   ```
3. Add **Cost Model** section with envelope calculation:
   ```
   | Resource | Unit Cost | Volume | Monthly Cost |
   |----------|-----------|--------|-------------|
   | [item]   | $X        | Y/day  | $Z          |
   | Total    |           |        | $W          |
   ```
4. Expand Interview Q&A to 8-10 questions. Each answer 2-3 sentences. Focus on judgment:
   - "When would you NOT use [this approach]?"
   - "What breaks first when [this system] scales to 10x traffic?"
   - "How would you debug [specific failure] in production?"
   - "What's the trade-off between [approach A] and [approach B]?"

**Pattern-specific failure scenarios and cost models:**

**14 — ab-testing:**
- Failure 1: Sample ratio mismatch — treatment group gets 60% instead of 50% due to cookie bug → inflate significance → wrong decision. Fix: monitor group sizes daily, auto-pause if ratio drifts >5%.
- Failure 2: Novelty effect — users engage more with any change in week 1 → premature winner → regression at week 4. Fix: run minimum 2 weeks, check week-over-week stability.
- Failure 3: Multiple testing problem — running 20 metrics, 1 will be significant by chance → p-hack. Fix: pre-register primary metric, Bonferroni correct secondary metrics.
- Cost model: 500K users/day × 30 days × $0.001 event storage = $15K/month; plus 2 engineer-weeks setup = $20K → $35K per experiment.

**15 — drift-detection:**
- Failure 1: PSI threshold too high (0.2 vs 0.1 standard) → drift undetected for 3 months → silent model degradation. Fix: use 0.1 for warning, 0.25 for critical.
- Failure 2: Monitoring only 5 features → drifting feature 12 undetected. Fix: monitor all features in top-30 importance, not just top-5.
- Failure 3: Alert fires but no runbook → oncall doesn't know what to do → delayed response. Fix: every alert links to runbook with 3-step triage.
- Cost model: 100 features × hourly PSI check × $0.0001/compute = $72/month. Shadow model inference: 10% traffic × $0.002/request × 1M requests/day × 30 days = $6K/month.

**16 — monitoring-and-observability:**
- Failure 1: Metric cardinality explosion — label {user_id} on per-request metric → 10M time series → Prometheus OOM. Fix: never use high-cardinality labels on per-request metrics; use aggregation.
- Failure 2: Alert fatigue — 200 alerts fire per day, oncall ignores them → real outage missed. Fix: alert only on SLO burn rate, not individual metrics.
- Failure 3: Dashboard shows p50 latency as healthy while p99 is 10s. Fix: always plot p50/p95/p99 on same panel.
- Cost model: 1M metrics/day × $0.10/1000 metrics (Datadog) = $100/day = $3K/month; dashboard maintenance = 0.5 engineer/week.

**17 — model-debugging:**
- Failure 1: Debugging on full dataset → hours to iterate. Fix: create representative 1% debug dataset with all failure modes present.
- Failure 2: Error analysis on random sample → biased toward majority class → misses rare but important failures. Fix: stratified sampling by prediction confidence buckets.
- Failure 3: Fix bug in feature code, retrain, same error → bug exists in serving code too. Fix: feature parity testing between training and serving pipelines.
- Cost model: 1 ML engineer debugging × $200/hour × 8 hours/week = $1,600/week; tooling (notebooks, labeling) $500/month → $7K/month total.

**18 — model-explainability:**
- Failure 1: SHAP computation on full dataset → 48 hours → not actionable. Fix: TreeExplainer on 1K sample for exploration, save full SHAP for compliance reports only.
- Failure 2: LIME explanations inconsistent across runs → can't reproduce for compliance. Fix: fix LIME random seed, log explanation with model version.
- Failure 3: Explainability post-hoc only → doesn't catch spurious correlations until production. Fix: run SHAP on validation set before deployment, flag features with unexpected high importance.
- Cost model: SHAP on 100K samples × 50 features = 10M SHAP values × $0.0001/value = $1K/run; scheduled weekly = $4K/month.

**19 — interpretability:**
- Failure 1: Attention weights misinterpreted as importance → wrong debugging direction. Fix: use gradient × attention (attention rollout) not raw attention for importance.
- Failure 2: Probing classifier shows concept present, assume model "understands" → probing accuracy inflated by linear correlation. Fix: causally verify with activation patching.
- Failure 3: Interpretability tool from paper doesn't reproduce on production model → different architecture. Fix: verify tool compatibility before building workflow around it.
- Cost model: Activation analysis compute on 10K examples × 12 layers × 768 dims = 92M floats × $0.0001/MB = $9/run; quarterly audit = $36/year.

- [ ] **Step 1: Expand pattern 14 (ab-testing.md) with all 4 additions**
- [ ] **Step 2: Validate word count ≥ 1800**
  ```bash
  words=$(wc -w < system-design/patterns/14-ab-testing.md); [ "$words" -ge 1800 ] && echo "PASS: $words" || echo "FAIL: $words"
  ```
- [ ] **Step 3: Expand patterns 15-19 one at a time, validate each**
- [ ] **Step 4: Commit**
  ```bash
  git add system-design/patterns/14-*.md system-design/patterns/15-*.md system-design/patterns/16-*.md system-design/patterns/17-*.md system-design/patterns/18-*.md system-design/patterns/19-*.md
  git commit -m "feat: expand system-design patterns 14-19 to 1800+ words with failure scenarios and cost models"
  ```

### Task B3: Expand patterns 20-25

**Pattern-specific content:**

**20 — feature-importance-tracking:**
- Failure 1: Feature importance changes post-deployment → model silently uses different signal → accuracy degrades. Root cause: feature distribution shift. Fix: track top-10 SHAP feature importance weekly, alert if rank changes by >3 positions.
- Failure 2: Permutation importance computed on test set → leaks test set structure into feature selection → overfitting. Fix: compute on validation set, never test.
- Cost model: SHAP importance tracking: 10K samples × weekly × $0.001/sample = $10K/year.

**21 — reproducibility:**
- Failure 1: Same code, different hardware → different results (non-deterministic CUDA ops). Fix: `torch.use_deterministic_algorithms(True)`, accept 10-15% slower training.
- Failure 2: Dataset versioned but preprocessing code not versioned → can't reproduce 6-month-old result. Fix: hash preprocessing pipeline code into experiment metadata.
- Cost model: DVC storage for 1TB dataset history × $0.023/GB/month = $23/month; experiment tracking (MLflow hosted) = $200/month.

**22 — cost-optimization:**
- Failure 1: Spot instance preemption during final epoch → 40-hour training run lost. Fix: checkpoint every 30 minutes, use restart-from-checkpoint on new instance.
- Failure 2: Development cluster runs 24/7 → $10K/month wasted. Fix: auto-shutdown idle clusters after 1 hour, use pre-emptible instances for non-critical jobs.
- Cost model: 8× A100 on-demand = $32/hr vs spot = $9.60/hr (70% savings). 100hr/week × 52 = 5200hr/year × $22.40 savings = $116K/year saved.

**23 — production-readiness:**
- Failure 1: Model passes offline eval but crashes under production load → not load-tested. Fix: load test at 2× expected peak QPS before deployment.
- Failure 2: Rollback procedure untested → takes 4 hours to roll back during incident. Fix: test rollback quarterly, document procedure, target <15 minute rollback.
- Cost model: Load testing: 4 engineer-hours × $200/hr = $800 per release. Staging environment: $2K/month. Total readiness overhead: $5K/month.

**24 — bias-detection:**
- Failure 1: Bias metrics computed on overall dataset → disparity in 5% subgroup masked by majority. Fix: always compute metrics per demographic group, not just overall.
- Failure 2: Historical data has proxy discrimination (zip code → race) → model learns illegal proxy. Fix: audit feature correlation with protected attributes, drop features with correlation >0.3.
- Cost model: Fairness audit: 2 weeks × $10K/week ML engineer = $20K per audit. Continuous bias monitoring: $500/month compute.

**25 — fairness-metrics:**
- Failure 1: Optimize for demographic parity → accuracy drops 8% on majority group → business rejects. Fix: use equalized odds as starting point, negotiate acceptable accuracy-fairness tradeoff with product.
- Failure 2: Fairness metric improves on internal benchmark but degrades on new deployment population. Fix: test fairness metrics on deployment population distribution, not just benchmark.
- Cost model: Labeling for fairness ground truth: 50K samples × $0.10/sample = $5K per dataset. Annual fairness reporting: 1 week engineer = $10K.

- [ ] **Step 1: Expand patterns 20-25 one at a time with content above**
- [ ] **Step 2: Validate each hits 1800+ words**
  ```bash
  for f in system-design/patterns/2[0-5]-*.md; do
      words=$(wc -w < "$f"); [ "$words" -ge 1800 ] && echo "PASS $words: $(basename $f)" || echo "FAIL $words: $(basename $f)"
  done
  ```
- [ ] **Step 3: Commit**
  ```bash
  git add $(for i in 20 21 22 23 24 25; do echo system-design/patterns/${i}-*.md; done)
  git commit -m "feat: expand system-design patterns 20-25 to 1800+ words with failure scenarios and cost models"
  ```

### Task B4: Expand patterns 26-31

**Pattern-specific content:**

**26 — data-governance:**
- Failure 1: PII leaks into training data → regulatory fine. Root: no automated PII scan in ingestion pipeline. Fix: Presidio/regex scan on all new data, quarantine matches for human review.
- Cost model: DLP scanning: 1TB/day × $0.001/GB = $1/day = $365/year. Data catalog (Collibra): $5K/month. Governance program: 1 FTE = $200K/year.

**27 — ml-governance:**
- Failure 1: Model deployed without approval → audit finds no documentation → regulatory issue. Fix: model cards required, deployment blocked until signed off by model review board.
- Cost model: Model review: 4 hours × $200/hr × 20 models/year = $16K/year. Governance tooling (Weights&Biases Teams): $3K/month.

**28 — privacy-preserving-ml:**
- Failure 1: Federated model vulnerable to gradient inversion attack → client data reconstructed from gradient updates. Fix: add DP-SGD noise (ε=1.0) to gradients before aggregation.
- Failure 2: Secure aggregation trusted server assumption violated → server operator can see individual gradients. Fix: use cryptographic secure aggregation (SecAgg protocol).
- Cost model: DP-SGD overhead: 2-3× longer training. Secure aggregation cryptography: 10-30% communication overhead. DP accuracy cost: 1-3% accuracy drop at ε=1.0.

**29 — differential-privacy:**
- Failure 1: ε budget exhausted after 100 queries → no more queries allowed → analytics team blocked. Fix: plan ε budget allocation upfront, use composition theorem to track.
- Failure 2: Adding DP noise kills signal on small subgroups (n<50) → analytics useless for minorities. Fix: minimum group size of 100 before reporting, suppress smaller groups.
- Cost model: DP-SGD on ImageNet: adds 40% training time = $0.40 × GPU-hours. Privacy accountant: negligible.

**30 — federated-learning:**
- Failure 1: Stragglers — 10% of clients take 10× longer → synchronous aggregation blocked. Fix: async aggregation with FedAsync, or synchronous with 20% straggler tolerance (drop slow clients).
- Failure 2: Client data heterogeneity → global model converges to majority distribution → minority clients get poor accuracy. Fix: personalized FL (pFedMe), local fine-tuning on top of global model.
- Cost model: Central server: $2K/month compute. Communication bandwidth: 100K clients × 100MB model × 10 rounds = 100TB/round × $0.09/GB = $9K/round.

**31 — disaster-recovery:**
- Failure 1: Backup restoration tested annually → restoration procedure bitrot → actual recovery takes 6 hours not 1. Fix: monthly restore drill, measure actual RTO against SLA.
- Failure 2: RPO assumed 1 hour (last backup) → database replication lag was 4 hours → lost 4 hours of data. Fix: continuously monitor replication lag, alert if lag > 0.5× RPO target.
- Cost model: Multi-region standby: 2× primary compute cost = $10K/month overhead. Cold standby: $500/month but RTO = 1-2 hours. RPO/RTO requirements drive 5-20× cost differences.

- [ ] **Step 1: Expand patterns 26-31 one at a time**
- [ ] **Step 2: Validate all 6 hit 1800+ words**
  ```bash
  for f in system-design/patterns/2[6-9]-*.md system-design/patterns/3[01]-*.md; do
      words=$(wc -w < "$f"); [ "$words" -ge 1800 ] && echo "PASS $words: $(basename $f)" || echo "FAIL $words: $(basename $f)"
  done
  ```
- [ ] **Step 3: Commit**
  ```bash
  git add $(for i in 26 27 28 29 30 31; do echo system-design/patterns/${i}-*.md; done)
  git commit -m "feat: expand system-design patterns 26-31 to 1800+ words with failure scenarios and cost models"
  ```

---

## PHASE 1 — AGENT 3: Sub-project C

> **Agent 3 scope:** Create 9 new data-structure files + 12 algorithm files + update coding/README.md. Follow the format of the existing `coding/data-structures/arrays-strings.md`.

### Task C1: Read reference and understand format

- [ ] **Step 1: Read existing reference file**

  Read `coding/data-structures/arrays-strings.md` to understand exact format. Every new file must follow the same structure:
  1. Topic header + one-line description
  2. **Core Patterns table** — `| Pattern | When to use | Time | Space |`
  3. **Python Implementation** — complete typed + docstring implementations
  4. **Complexity Summary** — table of operations
  5. **Interview Recognition Template** — "If you see X, think Y"
  6. **Worked Examples** — 3-5 problems with solutions (not just stubs)

- [ ] **Step 2: Read coding/README.md**

  Read `coding/README.md` to understand current navigation structure before modifying it.

### Task C2: Create data-structure files (9 new files)

- [ ] **Step 1: Create `coding/data-structures/linked-lists.md`**

  Must include:
  - Core patterns: fast/slow pointers, reverse in-place, detect cycle (Floyd's algorithm), merge sorted lists
  - Python `ListNode` class + `reverse_list`, `has_cycle`, `merge_sorted`, `find_middle` implementations with type hints
  - Complexity table: access O(n), insert O(1) with pointer, delete O(1) with pointer, search O(n)
  - Recognition template: "Two-pointer problems on lists usually use fast/slow; reversal problems use prev/curr/next"
  - 4 worked examples: reverse list, detect cycle, find middle, merge two sorted lists

- [ ] **Step 2: Create `coding/data-structures/stacks-queues.md`**

  Must include:
  - Core patterns: monotonic stack (next greater element), queue via two stacks, BFS queue, deque for sliding window
  - Python: `Stack` class, `Queue` (collections.deque), `MonotonicStack`, `MinStack`
  - Complexity: push O(1), pop O(1), peek O(1), search O(n)
  - Recognition template: "Nested structure (parentheses, expressions) → stack. Level-by-level traversal → queue."
  - 4 worked examples: valid parentheses, next greater element, min stack, sliding window max

- [ ] **Step 3: Create `coding/data-structures/trees.md`**

  Must include:
  - Core patterns: DFS (pre/in/post order), BFS level-order, path problems, LCA, BST operations
  - Python: `TreeNode`, `inorder_traversal` (recursive + iterative), `level_order`, `lca`, `is_valid_bst`
  - Complexity: search O(h), insert O(h), delete O(h) where h = log n (balanced)
  - Recognition template: "Parent-child relationship → tree. Sorted structure → BST. Level-by-level → BFS."
  - 5 worked examples: inorder traversal, max depth, LCA, validate BST, path sum

- [ ] **Step 4: Create `coding/data-structures/graphs.md`**

  Must include:
  - Core patterns: adjacency list vs matrix, DFS/BFS traversal, cycle detection, topological sort, connected components
  - Python: `Graph` adjacency list class, `dfs`, `bfs`, `has_cycle`, `topological_sort`, `num_components`
  - Complexity: DFS/BFS O(V+E), topological sort O(V+E)
  - Recognition template: "Relationships/dependencies → graph. Shortest path → BFS (unweighted) or Dijkstra (weighted)."
  - 5 worked examples: number of islands, course schedule, clone graph, number of connected components, bipartite check

- [ ] **Step 5: Create `coding/data-structures/heaps.md`**

  Must include:
  - Core patterns: top-K elements, K-th largest/smallest, merge K sorted lists, sliding window median
  - Python: `heapq` usage, `MinHeap`/`MaxHeap` wrappers, `kth_largest`, `merge_k_sorted`, `running_median`
  - Complexity: push O(log n), pop O(log n), peek O(1)
  - Recognition template: "Top-K or running order → heap. Two heaps for median. Priority queue for Dijkstra."
  - 4 worked examples: kth largest, top K frequent, merge K sorted lists, find median from data stream

- [ ] **Step 6: Create `coding/data-structures/hash-tables.md`**

  Must include:
  - Core patterns: frequency counting, two-sum pattern, grouping/anagram detection, caching/memoization
  - Python: `Counter`, `defaultdict`, `OrderedDict`, LRU cache implementation
  - Complexity: insert O(1) avg, lookup O(1) avg, delete O(1) avg; worst case O(n)
  - Recognition template: "Count frequencies → Counter. Need O(1) lookup → dict. Preserve insertion order → OrderedDict."
  - 5 worked examples: two sum, group anagrams, top K frequent, LRU cache, longest consecutive sequence

- [ ] **Step 7: Create `coding/data-structures/tries.md`**

  Must include:
  - Core patterns: prefix search, word insertion/lookup, wildcard search, autocomplete
  - Python: `TrieNode`, `Trie` with `insert`, `search`, `starts_with`, `get_words_with_prefix`
  - Complexity: insert O(m), search O(m), prefix O(m) where m = word length
  - Recognition template: "Prefix matching or autocomplete → Trie. Shared prefix optimization → Trie over set."
  - 3 worked examples: implement Trie, word search II (with Trie), replace words

- [ ] **Step 8: Create `coding/data-structures/union-find.md`**

  Must include:
  - Core patterns: connected components, cycle detection in undirected graph, dynamic connectivity
  - Python: `UnionFind` with path compression + union by rank. `find`, `union`, `connected`, `num_components`
  - Complexity: find O(α(n)) ≈ O(1) amortized, union O(α(n))
  - Recognition template: "Grouping elements, merging sets, detecting cycles in undirected → Union-Find."
  - 3 worked examples: number of provinces, redundant connection, accounts merge

- [ ] **Step 9: Create `coding/data-structures/segment-trees.md`**

  Must include:
  - Core patterns: range sum query, range min/max query, point update, lazy propagation for range updates
  - Python: `SegmentTree` array-based with `build`, `query(l,r)`, `update(i, val)`. Lazy propagation variant.
  - Complexity: build O(n), query O(log n), update O(log n)
  - Recognition template: "Range queries with updates → Segment Tree or BIT. Static range queries → prefix sums."
  - 3 worked examples: range sum query, range min query with point update, count of smaller numbers after self

- [ ] **Step 10: Validate all 9 files exist and are non-empty**
  ```bash
  for f in linked-lists stacks-queues trees graphs heaps hash-tables tries union-find segment-trees; do
      path="coding/data-structures/${f}.md"
      words=$(wc -w < "$path" 2>/dev/null || echo 0)
      [ "$words" -ge 400 ] && echo "PASS ($words words): $f" || echo "FAIL ($words words): $f"
  done
  ```

- [ ] **Step 11: Commit**
  ```bash
  git add coding/data-structures/
  git commit -m "feat: add 9 data structure topic files (linked-lists through segment-trees)"
  ```

### Task C3: Create algorithm files (12 new files)

- [ ] **Step 1: Create `coding/algorithms/sorting.md`**

  Must include: QuickSort, MergeSort, HeapSort, TimSort (Python's sort). Complexity comparison table. When to use each. Custom comparator patterns. 3 worked examples: sort by multiple keys, Dutch National Flag, sort matrix diagonals.

- [ ] **Step 2: Create `coding/algorithms/binary-search.md`**

  Must include: classic binary search, left/right boundary search, search on answer (monotonic function), rotated array search. Template: `lo, hi, mid = 0, len(a)-1, (lo+hi)//2`. Recognition: "Search in sorted space or minimizing/maximizing monotonic function → binary search on answer." 4 worked examples: search insert position, find minimum in rotated array, koko eating bananas, median of two sorted arrays.

- [ ] **Step 3: Create `coding/algorithms/sliding-window.md`**

  Must include: fixed-size window, variable-size window (expand/shrink), two-pointer window. Template: `l=0; for r in range(n): window.add(a[r]); while invalid: window.remove(a[l]); l+=1`. 4 worked examples: max sum subarray of size k, longest substring without repeating, minimum window substring, fruit into baskets.

- [ ] **Step 4: Create `coding/algorithms/two-pointers.md`**

  Must include: opposite-end pointers (sorted array), same-direction (fast/slow), partition pointer. Recognition template: "Sorted array + find pair/triplet with constraint → two pointers. Cycle detection → fast/slow." 4 worked examples: two sum II (sorted), 3sum, container with most water, partition list.

- [ ] **Step 5: Create `coding/algorithms/dynamic-programming.md`**

  Must include: top-down (memoization) vs bottom-up (tabulation), state definition framework, common patterns (0/1 knapsack, LCS, LIS, coin change, matrix DP). Template: "State: dp[i][j] = X. Transition: dp[i][j] = f(dp[i-1][j], dp[i][j-1])". 5 worked examples: climbing stairs, coin change, longest common subsequence, 0/1 knapsack, edit distance.

- [ ] **Step 6: Create `coding/algorithms/backtracking.md`**

  Must include: template (`choose → explore → unchoose`), pruning strategies, permutations vs combinations vs subsets. Template:
  ```python
  def backtrack(start, path):
      if done: result.append(path[:])
      for i in range(start, n):
          path.append(candidates[i])
          backtrack(i+1, path)
          path.pop()
  ```
  4 worked examples: subsets, permutations, combination sum, N-queens.

- [ ] **Step 7: Create `coding/algorithms/greedy.md`**

  Must include: greedy choice property, activity selection, interval scheduling, Huffman coding, when greedy fails. Recognition: "Locally optimal → globally optimal (verify with exchange argument). Intervals → sort by end time." 4 worked examples: jump game, meeting rooms II, task scheduler, minimum number of arrows.

- [ ] **Step 8: Create `coding/algorithms/recursion.md`**

  Must include: base case + recursive case, call stack analysis, tail recursion, common patterns (tree traversal, divide-and-conquer, memoized recursion). Conversion to iterative using explicit stack. 3 worked examples: fibonacci (naive vs memo), power function, flatten nested list.

- [ ] **Step 9: Create `coding/algorithms/graph-traversal.md`**

  Must include: DFS (recursive + iterative), BFS (shortest path), Dijkstra (weighted), Bellman-Ford (negative weights), topological sort (Kahn's algorithm). Complexity comparison table. 5 worked examples: word ladder, network delay time, cheapest flights within K stops, alien dictionary, parallel courses.

- [ ] **Step 10: Create `coding/algorithms/divide-conquer.md`**

  Must include: divide-conquer template, master theorem for complexity analysis, merge sort as canonical example, quick select. T(n) = aT(n/b) + O(n^d) → complexity formula. 3 worked examples: merge sort, majority element, maximum subarray (Kadane counts but also divide-conquer version).

- [ ] **Step 11: Create `coding/algorithms/bit-manipulation.md`**

  Must include: common bit tricks table (`n & (n-1)` removes lowest set bit, `n & (-n)` isolates lowest set bit, `n ^ n = 0`), XOR applications, bit masking for subsets. 4 worked examples: single number, count bits, reverse bits, subsets via bitmask.

- [ ] **Step 12: Create `coding/algorithms/string-patterns.md`**

  Must include: KMP (failure function), rolling hash (Rabin-Karp), palindrome (expand-around-center vs Manacher), anagram detection. Recognition: "Pattern matching → KMP or rolling hash. Substring search → sliding window + hash. Palindrome → expand around center." 4 worked examples: implement strStr(), longest palindromic substring, find all anagrams, repeated DNA sequences.

- [ ] **Step 13: Validate all 12 algorithm files exist and meet minimum length**
  ```bash
  for f in sorting binary-search sliding-window two-pointers dynamic-programming backtracking greedy recursion graph-traversal divide-conquer bit-manipulation string-patterns; do
      path="coding/algorithms/${f}.md"
      words=$(wc -w < "$path" 2>/dev/null || echo 0)
      [ "$words" -ge 400 ] && echo "PASS ($words words): $f" || echo "FAIL ($words words): $f"
  done
  ```

- [ ] **Step 14: Commit**
  ```bash
  git add coding/algorithms/
  git commit -m "feat: add coding/algorithms/ with 12 algorithm pattern files"
  ```

### Task C4: Update coding/README.md

- [ ] **Step 1: Update README to list all topics**

  Read current `coding/README.md`, then update it to list all 10 DS files and 12 algorithm files with one-line descriptions and links.

- [ ] **Step 2: Commit**
  ```bash
  git add coding/README.md
  git commit -m "docs: update coding/README.md with complete DS and algorithms topic index"
  ```

---

## PHASE 1 VALIDATION GATE

> Run this before dispatching Phase 2. All checks must pass.

### Task V1: Validate Phase 1 completion

- [ ] **Step 1: Validate modern-ai notebooks 46, 48-55**

  ```bash
  cd /home/sbisw/github/interviewprep-ml
  python3 -c "
  import json, glob
  targets = ['46','48','49','50','51','52','53','54','55']
  results = []
  for f in sorted(glob.glob('modern-ai/notebooks/[45][0-9]-*.ipynb')):
      num = f.split('/')[-1].split('-')[0]
      if num in targets:
          nb = json.load(open(f))
          lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type']=='code')
          cells = len(nb['cells'])
          ok = cells == 16 and lines >= 600
          results.append(ok)
          print(f'{'PASS' if ok else 'FAIL'}: {f.split('/')[-1]} — {cells} cells, {lines} lines')
  print(f'Sub-project A: {sum(results)}/{len(results)} passing')
  "
  ```

  Expected: 9/9 PASS.

- [ ] **Step 2: Validate system-design patterns 14-31**

  ```bash
  pass=0; fail=0
  for f in system-design/patterns/1[4-9]-*.md system-design/patterns/2[0-9]-*.md system-design/patterns/3[01]-*.md; do
      words=$(wc -w < "$f")
      if [ "$words" -ge 1800 ]; then echo "PASS ($words): $(basename $f)"; ((pass++))
      else echo "FAIL ($words): $(basename $f)"; ((fail++))
      fi
  done
  echo "Sub-project B: $pass passing, $fail failing"
  ```

  Expected: 18/18 PASS.

- [ ] **Step 3: Validate coding section**

  ```bash
  ds=$(ls coding/data-structures/*.md | wc -l)
  algo=$(ls coding/algorithms/*.md 2>/dev/null | wc -l)
  echo "Data structures: $ds (need 10)"
  echo "Algorithms: $algo (need 12)"
  [ "$ds" -ge 10 ] && [ "$algo" -ge 12 ] && echo "Sub-project C: PASS" || echo "Sub-project C: FAIL"
  ```

  Expected: 10 DS files, 12 algo files.

- [ ] **Step 4: Fix any failures before proceeding to Phase 2**

  If any check fails, re-run the relevant agent task for that file/pattern before continuing.

---

## PHASE 2 — AGENT 4: Sub-project D

> **Agent 4 scope:** Create 40 notebooks in `ml/notebooks/`. One notebook per concept in `ml/concepts/`, 12-cell format, modeled after `ai/notebooks/` section.

### Task D1: Set up reference and notebook mapping

- [ ] **Step 1: Read gold standard notebook**

  Read `ai/notebooks/01-gradient-descent.ipynb` to understand the exact 12-cell format expected.

- [ ] **Step 2: Confirm full notebook name mapping**

  The 40 notebooks map from alphabetically-sorted `ml/concepts/` files:

  | # | Notebook filename | Source concept |
  |---|------------------|----------------|
  | 01 | `01-activation-functions.ipynb` | activation-functions.md |
  | 02 | `02-attention-mechanism.ipynb` | attention-mechanism.md |
  | 03 | `03-batch-normalization.ipynb` | batch-normalization.md |
  | 04 | `04-class-imbalance.ipynb` | class-imbalance.md |
  | 05 | `05-cnns.ipynb` | cnns.md |
  | 06 | `06-cross-validation-strategies.ipynb` | cross-validation-strategies.md |
  | 07 | `07-data-leakage.ipynb` | data-leakage.md |
  | 08 | `08-distributed-training.ipynb` | distributed-training.md |
  | 09 | `09-domain-adaptation.ipynb` | domain-adaptation.md |
  | 10 | `10-dropout.ipynb` | dropout.md |
  | 11 | `11-early-stopping.ipynb` | early-stopping.md |
  | 12 | `12-ensemble-methods.ipynb` | ensemble-methods.md |
  | 13 | `13-evaluation-metrics.ipynb` | evaluation-metrics.md |
  | 14 | `14-feature-engineering.ipynb` | feature-engineering.md |
  | 15 | `15-gradient-accumulation.ipynb` | gradient-accumulation.md |
  | 16 | `16-hyperparameter-tuning.ipynb` | hyperparameter-tuning.md |
  | 17 | `17-knowledge-distillation.ipynb` | knowledge-distillation.md |
  | 18 | `18-layer-normalization.ipynb` | layer-normalization.md |
  | 19 | `19-learning-rate-schedules.ipynb` | learning-rate-schedules.md |
  | 20 | `20-loss-functions.ipynb` | loss-functions.md |
  | 21 | `21-meta-learning.ipynb` | meta-learning.md |
  | 22 | `22-mixed-precision-training.ipynb` | mixed-precision-training.md |
  | 23 | `23-model-compression.ipynb` | model-compression.md |
  | 24 | `24-model-selection.ipynb` | model-selection.md |
  | 25 | `25-momentum-and-acceleration.ipynb` | momentum-and-acceleration.md |
  | 26 | `26-naive-bayes.ipynb` | naive-bayes.md |
  | 27 | `27-neural-networks.ipynb` | neural-networks.md |
  | 28 | `28-optimization.ipynb` | optimization.md |
  | 29 | `29-overfitting-underfitting.ipynb` | overfitting-underfitting.md |
  | 30 | `30-probability-statistics.ipynb` | probability-statistics.md |
  | 31 | `31-pruning.ipynb` | pruning.md |
  | 32 | `32-quantization.ipynb` | quantization.md |
  | 33 | `33-regularization.ipynb` | regularization.md |
  | 34 | `34-rnns-lstms.ipynb` | rnns-lstms.md |
  | 35 | `35-supervised-learning.ipynb` | supervised-learning.md |
  | 36 | `36-support-vector-machines.ipynb` | support-vector-machines.md |
  | 37 | `37-transfer-learning.ipynb` | transfer-learning.md |
  | 38 | `38-transformers.ipynb` | transformers.md |
  | 39 | `39-unsupervised-learning.ipynb` | unsupervised-learning.md |
  | 40 | `40-weight-initialization.ipynb` | weight-initialization.md |

### Task D2: Create notebooks 01-10

For each notebook: read corresponding `ml/concepts/` file → create 12-cell notebook using nbformat → save to `ml/notebooks/`.

**12-cell structure for every notebook:**
```python
import nbformat

cells = [
    nbformat.v4.new_markdown_cell("# [Topic Name]\n\n## Learning Objectives\n1. ...\n2. ...\n3. ...\n4. ..."),
    nbformat.v4.new_code_cell("import numpy as np\nimport matplotlib.pyplot as plt\nimport torch\n...\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"),
    nbformat.v4.new_markdown_cell("## Level 1: Basic [Topic]"),
    nbformat.v4.new_code_cell("# 20-40 lines numpy implementation"),
    nbformat.v4.new_markdown_cell("## Level 2: Advanced [Topic]"),
    nbformat.v4.new_code_cell("# 60-100 lines torch/sklearn implementation with error handling"),
    nbformat.v4.new_markdown_cell("## Real-World Example 1: [Use case]"),
    nbformat.v4.new_code_cell("# 40-60 lines"),
    nbformat.v4.new_markdown_cell("## Real-World Example 2: [Use case]"),
    nbformat.v4.new_code_cell("# 40-60 lines"),
    nbformat.v4.new_markdown_cell("## Real-World Example 3: [Use case]"),
    nbformat.v4.new_code_cell("# 40-60 lines"),
    nbformat.v4.new_markdown_cell("## Comparison: When to Use What\n\n| Method | Use when | Trade-off |\n|--------|----------|-----------|"),
    nbformat.v4.new_code_cell("# matplotlib comparison plot"),
    nbformat.v4.new_markdown_cell("## Key Takeaways\n..."),
    nbformat.v4.new_markdown_cell("## Exercises\n1. ...\n2. ...\n3. ..."),
]
nb = nbformat.v4.new_notebook(cells=cells)
nbformat.write(nb, 'ml/notebooks/01-activation-functions.ipynb')
```

**Content guides for notebooks 01-10:**
- **01 activation-functions:** L1: ReLU/sigmoid/tanh from scratch numpy. L2: compare activations in torch MLP. RW1: dying ReLU diagnosis. RW2: GELU in transformer. RW3: activation function ablation study.
- **02 attention-mechanism:** L1: scaled dot-product attention numpy (QKV). L2: multi-head attention torch. RW1: self-attention in sequence classification. RW2: cross-attention encoder-decoder. RW3: attention visualization heatmap.
- **03 batch-normalization:** L1: BN forward/backward numpy. L2: BN vs LayerNorm vs GroupNorm comparison torch. RW1: BN in CNN training stability. RW2: BN at inference (frozen stats). RW3: BN vs no-BN training curves.
- **04 class-imbalance:** L1: SMOTE from scratch numpy. L2: class-weighted loss, SMOTE, undersampling comparison sklearn. RW1: fraud detection with heavy imbalance (1:1000). RW2: focal loss for object detection. RW3: threshold tuning for precision-recall tradeoff.
- **05 cnns:** L1: 2D convolution from scratch numpy. L2: CNN for CIFAR-10 in torch (conv+bn+relu+pool). RW1: transfer learning from ResNet. RW2: depthwise separable convolutions. RW3: CNN feature map visualization.
- **06 cross-validation-strategies:** L1: k-fold split numpy. L2: sklearn StratifiedKFold, TimeSeriesSplit, GroupKFold comparison. RW1: nested CV for hyperparameter tuning. RW2: temporal CV for time series. RW3: CV variance analysis.
- **07 data-leakage:** L1: demonstrate leakage with numpy (scaler fit on full dataset). L2: sklearn Pipeline that prevents leakage vs manual that leaks. RW1: target encoding leakage. RW2: temporal leakage in time series. RW3: train/val/test split audit tool.
- **08 distributed-training:** L1: gradient averaging simulation numpy (2 workers). L2: torch DistributedDataParallel mock. RW1: data parallelism throughput scaling. RW2: gradient checkpointing for memory. RW3: mixed precision + gradient accumulation.
- **09 domain-adaptation:** L1: domain shift visualization (2D toy numpy). L2: CORAL domain adaptation (match covariance). RW1: fine-tuning for domain shift. RW2: adversarial domain adaptation (DANN). RW3: pseudo-label self-training.
- **10 dropout:** L1: dropout forward/backward numpy. L2: dropout regularization comparison torch (rates 0, 0.1, 0.3, 0.5). RW1: Monte Carlo dropout for uncertainty estimation. RW2: variational dropout. RW3: scheduled dropout (curriculum).

- [ ] **Step 1: Create `ml/notebooks/` directory**
  ```bash
  mkdir -p ml/notebooks
  ```

- [ ] **Step 2: Create notebooks 01-10 using content guides above**
- [ ] **Step 3: Validate**
  ```bash
  python3 -c "
  import json, glob
  for f in sorted(glob.glob('ml/notebooks/0[1-9]-*.ipynb') + glob.glob('ml/notebooks/10-*.ipynb')):
      nb = json.load(open(f))
      cells = len(nb['cells'])
      md = sum(1 for c in nb['cells'] if c['cell_type']=='markdown')
      code = sum(1 for c in nb['cells'] if c['cell_type']=='code')
      print(f'{'PASS' if cells>=12 else 'FAIL'}: {f.split(\"/\")[-1]} — {cells} cells ({md} md, {code} code)')
  "
  ```

- [ ] **Step 4: Commit**
  ```bash
  git add ml/notebooks/0[1-9]-*.ipynb ml/notebooks/10-*.ipynb
  git commit -m "feat: add ml/notebooks 01-10 (activation functions through dropout)"
  ```

### Task D3: Create notebooks 11-20

**Content guides:**
- **11 early-stopping:** L1: loss tracking + patience counter numpy. L2: torch training loop with EarlyStopping callback. RW1: early stopping with LR reduce-on-plateau. RW2: model checkpointing best weights. RW3: early stopping vs full training overfitting curves.
- **12 ensemble-methods:** L1: voting ensemble (majority + soft) numpy. L2: bagging vs boosting vs stacking sklearn. RW1: random forest feature importance. RW2: XGBoost gradient boosting. RW3: stacking meta-learner with sklearn.
- **13 evaluation-metrics:** L1: precision/recall/F1/AUC from scratch numpy. L2: sklearn classification + regression metrics, confusion matrix. RW1: imbalanced class metrics (macro vs micro F1). RW2: ranking metrics (NDCG, MRR). RW3: business metric alignment (precision vs recall trade-off for spam).
- **14 feature-engineering:** L1: polynomial features + interaction terms numpy. L2: sklearn FeatureUnion, ColumnTransformer, PolynomialFeatures. RW1: time series feature extraction (lag, rolling stats). RW2: categorical encoding (ordinal vs one-hot vs target). RW3: feature selection (variance threshold + mutual information).
- **15 gradient-accumulation:** L1: gradient accumulation simulation numpy. L2: torch training loop with accumulation steps. RW1: large effective batch with accumulation. RW2: mixed precision + gradient accumulation. RW3: gradient clipping with accumulation.
- **16 hyperparameter-tuning:** L1: grid search numpy simulation. L2: sklearn GridSearchCV vs RandomSearchCV vs Optuna. RW1: Optuna with pruning for early stopping bad trials. RW2: Bayesian optimization (Optuna TPE). RW3: hyperparameter importance analysis.
- **17 knowledge-distillation:** L1: soft label distillation numpy (temperature scaling). L2: teacher-student in torch with KL divergence loss. RW1: BERT → DistilBERT style distillation. RW2: intermediate layer distillation (hint loss). RW3: distillation with data augmentation.
- **18 layer-normalization:** L1: LayerNorm forward backward numpy. L2: LayerNorm vs BatchNorm vs GroupNorm comparison torch. RW1: LayerNorm in transformer block. RW2: RMSNorm (LLaMA-style). RW3: pre-norm vs post-norm architecture comparison.
- **19 learning-rate-schedules:** L1: warmup + cosine schedule numpy. L2: torch schedulers (StepLR, CosineAnnealing, OneCycleLR, ReduceLROnPlateau). RW1: warmup + cosine for transformer training. RW2: cyclical LR for faster convergence. RW3: LR finder (fast.ai method).
- **20 loss-functions:** L1: MSE/MAE/Huber/BCE/CE from scratch numpy. L2: torch loss comparison on regression and classification tasks. RW1: focal loss for class imbalance. RW2: contrastive loss for metric learning. RW3: custom loss function with auxiliary term.

- [ ] **Step 1: Create notebooks 11-20**
- [ ] **Step 2: Validate all 10 have ≥12 cells**
  ```bash
  python3 -c "
  import json, glob
  for f in sorted(glob.glob('ml/notebooks/1[1-9]-*.ipynb') + glob.glob('ml/notebooks/20-*.ipynb')):
      nb = json.load(open(f)); cells = len(nb['cells'])
      print(f'{'PASS' if cells>=12 else 'FAIL'}: {f.split(\"/\")[-1]} {cells} cells')
  "
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add ml/notebooks/1[1-9]-*.ipynb ml/notebooks/20-*.ipynb
  git commit -m "feat: add ml/notebooks 11-20 (early-stopping through loss-functions)"
  ```

### Task D4: Create notebooks 21-30

**Content guides:**
- **21 meta-learning:** L1: MAML inner/outer loop simulation numpy. L2: prototypical networks torch (few-shot classification). RW1: Model-Agnostic Meta-Learning for few-shot image classification. RW2: meta-learning for fast adaptation to new domains. RW3: zero-shot vs few-shot performance comparison.
- **22 mixed-precision-training:** L1: float16 overflow/underflow demo numpy. L2: torch AMP (autocast + GradScaler). RW1: FP16 training on ResNet — speedup + memory comparison. RW2: loss scaling to prevent underflow. RW3: BF16 vs FP16 stability comparison.
- **23 model-compression:** L1: weight magnitude pruning numpy (remove bottom-k weights). L2: torch structured pruning + quantization pipeline. RW1: post-training quantization on BERT. RW2: knowledge distillation + pruning combined. RW3: compression ratio vs accuracy tradeoff plot.
- **24 model-selection:** L1: AIC/BIC/cross-val score comparison numpy. L2: sklearn model selection pipeline (LinearSVC vs SVM vs RF vs GBT). RW1: nested CV for unbiased model selection. RW2: bias-variance decomposition for model selection. RW3: no-free-lunch demonstration (no single best model).
- **25 momentum-and-acceleration:** L1: SGD vs momentum vs Nesterov update numpy. L2: torch optimizer comparison (SGD, SGD+momentum, Adam, AdaGrad). RW1: momentum warmup for training stability. RW2: gradient explosion with high momentum. RW3: optimizer convergence curves comparison.
- **26 naive-bayes:** L1: GaussianNB from scratch numpy (prior + likelihood + posterior). L2: sklearn MultinomialNB, GaussianNB, BernoulliNB for text classification. RW1: spam detection with MultinomialNB. RW2: Laplace smoothing effect on rare words. RW3: NB vs logistic regression on text classification.
- **27 neural-networks:** L1: 2-layer MLP from scratch numpy (forward + backprop). L2: torch nn.Module MLP with activations, batch norm, dropout. RW1: MLP on tabular data (UCI Heart Disease). RW2: MLP vs tree models comparison. RW3: architecture search (layers × width grid).
- **28 optimization:** L1: SGD/Adam/Adagrad/RMSProp gradient update numpy. L2: torch optimizer comparison on synthetic non-convex. RW1: Adam with weight decay (AdamW) for LLM fine-tuning. RW2: gradient clipping for RNN training. RW3: loss landscape visualization (2D projection).
- **29 overfitting-underfitting:** L1: polynomial regression bias-variance numpy. L2: learning curves (train vs val loss) with sklearn. RW1: regularization ladder (no reg → L2 → dropout → early stop). RW2: dataset size effect on generalization. RW3: bias-variance tradeoff plot for different model complexities.
- **30 probability-statistics:** L1: MLE/MAP estimation numpy (Gaussian parameters). L2: scipy.stats distributions, hypothesis testing, bootstrap CI. RW1: A/B test statistical significance (t-test + effect size). RW2: Bayesian updating with priors. RW3: calibration plot (predicted probability vs actual frequency).

- [ ] **Step 1: Create notebooks 21-30**
- [ ] **Step 2: Validate**
  ```bash
  python3 -c "
  import json, glob
  for f in sorted(glob.glob('ml/notebooks/2[1-9]-*.ipynb') + glob.glob('ml/notebooks/30-*.ipynb')):
      nb = json.load(open(f)); cells = len(nb['cells'])
      print(f'{'PASS' if cells>=12 else 'FAIL'}: {f.split(\"/\")[-1]} {cells} cells')
  "
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add ml/notebooks/2[1-9]-*.ipynb ml/notebooks/30-*.ipynb
  git commit -m "feat: add ml/notebooks 21-30 (meta-learning through probability-statistics)"
  ```

### Task D5: Create notebooks 31-40

**Content guides:**
- **31 pruning:** L1: weight magnitude pruning + sparsity measurement numpy. L2: torch.nn.utils.prune structured + unstructured pruning. RW1: iterative pruning with retraining (lottery ticket). RW2: BERT head pruning (remove unimportant attention heads). RW3: pruning ratio vs accuracy/latency tradeoff.
- **32 quantization:** L1: INT8 quantization (scale + zero_point) numpy. L2: torch post-training quantization (dynamic + static). RW1: GPTQ-style weight quantization for LLM. RW2: QAT (quantization-aware training) on ResNet. RW3: INT8 vs INT4 accuracy/throughput comparison.
- **33 regularization:** L1: L1 and L2 penalty numpy (weight update with penalty). L2: torch L1/L2/elastic net, dropout comparison on same architecture. RW1: weight decay as L2 (and why AdamW matters). RW2: L1 regularization for sparse feature selection. RW3: regularization strength vs train/val loss curves.
- **34 rnns-lstms:** L1: vanilla RNN forward pass numpy (sequence → hidden states). L2: LSTM cell implementation torch, bidirectional LSTM for sequence classification. RW1: sentiment analysis with LSTM. RW2: time series forecasting with stacked LSTM. RW3: LSTM vs transformer on sequential tasks.
- **35 supervised-learning:** L1: linear classification decision boundary numpy. L2: sklearn logistic regression, SVM, decision tree, random forest pipeline. RW1: feature importance comparison across models. RW2: calibrated probability outputs (Platt scaling). RW3: learning curve (train size vs accuracy).
- **36 support-vector-machines:** L1: linear SVM margin numpy (hard margin derivation). L2: sklearn SVC with kernels (linear, RBF, poly). RW1: SVM for text classification (TF-IDF + LinearSVC). RW2: kernel trick visualization (non-linear boundaries). RW3: SVM vs logistic regression on high-dimensional data.
- **37 transfer-learning:** L1: feature extraction simulation (freeze layers) numpy. L2: torch pretrained ResNet, freeze backbone, train classification head. RW1: fine-tuning pretrained BERT for sentiment. RW2: layer-wise learning rate decay. RW3: transfer learning data efficiency (accuracy vs dataset size).
- **38 transformers:** L1: transformer block from scratch numpy (attention + FFN + residual + LN). L2: torch Transformer encoder for classification. RW1: HuggingFace AutoModel for text classification. RW2: positional encoding ablation. RW3: transformer vs LSTM on long sequences.
- **39 unsupervised-learning:** L1: k-means from scratch numpy (lloyd's algorithm). L2: sklearn KMeans, DBSCAN, PCA, t-SNE, GMM comparison. RW1: customer segmentation with KMeans. RW2: anomaly detection with Isolation Forest. RW3: dimensionality reduction comparison (PCA vs UMAP vs t-SNE).
- **40 weight-initialization:** L1: Xavier vs He vs random initialization numpy (forward pass variance). L2: torch model with different init schemes, training curve comparison. RW1: dead neuron diagnosis from bad init. RW2: init for residual networks (small init). RW3: init effect on gradient flow (gradient norm per layer).

- [ ] **Step 1: Create notebooks 31-40**
- [ ] **Step 2: Validate all 10 and full set**
  ```bash
  python3 -c "
  import json, glob
  all_files = sorted(glob.glob('ml/notebooks/*.ipynb'))
  print(f'Total notebooks: {len(all_files)} (need 40)')
  passed = 0
  for f in all_files:
      nb = json.load(open(f)); cells = len(nb['cells'])
      ok = cells >= 12
      if ok: passed += 1
      else: print(f'FAIL: {f.split(\"/\")[-1]} {cells} cells')
  print(f'Passed: {passed}/{len(all_files)}')
  "
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add ml/notebooks/3[1-9]-*.ipynb ml/notebooks/40-*.ipynb
  git commit -m "feat: add ml/notebooks 31-40 (pruning through weight-initialization)"
  ```

---

## PHASE 2 — AGENT 5: Sub-project E

> **Agent 5 scope:** Create 3 Mermaid diagram files for each of 27 arch-review systems (systems 03, 04, 06-30). 81 files total in `arch-review/diagrams/`.

### Task E1: Read reference and understand format

- [ ] **Step 1: Read existing diagram files for format**

  Read these three existing files to understand the exact format to follow:
  - `arch-review/diagrams/01-customer-service-01-system-architecture.md`
  - `arch-review/diagrams/01-customer-service-02-application-architecture.md`
  - `arch-review/diagrams/01-customer-service-03-process-flow.md`

  Every diagram file must:
  - Start with a `# [System Name] — [Diagram Type]` heading
  - Contain exactly one Mermaid code block
  - Use `graph TD` for architecture diagrams, `sequenceDiagram` for process flows
  - Have 8-15 nodes (not too sparse, not overcrowded)
  - Use light backgrounds and dark text — no colors like red/green/blue
  - Contain NO Unicode characters (no emojis, no special symbols)

- [ ] **Step 2: Read all 27 system files to understand content**

  Read `arch-review/systems/` files for each system to extract: key components, data flow, main actors, critical paths.

### Task E2: Create diagrams for systems 03-15

For each system, create 3 files. Use the system file content to determine the components and flow.

**Systems and their key components (derived from reading system files):**

**03 — llm-api-gateway:** Client → Gateway → Router → Provider pool (OpenAI/Anthropic/Cohere). Cache layer. Auth/rate-limit. Cost tracker.

**04 — llm-finetuning-platform:** Data ingestion → preprocessing → training cluster (GPU) → evaluation → model registry → deployment.

**06 — realtime-translation-service:** Audio/text input → ASR (if audio) → translation engine → post-processing → output. Latency SLA < 500ms.

**07 — llm-content-moderation:** Content → classifier (fast) → detailed reviewer (slow, if flagged) → action (allow/flag/block) → appeal queue.

**08 — ai-legal-document-analysis:** Document upload → OCR/parse → section extractor → LLM analysis per section → risk scoring → report generation.

**09 — medical-diagnosis-assistant:** Patient data → de-identification → feature extraction → diagnostic model → confidence scoring → clinician review → recommendation.

**10 — ai-semantic-search-engine:** Query → embedding → vector search (ANN) → re-ranking → results. Indexing pipeline: documents → chunking → embedding → vector store.

**11 — autonomous-research-agent:** Task → planner → tool selection → web search/code/data tools → synthesizer → report. Memory store for context.

**12 — multi-agent-software-dev:** PM agent → architect agent → coder agents (parallel) → reviewer agent → test agent → deploy agent.

**13 — ai-customer-automation-agent:** Customer message → intent classifier → routing (FAQ/human/agent) → agent with tools (CRM, order system) → response → feedback loop.

**14 — autonomous-data-analysis-agent:** Data upload → schema detection → analysis planner → SQL/Python tool executor → insight generator → visualization → report.

**15 — supply-chain-optimization-agent:** Demand signal → forecaster → inventory optimizer → supplier agent → logistics agent → action recommender → human approval.

- [ ] **Step 1: Create 3 diagram files for system 03 (llm-api-gateway)**

  File 1: `arch-review/diagrams/03-llm-api-gateway-01-system-architecture.md`
  ```markdown
  # Multi-Provider LLM API Gateway — System Architecture

  ```mermaid
  graph TD
      Client[Client Application] --> Gateway[API Gateway\nAuth + Rate Limit]
      Gateway --> Cache[Response Cache\nRedis]
      Cache --> Hit{Cache Hit?}
      Hit -->|Yes| Client
      Hit -->|No| Router[Intelligent Router\nCost + Latency]
      Router --> OpenAI[OpenAI GPT-4]
      Router --> Anthropic[Anthropic Claude]
      Router --> Cohere[Cohere Command]
      Router --> Mistral[Mistral 7B]
      OpenAI --> Aggregator[Response Aggregator]
      Anthropic --> Aggregator
      Cohere --> Aggregator
      Mistral --> Aggregator
      Aggregator --> Logger[Cost and Usage Logger]
      Aggregator --> Cache
      Logger --> Dashboard[Cost Dashboard]
  ```
  ```

  File 2: `arch-review/diagrams/03-llm-api-gateway-02-application-architecture.md`
  ```markdown
  # Multi-Provider LLM API Gateway — Application Architecture

  ```mermaid
  graph TD
      API[REST API Layer\nFastAPI] --> Auth[Auth Middleware\nJWT + API Keys]
      Auth --> RateLimit[Rate Limiter\nRedis Sliding Window]
      RateLimit --> RequestNorm[Request Normalizer\nUnified Schema]
      RequestNorm --> CacheCheck[Cache Lookup\nSHA256 Key]
      CacheCheck --> RoutingEngine[Routing Engine\nCost + Latency Model]
      RoutingEngine --> ProviderPool[Provider Pool\nCircuit Breakers]
      ProviderPool --> Fallback[Fallback Handler\nRetry + Alternate]
      Fallback --> ResponseNorm[Response Normalizer]
      ResponseNorm --> CacheWrite[Cache Writer\nTTL 1 hour]
      ResponseNorm --> UsageTracker[Usage Tracker\nCost Allocation]
  ```
  ```

  File 3: `arch-review/diagrams/03-llm-api-gateway-03-process-flow.md`
  ```markdown
  # Multi-Provider LLM API Gateway — Process Flow

  ```mermaid
  sequenceDiagram
      participant C as Client
      participant G as Gateway
      participant Cache as Redis Cache
      participant R as Router
      participant P as Provider
      C->>G: POST /v1/chat/completions
      G->>G: Authenticate + rate check
      G->>Cache: Lookup request hash
      alt Cache hit
          Cache-->>G: Cached response
          G-->>C: 200 OK (cached)
      else Cache miss
          G->>R: Route request
          R->>R: Score providers by cost and latency
          R->>P: Forward to selected provider
          P-->>R: LLM response
          R-->>G: Normalized response
          G->>Cache: Store response (TTL=3600s)
          G-->>C: 200 OK
      end
  ```
  ```

- [ ] **Step 2: Create 3 diagram files for systems 04, 06-15 (11 systems × 3 = 33 files)**

  For each system: read the system file from `arch-review/systems/`, identify 8-12 key components and data flow, create 3 Mermaid files following the same format as system 03 above.

  Strict Mermaid rules for all files:
  - No emojis or Unicode — write "GPU Cluster" not "GPU Cluster ⚡"
  - Node labels use \n for line breaks in graph nodes, not special chars
  - `graph TD` for architecture; `sequenceDiagram` for process flow
  - 8-15 nodes per diagram

- [ ] **Step 3: Validate systems 03-15**
  ```bash
  for sys in 03 04 06 07 08 09 10 11 12 13 14 15; do
      count=$(ls arch-review/diagrams/${sys}-*.md 2>/dev/null | wc -l)
      [ "$count" -eq 3 ] && echo "PASS: system $sys ($count files)" || echo "FAIL: system $sys ($count files)"
  done
  ```

- [ ] **Step 4: Commit**
  ```bash
  git add arch-review/diagrams/03-*.md arch-review/diagrams/04-*.md
  git add $(for i in 06 07 08 09 10 11 12 13 14 15; do echo arch-review/diagrams/${i}-*.md; done)
  git commit -m "feat: add arch-review Mermaid diagrams for systems 03-15 (39 files)"
  ```

### Task E3: Create diagrams for systems 16-30

**Key components per system:**

**16 — financial-analysis-multi-agent:** Data ingestion (market feeds, filings) → analysis agents (fundamental, technical, sentiment) → aggregator → risk assessor → portfolio optimizer → trade recommendation.

**17 — autonomous-security-threat-response:** SIEM → threat detector → triage agent → investigation agents (log analysis, network, endpoint) → response orchestrator → remediation actions → SOC notification.

**18 — ai-document-processing-workflow:** Document intake → OCR/parse → classifier → extraction agents (entities, tables, clauses) → validation → output (structured JSON) → downstream systems.

**19 — multi-agent-content-creation:** Brief → research agent (web search) → outline agent → writing agents (parallel sections) → editor agent → SEO agent → publishing agent.

**20 — autonomous-db-query-agent:** NL question → schema loader → SQL generator → query validator → executor → result formatter → explanation generator.

**21 — realtime-fraud-detection:** Transaction → feature extractor → rule engine (fast, <1ms) → ML model (medium, <10ms) → graph model (slow, <100ms) → decision → action (allow/flag/block).

**22 — personalized-recommendation-engine:** User event stream → feature store → candidate generator (collaborative + content) → ranker (LTR model) → filter (business rules) → served recommendations → feedback loop.

**23 — realtime-video-understanding:** Video stream → frame extractor → object detector → action recognizer → scene graph builder → query interface → real-time alerts.

**24 — multimodal-ai-platform:** Image/text/audio input → modality encoders → fusion layer → joint embedding space → task-specific heads (VQA, captioning, retrieval) → output.

**25 — ai-observability-platform:** Model predictions → logging pipeline → feature monitor (PSI) → prediction monitor (drift) → performance monitor (accuracy) → alert router → dashboard.

**26 — intelligent-ecommerce-platform:** Search query → query understanding → product retrieval → ranking (LTR) → personalization layer → A/B test allocator → result page.

**27 — realtime-anomaly-detection:** Time series streams → feature extraction → isolation forest / autoencoder → anomaly score → threshold detector → alert → root cause analysis agent.

**28 — ai-news-feed-personalization:** User signal (clicks, dwell) → content embedder → user profiler → candidate retrieval → ranker → diversity re-ranker → feed → feedback collector.

**29 — speech-nlp-pipeline:** Audio → VAD → ASR → punctuation restorer → NER → intent classifier → entity linker → structured output.

**30 — llmops-platform:** Model training job → experiment tracker → model registry → evaluation pipeline → deployment gating → A/B traffic router → monitoring → alerting → retraining trigger.

- [ ] **Step 1: Create 3 diagram files for each of systems 16-30 (15 systems × 3 = 45 files)**

  For each system: read `arch-review/systems/{NN}-*.md`, extract key components and flow, create 3 Mermaid files (system architecture, application architecture, process flow).

- [ ] **Step 2: Validate all 27 systems have complete diagram sets**
  ```bash
  total=0; pass=0
  for sys in 03 04 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
      count=$(ls arch-review/diagrams/${sys}-*.md 2>/dev/null | wc -l)
      total=$((total+1))
      if [ "$count" -eq 3 ]; then pass=$((pass+1))
      else echo "FAIL: system $sys has $count/3 files"
      fi
  done
  echo "Sub-project E: $pass/$total systems complete"
  echo "Total diagram files: $(ls arch-review/diagrams/*.md | grep -v README | wc -l) (need 90)"
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add $(for i in 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do echo arch-review/diagrams/${i}-*.md; done)
  git commit -m "feat: add arch-review Mermaid diagrams for systems 16-30 (45 files)"
  ```

---

## FINAL CLEANUP

### Task F1: Update README with accurate counts

- [ ] **Step 1: Run final validation across all sub-projects**

  ```bash
  cd /home/sbisw/github/interviewprep-ml

  echo "=== Sub-project A: modern-ai notebooks ==="
  python3 -c "
  import json, glob
  ok = 0
  for n in ['46','48','49','50','51','52','53','54','55']:
      f = next(iter(glob.glob(f'modern-ai/notebooks/{n}-*.ipynb')), None)
      if f:
          nb = json.load(open(f))
          lines = sum(len(''.join(c['source']).split('\n')) for c in nb['cells'] if c['cell_type']=='code')
          if lines >= 600: ok += 1
          else: print(f'FAIL: {f} ({lines} lines)')
  print(f'{ok}/9 passing')
  "

  echo "=== Sub-project B: system-design patterns ==="
  pass=0
  for f in system-design/patterns/1[4-9]-*.md system-design/patterns/2[0-9]-*.md system-design/patterns/3[01]-*.md; do
      words=$(wc -w < "$f"); [ "$words" -ge 1800 ] && ((pass++)) || echo "FAIL: $(basename $f) ($words words)"
  done
  echo "$pass/18 passing"

  echo "=== Sub-project C: coding section ==="
  echo "DS files: $(ls coding/data-structures/*.md | wc -l)/10"
  echo "Algo files: $(ls coding/algorithms/*.md 2>/dev/null | wc -l)/12"

  echo "=== Sub-project D: ml notebooks ==="
  echo "Notebooks: $(ls ml/notebooks/*.ipynb 2>/dev/null | wc -l)/40"

  echo "=== Sub-project E: arch-review diagrams ==="
  echo "Diagram files: $(ls arch-review/diagrams/*.md | grep -v README | wc -l)/90"
  ```

- [ ] **Step 2: Update README.md section table**

  Read `README.md`. Find the "What's Inside" table and update these rows:
  - Machine Learning: change to "40 concepts + 40 implementation notebooks"
  - System Design: change to "31 ML/AI system design patterns (1800-2400 words each) + architecture reviews of 30 real systems with diagrams"
  - Coding Interview Prep: change to "10 data structure topic files + 12 algorithm pattern files"

- [ ] **Step 3: Final commit**
  ```bash
  git add README.md
  git commit -m "docs: update README with accurate file counts after Phase 1 and Phase 2 completion"
  ```

- [ ] **Step 4: Run existing test suite**
  ```bash
  python3 -m pytest tests/ -v 2>&1 | tail -20
  ```

  Investigate and fix any failures before declaring done.
