# Cheat Sheets

Quick-reference markdown files for interview preparation. Each file is designed to be read in under 10 minutes and cover the key facts, formulas, and decision rules you need to recall under pressure.

---

## Contents

| File | Description |
|------|-------------|
| [ml-algorithms.md](ml-algorithms.md) | Master comparison table of 10 algorithms: type, train/inference complexity, best-for, avoid-when, key hyperparameters, common pitfalls |
| [interview-formulas.md](interview-formulas.md) | All whiteboard formulas: loss functions, gradient descent, Adam, attention, precision/recall/F1, NDCG, statistics (CI, sample size, Cohen's d), RL (Bellman, Q-learning, PPO), KL divergence, LoRA |
| [neural-network-architectures.md](neural-network-architectures.md) | CNN, RNN/LSTM, Transformer, ViT, Diffusion Model, GAN — comparison table + ASCII diagrams + key papers + pitfalls |
| [optimizers.md](optimizers.md) | SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW, LAMB — full update rules, LR schedule comparison, gradient clipping guide |
| [evaluation-metrics.md](evaluation-metrics.md) | Tables by task type — classification (Accuracy/Precision/Recall/F1/AUC/MCC), regression (MSE/RMSE/MAE/MAPE/R2), ranking (MRR/NDCG/MAP), generation (BLEU/ROUGE/BERTScore), clustering (Silhouette/ARI/NMI) |
| [llm-models.md](llm-models.md) | GPT-4o, Claude 3.5 Sonnet, Llama 3.1 70B, Mistral 7B, Gemini 1.5 Pro, and more — params, context, strengths, cost, fine-tuning support, model selection guide |
| [regularization.md](regularization.md) | L1/L2/Elastic Net formulas, Weight Decay vs L2 distinction, Dropout/BatchNorm/LayerNorm, Label Smoothing, Data Augmentation (Mixup, CutMix), selection guide |
| [deployment-strategies.md](deployment-strategies.md) | Blue-green, Canary, Shadow, Rolling Update, Feature Flags, A/B Test — decision matrix with rollout speed, risk, rollback time + ML-specific considerations (model warm-up, KV cache, feature store lag) |
| [system-design-quick-ref.md](system-design-quick-ref.md) | Capacity estimation examples (1M QPS embedding, LLM serving), latency budgets, ML system components checklist, Two-Tower/Cascade/RAG/Ensemble patterns, back-of-envelope memory and throughput formulas |
| [python-ml-snippets.md](python-ml-snippets.md) | Copy-paste code: sklearn pipeline, PyTorch training loop, PyTorch evaluation, HuggingFace inference, NumPy softmax, confusion matrix, cross-validation, gradient accumulation, mixed precision, early stopping, embedding LRU cache |

---

## How to Use Before an Interview

### 30 minutes before
1. Scan **interview-formulas.md** — you should be able to write every formula from memory
2. Scan **ml-algorithms.md** Master Table — can you fill in the columns from memory?
3. Scan **evaluation-metrics.md** — do you know which metric to use for each task type?

### Night before (1 hour)
1. Read **neural-network-architectures.md** — draw the ASCII diagrams on paper from memory
2. Read **optimizers.md** — trace through Adam update rule step by step
3. Read **deployment-strategies.md** — practice explaining blue-green vs canary vs shadow out loud
4. Read **system-design-quick-ref.md** — practice the back-of-envelope calculations with different numbers

### During the interview
- For ML system design: start with the **components checklist** from system-design-quick-ref.md
- For metric selection: think through the task type → metric table from evaluation-metrics.md
- For architecture choice: use the comparison tables from ml-algorithms.md or neural-network-architectures.md
- For any formula: reconstruct from the derivation, not memorization (e.g., Adam = momentum + RMSProp)

### Connecting to code
- When asked to implement something, use **python-ml-snippets.md** as a mental template
- Always mention: device management, batching, error handling — these signal production awareness
