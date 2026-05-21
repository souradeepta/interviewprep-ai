# RNNs and LSTMs

## TL;DR
Recurrent Neural Networks process sequential data by maintaining a hidden state across timesteps.
LSTMs solve the vanishing gradient problem with gating mechanisms. Largely superseded by
transformers for long sequences, but still used in streaming and time-series applications.

## Core Intuition
Reading a sentence word by word with a "working memory." The problem: memory decays. LSTMs add a
long-term memory cell with explicit gates controlling what to write, read, and forget.

## How It Works

**Vanilla RNN:** $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$
Problem: gradient of $h_t$ w.r.t. $h_{t-k}$ involves $W_{hh}^k$ — explodes or vanishes.

**LSTM gates** (all use sigmoid → values in [0,1]):
- Forget: $f_t = \sigma(W_f [h_{t-1}, x_t])$ — what to erase from cell state
- Input: $i_t = \sigma(W_i [h_{t-1}, x_t])$ — what new info to write
- Output: $o_t = \sigma(W_o [h_{t-1}, x_t])$ — what to output
- Cell: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
- Hidden: $h_t = o_t \odot \tanh(c_t)$

**GRU:** simplified LSTM with 2 gates (reset, update). Fewer params, similar performance.

## Key Properties / Trade-offs
- Non-parallelizable across timesteps — slow on GPUs
- Transformers dominate for sequences > ~200 tokens
- RNNs still used in: streaming audio, online RL, edge devices

## Common Mistakes / Gotchas
- Vanilla RNNs can't capture long-range dependencies — always use LSTM/GRU
- Not clipping gradients — exploding gradients are common in RNN training
- LSTM has two states: $h_t$ (hidden) and $c_t$ (cell) — both matter

## Code Example
```python
import torch.nn as nn
lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
# input shape: (batch, seq_len, features)
# output: hidden state at each timestep, final hidden + cell state
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What problem does LSTM solve?" | Vanilla RNN suffers vanishing/exploding gradients. LSTM adds a cell state with forget/input/output gates — gradient highways for long-range dependencies. |
| "Why did transformers replace RNNs?" | Transformers process all positions in parallel (no sequential bottleneck), scale better on GPUs, and handle longer context via attention. |
| "LSTM vs GRU?" | GRU merges cell/hidden state, uses 2 gates instead of 3. Fewer parameters, similar performance. |

## Related Topics
- [Attention Mechanism](attention-mechanism.md) — [Transformers](transformers.md)

## Resources
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Colah. Best LSTM explainer.
