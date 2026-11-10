# RNNs and LSTMs

**TL;DR:** Recurrent networks process sequences by updating a hidden state over time; LSTMs use gates and a cell state to preserve useful information across longer spans.

## Prerequisites

Prerequisites: [neural networks](neural-networks.md), sequence data, and backpropagation. You can write the recurrent update, explain why gradients vanish or explode through time, distinguish hidden and cell states in an LSTM, compare RNNs with attention-based models, and choose a suitable sequence split and streaming boundary. This ML note covers recurrent computation; the NLP and RL tracks apply sequence methods to their domains.

## Learning Outcomes

- Explain the central assumptions and mechanism of rnns lstms.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Reading a sequence is like processing a story one sentence at a time while maintaining a compact notebook of relevant context. A recurrent neural network (RNN) updates that notebook after each input, reusing the same learned transformation at every step. This sharing makes it naturally handle variable sequence lengths and online streams. However, its fixed-size state is a bottleneck: information from far back must survive repeated transformations.

An LSTM gives the notebook explicit controls. A forget gate decides what old memory to retain, an input gate controls what new information to write, and an output gate chooses what to expose as the current hidden representation. These gates do not guarantee perfect long-term memory; they create a path that can carry gradients more directly than a vanilla recurrent transformation.

## How It Works

For input xₜ and previous hidden state hₜ₋₁, a vanilla RNN computes hₜ=tanh(Wₓxₜ+Wₕhₜ₋₁+b). The same weights are used at each timestep. Backpropagation through time unrolls this recurrence and propagates gradients across sequence positions. Repeated multiplication by recurrent Jacobians can shrink gradients toward zero or amplify them exponentially, making long dependencies hard to learn. Truncated BPTT limits the number of steps through which gradients flow, saving memory and compute but also limiting direct credit assignment.

An LSTM computes input gate iₜ, forget gate fₜ, output gate oₜ, and candidate update gₜ from xₜ and hₜ₋₁ using learned affine maps and sigmoid/tanh nonlinearities. It updates cell state cₜ=fₜ⊙cₜ₋₁+iₜ⊙gₜ and hidden state hₜ=oₜ⊙tanh(cₜ). The additive cell update allows information and gradients to persist when the forget gate is near one. A GRU combines some gates and omits a separate cell state, often with fewer parameters. Both still process positions sequentially during training and inference.

For batch size B, sequence length T, input width D, and hidden width H, a basic RNN layer costs approximately O(BT(DH+H²)); LSTM gate projections multiply the constant by roughly four. Training stores intermediate activations for backpropagation, so memory scales with T unless checkpointing or truncated BPTT is used. Packed sequences or masks handle padding, but loss and hidden-state logic must respect true lengths.

RNNs remain useful for streaming inputs when each new observation arrives incrementally and low-latency state updates matter. They are also common in small-data time series, embedded systems, and legacy pipelines. Transformers parallelize sequence positions during training and provide direct content-based access across context, but their attention cost and cache memory can be large. Convolutional sequence models offer parallel local processing. Architecture choice depends on sequence length, data, hardware, and online constraints.

Batch construction affects recurrent models more than a single tensor shape suggests. Sequences are often padded to the longest member of a batch, and the model must avoid treating those padding steps as real observations. Packed-sequence implementations can skip padded computation; explicit masks can protect pooling and loss calculations. For stateful training, hidden and cell states may be carried between adjacent chunks of one stream, but gradients are usually detached at chunk boundaries to bound memory. The state should never cross between unrelated users or documents simply because they share a batch.

Bidirectional RNNs read the sequence in both directions and concatenate context, which is useful for offline tasks where the complete sequence is available. They cannot be used unchanged for strictly causal streaming prediction because the reverse direction consumes future tokens. This distinction is often missed when moving a model from an offline benchmark to a live service. Likewise, teacher-forced next-token training and autoregressive inference expose different histories: generation feeds previous predictions, so errors can compound. Evaluate under the inference procedure the system will actually use.

## Trade-offs and Selection Boundary

| Model | Strength | Boundary |
|---|---|---|
| Vanilla RNN | Few parameters and natural streaming state | Long-range gradients vanish/explode; limited memory |
| LSTM | Gated memory helps preserve longer dependencies | Sequential compute and larger parameter/state cost |
| GRU | Fewer gates and often competitive accuracy | Still sequential; may be less expressive for some tasks |
| Transformer | Parallel training and direct long-range interactions | Attention compute/cache grow with sequence length |
| Temporal CNN | Parallel local temporal patterns | Long context requires depth/dilation or hierarchy |

Choose an LSTM/GRU when streaming state or moderate sequence lengths make recurrence appropriate, and define state reset boundaries between independent records. Prefer transformers for large-scale text or long-range dependencies when compute permits. Never let a user's future events enter an earlier state or split; use forward temporal validation for forecasts.

## Implementation

```python
import math

def sigmoid(x): return 1/(1+math.exp(-x))
def lstm_step(x, h, c, params):
    # Scalar illustration: each gate uses one shared affine input.
    i,f,o,g = params
    i, f, o = sigmoid(i*x), sigmoid(f*x), sigmoid(o*x)
    g = math.tanh(g*x)
    c_new = f*c + i*g
    h_new = o*math.tanh(c_new)
    return h_new, c_new

h, c = 0.0, 0.0
h, c = lstm_step(1.0, h, c, (1.0, 4.0, 1.0, .5))
assert -1 <= h <= 1 and c != 0
h2, c2 = lstm_step(0.0, h, c, (1.0, 4.0, 1.0, .5))
assert abs(c2) > 0  # a closed input gate need not erase stored cell state
```

## Failure Modes and Diagnostics

1. **Exploding gradients:** updates become non-finite on long sequences. Track gradient norms by time span and use clipping or lower learning rate after checking data and initialization.
2. **Vanishing long-range signal:** model learns recent context only. Compare performance by dependency distance; inspect gate saturation and try shorter task framing or attention.
3. **Padding affects state or loss:** predictions vary with extra padded timesteps. Use masks/packed sequences and test identical examples at different padded lengths.
4. **State leaks across independent sessions:** a new user inherits hidden state from a previous stream. Define and test reset boundaries explicitly.
5. **Random split on time series:** future patterns leak into training. Use chronological splits with a gap appropriate to the target horizon.

## Interview Quick Reference

| Question | Answer |
|---|---|
| How does an LSTM help with long dependencies? | Its gated additive cell-state update provides a path that can preserve information and gradients across steps. |
| Why are RNNs hard to parallelize? | Each hidden state depends on the previous timestep, creating a sequential dependency. |
| Design follow-up: predict from a live sensor stream. | Maintain per-stream state, reset it at session boundaries, validate forward in time, and compare LSTM with a causal convolution or transformer under latency limits. |

## Related Concepts

- [Attention mechanism](attention-mechanism.md), [transformers](transformers.md), and [gradient accumulation](gradient-accumulation.md)
- [Matching notebook](../notebooks/34-rnns-lstms.ipynb)
- [Interview practice: sequence attention](../interview-prep/ml-coding-questions.md#q2-implement-scaled-dot-product-attention-from-scratch)

## References

- Hochreiter and Schmidhuber, [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735).
- Cho et al., [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078).
