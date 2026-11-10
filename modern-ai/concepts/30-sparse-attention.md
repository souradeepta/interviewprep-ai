# Sparse Attention Patterns

**TL;DR:** Sparse attention restricts query-key connections to reduce work, but the mask encodes which dependencies remain reachable and therefore which tasks may lose information.

## Prerequisites

[Attention](11-flash-attention.md) and [long context management](27-long-context-management.md).

## Learning Outcomes

- Construct causal, sliding-window and global-local attention masks.
- Estimate how mask structure changes the number of query-key edges.
- Choose an attention pattern based on dependency structure and kernel support.
- Test a mask for causality and reachability requirements.

## Core Intuition

A meeting participant may need to listen to nearby turns plus a few agenda-setting speakers. A local attention window models nearby turns; global tokens provide shortcuts. Making the meeting cheaper by silencing distant participants is only helpful if the task does not depend on their information.

## How It Works

For sequence length n, dense attention has n² query-key score positions per head. A causal dense mask keeps n(n+1)/2 positions. A causal sliding window of width w keeps at most roughly nw positions, giving O(nw) work for fixed w. A global-local pattern adds connections to selected global positions; its connectivity and count depend on mask design. These counts describe mathematical edges, not wall-clock time: irregular sparse kernels may run slower than dense optimized kernels at moderate lengths.

The mask M_ij controls whether query i can attend to key j. A causal model requires j≤i. A sliding causal mask additionally requires i-j≤w. Global tokens can provide long-range paths, but stacking layers, positional encoding and residual streams affect information flow. Bidirectional encoders and causal decoders have different valid masks.

Complexity follows the number of permitted edges only when the implementation can exploit the pattern. A boolean mask applied to a dense matrix multiplication may still compute every score and merely replace disallowed entries afterward. Real savings need a kernel that skips blocks or gathers only selected keys. Block-sparse layouts often trade fine-grained sparsity for hardware-friendly regularity. Benchmark the complete forward pass, including mask creation and layout conversion, because those costs can dominate at small sequence lengths.

Mask semantics must be written down precisely. Does a window of width w include the current token, w previous tokens, or w total positions? Are global tokens visible to every query and are their queries also global? Off-by-one errors are common and can leak future tokens or remove the latest context. Unit tests should assert causality, expected row degree, diagonal inclusion, global connectivity and behavior at sequence boundaries. Test length one, a window larger than the sequence, and empty global-token sets.

Reachability across layers is a useful diagnostic. With a local window, information may travel only a limited distance per layer unless global links or other skip paths exist. An input far away from a query can become reachable after multiple layers, but may be attenuated or compete with many local tokens. Draw or compute the receptive field for the actual number of layers and windows. For tasks like document question answering, test pairs whose supporting evidence is separated by more than one local window.

Sparse patterns encode assumptions about data structure. A sliding window is plausible for local continuity in speech or nearby text. Global-local patterns suit documents with special summary or delimiter tokens. Random edges can improve graph connectivity but complicate determinism and reproducibility. Block patterns align with hardware but can split semantically related spans. Pattern selection should be based on task probes and data layout, not on an attractive asymptotic expression alone.

Finally, distinguish sparse attention from memory-efficient exact attention. Tiled exact algorithms can reduce materialized memory traffic while still considering all causal pairs. Sparse attention changes which pairs may interact and can change model behavior. If the objective is only to reduce memory footprint, an exact fused kernel may preserve the dense receptive field and be a safer first comparison. If the objective requires much longer sequences, sparsity may be necessary, with explicit quality tests for information lost by the mask.

## Trade-offs

| Pattern | Fit | Lost dependency |
|---|---|---|
| Dense causal | General autoregressive modeling | Quadratic edge count |
| Sliding window | Local continuity and bounded cache | Direct access to distant evidence |
| Global-local | Long documents with selected anchors | Anchor selection and irregular work |
| Block sparse | Structured long sequences and kernel support | Dependencies crossing block boundaries |

Choose sparse attention when the task structure supports locality or anchors and the runtime has an efficient kernel. Keep a dense baseline at representative sequence lengths for comparison.

## Implementation with Python

Run `python3 modern-ai/implementations/30-sparse-attention.py`. It builds tiny boolean masks and reports their density. Inspect exact rows to see which edges disappear; the Python loops are for explanation, not performance benchmarking.

```python
import runpy
mask = runpy.run_path("modern-ai/implementations/30-sparse-attention.py")["mask"]
m = mask(6, "sliding", window=2)
assert all(not m[q][k] for q in range(6) for k in range(6) if k > q)
```

## Failure Modes

| Failure | Diagnostic | Mitigation |
|---|---|---|
| Long-range fact becomes unreachable | Dependency probe across distant positions | Add global links or increase window; evaluate retrieval tasks |
| Mask violates causality | Assert no key index exceeds query index | Test mask invariants before model execution |
| Sparse kernel is slower | End-to-end latency versus dense kernel | Benchmark supported block patterns at target lengths |

- Long evidence is unreachable: run distance-controlled probes and inspect the layerwise receptive field; add global paths or enlarge the window.
- Mask leaks future context: assert every visible key index is no greater than its query for causal models; test boundary lengths.
- Sparse path is slower: time the supported kernel at target shapes and retain a dense fallback where sparse overhead dominates.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| How does a fixed causal window change asymptotic edge count? | From O(n²) to O(nw); ask what happens if w scales with n. |
| Why may sparse attention be slower? | Irregular memory access and kernel overhead; follow up with a benchmark plan. |
| How do global tokens help? | They create long-range paths; ask how you would test that anchors preserve needed evidence. |

## Related Concepts

See [FlashAttention](11-flash-attention.md), [long context](27-long-context-management.md), [notebook 30](../notebooks/30-sparse-attention.ipynb), and [inference interview practice](../interview-prep/inference-systems.md#sparse-attention).

An attention mask is part of the model's inductive bias and belongs in evaluation and release metadata. Record window sizes, selected global positions, block layout and mask version with the checkpoint. A window change can alter runtime and output behavior even when weights are identical. For reproducible comparisons, hold tokenization, positions, decoding and model weights constant while changing only the mask, then evaluate local and long-distance dependency probes.

## References

[Longformer](https://arxiv.org/abs/2004.05150) and [BigBird](https://arxiv.org/abs/2007.14062).
