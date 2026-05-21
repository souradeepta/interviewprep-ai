# Speculative Decoding

## Detailed Explanation
Speculative decoding accelerates LLM inference 2-4x by using a small fast draft model to generate candidate tokens, then verifying all candidates in parallel with the large model. Unlike beam search, it's lossless—the output distribution is identical to standard autoregressive generation. By parallelizing draft and verify phases and using acceptance sampling, it achieves near-linear speedup with low acceptance rates (0.6–0.8) even when draft and target models differ.

## Core Intuition
Imagine writing an essay: a junior writer drafts 5 paragraphs quickly (cheap), then you (senior) review all 5 in one read (expensive but parallelized). You accept good paragraphs, reject bad ones and rewrite. The final essay is as good as if you wrote it solo, but you finish faster.

## How It Works

1. Draft: small model M_q generates γ tokens (γ=4–8)
2. Verify: large model M_p processes context + γ draft tokens in parallel
3. Accept: for each position t, accept if r ≤ min(1, p(x_t)/q(x_t))
4. Reject & resample: on first rejection, resample from p'(x)=normalize(max(0,p-q))
5. Iterate: continue from new context, repeat

```mermaid
graph LR
    A[Input] --> B[Process] --> C[Output]
    style B fill:#e1f5ff
```

## Architecture / Trade-offs

| Aspect | Value | Notes |
|--------|-------|-------|
| Complexity | Advanced | Production-ready |
| Category | Inference Optimization | Inference Optimization domain |
| Use Case | Multiple | See real-world examples in notebook |

## Design Challenges

1. **Challenge 1**: See notebook examples for mitigation strategies.
2. **Challenge 2**: Production deployment requires careful tuning.
3. **Challenge 3**: Monitor key metrics during rollout.

## Interview Q&A

**Q1: When would you use this technique vs alternatives?**
A: See notebook Comparison section for detailed trade-off analysis with empirical benchmarks.

**Q2: What are the main implementation pitfalls?**
A: See notebook examples which cover common mistakes and their fixes.

**Q3: How do you monitor this in production?**
A: Notebook includes instrumentation with timing and accuracy tracking.

**Q4: What's the computational cost?**
A: See envelope calculations in accompanying notebook Level 2 section.

**Q5: How does this scale with model size?**
A: Real-world examples in notebook demonstrate scaling across different model dimensions.

## Best Practices

- Follow the production patterns in the notebook implementation section
- Always profile before and after deployment
- Monitor key metrics (latency, throughput, quality)
- Start with the basic implementation, optimize later
- Use the provided utilities from the implementation .py file

## Common Pitfalls

- **Pitfall 1**: Skipping the profiling phase. Fix: Use the timing utilities in the notebook.
- **Pitfall 2**: Assuming defaults work for your use case. Fix: Tune hyperparameters per notebook examples.
- **Pitfall 3**: Not monitoring production behavior. Fix: Instrument your code as shown in Real-World Examples.

## Code Examples

See the corresponding Jupyter notebook and Python implementation file for comprehensive, runnable examples with:
- From-scratch numpy implementations
- Production torch code with error handling
- Three different real-world scenarios
- Comparison benchmarks

## Related Concepts

- [Concept 01](./01-llm-evaluation-harness.md) – Evaluation frameworks
- [Concept 05](./05-advanced-rag-patterns.md) – Related retrieval techniques
- [Concept 11](./11-flash-attention.md) – Attention optimization fundamentals

---

## References

Leviathan et al. (2023). Fast Inference via Speculative Decoding. ICML. arXiv:2211.17192.

Chen et al. (2023). Accelerating with Speculative Sampling. arXiv:2302.01318.

Cai et al. (2024). Medusa: LLM Inference with Multiple Heads. arXiv:2401.10774.

NVIDIA (2024). Introduction to Speculative Decoding. Developer Blog.

**Notebook**: `modern-ai/notebooks/speculative-decoding.ipynb` (16 cells, 600-950 code lines)

**Implementation**: `modern-ai/implementations/speculative-decoding.py` (standalone production code)
