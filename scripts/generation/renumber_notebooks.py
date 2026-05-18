#!/usr/bin/env python3
"""Renumber notebook files to match the new concept ordering."""

import os

BASE = "/home/sbisw/github/interviewprep-ml"

LLM_ORDER = [
    "tokenization", "embeddings", "pretraining", "finetuning", "instruction-tuning",
    "rlhf", "dpo", "lora", "adapters", "prefix-tuning", "parameter-efficient-finetuning",
    "prompting", "few-shot-learning", "zero-shot-learning", "in-context-learning",
    "chain-of-thought", "prompt-optimization", "rag", "retrieval-augmented-generation",
    "vector-databases", "semantic-search", "semantic-caching", "kv-cache",
    "attention-optimization", "context-window", "continuous-batching",
    "speculative-decoding", "inference-optimization", "token-optimization",
    "quantization", "multimodal", "evaluation",
]

AGENTIC_ORDER = [
    "what-is-an-agent", "agent-loops", "tool-use", "tool-calling", "function-calling",
    "structured-output", "planning-reasoning", "react-reasoning-acting", "tree-of-thought",
    "mcts-for-agents", "memory-types", "agent-memory-management", "agent-communication",
    "multi-agent-systems", "hierarchical-agents", "cooperative-agents", "competitive-agents",
    "autonomous-agents", "context-window-management", "agent-routing", "agent-persistence",
    "skill-composition", "knowledge-graphs", "reflection-and-self-improvement",
    "retrieval-augmented-generation", "error-recovery", "agent-debugging", "agent-testing",
    "agent-evals", "agent-monitoring", "observability-for-agents", "tracing-agents",
    "latency-optimization-agents", "agent-cost-optimization", "agent-prompt-engineering",
    "safety-alignment", "human-agent-collaboration", "coding-agents", "code-analysis-agents",
    "customer-service-agents", "medical-agents", "legal-document-agents", "finance-agents",
    "logistics-agents", "data-analysis-agents", "research-agents", "content-moderation-agents",
    "recommendation-agents", "multimodal-agents", "real-time-agents", "web-agents",
    "simulation-for-agents",
]

def renumber_notebooks(nb_dir, order, keep_00=False):
    renamed = 0
    # First, build all renames to avoid conflicts during rename
    renames = []
    for i, slug in enumerate(order, 1):
        new_name = f"{i:02d}-{slug}.ipynb"
        new_path = f"{nb_dir}/{new_name}"
        if os.path.exists(new_path):
            # Already correctly named
            renamed += 1
            continue
        # Find old file - could be old numbered or unnumbered
        found = None
        # Try unnumbered
        if os.path.exists(f"{nb_dir}/{slug}.ipynb"):
            found = f"{nb_dir}/{slug}.ipynb"
        else:
            # Try any file ending in -{slug}.ipynb
            for f in os.listdir(nb_dir):
                if f.endswith(f"-{slug}.ipynb") and f != new_name:
                    found = f"{nb_dir}/{f}"
                    break
        if found:
            renames.append((found, new_path))
        else:
            print(f"  WARNING: no notebook found for {slug}")

    # Use temp names to avoid conflicts
    temp_renames = []
    for old, new in renames:
        tmp = old + ".tmp"
        os.rename(old, tmp)
        temp_renames.append((tmp, new))

    for tmp, new in temp_renames:
        os.rename(tmp, new)
        renamed += 1

    return renamed

print("=== Renumbering notebooks ===\n")

llm_nb = f"{BASE}/llm/notebooks"
agentic_nb = f"{BASE}/agentic-ai/notebooks"

n = renumber_notebooks(llm_nb, LLM_ORDER)
print(f"llm/notebooks: {n} notebooks processed")

n = renumber_notebooks(agentic_nb, AGENTIC_ORDER)
print(f"agentic-ai/notebooks: {n} notebooks processed")

print("\n✅ Done!")
