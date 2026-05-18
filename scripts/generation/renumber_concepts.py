#!/usr/bin/env python3
"""Renumber concept files in llm/concepts, agentic-ai/concepts, and system-design/patterns."""

import os
import re

BASE = "/home/sbisw/github/interviewprep-ml"

# ── LLM/CONCEPTS ──────────────────────────────────────────────────────────────
LLM_ORDER = [
    "tokenization",
    "embeddings",
    "pretraining",
    "finetuning",
    "instruction-tuning",
    "rlhf",
    "dpo",
    "lora",
    "adapters",
    "prefix-tuning",
    "parameter-efficient-finetuning",
    "prompting",
    "few-shot-learning",
    "zero-shot-learning",
    "in-context-learning",
    "chain-of-thought",
    "prompt-optimization",
    "rag",
    "retrieval-augmented-generation",
    "vector-databases",
    "semantic-search",
    "semantic-caching",
    "kv-cache",
    "attention-optimization",
    "context-window",
    "continuous-batching",
    "speculative-decoding",
    "inference-optimization",
    "token-optimization",
    "quantization",
    "multimodal",
    "evaluation",
]

# ── AGENTIC-AI/CONCEPTS ────────────────────────────────────────────────────────
AGENTIC_ORDER = [
    "what-is-an-agent",
    "agent-loops",
    "tool-use",
    "tool-calling",
    "function-calling",
    "structured-output",
    "planning-reasoning",
    "react-reasoning-acting",
    "tree-of-thought",
    "mcts-for-agents",
    "memory-types",
    "agent-memory-management",
    "agent-communication",
    "multi-agent-systems",
    "hierarchical-agents",
    "cooperative-agents",
    "competitive-agents",
    "autonomous-agents",
    "context-window-management",
    "agent-routing",
    "agent-persistence",
    "skill-composition",
    "knowledge-graphs",
    "reflection-and-self-improvement",
    "retrieval-augmented-generation",
    "error-recovery",
    "agent-debugging",
    "agent-testing",
    "agent-evals",
    "agent-monitoring",
    "observability-for-agents",
    "tracing-agents",
    "latency-optimization-agents",
    "agent-cost-optimization",
    "agent-prompt-engineering",
    "safety-alignment",
    "human-agent-collaboration",
    "coding-agents",
    "code-analysis-agents",
    "customer-service-agents",
    "medical-agents",
    "legal-document-agents",
    "finance-agents",
    "logistics-agents",
    "data-analysis-agents",
    "research-agents",
    "content-moderation-agents",
    "recommendation-agents",
    "multimodal-agents",
    "real-time-agents",
    "web-agents",
    "simulation-for-agents",
]

# ── SYSTEM-DESIGN/PATTERNS ─────────────────────────────────────────────────────
SYSDESIGN_ORDER = [
    "mlops-overview",
    "data-pipelines",
    "feature-store",
    "model-registry",
    "model-serving",
    "model-versioning",
    "online-vs-batch-inference",
    "inference-caching",
    "request-batching",
    "load-balancing",
    "blue-green-deployment",
    "canary-deployment",
    "shadow-mode",
    "ab-testing",
    "drift-detection",
    "monitoring-and-observability",
    "model-debugging",
    "model-explainability",
    "interpretability",
    "feature-importance-tracking",
    "reproducibility",
    "cost-optimization",
    "production-readiness",
    "bias-detection",
    "fairness-metrics",
    "data-governance",
    "ml-governance",
    "privacy-preserving-ml",
    "differential-privacy",
    "federated-learning",
    "disaster-recovery",
]

SECTIONS = [
    (f"{BASE}/llm/concepts",          LLM_ORDER),
    (f"{BASE}/agentic-ai/concepts",   AGENTIC_ORDER),
    (f"{BASE}/system-design/patterns", SYSDESIGN_ORDER),
]

def build_rename_map(directory, order):
    """Build {old_basename_no_ext: new_filename_no_ext}."""
    mapping = {}
    for i, slug in enumerate(order, 1):
        new_name = f"{i:02d}-{slug}"
        mapping[slug] = new_name
    return mapping

def update_links_in_file(filepath, rename_map):
    """Replace old .md filenames with new numbered filenames in markdown links."""
    with open(filepath, 'r') as f:
        content = f.read()

    changed = False
    for old_slug, new_slug in rename_map.items():
        # Match markdown links like ./old-slug.md or (old-slug.md)
        patterns = [
            (rf'\./({re.escape(old_slug)}\.md)', f'./{new_slug}.md'),
            (rf'\(({re.escape(old_slug)}\.md)\)', f'({new_slug}.md)'),
            (rf'\[([^\]]+)\]\({re.escape(old_slug)}\.md\)', lambda m, ns=new_slug: f'[{m.group(1)}]({ns}.md)'),
        ]
        for pat, repl in patterns:
            new_content = re.sub(pat, repl, content)
            if new_content != content:
                content = new_content
                changed = True

    if changed:
        with open(filepath, 'w') as f:
            f.write(content)
    return changed

def process_directory(directory, order):
    rename_map = build_rename_map(directory, order)

    # Check all files exist
    missing = []
    for slug in order:
        if not os.path.exists(f"{directory}/{slug}.md"):
            missing.append(slug)
    if missing:
        print(f"  WARNING: missing files in {directory}: {missing}")

    # Step 1: Rename files (old slug -> new numbered name)
    renamed = 0
    for old_slug, new_slug in rename_map.items():
        old_path = f"{directory}/{old_slug}.md"
        new_path = f"{directory}/{new_slug}.md"
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            renamed += 1

    print(f"  Renamed {renamed} files in {directory}")

    # Step 2: Update links in all .md files in directory
    updated = 0
    for fname in os.listdir(directory):
        if fname.endswith('.md'):
            changed = update_links_in_file(f"{directory}/{fname}", rename_map)
            if changed:
                updated += 1

    print(f"  Updated links in {updated} files")

    return rename_map

# Also update README files that reference these concepts
def update_readme(readme_path, rename_map):
    if not os.path.exists(readme_path):
        return False
    with open(readme_path, 'r') as f:
        content = f.read()
    original = content
    for old_slug, new_slug in rename_map.items():
        content = re.sub(rf'(?<!\d-){re.escape(old_slug)}\.md', f'{new_slug}.md', content)
    if content != original:
        with open(readme_path, 'w') as f:
            f.write(content)
        print(f"  Updated README: {readme_path}")
        return True
    return False

print("=== Renumbering concept files ===\n")

all_rename_maps = {}
for directory, order in SECTIONS:
    print(f"\nProcessing: {directory}")
    rm = process_directory(directory, order)
    all_rename_maps[directory] = rm

# Update top-level README files
print("\n=== Updating README files ===")
for directory, order in SECTIONS:
    rm = all_rename_maps[directory]
    parent = os.path.dirname(directory)
    for readme in ["README.md", "readme.md"]:
        update_readme(f"{parent}/{readme}", rm)
        update_readme(f"{directory}/{readme}", rm)

# Update the main README
update_readme(f"{BASE}/README.md", {k: v for rm in all_rename_maps.values() for k, v in rm.items()})

print("\n✅ Done! All concept files renumbered.")
