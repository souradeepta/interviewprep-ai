#!/usr/bin/env python3
"""Apply reviewed, deterministic repairs for legacy generated links."""

from pathlib import Path


REPLACEMENTS = {
    "./01-descriptive-statistics.md": "./01-probability-fundamentals.md",
    "./02-probability-distributions.md": "./02-distributions-reference.md",
    "./03-hypothesis-testing.md": "./05-hypothesis-testing.md",
    "./04-bayesian-statistics.md": "./03-bayesian-inference.md",
    "./05-regression-analysis.md": "./15-statistical-ml-connections.md",
    "./07-regression-analysis.md": "./15-statistical-ml-connections.md",
    "./08-bayesian-inference.md": "./03-bayesian-inference.md",
    "./01-descriptive-statistics.md": "./01-probability-fundamentals.md",
    "./01-markov-decision-process.md": "./01-markov-decision-processes.md",
    "./03-markov-decision-processes.md": "./01-markov-decision-processes.md",
    "./05-q-learning.md": "./06-q-learning.md",
    "./15-temporal-difference.md": "./05-temporal-difference-learning.md",
    "../../../llm/concepts/32-reward-modeling.md": "../../../llm/concepts/06-rlhf.md",
    "../llm/concepts/02-attention-mechanisms.md": "../../../llm/concepts/24-attention-optimization.md",
    "../../llm/concepts/12-attention-mechanisms.md": "../../../llm/concepts/24-attention-optimization.md",
    "../../llm/concepts/04-scaling-laws.md": "../../../llm/concepts/03-pretraining.md",
    "../../llm/concepts/05-lora.md": "../../../llm/concepts/08-lora.md",
    "../vision/concepts/02-vision-transformer.md": "../../cv/concepts/04-vision-transformers.md",
    "../retrieval/concepts/rag.md": "../retrieval/concepts/01-rag.md",
    "../../retrieval/concepts/rag.md": "../retrieval/concepts/01-rag.md",
    "../retrieval/concepts/01-rag.md": "../../retrieval/concepts/01-rag.md",
    "../foundation-models/concepts/cot.md": "../../../llm/concepts/16-chain-of-thought.md",
    "../foundation-models/concepts/few-shot.md": "../../../llm/concepts/13-few-shot-learning.md",
    "../agents/concepts/planning.md": "../../../agentic-ai/concepts/07-planning-reasoning.md",
    "../foundation-models/concepts/ranking.md": "../../../llm/concepts/21-semantic-search.md",
    "../../../agentic-ai/concepts/XX-policy-optimization.md": "../../../agentic-ai/concepts/07-planning-reasoning.md",
    "../../agentic-ai/concepts/XX-prompt-engineering.md": "../../../agentic-ai/concepts/35-agent-prompt-engineering.md",
    "../../agentic-ai/concepts/XX-few-shot-learning.md": "../../../llm/concepts/13-few-shot-learning.md",
    "../../agentic-ai/concepts/XX-tool-use.md": "../../../agentic-ai/concepts/03-tool-use.md",
    "../../agentic-ai/concepts/XX-agent-loops.md": "../../../agentic-ai/concepts/02-agent-loops.md",
    "../../agentic-ai/concepts/XX-planning.md": "../../../agentic-ai/concepts/07-planning-reasoning.md",
    "../../agentic-ai/concepts/XX-search-algorithms.md": "../../../agentic-ai/concepts/58-advanced-reasoning-variants.md",
    "../../llm/concepts/XX-in-context-learning.md": "../../../llm/concepts/15-in-context-learning.md",
    "../../foundation-models/concepts/XX-scaling-laws.md": "../../nlp/concepts/04-scaling-laws.md",
    "../../nlp/concepts/XX-generation-metrics.md": "../../nlp/concepts/02-bert.md",
    "../../foundation-models/concepts/XX-benchmarking.md": "../../nlp/concepts/02-bert.md",
    "../../../llm/concepts/15-lora.md": "../../../llm/concepts/08-lora.md",
    "../../nlp/concepts/xx-dpr.md": "../../nlp/concepts/08-information-retrieval.md",
    "../../../ml/concepts/xx-faiss.md": "../../../ml/concepts/feature-engineering.md",
    "../../../ai/concepts/xx-contrastive-learning.md": "../../../cv/concepts/05-contrastive-learning-vision.md",
    "../../../llm/concepts/xx-multimodal-transformers.md": "../../../llm/concepts/31-multimodal.md",
    "../../../cv/concepts/xx-vision-transformers.md": "../../../cv/concepts/04-vision-transformers.md",
    "../../../ml/concepts/xx-zero-shot-learning.md": "../../../llm/concepts/14-zero-shot-learning.md",
    "./02-prompt-engineering.md": "../../../llm/concepts/12-prompting.md",
    "./04-fine-tuning.md": "../../../llm/concepts/04-finetuning.md",
    "../../vision/concepts/02-vit.md": "../../vision/concepts/02-vision-transformer.md",
    "../../modern-ai/concepts/13-parameter-efficient-fine-tuning.md": "../../../llm/concepts/11-parameter-efficient-finetuning.md",
    "../../../coding/ml-coding/implement-attention.md": "../interview-prep/ml-coding-questions.md",
    "../../../coding/ml-coding/implement-transformer.md": "../interview-prep/ml-coding-questions.md",
    "../../system-design/patterns/ab-testing.md": "../../system-design/patterns/14-ab-testing.md",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    changed = 0
    for path in root.rglob("*.md"):
        if ".git" in path.parts or ".claude" in path.parts or "docs" in path.parts:
            continue
        text = path.read_text(errors="replace")
        updated = text
        for old, new in REPLACEMENTS.items():
            updated = updated.replace(f"]({old})", f"]({new})")
        if "ml/concepts" in str(path):
            updated = updated.replace("](other.md)", "](../README.md)")
            updated = updated.replace("](url)", "](../README.md)")
        if "ai/concepts" in str(path):
            updated = updated.replace("](./XX-related-1.md)", "](./04-optimization-algorithms.md)")
            updated = updated.replace("](./XX-related-2.md)", "](../ml/concepts/optimization.md)")
            updated = updated.replace("](./XX-related-3.md)", "](../stats/concepts/01-probability-fundamentals.md)")
        if updated != text:
            path.write_text(updated)
            changed += 1
    print(f"updated {changed} Markdown file(s)")


if __name__ == "__main__":
    main()
