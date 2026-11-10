#!/usr/bin/env python3
"""Report on-disk instructional artifact counts and audited manifest coverage."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKS = {
    "AI fundamentals": "ai",
    "Machine learning": "ml",
    "LLM": "llm",
    "Agentic AI": "agentic-ai",
    "Modern AI": "modern-ai",
    "MLOps": "mlops",
    "Reinforcement learning": "rl",
    "Statistics": "stats",
    "Computer vision": "cv",
    "NLP": "nlp",
}
ARTIFACTS = {
    "concepts": ("concepts", "*.md"),
    "notebooks": ("notebooks", "*.ipynb"),
    "implementations": ("implementations", "*.py"),
}


def count_files(track: str, folder: str, suffix: str) -> int:
    base = ROOT / track / folder
    return (
        sum(1 for path in base.rglob(suffix) if path.is_file() and path.name.lower() != "readme.md")
        if base.exists() else 0
    )


def main() -> None:
    manifest = json.loads((ROOT / "data/curriculum_manifest.json").read_text())
    by_track = Counter(item["track"] for item in manifest["lessons"])
    print("| Track | Concepts | Notebooks | Python implementations | Manifest lessons |")
    print("|---|---:|---:|---:|---:|")
    for label, track in TRACKS.items():
        values = [count_files(track, *ARTIFACTS[key]) for key in ARTIFACTS]
        print(f"| {label} (`{track}`) | {values[0]} | {values[1]} | {values[2]} | {by_track[track]} |")
    print("\nCounts are recursive on-disk files at generation time. They do not establish quality or completeness.")
    print("Manifest lesson counts cover required and baseline lessons in ten tracks; unindexed artifacts are outside that denominator.")
    families = [
        ("Papers", "papers", {"paper notes": "**/*.md", "notebooks": "**/*.ipynb"}),
        ("System design", "system-design", {"patterns": "patterns/**/*.md", "other guides": "**/*.md"}),
        ("Architecture review", "arch-review", {"systems": "systems/**/*.md", "post-mortems": "post-mortems/**/*.md", "diagram files": "diagrams/**/*", "animation source files": "animation-code/**/*", "animation records": "animations/**/*", "animated diagrams": "animated-diagrams/**/*"}),
        ("Coding guides", "coding", {"guides": "**/*.md"}),
        ("Cheat sheets", "cheatsheets", {"guides": "**/*.md"}),
        ("Projects", "projects", {"briefs": "**/*.md"}),
    ]
    print("\n| Advertised family | Artifact type | Files |")
    print("|---|---|---:|")
    for label, directory, artifact_types in families:
        seen: set[Path] = set()
        for artifact_type, pattern in artifact_types.items():
            paths = {
                p for p in (ROOT / directory).glob(pattern)
                if p.is_file() and p.name.lower() != "readme.md"
            }
            if artifact_type == "other guides":
                paths -= seen
            print(f"| {label} | {artifact_type} | {len(paths)} |")
            seen |= paths


if __name__ == "__main__":
    main()
