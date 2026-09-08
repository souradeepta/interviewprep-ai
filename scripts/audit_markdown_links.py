#!/usr/bin/env python3
"""Audit local Markdown links without requiring third-party packages."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


LINK_RE = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Problem:
    source: str
    line: int
    target: str
    kind: str


def slugify_heading(heading: str) -> str:
    heading = re.sub(r"[`*_~]", "", heading).strip().lower()
    heading = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"\s+", "-", heading)


def headings(path: Path) -> set[str]:
    found: set[str] = set()
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            found.add(slugify_heading(match.group(1)))
    return found


def markdown_files(root: Path, include_history: bool) -> list[Path]:
    excluded = {root / ".git", root / ".claude", root / ".pytest_cache"}
    history = (root / "docs" / "sessions", root / "docs" / "superpowers")
    files = []
    for path in root.rglob("*.md"):
        if any(parent in path.parents for parent in excluded):
            continue
        if not include_history and any(path == folder or folder in path.parents for folder in history):
            continue
        files.append(path)
    return sorted(files)


def audit(root: Path, include_history: bool = False) -> list[Problem]:
    problems: list[Problem] = []
    for source in markdown_files(root, include_history):
        in_fence = False
        for line_number, line in enumerate(source.read_text(errors="replace").splitlines(), 1):
            if FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            # Markdown-looking calls inside inline code are code examples.
            searchable = re.sub(r"`[^`]*`", "", line)
            for raw_target in LINK_RE.findall(searchable):
                target = raw_target.strip().strip("<>")
                if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_part, _, anchor = target.partition("#")
                destination = (source.parent / path_part).resolve() if path_part else source
                if not destination.exists():
                    problems.append(Problem(str(source.relative_to(root)), line_number, target, "missing-file"))
                elif anchor and destination.is_file() and anchor.lower() not in {h.lower() for h in headings(destination)}:
                    problems.append(Problem(str(source.relative_to(root)), line_number, target, "missing-anchor"))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-history", action="store_true", help="also audit dated session/spec/plan snapshots")
    parser.add_argument("--json", action="store_true", help="emit machine-readable problem records")
    args = parser.parse_args()
    problems = audit(args.root.resolve(), args.include_history)
    if args.json:
        import json

        print(json.dumps([asdict(problem) for problem in problems], indent=2))
    else:
        for problem in problems:
            print(f"{problem.kind}: {problem.source}:{problem.line}: {problem.target}")
        print(f"Checked active Markdown; found {len(problems)} problem(s).", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
