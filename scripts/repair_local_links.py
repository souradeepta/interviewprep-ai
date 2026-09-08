#!/usr/bin/env python3
"""Repair only missing Markdown links with an unambiguous repository match."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LINK_RE = re.compile(r"(?<!!)(\[[^]]*\]\()([^)]+)(\))")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
NUMBER_PREFIX = re.compile(r"^\d+-")


def normalized_name(name: str) -> str:
    return NUMBER_PREFIX.sub("", name.lower())


def candidate_files(index: dict[str, list[Path]], normalized_index: dict[str, list[Path]], target: str) -> list[Path]:
    basename = Path(target.split("#", 1)[0]).name
    if not basename or "XX" in basename or basename in {"other.md", "url"}:
        return []
    exact = index.get(basename, [])
    if exact:
        return exact
    normalized = normalized_name(basename)
    return normalized_index.get(normalized, [])


def files_to_scan(root: Path) -> list[Path]:
    excluded = {root / ".git", root / ".claude", root / ".pytest_cache"}
    history = (root / "docs" / "sessions", root / "docs" / "superpowers")
    return sorted(
        p for p in root.rglob("*.md")
        if not any(parent in p.parents for parent in excluded)
        and not any(p == folder or folder in p.parents for folder in history)
    )


def repair(root: Path, apply: bool) -> tuple[int, int]:
    changed = 0
    ambiguous = 0
    files = [
        p for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts and ".claude" not in p.parts
    ]
    index: dict[str, list[Path]] = {}
    normalized_index: dict[str, list[Path]] = {}
    for path in files:
        index.setdefault(path.name, []).append(path)
        normalized_index.setdefault(normalized_name(path.name), []).append(path)
    for source in files_to_scan(root):
        lines = source.read_text(errors="replace").splitlines(keepends=True)
        in_fence = False
        output: list[str] = []
        file_changed = False
        for line in lines:
            if FENCE_RE.match(line):
                in_fence = not in_fence
            if in_fence:
                output.append(line)
                continue

            def replace(match: re.Match[str]) -> str:
                nonlocal ambiguous, file_changed
                target = match.group(2).strip().strip("<>")
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    return match.group(0)
                path_part, _, anchor = target.partition("#")
                destination = (source.parent / path_part).resolve() if path_part else source
                if destination.exists():
                    return match.group(0)
                candidates = candidate_files(index, normalized_index, target)
                if len(candidates) != 1:
                    if len(candidates) > 1:
                        ambiguous += 1
                    return match.group(0)
                relative = candidates[0].relative_to(source.parent.resolve()) if candidates[0].is_relative_to(source.parent.resolve()) else Path(__import__('os').path.relpath(candidates[0], source.parent))
                replacement = str(relative)
                if not replacement.startswith("."):
                    replacement = "./" + replacement
                if anchor:
                    replacement += "#" + anchor
                file_changed = True
                return match.group(1) + replacement + match.group(3)

            output.append(LINK_RE.sub(replace, line))
        if file_changed:
            changed += 1
            if apply:
                source.write_text("".join(output))
            else:
                print(source)
    return changed, ambiguous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="write unambiguous repairs")
    args = parser.parse_args()
    changed, ambiguous = repair(args.root.resolve(), args.apply)
    mode = "updated" if args.apply else "would update"
    print(f"{mode} {changed} Markdown file(s); left {ambiguous} ambiguous match(es) unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
