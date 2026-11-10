#!/usr/bin/env python3
"""Validate manifest-owned curriculum inventory and content contracts."""

from __future__ import annotations

import argparse
import io
import json
import keyword
import re
import sys
import tokenize
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/curriculum_manifest.json"
SCAFFOLD_PHRASES = (
    "[To be filled",
    "Important property 1",
    "Mistake 1",
    "[Key insight]",
    "Core idea: understand the fundamental principle",
    "See notebook",
    "Challenge 1",
    "1. Core concept",
    "Modify parameters",
)
CONCEPT_SECTIONS = (
    "Prerequisites",
    "Learning Outcomes",
    "Core Intuition",
    "How It Works",
    "Trade-offs",
    "Implementation",
    "Failure Modes",
    "Interview Quick Reference",
    "Related Concepts",
)
PRACTICE_HEADINGS = (
    "Prompt",
    "Role/Domain",
    "Difficulty",
    "Expected Answer",
    "Evaluation",
    "Follow-up",
    "Common Mistake",
)
CAPSTONE_HEADINGS = (
    "Problem Brief",
    "Data Contract",
    "Deliverables",
    "Evaluation Rubric",
    "Timebox",
    "Extension Prompt",
    "Prerequisites",
)


def _manifest_data(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"cannot read manifest {path}: {exc}"]
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        return None, ["manifest must be an object with schema_version 1"]
    if not isinstance(data.get("lessons"), list):
        return None, ["manifest lessons must be a list"]
    if not isinstance(data.get("entry_lessons"), list):
        errors.append("manifest entry_lessons must be a list")
    return data, errors


def _lesson_artifact_paths(lesson: dict[str, Any]) -> list[tuple[str, str]]:
    fields = ("concept", "notebook", "implementation", "roadmap", "practice", "capstone")
    refs: list[tuple[str, str]] = []
    for field in fields:
        value = lesson.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and item.strip() and "\x00" not in item:
                refs.append((field, item.split("#", 1)[0]))
    return refs


def _validate_manifest(data: dict[str, Any], root: Path, track: str | None) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    all_lessons = data["lessons"]
    ids: dict[str, dict[str, Any]] = {}
    owned_paths: dict[str, dict[str, str]] = {field: {} for field in ("concept", "notebook", "implementation")}
    for index, lesson in enumerate(all_lessons):
        if not isinstance(lesson, dict):
            errors.append(f"lessons[{index}] must be an object")
            continue
        lesson_id = lesson.get("id")
        if not isinstance(lesson_id, str) or not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", lesson_id):
            errors.append(f"lessons[{index}] has invalid stable id {lesson_id!r}")
            continue
        if lesson_id in ids:
            errors.append(f"duplicate lesson id: {lesson_id}")
        ids[lesson_id] = lesson
        if lesson.get("track") != lesson_id.split("/", 1)[0]:
            errors.append(f"{lesson_id}: track must match the first ID segment")
        if lesson.get("quality_status") not in {"baseline", "required"}:
            errors.append(f"{lesson_id}: quality_status must be baseline or required")
        if not isinstance(lesson.get("prerequisites"), list):
            errors.append(f"{lesson_id}: prerequisites must be a list")
        elif not lesson["prerequisites"] and lesson_id not in data.get("entry_lessons", []):
            errors.append(f"{lesson_id}: empty prerequisites are allowed only for entry_lessons")
        if not isinstance(lesson.get("outcomes"), list) or len(lesson["outcomes"]) < 2:
            errors.append(f"{lesson_id}: outcomes must contain at least two items")
        if not isinstance(lesson.get("estimated_minutes"), int) or lesson["estimated_minutes"] <= 0:
            errors.append(f"{lesson_id}: estimated_minutes must be a positive integer")
        for field in ("concept", "notebook", "implementation", "roadmap", "practice", "capstone"):
            value = lesson.get(field)
            if value is None:
                continue
            if value == [] and field not in owned_paths:
                continue
            if field in owned_paths and not isinstance(value, str):
                errors.append(f"{lesson_id}: {field} must be a single path")
                continue
            values = value if isinstance(value, list) else [value]
            if not values or any(not isinstance(item, str) or not item.strip() for item in values):
                errors.append(f"{lesson_id}: {field} must be a non-empty path or list of paths")
                continue
            if any("\x00" in item for item in values):
                errors.append(f"{lesson_id}: {field} path contains a NUL character")
                continue
            if field in owned_paths:
                for item in values:
                    path = item.split("#", 1)[0]
                    resolved = str((root / path).resolve())
                    owner = owned_paths[field].get(resolved)
                    if owner and owner != lesson_id:
                        errors.append(f"{lesson_id}: duplicate {field} path {path} also owned by {owner}")
                    owned_paths[field][resolved] = lesson_id

    for lesson_id, lesson in ids.items():
        prerequisites = lesson.get("prerequisites", [])
        if isinstance(prerequisites, list):
            for prerequisite in prerequisites:
                if not isinstance(prerequisite, str) or prerequisite not in ids:
                    errors.append(f"{lesson_id}: unresolved prerequisite {prerequisite!r}")

    state: dict[str, int] = {}
    def visit(node: str, stack: list[str]) -> None:
        if state.get(node) == 1:
            cycle = " -> ".join(stack + [node])
            errors.append(f"prerequisite cycle: {cycle}")
            return
        if state.get(node) == 2:
            return
        state[node] = 1
        parents = ids[node].get("prerequisites", [])
        for parent in parents if isinstance(parents, list) else []:
            if isinstance(parent, str) and parent in ids:
                visit(parent, stack + [node])
        state[node] = 2
    for lesson_id in ids:
        visit(lesson_id, [])

    selected = [lesson for lesson in all_lessons if isinstance(lesson, dict) and (track is None or lesson.get("track") == track)]
    for lesson in selected:
        lesson_id = lesson.get("id", "<unknown>")
        for field, relpath in _lesson_artifact_paths(lesson):
            target = (root / relpath).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{lesson_id}: {field} path escapes repository: {relpath}")
                continue
            if not target.is_file():
                errors.append(f"{lesson_id}: missing {field} artifact {relpath}")

        notebook = lesson.get("notebook")
        if isinstance(notebook, str) and notebook and "\x00" not in notebook:
            notebook_path = root / notebook
            try:
                nb = json.loads(notebook_path.read_text(encoding="utf-8"))
                if not isinstance(nb, dict) or nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
                    errors.append(f"{lesson_id}: notebook must be valid v4 JSON with a cells list ({notebook})")
                else:
                    errors.extend(_check_notebook_json_shape(lesson_id, nb, notebook))
                    if lesson.get("quality_status") == "required":
                        errors.extend(_check_notebook(lesson_id, nb, notebook))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{lesson_id}: malformed notebook JSON {notebook}: {exc}")

        if lesson.get("quality_status") != "required":
            continue
        for field in ("concept", "roadmap", "practice", "capstone"):
            value = lesson.get(field)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if not isinstance(item, str) or not item:
                    continue
                target = root / item.split("#", 1)[0]
                if target.suffix.lower() in {".md", ".markdown"} and target.is_file():
                    content = target.read_text(encoding="utf-8")
                    errors.extend(_check_links(lesson_id, target, content, root))
                    if field == "concept":
                        errors.extend(_check_concept(lesson_id, content, item, lesson.get("lesson_type")))
                    if field == "practice":
                        errors.extend(_check_headings(lesson_id, content, PRACTICE_HEADINGS, "practice"))
                    if field == "capstone":
                        errors.extend(_check_headings(lesson_id, content, CAPSTONE_HEADINGS, "capstone"))
    return errors, selected


def _prose_word_count(markdown: str) -> int:
    text = re.sub(r"```.*?```|~~~.*?~~~", " ", markdown, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+.*$", " ", text, flags=re.M)
    text = re.sub(r"!?\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return len(re.findall(r"\b[\w'-]+\b", text))


def _check_concept(lesson_id: str, content: str, path: str, lesson_type: str | None = None) -> list[str]:
    errors: list[str] = []
    headings = [(m.group(1).strip().lower(), m.start()) for m in re.finditer(r"^##\s+(.+?)\s*$", content, re.M)]
    heading_text = [name for name, _ in headings]
    for section in CONCEPT_SECTIONS:
        if not any(section.lower() in heading for heading in heading_text):
            errors.append(f"{lesson_id}: concept {path} missing section {section!r}")
    positions = []
    for section in CONCEPT_SECTIONS:
        matches = [pos for name, pos in headings if section.lower() in name]
        if matches:
            positions.append(matches[0])
    if len(positions) == len(CONCEPT_SECTIONS) and positions != sorted(positions):
        errors.append(f"{lesson_id}: concept sections are out of contract order ({path})")
    first_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not first_lines or not first_lines[0].startswith("# "):
        errors.append(f"{lesson_id}: concept {path} must start with a level-one title")
    if len(first_lines) < 2 or not re.match(r"(?:TL;DR\s*:?\s*)?\S.+[.!?]$", first_lines[1], re.I):
        errors.append(f"{lesson_id}: concept {path} must have a one-sentence TL;DR immediately after the title")
    for phrase in SCAFFOLD_PHRASES:
        if phrase.lower() in content.lower():
            errors.append(f"{lesson_id}: concept {path} contains scaffold phrase {phrase!r}")
    words = _prose_word_count(content)
    reference = lesson_type == "reference"
    minimum = 650 if reference else 900
    if words < minimum:
        errors.append(f"{lesson_id}: concept {path} has {words} prose words; minimum is {minimum}")
    urls = re.findall(r"https?://[^\s)>]+", content)
    if len(urls) < 2:
        errors.append(f"{lesson_id}: concept {path} needs at least two authoritative references")
    if not re.search(r"```(?:python|py)\b", content, re.I) and "executable pseudocode" not in content.lower():
        errors.append(f"{lesson_id}: concept {path} needs a Python example or labeled executable pseudocode")
    outcomes = _section(content, "Learning Outcomes")
    outcome_lines = [line for line in outcomes.splitlines() if re.match(r"\s*(?:[-*]|\d+[.)])\s+\S", line)]
    if not 3 <= len(outcome_lines) <= 5:
        errors.append(f"{lesson_id}: concept {path} needs three to five measurable learning outcomes")
    tradeoffs = _section(content, "Trade-offs")
    table_rows = [line for line in tradeoffs.splitlines() if line.strip().startswith("|") and not re.match(r"\s*\|[\s:|-]+\|\s*$", line)]
    if len(table_rows) < 4:  # header plus at least three topic-specific alternatives
        errors.append(f"{lesson_id}: concept {path} needs a trade-off table with at least three data rows")
    failures = _section(content, "Failure Modes")
    failure_lines = [line for line in failures.splitlines() if re.match(r"\s*(?:[-*]|\d+[.)])\s+\S", line)]
    if len(failure_lines) < 3:
        errors.append(f"{lesson_id}: concept {path} needs at least three failure modes with diagnostics or mitigations")
    interview = _section(content, "Interview Quick Reference")
    interview_rows = [line for line in interview.splitlines() if line.strip().startswith("|") and not re.match(r"\s*\|[\s:|-]+\|\s*$", line)]
    if len(interview_rows) < 4:
        errors.append(f"{lesson_id}: concept {path} needs at least three interview Q&A rows")
    if not re.search(r"\b(debug|design|follow-up)\b", interview, re.I):
        errors.append(f"{lesson_id}: concept {path} needs a design or debugging interview follow-up")
    return errors


def _section(markdown: str, name: str) -> str:
    match = re.search(rf"^##\s+[^\n]*{re.escape(name)}[^\n]*\n(.*?)(?=^##\s+|\Z)", markdown, re.M | re.I | re.S)
    return match.group(1) if match else ""


def _check_notebook(lesson_id: str, notebook: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    cells = notebook["cells"]
    nonempty = [cell for cell in cells if isinstance(cell, dict) and _cell_source(cell).strip()]
    if len(nonempty) < 9:
        errors.append(f"{lesson_id}: notebook {path} has {len(nonempty)} non-empty cells; minimum is 9")
    code = "\n".join(_cell_source(cell) for cell in nonempty if cell.get("cell_type") == "code")
    all_source = "\n".join(_cell_source(cell) for cell in nonempty)
    if not code.strip():
        errors.append(f"{lesson_id}: notebook {path} has no executable code cell")
    if not re.search(r"\bassert\b", code):
        errors.append(f"{lesson_id}: notebook {path} has no assertion")
    if not re.search(r"\bexercise|try it|your turn\b", all_source, re.I):
        errors.append(f"{lesson_id}: notebook {path} has no exercise")
    for label, pattern in (
        ("title/prerequisites/time/objectives", r"title|prerequisite|estimated time|objective"),
        ("baseline implementation", r"baseline|from scratch|transparent implementation"),
        ("comparison or metric", r"comparison|metric|visualiz"),
        ("edge or failure cases", r"edge case|failure case|failure mode"),
        ("summary, next lesson, and practice", r"summary|next lesson|practice"),
    ):
        if not re.search(pattern, all_source, re.I):
            errors.append(f"{lesson_id}: notebook {path} is missing {label}")
    for phrase in SCAFFOLD_PHRASES:
        if phrase.lower() in all_source.lower():
            errors.append(f"{lesson_id}: notebook {path} contains scaffold phrase {phrase!r}")
    return errors


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(part for part in source if isinstance(part, str))
    return ""


def _has_assertion_token(notebook: dict[str, Any]) -> bool:
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        try:
            if any(token.type == tokenize.NAME and token.string == "assert" and keyword.iskeyword(token.string)
                   for token in tokenize.generate_tokens(io.StringIO(_cell_source(cell)).readline)):
                return True
        except (tokenize.TokenError, IndentationError):
            continue
    return False


def _check_notebook_json_shape(lesson_id: str, notebook: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict):
            errors.append(f"{lesson_id}: notebook {path} cell {index} must be an object")
            continue
        if cell.get("cell_type") not in {"markdown", "code", "raw"}:
            errors.append(f"{lesson_id}: notebook {path} cell {index} has invalid cell_type")
        source = cell.get("source")
        if not isinstance(source, (str, list)) or (isinstance(source, list) and not all(isinstance(part, str) for part in source)):
            errors.append(f"{lesson_id}: notebook {path} cell {index} has invalid source")
        if not isinstance(cell.get("metadata"), dict):
            errors.append(f"{lesson_id}: notebook {path} cell {index} metadata must be an object")
        if cell.get("cell_type") == "code" and not isinstance(cell.get("execution_count"), (int, type(None))):
            errors.append(f"{lesson_id}: notebook {path} code cell {index} has invalid execution_count")
        if cell.get("cell_type") == "code" and not isinstance(cell.get("outputs"), list):
            errors.append(f"{lesson_id}: notebook {path} code cell {index} outputs must be a list")
    return errors


def _check_headings(lesson_id: str, content: str, required: tuple[str, ...], kind: str) -> list[str]:
    errors = []
    normalized = content.lower()
    for heading in required:
        if heading.lower() not in normalized:
            errors.append(f"{lesson_id}: {kind} page missing required heading/content {heading!r}")
    return errors


def _check_links(lesson_id: str, source_path: Path, content: str, root: Path) -> list[str]:
    errors: list[str] = []
    for match in re.finditer(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", content):
        raw = match.group(1).strip().split()[0].strip("<>")
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc or (not parsed.path and not parsed.fragment):
            continue
        target_path = (source_path.parent / unquote(parsed.path)).resolve() if parsed.path else source_path.resolve()
        try:
            target_path.relative_to(root.resolve())
        except ValueError:
            errors.append(f"{lesson_id}: link escapes repository in {source_path.relative_to(root)}: {raw}")
            continue
        if not target_path.exists():
            errors.append(f"{lesson_id}: broken local link in {source_path.relative_to(root)}: {raw}")
            continue
        if parsed.fragment and target_path.is_file() and target_path.suffix.lower() in {".md", ".markdown"}:
            target_text = target_path.read_text(encoding="utf-8")
            anchors: set[str] = set()
            counts: dict[str, int] = {}
            for heading in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", target_text, re.M):
                slug = _slug(heading.group(1))
                count = counts.get(slug, 0)
                anchors.add(f"{slug}-{count}" if count else slug)
                counts[slug] = count + 1
            if unquote(parsed.fragment).lower() not in anchors:
                errors.append(f"{lesson_id}: broken local anchor in {source_path.relative_to(root)}: {raw}")
    return errors


def _slug(heading: str) -> str:
    heading = re.sub(r"[`*_~]", "", heading).strip().lower()
    heading = re.sub(r"[^\w -]", "", heading)
    return re.sub(r"\s+", "-", heading.strip())


def _unindexed_artifacts(root: Path, lessons: list[Any], selected_track: str | None = None) -> dict[str, dict[str, list[str]]]:
    """Return concept, notebook, and implementation files not named by the manifest."""
    indexed: dict[str, set[str]] = {}
    for lesson in lessons:
        if not isinstance(lesson, dict):
            continue
        track = lesson.get("track")
        if not isinstance(track, str) or not re.fullmatch(r"[a-z0-9-]+", track) or (selected_track is not None and track != selected_track):
            continue
        refs = indexed.setdefault(track, set())
        for field, path in _lesson_artifact_paths(lesson):
            if field in {"concept", "notebook", "implementation"}:
                refs.add((root / path).resolve().as_posix())
    result: dict[str, dict[str, list[str]]] = {}
    patterns = {"concepts": "concepts/**/*.md", "notebooks": "notebooks/**/*.ipynb", "implementations": "implementations/**/*.py"}
    tracks = [selected_track] if selected_track is not None else list(indexed)
    for track in tracks:
        refs = indexed.get(track, set())
        result[track] = {}
        folder = (root / track).resolve()
        try:
            folder.relative_to(root.resolve())
        except ValueError:
            result[track] = {kind: [] for kind in patterns}
            continue
        for kind, pattern in patterns.items():
            files = sorted(path for path in folder.glob(pattern) if path.is_file() and path.name.lower() != "readme.md") if folder.exists() else []
            missing = [path.relative_to(root).as_posix() for path in files if path.resolve().as_posix() not in refs]
            result[track][kind] = missing
    return result


def audit(root: Path = ROOT, manifest_path: Path = DEFAULT_MANIFEST, track: str | None = None) -> dict[str, Any]:
    data, errors = _manifest_data(manifest_path)
    if data is not None:
        manifest_errors, selected = _validate_manifest(data, root, track)
        errors.extend(manifest_errors)
        inventory: dict[str, dict[str, int]] = {}
        warnings: list[str] = []
        for lesson in data["lessons"]:
            if not isinstance(lesson, dict):
                continue
            name = lesson.get("track") if isinstance(lesson.get("track"), str) else "<unknown>"
            counts = inventory.setdefault(name, {"required": 0, "baseline": 0, "excluded": 0, "missing": 0})
            if track is not None and name != track:
                counts["excluded"] += 1
                continue
            status = lesson.get("quality_status")
            if status in ("required", "baseline"):
                counts[status] += 1
            # Missing counts lessons with at least one declared artifact absent.
            if any(not (root / path).is_file() for _, path in _lesson_artifact_paths(lesson)):
                counts["missing"] += 1
            concept = lesson.get("concept")
            if status == "required" and isinstance(concept, str):
                path = root / concept.split("#", 1)[0]
                if path.is_file():
                    sentences = re.split(r"(?<=[.!?])\s+", path.read_text(encoding="utf-8"))
                    frequencies: dict[str, int] = {}
                    for sentence in sentences:
                        normalized = " ".join(sentence.lower().split())
                        if len(normalized.split()) >= 8:
                            frequencies[normalized] = frequencies.get(normalized, 0) + 1
                    if max(frequencies.values(), default=0) >= 5:
                        warnings.append(f"{lesson.get('id', '<unknown>')}: concept {concept} repeats a sentence at least five times; review for generic filler")
            notebook_path = lesson.get("notebook")
            if status == "required" and isinstance(notebook_path, str) and "\x00" not in notebook_path:
                try:
                    nb = json.loads((root / notebook_path).read_text(encoding="utf-8"))
                    if isinstance(nb, dict) and isinstance(nb.get("cells"), list) and not _has_assertion_token(nb):
                        warnings.append(f"{lesson.get('id', '<unknown>')}: notebook {notebook_path} has no detectable Python assertion token; review its tests and syntax")
                except (OSError, json.JSONDecodeError):
                    pass  # The manifest validator reports malformed or missing notebooks.
        return {
            "ok": not errors,
            "track": track,
            "lesson_count": len(selected),
            "required_count": sum(1 for lesson in selected if lesson.get("quality_status") == "required"),
            "baseline_count": sum(1 for lesson in selected if lesson.get("quality_status") == "baseline"),
            "inventory_by_track": inventory,
            "unindexed_artifacts_by_track": _unindexed_artifacts(root, data["lessons"], track),
            "inventory_definition": "Counts cover manifest lessons only; excluded means outside the selected track, and missing counts selected lessons with absent declared artifacts (overlaps status counts). Unindexed lessons are not counted.",
            "human_review": "Structural checks and warnings do not verify technical accuracy, topic-specific teaching, references, or executable behavior; independent human review is required.",
            "warnings": warnings,
            "errors": errors,
        }
    return {"ok": False, "track": track, "lesson_count": 0, "required_count": 0, "baseline_count": 0, "inventory_by_track": {}, "unindexed_artifacts_by_track": {}, "warnings": [], "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--track", help="audit only lessons in this track")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    result = audit(ROOT, args.manifest, args.track)
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Curriculum audit {'passed' if result['ok'] else 'failed'}: {result['lesson_count']} selected manifest lessons")
        for track, counts in sorted(result["inventory_by_track"].items()):
            print(f"  {track}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
        for track, groups in sorted(result.get("unindexed_artifacts_by_track", {}).items()):
            unindexed = {kind: len(paths) for kind, paths in groups.items() if paths}
            if unindexed:
                print(f"  {track} unindexed files: " + ", ".join(f"{kind}={count}" for kind, count in unindexed.items()))
        if "inventory_definition" in result:
            print(result["inventory_definition"])
            print(result["human_review"])
        for warning in result["warnings"]:
            print(f"Warning: {warning}", file=sys.stderr)
        for error in result["errors"]:
            print(f"- {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
