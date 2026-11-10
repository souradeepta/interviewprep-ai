#!/usr/bin/env python3
"""Normalize tracked notebooks to valid nbformat v4.5 cell schemas."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def normalize(notebook: dict) -> dict:
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("expected a v4 notebook with a cells list")
    notebook["nbformat_minor"] = 5
    seen: set[str] = set()
    for index, cell in enumerate(notebook["cells"]):
        cell_type = cell.get("cell_type")
        if isinstance(cell_type, dict):
            cell_type = cell_type.get("text/plain")
            cell["cell_type"] = cell_type
        cell.setdefault("metadata", {})
        if "parameters" in cell:
            cell["metadata"]["legacy_parameters"] = cell.pop("parameters")
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
            cell_id = f"cell-{index:04d}"
            cell["id"] = cell_id
        seen.add(cell_id)
        if cell_type == "code":
            cell.pop("execute_count", None)
            cell.pop("language", None)
            cell.setdefault("execution_count", None)
            cell.setdefault("outputs", [])
        elif cell_type in {"markdown", "raw"}:
            for key in ("execute_count", "execution_count", "outputs", "language"):
                cell.pop(key, None)
        else:
            raise ValueError(f"unsupported cell type: {cell_type!r}")
    return notebook


def main() -> None:
    changed = 0
    for relative in subprocess.check_output(["git", "ls-files", "*.ipynb"], cwd=ROOT, text=True).splitlines():
        path = ROOT / relative
        raw = path.read_text(encoding="utf-8")
        notebook = json.loads(raw)
        try:
            nbformat.validate(nbformat.reads(raw, as_version=4))
            continue
        except nbformat.ValidationError:
            pass
        fixed = normalize(notebook)
        nbformat.validate(fixed)
        path.write_text(json.dumps(fixed, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        changed += 1
    print(f"Normalized {changed} notebooks; all tracked files validated as nbformat v4.5.")


if __name__ == "__main__":
    main()
