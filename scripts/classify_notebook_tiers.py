#!/usr/bin/env python3
"""Generate a reproducible execution-tier ledger for tracked notebooks."""
from __future__ import annotations

import json
import ast
import re
import subprocess
import sys
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_MODULES = {
    "numpy", "scipy", "sklearn", "matplotlib", "torch", "pytest", "nbformat",
}
RESOURCE_RULES = {
    "model download or loading": re.compile(r"from_pretrained\s*\(|load_model\s*\(|load_dataset\s*\(", re.I),
    "external HTTP request": re.compile(r"requests\.|urlopen\s*\(|https?://", re.I),
    "provider API or credential": re.compile(r"api[_-]?key|anthropic|openai\.\w+|google\.colab", re.I),
    "download or shell install": re.compile(r"(?:!|%)(?:pip|wget|curl)\b|\b(?:wget|curl)\s+", re.I),
    "accelerator-specific execution": re.compile(r"\.cuda\s*\(|device\s*=\s*['\"]cuda", re.I),
}
STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def modules(source: str) -> set[str]:
    result: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                result.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                result.add(node.args[0].value.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                result.add(node.args[0].value.split(".")[0])
        return result
    # IPython magics can make a cell invalid Python; remove comments and string
    # literals before scanning imports to avoid interpreting notebook prose.
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        cleaned = tokenize.untokenize(token for token in tokens if token.type not in {tokenize.COMMENT, tokenize.STRING})
    except (tokenize.TokenError, IndentationError):
        cleaned = re.sub(r"(?s)(\"\"\".*?\"\"\"|''' .*? ''')", "", source)
    for match in re.findall(r"^\s*import\s+([^\n;]+)|^\s*from\s+([\w.]+)\s+import\b", cleaned, re.M):
        clause = match[0] or match[1]
        result.update(part.strip().split(" as ")[0].split(".")[0] for part in clause.split(",") if part.strip())
    return result


def code_without_comments(source: str) -> str:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return tokenize.untokenize(token for token in tokens if token.type != tokenize.COMMENT)
    except (tokenize.TokenError, IndentationError):
        return re.sub(r"(?m)#.*$", "", source)


def resource_reasons(code_cells: list[str]) -> list[str]:
    found: set[str] = set()
    for source in code_cells:
        for line in source.splitlines():
            if re.match(r"^\s*[!%]\s*(?:pip|wget|curl)\b", line, re.I):
                found.add("download or shell install")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                if names & {"anthropic", "openai", "google"}:
                    found.add("provider API or credential")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in {"anthropic", "openai", "google"}:
                    found.add("provider API or credential")
            elif isinstance(node, ast.Call):
                func = node.func
                attr = func.attr if isinstance(func, ast.Attribute) else ""
                root = func.value.id if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else ""
                if attr in {"from_pretrained", "load_model", "load_dataset"}:
                    found.add("model download or loading")
                if attr == "urlopen" or (root == "requests" and attr in {"get", "post", "put", "delete", "request"}):
                    found.add("external HTTP request")
                if (root == "os" and attr == "system") or (root == "subprocess" and attr in {"run", "Popen", "call"}):
                    if any(isinstance(arg, ast.Constant) and isinstance(arg.value, str) and re.search(r"\b(?:pip|wget|curl)\b", arg.value) for arg in node.args):
                        found.add("download or shell install")
                if attr == "cuda" or (root == "torch" and attr == "device" and any(isinstance(arg, ast.Constant) and arg.value == "cuda" for arg in node.args)):
                    found.add("accelerator-specific execution")
    return [name for name in RESOURCE_RULES if name in found]


def alternative_check(reasons: list[str]) -> str:
    checks = []
    if "model download or loading" in reasons:
        checks.append("test configuration and tensor shapes with a synthetic in-memory fixture")
    if "external HTTP request" in reasons:
        checks.append("mock the transport and test request construction, timeout, and failure handling")
    if "provider API or credential" in reasons:
        checks.append("use a deterministic provider stub and test parsing, retries, and error paths")
    if "download or shell install" in reasons:
        checks.append("test the post-download processing on a checked-in tiny fixture")
    if "accelerator-specific execution" in reasons:
        checks.append("run a small CPU equivalent where supported and test device-selection behavior")
    if not checks:
        checks.append("test deterministic helpers with a small fixture under the dependency-specific CI job")
    return "Validate nbformat and source policy; " + "; ".join(checks) + "."


def classify(path: str) -> dict:
    notebook = json.loads((ROOT / path).read_text(encoding="utf-8"))
    code_cells = ["".join(cell.get("source", [])) for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    code = "\n".join(code_cells)
    used = set().union(*(modules(cell) for cell in code_cells))
    external = sorted(used - STDLIB - PINNED_MODULES)
    resource_flags = resource_reasons(code_cells)
    if not code.strip():
        tier, reason = 1, "No executable code cells; validate nbformat, links, policy, and content structure."
    elif external or resource_flags:
        tier = 3
        details = []
        if external:
            details.append("unavailable or unpinned imports: " + ", ".join(external))
        details.extend(resource_flags)
        reason = "; ".join(details)
    else:
        tier, reason = 2, "Uses only Python standard library and pinned notebook-smoke modules; execute in a clean kernel."
    alternative = None if tier == 2 else alternative_check(resource_flags)
    return {
        "path": path,
        "tier": tier,
        "reason": reason,
        "alternative_check": alternative,
        "detected_imports": sorted(used),
        "detected_resource_calls": resource_flags,
    }


def main() -> None:
    paths = subprocess.check_output(["git", "ls-files", "*.ipynb"], cwd=ROOT, text=True).splitlines()
    rows = [classify(path) for path in paths]
    tier3_representatives = [
        {"path": "agentic-ai/notebooks/01-what-is-an-agent.ipynb", "category": "provider API or credentials", "evidence_import": "anthropic", "check": "Test provider configuration parsing with a stub; never make a live call in default CI."},
        {"path": "papers/retrieval/notebooks/01-rag.ipynb", "category": "extra dependency and model download", "evidence_import": "sentence_transformers", "check": "Run the deterministic retrieval helper using a local toy corpus; validate ranking and empty-result behavior without fetching an embedding model."},
        {"path": "papers/code-systems/notebooks/01-codet5.ipynb", "category": "model download", "evidence_call": "model download or loading", "check": "Validate config and tensor-shape logic with an in-memory tiny model fixture; do not fetch weights."},
        {"path": "agentic-ai/notebooks/51-web-agents.ipynb", "category": "external HTTP", "evidence_call": "external HTTP request", "check": "Mock the request transport and verify payload, timeout, and failure handling."},
    ]
    output = {
        "schema_version": 1,
        "source": "git ls-files *.ipynb",
        "tier_definitions": {
            "1": "Structure and policy validation only; notebook has no executable code cells.",
            "2": "Clean-kernel execution in the pinned notebook-smoke environment.",
            "3": "Excluded from default execution due to recorded extra dependency, external resource, model, API, or accelerator requirement; an alternative check is recorded.",
        },
        "static_classification_note": "Tier assignments are generated from imports and source patterns; resource and dependency findings require human confirmation before Tier 3 exclusions are accepted.",
        "tier3_representatives": tier3_representatives,
        "notebooks": rows,
    }
    (ROOT / "data/notebook_execution_tiers.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    counts = {str(tier): sum(row["tier"] == tier for row in rows) for tier in (1, 2, 3)}
    print(f"Classified {len(rows)} tracked notebooks: {counts}")


if __name__ == "__main__":
    main()
