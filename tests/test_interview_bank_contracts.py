"""Stable structural contracts for ML/AI interview simulation banks."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARD_MARKERS = (
    "**Scenario and constraints:**",
    "**A strong candidate clarifies:**",
    "**Success criteria:**",
    "**Failure modes/edge cases:**",
    "**Interviewer follow-up:**",
    "**Rubric (0–3):**",
    "**Remediation:**",
)


def cards(path, expected_count):
    text = path.read_text()
    matches = list(re.finditer(r"^### Q(\d+) — .+$", text, re.MULTILINE))
    assert [int(match.group(1)) for match in matches] == list(range(1, expected_count + 1))
    result = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result.append(text[match.start():end])
    return result


def test_all_interview_banks_have_independent_simulation_cards():
    banks = (
        (ROOT / "llm/interview-prep/llm-interview-questions.md", 30),
        (ROOT / "agentic-ai/interview-prep/agent-interview-questions.md", 20),
        (ROOT / "system-design/interview-prep/question-bank.md", 25),
    )
    for path, count in banks:
        for card in cards(path, count):
            assert "**Round/time:**" in card
            for marker in CARD_MARKERS:
                assert marker in card, (path, marker, card[:100])


def test_active_document_counts_and_repair_link_are_current():
    readme = (ROOT / "llm/README.md").read_text()
    assert "45 Jupyter notebooks" in readme
    expansion = (ROOT / "docs/superpowers/specs/2026-09-07-ml-ai-interview-practice-expansion.md").read_text()
    assert "2026-09-07-interview-practice-correctness-repair.md" in expansion
    coding = (ROOT / "ml/interview-prep/ml-coding-questions.md").read_text()
    assert "Q1. Implement K-Nearest" in coding
    assert "Remaining high-value ML coding topics" not in coding
