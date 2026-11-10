# Curriculum Content Standard

This document defines what counts as a complete, interview-ready curriculum
artifact. The manifest at `data/curriculum_manifest.json` is the inventory and
prerequisite source of truth. A file existing on disk, appearing in a README,
or having a matching filename does not by itself make it complete.

## Manifest authoring

Every lesson has a stable `track/slug` ID, track, level, concept path, notebook
path, nullable implementation path, prerequisite IDs, at least two measurable
outcomes, one or more practice references, and a positive estimated time in
minutes. Practice, roadmap, and capstone paths may be added as their tracks
enter the audited curriculum. Use repository-relative paths. A prerequisite
list may be empty only for an ID listed in `entry_lessons`.

`quality_status` is `baseline` while a known legacy artifact awaits repair and
`required` once it must satisfy this standard. Both states validate manifest
shape, stable and unique IDs, prerequisite references and acyclicity, artifact
existence, and referenced notebook JSON as nbformat v4 with a `cells` list.
`required` additionally enforces the content and link contracts below. This
staged gate lets the repository inventory be recorded before every legacy
lesson is ready, without calling incomplete material complete. Promote a
lesson to `required` only after its concept, notebook, and required assessment
artifacts have been repaired.

## Concept note contract

Each concept note must use these sections in this order:

1. A level-one title and an immediately following, one-sentence TL;DR.
2. Prerequisites and three to five measurable learning outcomes.
3. Core intuition with a concrete analogy or counterexample.
4. How it works, including relevant notation, assumptions, and complexity.
5. A topic-specific decision or trade-off table with at least three rows.
6. One runnable, dependency-light Python example. An infrastructure-only topic
   may use clearly labeled executable pseudocode when Python would mislead.
7. At least three topic-specific failure modes, each paired with a diagnostic
   or mitigation.
8. An interview quick reference with at least three question and answer rows,
   including a design or debugging follow-up.
9. Related concepts with real relative links and at least two primary or
   authoritative references.

Write 900 to 1,800 substantive prose words. Headings, code, and reference URLs
do not count. A narrow reference page may contain 650 to 900 prose words only
when its manifest record declares `lesson_type: "reference"` and it still
meets every section requirement. Avoid repeating generic text to reach a word
target. Explain the role of this lesson in its track and make distinctions
from similarly named material explicit.

## Notebook contract

Every notebook is valid nbformat v4 JSON with at least nine non-empty cells,
including:

1. Title, prerequisites, estimated time, and measurable objectives.
2. Compact concept recap with assumptions.
3. A from-scratch or transparent baseline implementation.
4. Assertions for expected baseline behavior.
5. A realistic extension or included-library implementation where appropriate.
6. A comparison, metric, or visualization tied to the named topic.
7. At least two deliberately chosen edge or failure cases.
8. Exercises with expected properties or test cases.
9. Summary, next lessons, and linked practice.

Use a fixed random seed for stochastic examples. A default smoke notebook must
not require an API key. Write files only inside the notebook directory or
`/tmp`. A mocked external service must state what the mock cannot prove and
test the surrounding deterministic logic. Do not label a simulation as a
measured production result.

## Interview practice contract

Each question includes its prompt, role or domain, difficulty, a short
standalone expected answer, evaluation dimensions, at least one follow-up,
and a common mistake. Coding questions also include runnable test cases and
time and space complexity. Answers should teach sound reasoning without
delegating the answer to another artifact.

## Capstone contract

Each capstone includes a problem brief, supplied or synthetic data contract,
deliverables, an evaluation rubric, timebox, extension prompt, and links to
prerequisite lessons. State a local, executable success criterion. Rubric
dimensions and point values must be clear and total 100 points.

## Local links and scope

The audit follows links only from manifest-owned concept, roadmap, practice,
and capstone pages. Relative paths must resolve inside the repository, and a
linked Markdown heading anchor must exist. This scope avoids treating an
unrelated historical report or fixture as active curriculum content. Keep
fixtures out of the manifest unless a test specifically exercises a fixture
manifest.

## Audit commands

Run the default inventory audit, machine-readable report, or a single track:

```bash
python3 scripts/audit_curriculum.py
python3 scripts/audit_curriculum.py --format json
python3 scripts/audit_curriculum.py --track ml
```

The audit exits nonzero for malformed manifest structure, duplicate IDs,
missing paths, unresolved prerequisites, cycles, invalid notebook JSON, and
any required content or link contract violation. Tests in
`tests/test_curriculum_quality.py` pin section names, scaffold phrases, and
minimum lengths so these thresholds are explicit and reviewable.
