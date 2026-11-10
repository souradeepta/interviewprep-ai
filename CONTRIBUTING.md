# Contributing

Thank you for helping improve InterviewPrep AI. Contributions can add or repair
lessons, notebooks, practice material, capstones, navigation, or validation.

## Source of truth

For audited curriculum content, [`docs/CURRICULUM_CONTENT_STANDARD.md`](docs/CURRICULUM_CONTENT_STANDARD.md)
is the source of truth for lesson, notebook, practice, capstone, manifest, and
local-link requirements. Read the applicable contract before editing. This
guide explains how those requirements fit the repository workflow; it does not
define a competing template. Existing files vary by track, so use the standard
and nearby files for context instead of copying an old structure blindly.

The machine-readable inventory and prerequisite graph are in
[`data/curriculum_manifest.json`](data/curriculum_manifest.json). The current
automated curriculum audit covers the tracks and artifacts indexed there. A
file elsewhere in the repository is not automatically covered by that audit.
Do not describe an unaudited track or artifact as passing the curriculum audit.

## Adding or updating curriculum material

1. **Choose the learner outcome and location.** Check the relevant track README,
   roadmap, nearby lesson files, and manifest entry (if one exists). Keep the
   change focused enough for a reviewer to read the complete content.
2. **Add the concept.** Put it in the track's `concepts/` directory. For an
   audited lesson, use the concept-note contract in the content standard and
   keep the prerequisites and outcomes aligned with its manifest record. Link
   prerequisites and related concepts with repository-relative Markdown links.
3. **Add its notebook.** Put it in the track's `notebooks/` directory, following
   the local filename and numbering convention. Meet the notebook contract in
   the standard. Keep examples deterministic and API-free by default; document
   dependencies and any execution limitations. Do not commit execution outputs
   unless the relevant repository convention explicitly requires them.
4. **Add an implementation when it helps.** The manifest's `implementation`
   field is nullable. Use the track's existing `implementations/` layout when a
   separate implementation artifact is useful; do not add one merely to fill a
   template.
5. **Update the manifest for an audited lesson.** Add or update the stable
   `track/slug` ID, track, level, concept and notebook paths, nullable
   implementation path, prerequisite IDs, at least two measurable outcomes,
   practice references, estimated minutes, and `quality_status`. Add a
   prerequisite-free lesson to `entry_lessons` before assigning it an empty
   prerequisite list. Keep the roadmap's learner-facing sequence consistent
   with the manifest graph. Use `baseline` for known legacy material awaiting
   repair; use `required` only when the lesson meets the standard.
6. **Add or update practice.** Put interview questions in the relevant track's
   `interview-prep/` area and link them from the lesson and its manifest
   `practice` list when applicable. Each question in audited practice needs the
   prompt, domain or role, difficulty, standalone expected answer, evaluation
   dimensions, follow-up, and common mistake. Coding questions also need
   runnable test cases and time and space complexity.
7. **Add or update a capstone.** Follow the capstone contract: include the
   problem, data contract, deliverables, a 100-point rubric with clear point
   values, timebox, extension, prerequisite lesson links, and a local
   executable success criterion. Link it from the appropriate project or track
   README and learner path. If capstone paths are represented in the manifest
   for that track, update those records consistently.
8. **Update navigation and neighboring references.** Add links where learners
   enter the material, and check prerequisite, next-lesson, practice, and
   capstone links affected by the change. Keep counts and status claims in sync
   with the files they describe.

For material outside the manifest's current scope, follow the relevant track's
existing organization and clearly avoid implying that the curriculum audit
covers it. Do not add a legacy artifact to the manifest as `required` solely to
make it appear in the inventory.

## Local validation

Run the curriculum audit for changes to manifest-owned material. The default
command checks the full indexed inventory; `--track` narrows it while iterating.

```bash
python3 scripts/audit_curriculum.py
python3 scripts/audit_curriculum.py --track ml
python3 scripts/audit_curriculum.py --format json
```

Run the repository's active-Markdown local link and heading-anchor check when
adding or changing Markdown links. It excludes dated session and superpowers
history by default; it does not check external URLs.

```bash
python3 scripts/audit_markdown_links.py --root .
```

Also review the complete diff and whitespace errors:

```bash
git diff --check
git status --short
git diff -- CONTRIBUTING.md
```

For notebook edits, the curriculum audit validates notebook JSON and required
structure. Runtime execution is handled by the repository's test and CI
notebook smoke coverage; that coverage is representative, so it does not mean
every notebook has been executed. Follow the notebook's documented environment
and run it locally when practical. Record any execution skip and its reason in
the change description.

## Terra/Luna review cycle

Every substantial curriculum change needs an independent author and reviewer.
Use Terra and Luna as the roles in the project plan; whichever person or agent
implements a batch cannot be its sole approver. For a small documentation-only
change, the same sequence can be recorded in the pull request description.

1. **Agree on a bounded batch.** Record the learner outcome, lesson IDs and
   paths, expected files, explicit exclusions, dependencies, acceptance
   criteria, references, and validation commands. Terra checks scope,
   dependencies, and verifiable criteria; Luna checks learner progression,
   topic coverage, and whether the batch can be fully reviewed.
2. **Implement and hand off evidence.** The author keeps paired artifacts,
   manifest metadata, and navigation aligned. Provide a changed-artifact list
   and validation results with the handoff.
3. **Review independently.** The reviewer reads all changed instructional
   content and checks technical correctness and teaching value separately.
   Review prerequisites, examples, exercises and answers, sources, links, and
   execution evidence as applicable. Record findings with severity, path and
   section or cell, evidence, and requested disposition:
   - **Blocking:** incorrect or misleading material, broken learner path,
     missing required artifact, invalid code, or failed required gate.
   - **Major:** materially weak or generic explanation/example, unsupported
     claim, or missing assessment signal.
   - **Minor:** wording, formatting, or non-blocking navigation issue.
4. **Repair and re-review.** The author addresses blocking and major findings
   or proposes an evidence-based exception. The reviewer checks each changed
   finding and the nearby links, prerequisites, references, or tests affected
   by the repair. The author does not close their own blocking finding; send
   disagreements to the project owner with both positions and evidence.
5. **Close the batch.** Record final dispositions, checks run and their
   results, accepted follow-up issues with owners, and any known limitation.
   Close only when no blocking findings remain and the applicable acceptance
   criteria and validation gates pass.

## Pull request handoff

Describe the learner or contributor outcome, list changed tracks and artifact
types, identify manifest or navigation changes, and summarize validation
commands and results. Include the Terra/Luna review findings and dispositions
for curriculum batches. Call out notebook runtime skips, exceptions, or
remaining follow-up work with an owner. Keep unrelated work in separate
reviewable batches.
