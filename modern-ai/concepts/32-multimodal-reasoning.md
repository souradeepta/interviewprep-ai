# Multimodal Fusion and Missing Inputs

**TL;DR:** Multimodal systems combine evidence from different encoders, and must define how scores are aligned, weighted and renormalized when a modality is noisy or absent.

## Prerequisites

Embedding models and classification metrics.

## Learning Outcomes

- Distinguish early, joint-attention and late score fusion.
- Implement late fusion that renormalizes weights over present modalities.
- Diagnose missing-modality and disagreement cases without treating absence as a zero score.
- Identify calibration and slice tests needed before deployment.

## Core Intuition

A doctor combines a patient's words, an image and an audio description. If the image was never captured, interpreting its score as zero would incorrectly count evidence against a diagnosis. A missing modality means “no observation”; it should be excluded or handled by an explicit learned missingness policy. Conflicting available signals should also be surfaced.

## How It Works

Early fusion combines representations before prediction, allowing cross-modal interactions but requiring aligned dimensions and more joint training. Late fusion combines modality-specific logits or probabilities. For scores s_m and nonnegative weights a_m, the simple baseline is \(s=\sum_{m\in A}a_ms_m/\sum_{m\in A}a_m\), where A is the set of present modalities. This assumes scores are comparable and weights meaningful; raw logits from differently calibrated models often are not.

Joint attention or contrastive alignment can learn cross-modal relationships, but needs paired data and a consistent preprocessing contract. Missingness may correlate with the label, creating a shortcut. A robust system evaluates every modality subset likely to occur, reports coverage, and may return an abstention when evidence is insufficient or contradictory.

The modalities must first be represented on compatible scales. Text and image encoders can each emit a score that looks like a probability but has different calibration. Averaging them directly can let an overconfident channel dominate even when it is less accurate. Calibrate on a held-out set of paired examples, inspect reliability by modality availability, and compare score fusion against simple majority or single-modality baselines. A fusion improvement should persist across the subsets used in production, not only when every sensor is present.

Presence itself is a feature and a potential source of leakage. A camera may be missing because a device model is older, and the device model may correlate with a label. If the classifier uses missingness as a shortcut, it can appear strong on the current cohort and fail after deployment changes. Evaluate metrics conditioned on each modality subset and on the reason for missingness where known. Train with modality dropout or simulated outages when appropriate, but keep synthetic missingness patterns realistic.

Late fusion gives useful modularity: each channel can be tested, updated or disabled independently. It also assumes the channel predictions can be combined after each encoder has already compressed its input. If a question requires grounding a word in a specific image region or synchronizing speech with video, score-level combination may discard the alignment needed to answer. Cross-attention or token-level fusion retains interactions at a higher compute and data cost. Select the simplest architecture that passes alignment-specific evaluations.

Disagreement should be treated as a first-class output. A single weighted average hides whether all modalities support the same answer or one strong channel overrides two conflicting signals. Track pairwise disagreement, per-modality confidence and coverage. Thresholds for abstention need calibration against the cost of incorrect answers and unnecessary deferrals. In high-consequence cases, a missing critical modality may require a refusal or a request for new input rather than renormalizing remaining weights.

Data pairing and preprocessing can dominate modeling. Ensure examples align temporally and semantically; an audio clip paired with the wrong image teaches spurious correlations. Keep image resize/crop, audio sampling rate, text normalization and timestamps versioned. Split train and validation by subject, session or source to prevent near-duplicate leakage across modalities. Audit consent, retention and access policy per modality because an image or audio stream can contain sensitive information not present in its text transcription.

## Trade-offs

| Fusion | Strength | Boundary |
|---|---|---|
| Early feature concatenation | Learns interactions | Requires aligned features and joint training |
| Cross-attention | Flexible token-level grounding | Compute and data demands are higher |
| Late score fusion | Modular and easy to ablate | Calibration and interaction are limited |
| Missing-modality specialist | Explicit subset robustness | More models or training cases to maintain |

## Implementation with Python

Run `python3 modern-ai/implementations/32-multimodal-reasoning.py`. Three fixed scalar confidence signals are fused, then one modality is removed and the remaining weights renormalized. This demonstrates bookkeeping, not multimodal understanding.

```python
import runpy
fuse = runpy.run_path("modern-ai/implementations/32-multimodal-reasoning.py")["fuse"]
text_only = fuse(text=.7, image=None, audio=None)
assert text_only == .7
```

## Failure Modes

| Failure | Diagnostic | Mitigation |
|---|---|---|
| Missing channel treated as negative evidence | Evaluate modality ablations and null-input cases | Renormalize present weights or train an explicit missingness path |
| Miscalibrated channels dominate | Calibration curves per modality | Calibrate scores on held-out paired examples |
| Contradictory inputs produce confident answer | Disagreement and abstention rate | Add conflict-aware policy and inspect examples |

- Missing channel acts as negative evidence: test each availability subset and compare it with an explicit missing-input policy.
- A miscalibrated channel dominates: inspect modality-specific reliability and recalibrate using paired validation examples.
- Incorrect pairing teaches shortcuts: audit temporal and identity alignment, split by source/session and inspect disagreement cases.

## Interview Quick Reference

| Question | Expected answer / follow-up |
|---|---|
| When is late fusion a good baseline? | Independent encoders or intermittent inputs; ask how you calibrate their scores. |
| How should missing image data be represented? | As missingness, with a tested policy, not automatically a zero; follow up on informative missingness. |
| How do you debug a multimodal regression? | Slice by available modalities and disagreement; ask which paired data is required. |

## Related Concepts

See [multimodal fine-tuning](20-multimodal-finetuning.md), [model evaluation](01-llm-evaluation-harness.md), [notebook 32](../notebooks/32-multimodal-reasoning.ipynb), and [inference interview practice](../interview-prep/inference-systems.md#multimodal-fusion).

Evaluation should cover complete and incomplete sensor conditions. Build a matrix whose rows are modality subsets and whose columns include task quality, calibration, coverage and abstention. Rare subsets need targeted validation examples or realistic outage simulation. Do not infer robustness from random modality masking if production failures are correlated, such as low light degrading both images and OCR. Include corrupted-but-present inputs as well as absent inputs; a non-null tensor can still carry no useful evidence.

## References

[CLIP](https://arxiv.org/abs/2103.00020) and [Flamingo](https://arxiv.org/abs/2204.14198).
