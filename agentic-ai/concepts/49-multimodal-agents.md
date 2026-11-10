# Multimodal Agents
**TL;DR:** A multimodal agent combines text and non-text inputs while preserving modality provenance, alignment, and explicit behavior when a modality is missing.

## Prerequisites
Agent loops, tool boundaries, and evaluation.

## Learning Outcomes
- Explain the main modality fusion workflow and its state boundary.
- Compare an automated path with review or fallback using measured outcomes.
- Diagnose a representative failure using trace evidence.

## Core Intuition
Multimodal Agents is best understood as a controlled workflow with explicit evidence and a bounded decision. The system should expose what it knows, what it is allowed to do, and when a person or deterministic checker must take over. This framing prevents a fluent answer from being confused with a verified outcome.

## How It Works
Normalize timestamps, crops, units, and confidence before fusion. Make missingness explicit instead of filling absent modalities with zeros that resemble valid measurements. Preserve the ability to inspect each modality-specific result and calibrate confidence after fusion. A model that accepts image and text inputs does not prove that both were used correctly.
## Detailed Explanation

Multimodal agents process multiple data types: text, images, audio, video. Mechanisms: (1) modality fusion (combine signals), (2) cross-modal understanding, (3) joint reasoning. Advantages: richer context, better understanding, more capabilities. Challenges: alignment (which parts of image relate to which text), modality gaps (image resolution vs text precision), computational cost. Best for: visual question answering, medical imaging, autonomous vehicles, accessibility.


## Interview Q&A

**Q: How do you handle the case where different modalities provide conflicting information?**
A: Conflicting modalities are common: a document might have text saying "increase sales" but a chart showing declining sales. Design the agent to explicitly surface conflicts: "The text summary indicates X, but the accompanying chart shows Y—these appear to contradict each other." Don't silently resolve conflicts by preferring one modality. Ask for clarification or flag for human review. For automated pipelines, implement a consistency check between modalities and route inconsistent items for human verification.

**Q: What are the latency implications of multimodal agents and how do you optimize them?**
A: Image encoding typically adds 200-500ms per image. Video frame extraction and processing can add seconds. Audio transcription adds 500ms-2s. To optimize: cache encodings for repeated media (same image in multiple requests), process modalities in parallel where possible (encode all images while processing text), use lightweight specialized models for initial filtering (is this image relevant?), and reserve full multimodal analysis for items that pass initial screening.

**Q: How do you ensure a multimodal agent correctly attributes information to its source modality?**
A: Attribution is critical for debugging and user trust. Design the agent to cite which modality provided each piece of information: "Based on the image (timestamp 2:34 in the video)..." or "The PDF text on page 3 states..." Implement source tracking through the reasoning chain, not just in the final output. For RAG pipelines that mix text and images, store the modality and source location with each retrieved chunk. Test attribution accuracy by injecting known facts in specific modalities and verifying the agent correctly identifies the source.

**Q: What data preprocessing is needed for reliable multimodal agent performance?**
A: Images: normalize resolution (resize to model's expected input), handle EXIF orientation, convert to RGB (handle CMYK/grayscale), and strip malicious metadata. Audio: normalize volume levels, handle different sample rates, remove silence padding. Video: extract keyframes at consistent intervals, handle variable frame rates. Documents: extract text from PDFs accurately (handle scanned vs. digital), preserve table structure, handle multiple columns. Preprocessing quality significantly impacts downstream model quality—invest in a robust preprocessing pipeline.

**Q: How do you build evaluation datasets for multimodal agent quality?**
A: Construct test cases that require genuine multi-modal reasoning (not solvable from one modality alone). Include: questions where text and image together provide the answer, cases testing cross-modal reference ("click the button shown in step 3"), cases with conflicting information across modalities, and edge cases (low-quality images, audio with background noise, corrupted files). Label the expected reasoning path, not just the answer, to enable detailed error analysis. Benchmark against human performance on the same test cases.

**Q: When should you use a specialized model (OCR, speech-to-text, image classifier) vs. a general multimodal LLM?**
A: Specialized models for: tasks requiring high accuracy on a specific modality (OCR: >99% character accuracy needed; speech-to-text: low word error rate), structured extraction (reading tabular data from images), high-throughput preprocessing (transcribing millions of audio files), and when cost matters (specialized models are 5-10x cheaper than multimodal LLMs). General multimodal LLMs for: tasks requiring cross-modal reasoning, tasks where context from multiple modalities is needed, open-ended analysis where you don't know what to extract, and when output is a natural language response.
Evaluate each modality alone and in combination, including corrupted or absent input, misalignment, and subgroup conditions. Measure task quality, calibration, latency, memory, and modality-specific failure rates.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Deterministic baseline | Predictable behavior and low cost | Limited coverage of ambiguous cases |
| Model-assisted path | Handles flexible inputs | Requires evidence checks and evaluation |
| Human review | Resolves high-impact uncertainty | Capacity and response-time limits |
| Fallback or abstention | Bounds unsafe behavior | May defer useful work |

## Implementation
This runnable Python example shows a bounded policy gate relevant to the workflow. It models control logic only; it does not claim production quality.

```python
def fuse(scores):
    available = [score for score in scores.values() if score is not None]
    return sum(available) / len(available) if available else None

assert fuse({'text': 0.8, 'image': None}) == 0.8
```

## Failure Modes
- Multimodal Agents uses stale or incomplete input: log source and timestamp, validate freshness, and request a refresh before acting.
- Multimodal Agents produces an unsupported result: require evidence and a domain-specific validator; route unresolved cases to review.
- Multimodal Agents exceeds its resource or authority boundary: enforce explicit budgets and permissions in the controller, then report the stopped state.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| How do you handle the case where different modalities provide conflicting information? | Conflicting modalities are common: a document might have text saying "increase sales" but a chart showing declining sales. Design the agent to explicitly surface conflicts: "The text summary indicates X, but the accompanying chart shows Y—these appear to contradict each other." Don't silently resolve conflicts by preferring one modality. Ask for clarification or flag for human review. For automated pipelines, implement a consistency check between modalities and route inconsistent items for human verification. |
| What are the latency implications of multimodal agents and how do you optimize them? | Image encoding typically adds 200-500ms per image. Video frame extraction and processing can add seconds. Audio transcription adds 500ms-2s. To optimize: cache encodings for repeated media (same image in multiple requests), process modalities in parallel where possible (encode all images while processing text), use lightweight specialized models for initial filtering (is this image relevant?), and reserve full multimodal analysis for items that pass initial screening. |
| How do you ensure a multimodal agent correctly attributes information to its source modality? | Attribution is critical for debugging and user trust. Design the agent to cite which modality provided each piece of information: "Based on the image (timestamp 2:34 in the video)..." or "The PDF text on page 3 states..." Implement source tracking through the reasoning chain, not just in the final output. For RAG pipelines that mix text and images, store the modality and source location with each retrieved chunk. Test attribution accuracy by injecting known facts in specific modalities and verifying the agent correctly identifies the source. |

## Related Concepts
See [agent testing](28-agent-testing.md), [agent evaluation](29-agent-evals.md), and [agent monitoring](30-agent-monitoring.md).

## References
- [www.nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)
- [opentelemetry.io](https://opentelemetry.io/docs/)

## Further Topic Detail
## Detailed Explanation

Multimodal agents process multiple data types: text, images, audio, video. Mechanisms: (1) modality fusion (combine signals), (2) cross-modal understanding, (3) joint reasoning. Advantages: richer context, better understanding, more capabilities. Challenges: alignment (which parts of image relate to which text), modality gaps (image resolution vs text precision), computational cost. Best for: visual question answering, medical imaging, autonomous vehicles, accessibility.


## Interview Q&A

**Q: How do you handle the case where different modalities provide conflicting information?**
A: Conflicting modalities are common: a document might have text saying "increase sales" but a chart showing declining sales. Design the agent to explicitly surface conflicts: "The text summary indicates X, but the accompanying chart shows Y—these appear to contradict each other." Don't silently resolve conflicts by preferring one modality. Ask for clarification or flag for human review. For automated pipelines, implement a consistency check between modalities and route inconsistent items for human verification.

**Q: What are the latency implications of multimodal agents and how do you optimize them?**
A: Image encoding typically adds 200-500ms per image. Video frame extraction and processing can add seconds. Audio transcription adds 500ms-2s. To optimize: cache encodings for repeated media (same image in multiple requests), process modalities in parallel where possible (encode all images while processing text), use lightweight specialized models for initial filtering (is this image relevant?), and reserve full multimodal analysis for items that pass initial screening.

**Q: How do you ensure a multimodal agent correctly attributes information to its source modality?**
A: Attribution is critical for debugging and user trust. Design the agent to cite which modality provided each piece of information: "Based on the image (timestamp 2:34 in the video)..." or "The PDF text on page 3 states..." Implement source tracking through the reasoning chain, not just in the final output. For RAG pipelines that mix text and images, store the modality and source location with each retrieved chunk. Test attribution accuracy by injecting known facts in specific modalities and verifying the agent correctly identifies the source.

**Q: What data preprocessing is needed for reliable multimodal agent performance?**
A: Images: normalize resolution (resize to model's expected input), handle EXIF orientation, convert to RGB (handle CMYK/grayscale), and strip malicious metadata. Audio: normalize volume levels, handle different sample rates, remove silence padding. Video: extract keyframes at consistent intervals, handle variable frame rates. Documents: extract text from PDFs accurately (handle scanned vs. digital), preserve table structure, handle multiple columns. Preprocessing quality significantly impacts downstream model quality—invest in a robust preprocessing pipeline.

**Q: How do you build evaluation datasets for multimodal agent quality?**
A: Construct test cases that require genuine multi-modal reasoning (not solvable from one modality alone). Include: questions where text and image together provide the answer, cases testing cross-modal reference ("click the button shown in step 3"), cases with conflicting information across modalities, and edge cases (low-quality images, audio with background noise, corrupted files). Label the expected reasoning path, not just the answer, to enable detailed error analysis. Benchmark against human performance on the same test cases.

**Q: When should you use a specialized model (OCR, speech-to-text, image classifier) vs. a general multimodal LLM?**
A: Specialized models for: tasks requiring high accuracy on a specific modality (OCR: >99% character accuracy needed; speech-to-text: low word error rate), structured extraction (reading tabular data from images), high-throughput preprocessing (transcribing millions of audio files), and when cost matters (specialized models are 5-10x cheaper than multimodal LLMs). General multimodal LLMs for: tasks requiring cross-modal reasoning, tasks where context from multiple modalities is needed, open-ended analysis where you don't know what to extract, and when output is a natural language response.
