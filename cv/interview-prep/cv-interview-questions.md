# Computer Vision Interview Questions

Each prompt includes an expected answer and criteria for evaluating the reasoning. Scenarios emphasize data, metrics, distribution shift, and deployment constraints.

### 1. A classifier has strong accuracy but misses rare defects
**Prompt:** A factory image classifier reports 98% accuracy, but inspectors still find many missed defects. What do you investigate?
**Role/domain:** Computer vision; quality inspection. **Difficulty:** Medium.
**Expected answer:** Inspect class prevalence, confusion matrix, recall/precision at the operating threshold, label quality, and defect-specific slices. Choose a threshold from miss-versus-review costs and compare to a simple baseline; accuracy hides false negatives on the rare class.
**Evaluation:** Metric selection, label investigation, operational threshold.
**Follow-up:** How would you estimate uncertainty if only a few defects are labeled?
**Common mistake:** Celebrating accuracy without comparing it to an always-negative classifier.

### 2. Design augmentation for a small dataset
**Prompt:** You have 600 labeled photos of packaged goods. Which augmentations do you try, and how do you prevent label corruption?
**Role/domain:** CV training; data augmentation. **Difficulty:** Medium.
**Expected answer:** Apply label-preserving transformations plausible at deployment, such as modest crop, brightness, or orientation changes when orientation is irrelevant. Keep validation data unaugmented, inspect transformed examples, and transform boxes/masks consistently for detection/segmentation.
**Evaluation:** Domain assumptions, label consistency, validation discipline.
**Follow-up:** Which transformation could invalidate a left/right defect label?
**Common mistake:** Applying every available augmentation without checking invariance.

### 3. Diagnose annotation disagreement
**Prompt:** Annotators disagree on object boundaries and model IoU stalls. How do you tell whether the bottleneck is the model or label process?
**Role/domain:** CV data quality; debugging. **Difficulty:** Hard.
**Expected answer:** Sample disagreements and errors for adjudication, quantify inter-annotator variation, clarify boundary policy, and compare model error to human agreement. Repair ambiguous labels and re-evaluate on a versioned adjudicated set before changing architecture.
**Evaluation:** Measurement of label noise, targeted review, data versioning.
**Follow-up:** How should an ambiguous boundary count in evaluation?
**Common mistake:** Treating all annotations as perfect ground truth.

### 4. Choose a detection metric
**Prompt:** Explain why object detection uses average precision over IoU thresholds rather than only classification accuracy.
**Role/domain:** Object detection. **Difficulty:** Easy.
**Expected answer:** Detection must match predicted boxes to ground truth, account for localization quality, and rank confidence thresholds. AP summarizes precision-recall behavior; COCO-style mAP averages over classes and IoU thresholds, exposing localization quality that image-level accuracy misses.
**Evaluation:** Matching, localization, threshold/ranking interpretation.
**Follow-up:** What happens when duplicate boxes overlap one true object?
**Common mistake:** Confusing per-image class accuracy with object-level detection quality.

### 5. Debug a low mAP release
**Prompt:** A detector’s mAP falls after data collection expands to nighttime images. What slice analyses do you run?
**Role/domain:** CV deployment; debugging. **Difficulty:** Hard.
**Expected answer:** Break down by lighting, camera, object size, class, and occlusion; inspect label coverage and preprocessing parity; compare confidence calibration and localization error. Determine whether new data represents a true domain shift or annotation/schema change, then choose targeted data or robustness work.
**Evaluation:** Slice design, label/preprocessing checks, causal restraint.
**Follow-up:** How do you decide whether to retrain or use test-time adaptation?
**Common mistake:** Comparing only aggregate mAP and immediately increasing model capacity.

### 6. Compare IoU and Dice for segmentation
**Prompt:** When can Dice be easier to interpret than IoU, and what important cases do both metrics hide?
**Role/domain:** Segmentation. **Difficulty:** Medium.
**Expected answer:** Both measure overlap; Dice is (2|A∩B|/(|A|+|B|)), IoU is (|A∩B|/|A∪B|), and Dice is numerically larger for the same masks. Both can hide boundary quality, small-object failures, and class imbalance; report per-class and boundary-aware measures where relevant.
**Evaluation:** Correct relation, aggregation and limitations.
**Follow-up:** Why can mean Dice obscure a critical tiny class?
**Common mistake:** Reporting only a global pixel average dominated by background.

### 7. Handle extreme class imbalance
**Prompt:** A medical-image segmentation dataset has 0.2% positive pixels. How do you train and evaluate without misleading yourself?
**Role/domain:** CV; imbalanced segmentation. **Difficulty:** Hard.
**Expected answer:** Use patient-level splits, inspect positive-case coverage, consider weighted/focal or overlap losses, and sample patches carefully. Report sensitivity, precision, PR behavior, calibration/threshold, and per-patient results; never use pixel accuracy as the main measure.
**Evaluation:** Split safety, training choice, clinically meaningful metrics.
**Follow-up:** How could patch sampling distort predicted probabilities?
**Common mistake:** Oversampling positives and then assuming model scores remain calibrated to deployment prevalence.

### 8. Avoid leakage in video splits
**Prompt:** Randomly split frames from each video into train and test; the result is unusually high. What is wrong?
**Role/domain:** Video understanding; evaluation. **Difficulty:** Medium.
**Expected answer:** Neighboring frames share scene, subject, and near-duplicate pixels, so the test set leaks video identity. Split by video, subject, or acquisition session matching deployment, deduplicate near-identical content, then reassess performance.
**Evaluation:** Correlation unit, split design, duplicate detection.
**Follow-up:** What split would test generalization to a new camera?
**Common mistake:** Treating every frame as an independent sample.

### 9. Choose edge deployment trade-offs
**Prompt:** A model misses its edge-device latency target by 3×. What sequence of changes do you evaluate?
**Role/domain:** CV serving; edge deployment. **Difficulty:** Hard.
**Expected answer:** Profile preprocessing, transfer, inference, and postprocessing on the target device and representative inputs. Consider input resolution, architecture, quantization, pruning, or accelerator support; measure accuracy/latency/memory together and retain a fallback if the quality gate fails.
**Evaluation:** End-to-end profiling, target hardware, quality-aware trade-off.
**Follow-up:** Why might model FLOPs not predict wall-clock latency?
**Common mistake:** Reporting GPU benchmark latency from a desktop as edge-device performance.

### 10. Handle a camera distribution shift
**Prompt:** A model degrades after a camera firmware update changes color and sharpening. Outline a safe response.
**Role/domain:** Production CV; distribution shift. **Difficulty:** Hard.
**Expected answer:** Verify the change with raw examples and pipeline checks, quantify shift and errors by camera/version, preserve a rollback or conservative fallback, and label a representative sample. Test normalization/augmentation or retraining on a time-separated set before promotion.
**Evaluation:** Incident containment, shift diagnosis, representative evaluation.
**Follow-up:** How do you separate covariate shift from label-definition change?
**Common mistake:** Automatically adapting to the new distribution without checking labels or safety impact.

### 11. Set a threshold for object detection
**Prompt:** A warehouse detector feeds a human review queue with limited capacity. How do you choose confidence threshold and NMS settings?
**Role/domain:** Detection system design. **Difficulty:** Medium.
**Expected answer:** Tune on representative validation data against recall/precision and review capacity, stratified by class/size. Measure duplicate suppression and missed-object cost; freeze the policy before test evaluation and monitor queue load after deployment.
**Evaluation:** Capacity constraint, class-specific costs, valid tuning.
**Follow-up:** When would class-specific thresholds be justified?
**Common mistake:** Selecting a threshold on the test set or optimizing mAP alone despite a queue limit.

### 12. Investigate a mask boundary failure
**Prompt:** Segmentation overlap is acceptable, but downstream area estimates are biased for thin structures. What do you change in evaluation?
**Role/domain:** Segmentation; measurement. **Difficulty:** Hard.
**Expected answer:** Evaluate boundary distance, topology or area error in addition to overlap; stratify by object thickness and image resolution, review annotation convention, and assess the downstream measurement’s tolerance. A small boundary displacement can have large relative area error on thin structures.
**Evaluation:** Metric/task alignment, scale effects, label policy.
**Follow-up:** Which metric would detect disconnected fragments?
**Common mistake:** Assuming a good global Dice score guarantees accurate geometry.

### 13. Evaluate a multimodal image-text model
**Prompt:** A vision-language model scores well on image-caption retrieval. What tests reveal whether it grounds attributes rather than exploiting shortcuts?
**Role/domain:** Multimodal evaluation. **Difficulty:** Hard.
**Expected answer:** Build counterfactual pairs that change one visual attribute while keeping text context, test compositional and negation cases, balance templates, and measure by attribute/category. Include image-only/text-only ablations and leakage checks; retrieval scores alone do not prove grounding.
**Evaluation:** Counterfactual design, shortcut analysis, ablations.
**Follow-up:** How would you test sensitivity to an irrelevant background?
**Common mistake:** Treating aggregate benchmark rank as evidence of robust visual grounding.

### 14. Decide whether to use transfer learning
**Prompt:** You have 2,000 labeled images from a specialized domain and a pretrained vision backbone. How do you choose frozen features versus fine-tuning?
**Role/domain:** Applied CV; transfer learning. **Difficulty:** Medium.
**Expected answer:** Compare a frozen linear head and staged fine-tuning on a leakage-safe split, with augmentation and regularization appropriate to dataset size. Inspect calibration and class slices; domain distance, label quantity, and compute determine the choice rather than a universal rule.
**Evaluation:** Baseline, split, data/compute trade-off.
**Follow-up:** What indicates catastrophic forgetting in full fine-tuning?
**Common mistake:** Fine-tuning every layer by default on a small noisy dataset.

### 15. Review an annotation-to-release pipeline
**Prompt:** Design quality gates from incoming images through a detector release.
**Role/domain:** CV system design. **Difficulty:** Hard.
**Expected answer:** Version raw data and labels, validate schema/coverage/duplicates, split by deployment unit, run baseline and slice metrics, review high-impact errors, and promote an immutable artifact through shadow/canary with latency and quality gates. Keep a rollback and auditable annotation policy.
**Evaluation:** End-to-end controls, lineage, release safety.
**Follow-up:** Which gate catches train/serve image preprocessing mismatch?
**Common mistake:** Treating model evaluation as the only gate in the vision pipeline.
