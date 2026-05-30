# Post-Mortem: Fraud Model Demographic Bias Discovered in Regulatory Audit

## Incident Summary
**Date:** 2024-07-22 (audit finding date; model launched 3 months earlier)
**Duration:** 3 months of biased predictions before audit detection; 8 weeks to full remediation
**Business Impact:** Regulatory compliance risk; potential fines up to $2.1M; reputational risk; mandatory model halt for the 18-25 demographic pending retrain
**Severity:** P1 (Regulatory and compliance; executive escalation; external audit involvement)

---

## Timeline

| Time | Event |
|------|-------|
| 2024-01-10 | Fraud model v3 training complete; overall AUC = 0.891 on holdout set |
| 2024-01-15 | Standard pre-launch evaluation: overall metrics pass all thresholds |
| 2024-01-22 | Model deployed to production; all traffic (100% rollout after 5-day canary) |
| 2024-01-22 | Model processes all transactions including those from users aged 18-25 |
| 2024-04-15 | Regulatory auditor begins 90-day ML system review (routine compliance audit) |
| 2024-07-10 | Auditor requests model performance breakdown by demographic group |
| 2024-07-18 | Analysis completed: overall AUC = 0.891; users aged 18-25: AUC = 0.672 |
| 2024-07-22 | Audit finding issued: "Significant performance disparity in fraud detection across age groups" |
| 2024-07-23 | Incident declared; model halted for users 18-25 (fallback to rules-based system) |
| 2024-08-01 | Root cause confirmed: training data heavily skewed toward fraud patterns of older users |
| 2024-09-15 | Retrained model with stratified sampling; AUC 18-25 group = 0.861; redeployed |

---

## What Happened (Technical)

The fraud model was trained on 24 months of historical transaction data with binary fraud labels from the investigation team. The training dataset contained 8.2M labeled transactions. Users aged 18-25 accounted for 11% of total transactions but 28% of fraud cases for this age group went uninvestigated (because the investigation team historically prioritized high-value transactions, and younger users tend to have lower-value accounts).

This created a systematic labeling gap: for users aged 18-25, only high-confidence fraud cases were labeled as fraudulent. Low-confidence but genuine fraud (new account takeover schemes, peer-to-peer payment fraud common in this demographic) appeared as "legitimate" in training data. The model learned that patterns associated with young users were correlated with non-fraud, because the labeled positives in this group were systematically sparse and biased.

The model's overall AUC (0.891) appeared strong in pre-launch evaluation because the 18-25 group (11% of transactions) had its poor performance averaged out. The pre-launch evaluation report showed a single aggregate metric with no demographic breakdown.

In production, two impacts were observed: (1) genuine fraud from young users was under-detected (false negatives, causing financial loss), and (2) legitimate transactions from young users were incorrectly flagged at a higher rate in some score bands (false positives, causing user friction). The second issue reached customer support as complaints about "wrongly blocked transactions" from young-adult users, but these were attributed to fraud policy rather than model quality.

---

## Root Cause Analysis

**Contributing factors:**
1. Training data was not representative: investigation/labeling effort skewed toward high-value, high-certainty cases disproportionately associated with older users
2. Pre-launch evaluation only computed aggregate AUC; no demographic or segment-level breakdown was required or run
3. Labeling process had a historical selection bias (human investigators prioritized certain case types) that was never documented or accounted for
4. Feature engineering used `transaction_amount` and `account_age` heavily; both correlated with user age in ways that encoded demographic proxies
5. No protected-attribute analysis was part of the model risk management process
6. Customer complaints about blocked young-user transactions were routed to fraud policy team, not ML team

**5 Whys:**

Why did the model perform poorly for users aged 18-25?
The model had AUC=0.672 on this group, indicating poor calibration and discrimination for their fraud patterns.

Why was the model poorly calibrated for this demographic?
The training data had systematically incomplete fraud labels for this group — legitimate transactions and actual fraud were conflated due to labeling gaps.

Why did the labeling gaps exist for young users?
Human investigators historically prioritized high-value cases and skipped low-value young-user transactions, creating sparse and biased labels for this demographic.

Why wasn't this labeling bias detected before training?
No analysis of label completeness by demographic segment was performed; the dataset was treated as uniformly labeled.

Why wasn't the deployment evaluation slice-aware?
The pre-launch evaluation checklist required only aggregate AUC > 0.85 and false positive rate < 2%; no slice-based evaluation was mandated.

---

## What Went Well

- The regulatory audit process eventually caught the issue — external review serves as a backstop
- The ML team's rapid response (model halted within 24 hours of audit finding) demonstrated process responsiveness
- The model registry preserved the full training dataset and metadata, making root cause analysis possible
- The rules-based fallback system for the 18-25 demographic was operational and deployable quickly

---

## Action Items

| Item | Owner | Due | Status |
|------|-------|-----|--------|
| Mandatory slice-based evaluation: all models must report AUC by age group, geography, and account tenure | ML Platform | +3 weeks | Done |
| Add label completeness audit to training data pipeline: flag segments with >20% lower investigation rate | Data Engineering | +4 weeks | Done |
| Retrain with stratified sampling: ensure 18-25 group is proportionally sampled in training | ML Research | +6 weeks | Done |
| Add protected attribute analysis to model risk management checklist | ML Governance | +2 weeks | Done |
| Route fraud complaint metadata to ML team: flag if complaints disproportionately come from a demographic | Operations | +4 weeks | In progress |
| Implement continuous fairness monitoring: alert if group AUC gap > 0.1 in production | ML Infra | +8 weeks | In progress |

---

## Interview Discussion Points

**What would you have done differently?**
Mandatory slice-based evaluation before any model launch. Every model card should include performance metrics broken down by age group, geography, account tenure, and any other dimension that could reveal demographic disparities. Additionally, I would audit training data label quality by segment: if certain groups have systematically fewer or lower-quality labels, that's a training data problem that no algorithm can fix.

**How would you prevent this category of failure (demographic performance disparity)?**
Three practices: (1) **slice-based eval as a launch gate** — no model launches unless slice performance gaps are within threshold (e.g., AUC gap < 0.05 across demographic groups), (2) **label quality audit** — analyze label completeness and false-negative rate by segment before training, (3) **continuous fairness monitoring** — monitor group-level metrics in production with automated alerts. Also consider: use adversarial debiasing or fairness-constrained training if a group consistently underperforms.

**What does this reveal about ML model governance?**
ML model governance requires more than aggregate performance metrics. Production models have disparate real-world impacts on different populations, and aggregate AUC can hide severe disparities. This is why regulatory frameworks (EU AI Act, US EEOC guidelines) increasingly require impact assessments and slice-based evaluation as prerequisites for deploying high-stakes ML systems. Model cards and datasheets should include known limitations by demographic group.

**How does sampling bias in training data affect model fairness?**
When training data is labeled non-uniformly (some groups have higher-quality or more complete labels), the model learns to fit the well-labeled groups better. This is distinct from class imbalance — it's label noise that's correlated with a demographic attribute. Solutions: (1) active learning to identify and label underrepresented patterns, (2) importance weighting to upweight underrepresented groups, (3) stratified sampling to ensure equal label quality across groups.
