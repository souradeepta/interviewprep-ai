# Post-Mortem: Silent Data Pipeline Poisoning via Schema Change

## Incident Summary
**Date:** 2023-08-01 (schema change date); detected 2023-08-05
**Duration:** 4 days of silent model degradation; 6 hours to diagnose; 8 hours to fix pipeline
**Business Impact:** Revenue -12% for 4 days; estimated $3.1M revenue loss
**Severity:** P1 (Silent production degradation; significant revenue impact; data integrity failure)

---

## Timeline

| Time | Event |
|------|-------|
| 2023-07-28 | Upstream data engineering team announces ZIP code format change from INT to VARCHAR (ZIPPLUS4) in migration notes (not sent to ML team) |
| 2023-08-01 00:00 | Schema migration applied; ZIP code column changes from INT (e.g., 10001) to VARCHAR (e.g., "10001-1234") |
| 2023-08-01 00:00 | ML feature pipeline silently fails to parse VARCHAR ZIP codes; treats all new-format ZIPs as NaN |
| 2023-08-01 00:00 | NaN imputation kicks in: replaces NaN with median ZIP (imputed value from training) |
| 2023-08-01 00:00 | ~40% of production traffic affected (those with ZIPPLUS4 format in database) |
| 2023-08-01 00:00 | No alert fires: NaN rate per feature had no monitoring; null imputation is silent |
| 2023-08-01 09:00 | Revenue tracking shows slight dip; attributed to "weekend traffic patterns" |
| 2023-08-03 | Revenue trend continues downward; business team flags to engineering |
| 2023-08-05 08:00 | On-call engineer begins investigation; checks error logs — all clean |
| 2023-08-05 10:00 | Engineer logs sample of production feature vectors; notices `zip_code` = median value for 40% of requests |
| 2023-08-05 11:30 | Upstream data team queried; schema change identified as root cause |
| 2023-08-05 14:00 | Feature pipeline patched to parse ZIPPLUS4 format |
| 2023-08-05 18:00 | ZIP code feature restored; revenue begins recovering |
| 2023-08-06 | Full recovery to pre-incident revenue levels |

---

## What Happened (Technical)

The recommendation model used `zip_code` (encoded as an integer, then transformed to region-level features) as the 4th most important feature by SHAP value. The feature captured geographic purchasing patterns — urban vs rural, regional preferences, local demographic proxies.

The upstream data engineering team migrated the ZIP code database column from `INT` (5-digit ZIP, e.g., 10001) to `VARCHAR` (ZIPPLUS4 format, e.g., "10001-1234"). This change was part of a postal address standardization project. The change was documented in the data engineering team's migration notes but was not communicated to the ML team, and no schema compatibility check was in place.

The ML feature pipeline used `int(row['zip_code'])` to parse the field. When the VARCHAR field arrived, Python's `int()` conversion raised a `ValueError` on the hyphen in "10001-1234". The pipeline's error handling caught this exception and substituted `None`. The subsequent imputation step replaced `None` with the training-data median ZIP code value (computed once during pipeline initialization).

The result: for the 40% of users whose records had been migrated to ZIPPLUS4 format, the `zip_code` feature was replaced with a constant (the training median). The model received a constant geographic signal for nearly half its traffic, causing recommendation rankings to degrade for these users. Because the imputation was silent (no log, no alert), and because null rate monitoring was not in place for any feature, the failure was completely invisible to the ML team.

---

## Root Cause Analysis

**Contributing factors:**
1. No schema contract existed between the upstream data source and the ML feature pipeline
2. The upstream team made a breaking schema change without notifying downstream consumers
3. The feature pipeline's error handling substituted NaN silently rather than raising an alert
4. No per-feature null rate monitoring existed; the monitoring system had no visibility into NaN imputation frequency
5. NaN imputation with a constant value (training median) was a silent failure mode — no error, just wrong values
6. Revenue monitoring had insufficient granularity to separate model quality degradation from macro traffic patterns

**5 Whys:**

Why did revenue fall 12% for 4 days?
The `zip_code` feature was replaced with a constant value for 40% of production traffic, degrading recommendation quality for those users.

Why was `zip_code` replaced with a constant?
The feature pipeline silently failed to parse the new VARCHAR format and substituted a median imputation value.

Why did the pipeline fail silently?
Error handling caught the `ValueError` from `int()` parsing and substituted `None` without logging an alert; NaN imputation ran without monitoring.

Why didn't the schema change trigger any notification?
No schema contract or compatibility check existed between the data engineering team's database and the ML feature pipeline.

Why was there no schema contract?
The ML feature pipeline was built independently of the upstream data team; no formal interface agreement (schema versioning, change notification process) was established when the pipeline was created.

---

## What Went Well

- Once the root cause was identified, the fix was straightforward (update ZIP parsing to handle ZIPPLUS4)
- The on-call engineer's approach (logging sample feature vectors) was effective for identifying the issue
- The upstream data team responded quickly once contacted
- No data loss occurred; the issue was entirely a parsing failure, not a data integrity problem

---

## Action Items

| Item | Owner | Due | Status |
|------|-------|-----|--------|
| Implement schema contracts with Great Expectations: validate column types, allowed formats, and null rates at pipeline ingestion | Data Engineering | +3 weeks | Done |
| Add per-feature null rate monitoring: alert if null rate for any top-20 feature exceeds 5% (was 0% pre-incident) | ML Infra | +2 weeks | Done |
| Schema change notification process: upstream data teams must notify ML team 2 weeks before any schema change affecting ML-used columns | Platform | +1 week | Done |
| Replace silent NaN imputation with noisy failure: log ERROR for every NaN substitution on features in top-10 by importance | ML Infra | +2 weeks | Done |
| Add upstream schema version check to feature pipeline startup: fail fast if schema doesn't match expected contract | Data Engineering | +3 weeks | Done |
| Revenue monitoring granularity: add per-segment revenue tracking (device, geography) to catch localized degradation earlier | Analytics | +4 weeks | In progress |

---

## Interview Discussion Points

**What would you have done differently?**
Treat feature pipeline inputs as an API with a contract. Use a schema validation library (Great Expectations, Pandera, or Pydantic) at every ingestion point. The contract specifies: column names, types, allowed value ranges, and maximum null rates. Any schema change that violates the contract causes a pipeline failure (loud, fast) rather than silent NaN substitution. This turns a 4-day silent failure into a 5-minute loud failure.

**How would you prevent data pipeline poisoning?**
Three layers: (1) **schema contracts** — validate at ingestion, fail loudly on violation; (2) **null rate monitoring** — alert immediately when any feature's null rate increases significantly; (3) **upstream change notification process** — ML team must be in the review loop for any change to data sources they depend on. Also consider: feature lineage tracking (which upstream tables each feature comes from) makes impact analysis of schema changes instantaneous.

**What monitoring gaps does this reveal?**
Two gaps: (1) **data quality monitoring** — null rates, value distributions, type conformance should be checked at every pipeline stage, not just at model output. A feature being silently nulled is as bad as a model crash, but much harder to detect. (2) **granular revenue monitoring** — if revenue monitoring had been sliced by ZIP code region or user segment, the 40% of affected users' degradation would have been visible immediately rather than averaged out in aggregate metrics.

**What is Great Expectations and how does it help?**
Great Expectations is a Python library for data quality validation. You define "expectations" (column X is type int, values between 10000-99999, null rate < 1%), and the library runs them against incoming data. Failures raise exceptions or emit to a data observability dashboard. For ML pipelines, expectations should be defined at: (1) raw data ingestion (schema check), (2) post-feature-engineering (value range check), (3) model input (pre-prediction sanity check). This provides defense in depth against data quality issues at every stage.
