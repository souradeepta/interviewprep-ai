-- Unknown/not-yet-mature labels must not be counted as negative outcomes.
-- The naive rate is shown only as a debugging counterexample.
DROP VIEW IF EXISTS null_label_metric_audit;
CREATE TEMP VIEW null_label_metric_audit AS
WITH params AS (SELECT '2026-01-10 00:00' AS evaluation_cutoff),
windowed AS (
  SELECT p.prediction_id,
         CASE
           WHEN l.label_id IS NULL THEN 'unknown_not_observed'
           WHEN l.label_time <= datetime(p.pred_time, '+7 days')
            AND l.available_at <= (SELECT evaluation_cutoff FROM params)
            AND l.converted = 1 THEN 'mature_positive'
           WHEN l.label_time <= datetime(p.pred_time, '+7 days')
            AND l.available_at <= (SELECT evaluation_cutoff FROM params)
            AND l.converted = 0 THEN 'mature_negative'
           ELSE 'unknown_not_mature'
         END AS label_status
  FROM predictions p
  LEFT JOIN labels l ON l.attribution_key = p.attribution_key
    AND l.label_time > p.pred_time
    AND l.label_time <= datetime(p.pred_time, '+7 days')
), counts AS (
  SELECT COUNT(*) AS predictions,
         SUM(label_status = 'mature_positive') AS mature_positive,
         SUM(label_status = 'mature_negative') AS mature_negative,
         SUM(label_status LIKE 'unknown%') AS unknown_labels
  FROM windowed
)
SELECT predictions, mature_positive, mature_negative, unknown_labels,
       mature_negative + unknown_labels AS naive_negative_count,
       CAST(mature_negative AS REAL) /
         NULLIF(mature_positive + mature_negative, 0) AS mature_negative_rate,
       CAST(mature_negative + unknown_labels AS REAL) /
         NULLIF(predictions, 0) AS naive_negative_rate
FROM counts;

SELECT * FROM null_label_metric_audit;
