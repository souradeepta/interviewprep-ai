-- Assign an entity to exactly one split using its first prediction time.
-- Later rows for that entity inherit the assignment, preventing entity
-- leakage even when their individual timestamps cross a cutoff.
DROP VIEW IF EXISTS temporal_entity_splits;
CREATE TEMP VIEW temporal_entity_splits AS
WITH first_predictions AS (
  SELECT user_id, MIN(pred_time) AS first_pred_time
  FROM predictions
  GROUP BY user_id
), entity_splits AS (
  SELECT user_id, first_pred_time,
         CASE
           WHEN first_pred_time < '2026-01-02 00:00' THEN 'train'
           WHEN first_pred_time < '2026-01-03 00:00' THEN 'validation'
           ELSE 'test'
         END AS split
  FROM first_predictions
)
SELECT p.prediction_id, p.user_id, p.pred_time, s.first_pred_time, s.split
FROM predictions p
JOIN entity_splits s ON s.user_id = p.user_id
ORDER BY p.prediction_id;

SELECT * FROM temporal_entity_splits;
