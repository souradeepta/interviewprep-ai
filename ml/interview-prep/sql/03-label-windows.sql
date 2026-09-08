-- One row per prediction at a fixed evaluation cutoff. A label must be tied to
-- that prediction's attribution key, occur within the 7-day horizon, and be
-- available by the cutoff. Otherwise it remains unknown/not_mature.
DROP VIEW IF EXISTS label_window_results;
CREATE TEMP VIEW label_window_results AS
WITH params AS (SELECT '2026-01-10 00:00' AS evaluation_cutoff),
joined AS (
  SELECT p.prediction_id, p.user_id, p.pred_time,
         l.label_time, l.available_at, l.converted,
         CASE WHEN l.label_id IS NULL THEN 0 ELSE 1 END AS has_label
  FROM predictions p
  LEFT JOIN labels l ON l.attribution_key = p.attribution_key
    AND l.label_time > p.pred_time
    AND l.label_time <= datetime(p.pred_time, '+7 days')
)
SELECT j.prediction_id, j.user_id, j.pred_time,
       CASE
         WHEN j.has_label = 0 THEN 'unknown_not_observed'
         WHEN j.available_at > (SELECT evaluation_cutoff FROM params)
           THEN 'unknown_not_mature'
         WHEN j.converted = 1 THEN 'mature_positive'
         ELSE 'mature_negative'
       END AS label_status,
       CASE WHEN j.available_at <= (SELECT evaluation_cutoff FROM params)
            THEN j.converted ELSE NULL END AS label_7d
FROM joined j
ORDER BY j.prediction_id;

SELECT * FROM label_window_results;
