-- Compare the safe strict-before feature with the value an equality-inclusive
-- join would select. A same-time update is evidence of a point-in-time leak.
DROP VIEW IF EXISTS asof_boundary_results;
CREATE TEMP VIEW asof_boundary_results AS
SELECT p.prediction_id, p.user_id, p.pred_time,
       (
         SELECT f.spend_7d
         FROM feature_history f
         WHERE f.user_id = p.user_id AND f.feature_time < p.pred_time
         ORDER BY f.feature_time DESC LIMIT 1
       ) AS strict_prior_value,
       (
         SELECT f.spend_7d
         FROM feature_history f
         WHERE f.user_id = p.user_id AND f.feature_time = p.pred_time
         ORDER BY f.feature_time DESC LIMIT 1
       ) AS same_time_update,
       CASE WHEN EXISTS (
         SELECT 1
         FROM feature_history f
         WHERE f.user_id = p.user_id AND f.feature_time = p.pred_time
       ) THEN 1 ELSE 0 END AS equality_join_would_leak
FROM predictions p
ORDER BY p.prediction_id;

SELECT * FROM asof_boundary_results;
