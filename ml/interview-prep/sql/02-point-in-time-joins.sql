-- Use the latest feature strictly before prediction time; a feature at the
-- exact prediction timestamp is not available to that prediction.
DROP VIEW IF EXISTS point_in_time_feature_results;
CREATE TEMP VIEW point_in_time_feature_results AS
SELECT p.prediction_id, p.user_id, p.pred_time,
       (SELECT f.spend_7d FROM feature_history f
        WHERE f.user_id = p.user_id AND f.feature_time < p.pred_time
        ORDER BY f.feature_time DESC LIMIT 1) AS spend_7d
FROM predictions p ORDER BY p.prediction_id;

SELECT * FROM point_in_time_feature_results;
