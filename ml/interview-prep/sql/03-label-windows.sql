-- A label is mature only when it occurs after scoring and within the look-forward
-- window. Missing labels remain NULL, never an observed negative.
SELECT p.prediction_id, p.user_id,
       MAX(l.converted) AS label_7d
FROM predictions p
LEFT JOIN labels l ON l.user_id = p.user_id
  AND l.label_time > p.pred_time
  AND l.label_time <= datetime(p.pred_time, '+7 days')
GROUP BY p.prediction_id, p.user_id ORDER BY p.prediction_id;
