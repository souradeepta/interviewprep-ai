-- Exposure-level CTR uses only assigned users and never treats unexposed users
-- or future outcomes as negatives. Each assignment has exactly one arm; the
-- exposure foreign key prevents an arm mismatch.
DROP VIEW IF EXISTS exposure_ctr;
DROP VIEW IF EXISTS sample_ratio_summary;
CREATE TEMP VIEW exposure_ctr AS
SELECT e.experiment_id, e.variant, COUNT(*) AS exposures,
       SUM(e.clicked) AS clicks,
       CAST(SUM(e.clicked) AS REAL) / COUNT(*) AS ctr
FROM exposures e
JOIN experiment_assignments a
  ON a.experiment_id = e.experiment_id AND a.user_id = e.user_id
 AND a.variant = e.variant
GROUP BY e.experiment_id, e.variant;

CREATE TEMP VIEW sample_ratio_summary AS
WITH counts AS (
  SELECT experiment_id,
         SUM(CASE WHEN variant = 'control' THEN 1 ELSE 0 END) AS control_assignments,
         SUM(CASE WHEN variant = 'treatment' THEN 1 ELSE 0 END) AS treatment_assignments
  FROM experiment_assignments GROUP BY experiment_id
)
SELECT experiment_id, control_assignments, treatment_assignments,
       CAST(treatment_assignments AS REAL) / control_assignments AS observed_ratio,
       CASE WHEN ABS(CAST(treatment_assignments AS REAL) / control_assignments - 1.0) > 0.05
            THEN 1 ELSE 0 END AS sample_ratio_mismatch
FROM counts;

SELECT * FROM exposure_ctr ORDER BY experiment_id, variant;
SELECT * FROM sample_ratio_summary ORDER BY experiment_id;
