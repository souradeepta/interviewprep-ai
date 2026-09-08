-- Aggregate exposure-level CTR by randomized variant. Do not treat unexposed
-- users as negatives; the denominator is exposures.
SELECT variant, COUNT(*) AS exposures, SUM(clicked) AS clicks,
       CAST(SUM(clicked) AS REAL) / COUNT(*) AS ctr
FROM exposures GROUP BY variant ORDER BY variant;
