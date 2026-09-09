-- Event time is immutable historical truth; ingestion time determines whether
-- the row was available to an as-of training snapshot.
DROP VIEW IF EXISTS late_arrival_audit;
CREATE TEMP VIEW late_arrival_audit AS
WITH params AS (SELECT '2026-01-03 00:00' AS as_of_cutoff),
dedup AS (
  SELECT e.event_id, e.user_id, e.event_time,
         ROW_NUMBER() OVER (
           PARTITION BY e.event_id ORDER BY e.event_time, e.event_type
         ) AS delivery_rank
  FROM events e
)
SELECT e.event_id, e.user_id, e.event_time, i.ingested_at,
       CASE WHEN i.ingested_at <= (SELECT as_of_cutoff FROM params)
            THEN 'available_at_cutoff' ELSE 'late_arrival' END AS availability,
       CASE WHEN i.ingested_at <= (SELECT as_of_cutoff FROM params)
            THEN 0 ELSE 1 END AS is_late
FROM dedup e
JOIN event_ingestion i ON i.event_id = e.event_id
WHERE e.delivery_rank = 1
ORDER BY e.event_id;

SELECT * FROM late_arrival_audit;
