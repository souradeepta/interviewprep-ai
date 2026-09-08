-- Deduplicate delivery IDs before aggregating. Expected: u1=(3 events, 2 views),
-- u2=(1,1); NULL entities are excluded from a training feature table.
WITH dedup AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY event_time) AS rn
  FROM events WHERE user_id IS NOT NULL
)
SELECT user_id, COUNT(*) AS event_count,
       SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
       MAX(event_time) AS last_event
FROM dedup WHERE rn = 1
GROUP BY user_id ORDER BY user_id;
