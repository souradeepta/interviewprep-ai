-- Start a new session only when the gap is strictly greater than 30 minutes.
-- Duplicate deliveries and rows without a user identity are excluded first.
DROP VIEW IF EXISTS sessionized_events;
CREATE TEMP VIEW sessionized_events AS
WITH dedup AS (
  SELECT event_id, user_id, item_id, event_time, event_type,
         ROW_NUMBER() OVER (
           PARTITION BY event_id
           ORDER BY event_time, event_type, item_id
         ) AS delivery_rank
  FROM events
  WHERE user_id IS NOT NULL
), valid_events AS (
  SELECT event_id, user_id, item_id, event_time, event_type,
         LAG(event_time) OVER (
           PARTITION BY user_id ORDER BY event_time, event_id
         ) AS previous_event_time
  FROM dedup
  WHERE delivery_rank = 1
), boundaries AS (
  SELECT *, CASE
    WHEN previous_event_time IS NULL
      OR event_time > datetime(previous_event_time, '+30 minutes')
    THEN 1 ELSE 0 END AS starts_session
  FROM valid_events
), numbered AS (
  SELECT *, SUM(starts_session) OVER (
    PARTITION BY user_id ORDER BY event_time, event_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS session_number
  FROM boundaries
)
SELECT event_id, user_id, event_time, event_type, session_number
FROM numbered
ORDER BY user_id, event_time, event_id;

SELECT * FROM sessionized_events;
