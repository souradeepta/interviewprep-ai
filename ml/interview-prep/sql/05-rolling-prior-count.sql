-- Count deduplicated events for the same user in the 24 hours before each
-- event. The lower bound is inclusive and the current event is excluded.
-- NULL user IDs are quarantined rather than merged into a synthetic entity.
DROP VIEW IF EXISTS rolling_prior_event_counts;
CREATE TEMP VIEW rolling_prior_event_counts AS
WITH dedup AS (
  SELECT event_id, user_id, item_id, event_time, event_type,
         ROW_NUMBER() OVER (
           PARTITION BY event_id
           ORDER BY event_time, event_type, item_id
         ) AS delivery_rank
  FROM events
  WHERE user_id IS NOT NULL
), valid_events AS (
  SELECT event_id, user_id, item_id, event_time, event_type
  FROM dedup
  WHERE delivery_rank = 1
)
SELECT current.event_id, current.user_id, current.event_time,
       (
         SELECT COUNT(*)
         FROM valid_events prior
         WHERE prior.user_id = current.user_id
           AND prior.event_time >= datetime(current.event_time, '-24 hours')
           AND prior.event_time < current.event_time
       ) AS prior_events_24h
FROM valid_events current
ORDER BY current.user_id, current.event_time, current.event_id;

SELECT * FROM rolling_prior_event_counts;
