-- Preserve missing identities as a quarantine outcome. Never COALESCE NULL
-- user IDs to a shared sentinel that could combine unrelated people.
DROP VIEW IF EXISTS entity_id_quality_results;
CREATE TEMP VIEW entity_id_quality_results AS
WITH dedup AS (
  SELECT event_id, user_id, item_id, event_time, event_type,
         ROW_NUMBER() OVER (
           PARTITION BY event_id
           ORDER BY event_time, event_type, item_id
         ) AS delivery_rank
  FROM events
)
SELECT event_id, user_id, event_time, event_type,
       CASE WHEN user_id IS NULL THEN 'quarantined_missing_entity'
            ELSE 'valid_entity' END AS entity_status
FROM dedup
WHERE delivery_rank = 1
ORDER BY event_id;

SELECT * FROM entity_id_quality_results;
