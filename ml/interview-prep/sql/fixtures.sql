-- Synthetic SQLite fixtures. Timestamps are ISO-8601 UTC strings.
-- The data intentionally includes duplicate delivery, a missing entity, a late
-- event, a same-time feature, and a label that is not available at the cutoff.
PRAGMA foreign_keys = ON;
DROP TABLE IF EXISTS labels;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS event_ingestion;
DROP TABLE IF EXISTS exposures;
DROP TABLE IF EXISTS experiment_assignments;
DROP TABLE IF EXISTS feature_history;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS users;

CREATE TABLE users (user_id TEXT PRIMARY KEY, country TEXT NOT NULL);
CREATE TABLE items (item_id TEXT PRIMARY KEY, category TEXT NOT NULL);
CREATE TABLE events (
  event_id TEXT NOT NULL, user_id TEXT, item_id TEXT NOT NULL,
  event_time TEXT NOT NULL, event_type TEXT NOT NULL, value REAL
);
CREATE TABLE event_ingestion (
  event_id TEXT PRIMARY KEY, ingested_at TEXT NOT NULL
);
CREATE TABLE feature_history (
  user_id TEXT NOT NULL, feature_time TEXT NOT NULL, spend_7d REAL NOT NULL
);
CREATE TABLE predictions (
  prediction_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, pred_time TEXT NOT NULL,
  score REAL NOT NULL, attribution_key TEXT NOT NULL UNIQUE
);
CREATE TABLE labels (
  label_id TEXT PRIMARY KEY, attribution_key TEXT NOT NULL,
  label_time TEXT NOT NULL, available_at TEXT NOT NULL, converted INTEGER NOT NULL,
  FOREIGN KEY (attribution_key) REFERENCES predictions(attribution_key),
  UNIQUE (label_id)
);
CREATE TABLE experiment_assignments (
  experiment_id TEXT NOT NULL, user_id TEXT NOT NULL, variant TEXT NOT NULL,
  assigned_at TEXT NOT NULL, PRIMARY KEY (experiment_id, user_id)
);
CREATE TABLE exposures (
  exposure_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, user_id TEXT NOT NULL,
  variant TEXT NOT NULL, item_id TEXT NOT NULL, exposed_at TEXT NOT NULL,
  clicked INTEGER NOT NULL,
  FOREIGN KEY (experiment_id, user_id) REFERENCES experiment_assignments(experiment_id, user_id)
);

INSERT INTO users VALUES ('u1','US'),('u2','CA'),('u3','US');
INSERT INTO items VALUES ('i1','book'),('i2','tool'),('i3','book');
INSERT INTO events VALUES
 ('e1','u1','i1','2026-01-01 10:00','view',1.0),
 ('e2','u1','i1','2026-01-01 10:05','click',1.0),
 ('e2','u1','i1','2026-01-01 10:05','click',1.0), -- duplicate delivery
 ('e3','u2','i2','2026-01-02 12:00','view',2.0),
 ('e4',NULL,'i3','2026-01-03 12:00','view',1.0), -- missing entity
 ('e5','u1','i2','2025-12-31 23:59','view',3.0); -- late arrival
INSERT INTO event_ingestion VALUES
 ('e1','2026-01-01 10:01'),
 ('e2','2026-01-01 10:06'),
 ('e3','2026-01-02 12:01'),
 ('e4','2026-01-03 12:01'),
 ('e5','2026-01-03 00:05'); -- event_time precedes the as-of cutoff, ingestion does not
INSERT INTO feature_history VALUES
 ('u1','2025-12-31 00:00',2.0),('u1','2026-01-01 10:00',5.0),
 ('u1','2026-01-01 12:00',7.0),('u1','2026-01-02 00:00',9.0),
 ('u2','2026-01-01 00:00',1.0);
INSERT INTO predictions VALUES
 ('p1','u1','2026-01-01 12:00',0.8,'p1'),
 ('p2','u2','2026-01-02 12:00',0.4,'p2'),
 ('p3','u1','2026-01-03 00:00',0.7,'p3');
INSERT INTO labels VALUES
 ('l1','p1','2026-01-04 00:00','2026-01-05 00:00',1), -- mature positive
 ('l2','p2','2026-01-04 00:00','2026-01-05 00:00',0), -- mature negative
 ('l3','p3','2026-01-04 00:00','2026-01-12 00:00',1); -- future availability => unknown
INSERT INTO experiment_assignments VALUES
 ('search-v1','u1','control','2026-01-01 08:00'),
 ('search-v1','u2','control','2026-01-01 08:00'),
 ('search-v1','u3','treatment','2026-01-01 08:00');
INSERT INTO exposures VALUES
 ('x1','search-v1','u1','control','i1','2026-01-01 09:00',1),
 ('x2','search-v1','u2','control','i2','2026-01-01 09:00',0),
 ('x3','search-v1','u3','treatment','i1','2026-01-01 09:00',1),
 ('x4','search-v1','u1','control','i2','2026-01-02 09:00',0);
