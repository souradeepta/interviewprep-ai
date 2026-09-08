-- Synthetic SQLite fixtures. Timestamps are ISO-8601 UTC strings.
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS feature_history;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS labels;
DROP TABLE IF EXISTS exposures;

CREATE TABLE users (user_id TEXT, country TEXT);
CREATE TABLE items (item_id TEXT, category TEXT);
CREATE TABLE events (event_id TEXT, user_id TEXT, item_id TEXT, event_time TEXT, event_type TEXT, value REAL);
CREATE TABLE feature_history (user_id TEXT, feature_time TEXT, spend_7d REAL);
CREATE TABLE predictions (prediction_id TEXT, user_id TEXT, pred_time TEXT, score REAL);
CREATE TABLE labels (user_id TEXT, label_time TEXT, converted INTEGER);
CREATE TABLE exposures (user_id TEXT, variant TEXT, item_id TEXT, exposed_at TEXT, clicked INTEGER);

INSERT INTO users VALUES ('u1','US'),('u2','CA'),('u3','US');
INSERT INTO items VALUES ('i1','book'),('i2','tool'),('i3','book');
INSERT INTO events VALUES
 ('e1','u1','i1','2026-01-01 10:00','view',1.0),
 ('e2','u1','i1','2026-01-01 10:05','click',1.0),
 ('e2','u1','i1','2026-01-01 10:05','click',1.0), -- duplicate delivery
 ('e3','u2','i2','2026-01-02 12:00','view',2.0),
 ('e4',NULL,'i3','2026-01-03 12:00','view',1.0), -- missing entity
 ('e5','u1','i2','2025-12-31 23:59','view',3.0); -- late arrival
INSERT INTO feature_history VALUES
 ('u1','2025-12-31 00:00',2.0),('u1','2026-01-01 10:00',5.0),
 ('u1','2026-01-02 00:00',9.0),('u2','2026-01-01 00:00',1.0);
INSERT INTO predictions VALUES
 ('p1','u1','2026-01-01 12:00',0.8),('p2','u2','2026-01-02 12:00',0.4),
 ('p3','u1','2026-01-03 00:00',0.7);
INSERT INTO labels VALUES
 ('u1','2026-01-04 00:00',1),('u1','2026-01-20 00:00',1),('u2','2026-01-04 00:00',0);
INSERT INTO exposures VALUES
 ('u1','control','i1','2026-01-01 09:00',1),('u2','control','i2','2026-01-01 09:00',0),
 ('u3','treatment','i1','2026-01-01 09:00',1),('u1','treatment','i2','2026-01-02 09:00',0);
