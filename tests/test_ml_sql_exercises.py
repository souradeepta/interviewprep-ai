"""Result and leakage contracts for the SQLite interview drills."""

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "ml/interview-prep/sql"


def connection():
    db = sqlite3.connect(":memory:")
    db.executescript((SQL / "fixtures.sql").read_text())
    return db


def test_fixture_assignment_and_attribution_invariants():
    db = connection()
    assert db.execute("SELECT COUNT(*) FROM predictions GROUP BY attribution_key HAVING COUNT(*) > 1").fetchall() == []
    assert db.execute("""
        SELECT experiment_id, user_id, COUNT(*) FROM experiment_assignments
        GROUP BY experiment_id, user_id HAVING COUNT(*) != 1
    """).fetchall() == []
    assert db.execute("""
        SELECT e.exposure_id FROM exposures e
        JOIN experiment_assignments a USING (experiment_id, user_id)
        WHERE e.variant != a.variant
    """).fetchall() == []
    assert db.execute("""
        SELECT l.label_id FROM labels l
        LEFT JOIN predictions p ON p.attribution_key = l.attribution_key
        WHERE p.prediction_id IS NULL
    """).fetchall() == []


def test_feature_aggregation_deduplicates_deliveries_and_excludes_missing_entities():
    db = connection()
    db.executescript((SQL / "01-feature-aggregations.sql").read_text())
    rows = db.execute(
        "SELECT user_id, event_count, view_count, last_event "
        "FROM feature_aggregation_results ORDER BY user_id"
    ).fetchall()
    assert rows == [
        ("u1", 3, 2, "2026-01-01 10:05"),
        ("u2", 1, 1, "2026-01-02 12:00"),
    ]


def test_point_in_time_join_excludes_same_timestamp_feature():
    db = connection()
    db.executescript((SQL / "02-point-in-time-joins.sql").read_text())
    rows = db.execute(
        "SELECT prediction_id, spend_7d "
        "FROM point_in_time_feature_results ORDER BY prediction_id"
    ).fetchall()
    assert rows == [("p1", 5.0), ("p2", 1.0), ("p3", 9.0)]

    same_time_value = db.execute(
        "SELECT spend_7d FROM feature_history "
        "WHERE user_id = 'u1' AND feature_time = '2026-01-01 12:00'"
    ).fetchone()[0]
    assert same_time_value == 7.0
    assert rows[0][1] != same_time_value


def test_rolling_count_uses_only_prior_deduplicated_events_in_window():
    db = connection()
    db.executescript((SQL / "05-rolling-prior-count.sql").read_text())
    rows = db.execute(
        "SELECT event_id, user_id, prior_events_24h "
        "FROM rolling_prior_event_counts "
        "ORDER BY user_id, event_time, event_id"
    ).fetchall()
    assert rows == [
        ("e5", "u1", 0),
        ("e1", "u1", 1),
        ("e2", "u1", 2),
        ("e3", "u2", 0),
    ]

    assert db.execute(
        "SELECT COUNT(*) FROM rolling_prior_event_counts WHERE event_id = 'e2'"
    ).fetchone()[0] == 1
    assert db.execute(
        "SELECT COUNT(*) FROM rolling_prior_event_counts WHERE user_id IS NULL"
    ).fetchone()[0] == 0


def test_label_window_respects_attribution_and_availability_cutoff():
    db = connection()
    db.executescript((SQL / "03-label-windows.sql").read_text())
    rows = db.execute(
        "SELECT prediction_id, label_status, label_7d FROM label_window_results"
    ).fetchall()
    assert rows == [
        ("p1", "mature_positive", 1),
        ("p2", "mature_negative", 0),
        ("p3", "unknown_not_mature", None),
    ]
    db.execute("UPDATE labels SET available_at = '2026-01-20 00:00' WHERE label_id = 'l1'")
    assert db.execute("SELECT label_status FROM label_window_results WHERE prediction_id = 'p1'").fetchone()[0] == "unknown_not_mature"


def test_exposure_ctr_and_sample_ratio_mismatch_use_valid_assignments():
    db = connection()
    db.executescript((SQL / "04-ranking-and-experiment-analysis.sql").read_text())
    assert db.execute("SELECT variant, exposures, clicks FROM exposure_ctr ORDER BY variant").fetchall() == [
        ("control", 3, 1), ("treatment", 1, 1)
    ]
    assert db.execute("SELECT observed_ratio, sample_ratio_mismatch FROM sample_ratio_summary").fetchone() == (0.5, 1)


def test_each_reference_query_starts_from_a_fresh_database():
    for path in sorted(SQL.glob("[0-9][0-9]-*.sql")):
        db = connection()
        db.executescript(path.read_text())
        db.close()
