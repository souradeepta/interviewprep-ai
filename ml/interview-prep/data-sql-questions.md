# ML Data and SQL Interview Practice

Use SQLite and the synthetic [fixtures](sql/fixtures.sql). Each 15–20 minute
problem requires the query, expected rows, why a plausible alternative leaks
or miscounts, and one validation test. The five runnable query files are the
reference answers.

1. Deduplicate event deliveries and aggregate user features — [query](sql/01-feature-aggregations.sql).
2. Build a rolling 24-hour count without including the current event — [query](sql/05-rolling-prior-count.sql).
3. Sessionize events using a 30-minute inactivity gap — [query](sql/06-sessionization.sql).
4. Handle missing entity IDs without creating a shared training entity — [query](sql/07-missing-entity-ids.sql).
5. Join the latest feature strictly before scoring time — [query](sql/02-point-in-time-joins.sql).
6. Explain why an equality-inclusive as-of join can leak same-time updates — [query](sql/08-asof-boundary-check.sql).
7. Build a mature seven-day label window — [query](sql/03-label-windows.sql).
8. Split train/validation/test by time and entity, not random rows — [query](sql/10-temporal-entity-split.sql).
9. Handle late-arriving events and backfills without rewriting historical truth silently — [query](sql/11-late-arrival-audit.sql).
10. Compute top-K exposure and click-through rate by experiment arm — [query](sql/04-ranking-and-experiment-analysis.sql).
11. Detect sample-ratio mismatch and state its operational response — [query](sql/04-ranking-and-experiment-analysis.sql).
12. Debug a metric that improves after NULL rows are accidentally counted as negatives — [query](sql/09-null-label-metric-audit.sql).

All SQL is SQLite-compatible and uses only synthetic data. The eleven marked
query files are executable reference answers with tested result contracts. The
important interview skill is
proving that features and labels were available at the prediction cutoff; a
fast query that uses future information is incorrect.
