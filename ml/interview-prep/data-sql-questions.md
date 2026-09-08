# ML Data and SQL Interview Practice

Use SQLite and the synthetic [fixtures](sql/fixtures.sql). Each 15–20 minute
problem requires the query, expected rows, why a plausible alternative leaks
or miscounts, and one validation test. The four runnable query files are the
reference answers.

1. Deduplicate event deliveries and aggregate user features — [query](sql/01-feature-aggregations.sql).
2. Build a rolling count without including the current event.
3. Sessionize events using an inactivity gap.
4. Handle missing entity IDs without creating a shared training entity.
5. Join the latest feature strictly before scoring time — [query](sql/02-point-in-time-joins.sql).
6. Explain why an equality-inclusive as-of join can leak same-time updates.
7. Build a mature seven-day label window — [query](sql/03-label-windows.sql).
8. Split train/validation/test by time and entity, not random rows.
9. Handle late-arriving events and backfills without rewriting historical truth silently.
10. Compute top-K exposure and click-through rate by experiment arm — [query](sql/04-ranking-and-experiment-analysis.sql).
11. Detect sample-ratio mismatch and state its operational response.
12. Debug a metric that improves after NULL rows are accidentally counted as negatives.

All SQL is SQLite-compatible and uses only synthetic data. The four marked
query files are executable reference answers with tested result contracts; the
other eight are intentionally prompt-only drills until dedicated reference
queries and expected outputs are added. The important interview skill is
proving that features and labels were available at the prediction cutoff; a
fast query that uses future information is incorrect.
