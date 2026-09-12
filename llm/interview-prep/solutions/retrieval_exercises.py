"""Small deterministic retrieval exercises; no model or service calls."""

from __future__ import annotations


def select_chunk_window(chunks, *, window_size=3):
    """Return an adjacent same-document window around the best chunk."""
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    rows = list(chunks)
    if not rows:
        return []
    required = {"document_id", "chunk_id", "position", "score"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("each chunk needs document_id, chunk_id, position, and score")
    if any(not isinstance(row["score"], (int, float)) for row in rows):
        raise ValueError("scores must be numeric")
    anchor = max(enumerate(rows), key=lambda pair: (pair[1]["score"], -pair[0]))[1]
    same_doc = sorted((row for row in rows if row["document_id"] == anchor["document_id"]),
                      key=lambda row: row["position"])
    center = next(index for index, row in enumerate(same_doc) if row["chunk_id"] == anchor["chunk_id"])
    left = max(0, min(center - window_size // 2, len(same_doc) - window_size))
    return same_doc[left:left + min(window_size, len(same_doc))]


def filtered_retrieval(candidates, *, tenant_id, now, max_age, allowed_ids, k):
    """Apply tenant/ACL/freshness filters, deduplicate IDs, and return top-k."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0 or max_age < 0:
        raise ValueError("invalid retrieval limits")
    allowed_ids = set(allowed_ids)
    selected = {}
    for row in candidates:
        required = {"candidate_id", "tenant_id", "updated_at", "score"}
        if not required.issubset(row):
            raise ValueError("candidate is missing a required field")
        if row["tenant_id"] != tenant_id or row["candidate_id"] not in allowed_ids:
            continue
        if now - row["updated_at"] > max_age or row["updated_at"] > now:
            continue
        current = selected.get(row["candidate_id"])
        if current is None or row["score"] > current["score"]:
            selected[row["candidate_id"]] = row
    ranked = sorted(selected.values(), key=lambda row: (-row["score"], row["candidate_id"]))
    return ranked[:k]
