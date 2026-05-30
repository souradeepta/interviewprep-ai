# Post-Mortem: Vector DB Cold Start and Embedding Space Mismatch After Migration

## Incident Summary
**Date:** 2024-05-08
**Duration:** 6 hours of degraded retrieval quality; 1 hour of partial outage during index rebuild
**Business Impact:** RAG system returning incorrect answers for 6 hours; user complaints 4x normal; NPS survey scores dropped for cohort impacted; $150K estimated opportunity cost
**Severity:** P1 (User-facing quality outage; incorrect LLM answers generated from wrong retrieved context)

---

## Timeline

| Time | Event |
|------|-------|
| 2024-04-01 | Decision made to migrate RAG vector store from Pinecone to Weaviate (cost savings: $18K/month) |
| 2024-04-15 | Migration plan approved; includes embedding model "simplification" from text-embedding-3-large to text-embedding-3-small |
| 2024-04-30 | Migration runbook written; no rollback procedure documented |
| 2024-05-07 22:00 | Migration begins: Weaviate deployed, old Pinecone vectors exported |
| 2024-05-07 22:30 | Decision made to re-embed all 8.2M documents with text-embedding-3-small (different model = different vector space) |
| 2024-05-07 22:30 | Old vectors cannot be reused; re-embedding begins |
| 2024-05-08 01:00 | Re-embedding reaches 3.2M documents (39%); system switched to Weaviate despite incomplete index |
| 2024-05-08 02:00 | Traffic routed to new RAG system; retrieval quality immediately degrades |
| 2024-05-08 02:00 | User queries retrieve wrong documents; LLM generates hallucinated/wrong answers |
| 2024-05-08 03:30 | On-call engineer woken by user complaint spike; investigation begins |
| 2024-05-08 05:00 | Root cause identified: incomplete index + embedding model mismatch |
| 2024-05-08 06:00 | Re-embedding at 100%; retrieval quality recovers |
| 2024-05-08 07:00 | Traffic fully stable; incident closed |

---

## What Happened (Technical)

The team planned a cost-driven migration from Pinecone to Weaviate. The plan also included downgrading the embedding model from OpenAI `text-embedding-3-large` (3072 dimensions) to `text-embedding-3-small` (1536 dimensions) to reduce per-embedding cost.

The critical failure came from the interaction of two independent decisions: (1) switching vector database, and (2) switching embedding model. Vectors embedded with `text-embedding-3-large` exist in a 3072-dimensional space; vectors embedded with `text-embedding-3-small` exist in a 1536-dimensional space. These are **completely different vector spaces** — you cannot mix vectors from different embedding models in the same index. The existing Pinecone vectors (with the large model) could not be reused in the new system.

Re-embedding 8.2M documents with the new model required approximately 8 hours at the throughput available. At 1:00 AM, with only 39% of documents re-embedded, the team decided to cut over to the new system to meet the maintenance window deadline. For the next 5 hours, 61% of the document corpus was missing from the index. Retrieval queries either returned irrelevant documents (from the partial index) or no results, causing the LLM to generate answers from incorrect or absent context.

Additionally, no rollback procedure had been documented. When the on-call engineer confirmed the issue, the path back to Pinecone would have required waiting for the old service to be re-provisioned — a 2-hour process that was never needed because the re-embedding completed first.

The query-time quality monitoring had not been set up for the new Weaviate instance before cutover. The old Pinecone-based monitoring logged retrieval scores that were never migrated, leaving the team blind to retrieval quality for 90 minutes after cutover.

---

## Root Cause Analysis

**Contributing factors:**
1. Embedding model was changed at the same time as the database migration — two high-risk changes conflated into one maintenance window
2. No rollback procedure was documented in the migration runbook
3. Cutover proceeded at 39% index completeness; no completion gate existed
4. Query-time quality monitoring was not operational on the new system before traffic was routed to it
5. The embedding model change was treated as a cost optimization detail rather than a breaking change

**5 Whys:**

Why were RAG answers wrong for 6 hours?
The vector index was only 39% complete; 61% of the document corpus was missing from retrieval.

Why was the system cut over before the index was complete?
The team had a maintenance window deadline and decided to proceed, believing retrieval would be "mostly OK" with a partial index.

Why was the partial index cutover considered acceptable?
The migration runbook did not specify a minimum index completeness threshold; the decision was made informally under time pressure.

Why didn't the team have a rollback option?
The migration runbook did not include a rollback procedure for the vector database or embedding model change.

Why were two breaking changes (DB migration + embedding model change) bundled into one migration?
Cost pressure created urgency; the embedding model downgrade was added late to the migration plan without a separate risk assessment.

---

## What Went Well

- Re-embedding pipeline was parallelized effectively; reached 100% in 8 hours total
- On-call engineer identified root cause quickly (90 minutes from alert to diagnosis)
- No permanent data loss: the document corpus and old Pinecone index were preserved
- Incident drove immediate improvements to migration runbook standards

---

## Action Items

| Item | Owner | Due | Status |
|------|-------|-----|--------|
| Embedding model version pinning: pin embedding model in deployment config; require explicit approval to change | ML Infra | +1 week | Done |
| Minimum index completeness gate: do not route traffic until index is >= 99% complete | ML Platform | +1 week | Done |
| Separate embedding model migrations from DB migrations: any embedding model change is its own project | ML Governance | +2 weeks | Done |
| Document rollback procedure for all future infrastructure migrations | DevOps | +2 weeks | Done |
| Query-time quality monitoring must be operational and verified BEFORE traffic cutover | ML Platform | +2 weeks | Done |
| Shadow testing for embedding model changes: run new model in shadow for 2 weeks before any cutover | ML Research | +4 weeks | In progress |

---

## Interview Discussion Points

**What would you have done differently?**
Separate the two changes entirely. The database migration (Pinecone to Weaviate) is an infrastructure change that can be done transparently if you keep the same embedding model — you simply copy vectors, validate checksums, run a quality check, and cut over. The embedding model change is a semantic change that requires re-embedding everything and careful quality validation. Bundle them together and you double the risk with zero additional benefit.

**How would you prevent cold start and index completeness issues in future migrations?**
Three gates: (1) minimum index completeness threshold (99%) enforced by deployment automation, (2) retrieval quality parity check — compare top-K recall on a labeled test set between old and new system before cutover, (3) shadow mode first — route 1% of queries to new system while old system serves, compare retrieved document IDs and quality scores. Only promote when shadow quality matches production.

**What does this reveal about embedding model versioning?**
Embedding model versions must be treated like schema versions — changing them is a breaking change that invalidates all existing vectors. The embedding model version should be pinned in the same artifact manifest as the vector index, enforced at query time (reject queries if the query encoder version doesn't match the index encoder version). This is analogous to database schema migration: you need a migration plan, not just a software update.

**How do you monitor RAG retrieval quality in production?**
Three approaches: (1) **retrieval score monitoring** — log the similarity score of the top retrieved document for every query; alert if the rolling mean drops >10%, (2) **answer confidence proxy** — if the LLM produces "I don't know" or low-confidence answers at a higher rate, this often signals retrieval failure, (3) **sample-based human eval** — randomly sample 50 queries/day and have a human rate retrieval relevance; this gives a ground truth signal. All three should have been in place before the migration cutover.
