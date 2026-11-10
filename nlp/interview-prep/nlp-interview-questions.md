# NLP Interview Questions

Prompts cover classical NLP through retrieval and LLM boundaries. Scenario questions ask for evaluation and data decisions as well as model knowledge.

### 1. Choose a tokenization strategy
**Prompt:** A multilingual support corpus includes emoji, code-switched text, and rare product names. What tokenization risks do you inspect?
**Role/domain:** NLP; preprocessing. **Difficulty:** Medium.
**Expected answer:** Inspect fragmentation and unknown behavior by language/domain, normalization effects, vocabulary coverage, and token-count cost. Subword tokenization handles open vocabulary but may split names or scripts unevenly; preserve identifiers where task semantics require it and evaluate downstream behavior.
**Evaluation:** Coverage, normalization, language-specific analysis.
**Follow-up:** How might Unicode normalization change a product identifier?
**Common mistake:** Assuming a tokenizer’s nominal vocabulary coverage means equal quality across languages.

### 2. Explain embedding similarity limits
**Prompt:** Two words have high cosine similarity in an embedding space. What does that establish, and what does it not?
**Role/domain:** NLP; representation learning. **Difficulty:** Easy.
**Expected answer:** Their vectors point in similar directions under that embedding and training representation. It does not prove synonymy, identical context behavior, fairness, or task relevance; anisotropy, polysemy, and corpus/domain mismatch can distort similarity.
**Evaluation:** Geometric meaning, semantic limits, task validation.
**Follow-up:** How would contextual embeddings change the comparison?
**Common mistake:** Treating cosine similarity as a calibrated semantic probability.

### 3. Diagnose a retrieval recall drop
**Prompt:** Search recall falls after a corpus refresh even though document count increases. How do you debug it?
**Role/domain:** NLP retrieval; debugging. **Difficulty:** Hard.
**Expected answer:** Compare query/corpus versions, parsing and chunking, duplicate rates, embedding/index versions, filters, and relevance labels. Inspect misses by query type and language, evaluate recall@k against a fixed judged set, and isolate ingestion from ranking before changing the model.
**Evaluation:** Pipeline decomposition, fixed evaluation, slice diagnosis.
**Follow-up:** How could changed chunk boundaries lower recall without changing embeddings?
**Common mistake:** Increasing top-k without checking malformed or filtered-out documents.

### 4. Compare BM25 and dense retrieval
**Prompt:** When is BM25 likely to beat dense retrieval, and how would you evaluate a hybrid?
**Role/domain:** Information retrieval. **Difficulty:** Medium.
**Expected answer:** BM25 often excels at exact identifiers, rare terms, and lexical matching; dense retrieval can help paraphrases and semantic variation. Evaluate recall/ranking on judged queries, tune normalization and fusion on validation only, and inspect complementary error sets.
**Evaluation:** Query fit, fair comparison, fusion method.
**Follow-up:** Why can reciprocal-rank fusion help when score scales differ?
**Common mistake:** Comparing raw BM25 and cosine scores as if they shared a scale.

### 5. Debug a ranking regression
**Prompt:** A reranker improves NDCG overall but harms exact product-code queries. What do you do?
**Role/domain:** Search ranking; debugging. **Difficulty:** Hard.
**Expected answer:** Slice evaluation by query intent, exact-match behavior, language, and candidate recall; inspect training labels and feature dominance. Preserve lexical candidates or exact-match constraints as needed, tune on validation data, and report the trade-off rather than hiding the regression in aggregate NDCG.
**Evaluation:** Slice metrics, candidate-stage reasoning, decision transparency.
**Follow-up:** How do you avoid overfitting the product-code slice?
**Common mistake:** Shipping based only on aggregate ranking improvement.

### 6. Handle class imbalance in text classification
**Prompt:** A moderation classifier has 1% positive labels. Which metrics and split choices matter?
**Role/domain:** NLP classification. **Difficulty:** Medium.
**Expected answer:** Use user/document/time-aware splits to prevent duplicates crossing partitions. Report PR-AUC, precision/recall at the review budget, calibration and per-category performance; inspect label policy and threshold costs. Accuracy alone is uninformative at this prevalence.
**Evaluation:** Split leakage, prevalence-aware metrics, operating point.
**Follow-up:** What does a change in prevalence do to precision?
**Common mistake:** Optimizing ROC-AUC and ignoring the actual workload threshold.

### 7. Evaluate sequence labeling
**Prompt:** A named-entity tagger has high token accuracy but misses many complete organizations. What metric should supplement it?
**Role/domain:** NLP; sequence labeling. **Difficulty:** Medium.
**Expected answer:** Report exact span-level precision, recall, and F1 by entity type alongside token metrics; define treatment of partial overlaps and invalid BIO transitions. Review boundary and rare-entity errors because token accuracy is dominated by outside labels.
**Evaluation:** Span semantics, label scheme, class-specific behavior.
**Follow-up:** How do BIO constraints affect decoding?
**Common mistake:** Treating token-level accuracy as entity extraction quality.

### 8. Debug a multilingual quality gap
**Prompt:** A model’s aggregate F1 is stable, but a low-resource language has sharply degraded. How do you investigate and report it?
**Role/domain:** Multilingual NLP; fairness. **Difficulty:** Hard.
**Expected answer:** Verify sample size, annotation consistency, script/normalization, code-switching, and train/test representation by language. Report uncertainty and per-language metrics, collect representative labels, and evaluate adaptation without silently pooling away the gap.
**Evaluation:** Data diagnostics, uncertainty, transparent slice reporting.
**Follow-up:** When might translation-based evaluation mislead?
**Common mistake:** Assuming stable aggregate metrics imply stable language-level quality.

### 9. Choose an evaluation split for conversations
**Prompt:** A dataset contains many turns from the same conversation and user. How do you avoid leakage?
**Role/domain:** NLP evaluation; experiment design. **Difficulty:** Medium.
**Expected answer:** Split by conversation or user according to the generalization target, with temporal holdout if future deployment matters. Deduplicate near-identical text and freeze preprocessing; random turn-level splitting lets context and user-specific patterns leak across partitions.
**Evaluation:** Unit of generalization, temporal considerations, duplication.
**Follow-up:** What if evaluation specifically targets later turns from known users?
**Common mistake:** Randomly splitting utterances as independent examples.

### 10. Design a retrieval evaluation set
**Prompt:** Search logs have clicks but few explicit judgments. How would you build a useful retrieval test set?
**Role/domain:** Search/NLP; system design. **Difficulty:** Hard.
**Expected answer:** Sample queries across intents and slices, gather blinded relevance judgments with guidelines, pool candidates from multiple systems to reduce system-specific bias, and version labels. Treat clicks as biased implicit signals; use held-out queries and report uncertainty and judgment coverage.
**Evaluation:** Sampling, judgment design, bias awareness.
**Follow-up:** How do you handle disagreement between assessors?
**Common mistake:** Calling clicked results ground-truth relevant and unclicked results irrelevant.

### 11. Diagnose a tokenizer-induced cost spike
**Prompt:** API spend rises after a model switch even though character counts are unchanged. What do you inspect?
**Role/domain:** NLP/LLM operations; debugging. **Difficulty:** Medium.
**Expected answer:** Measure token counts by language, format, and request type using each tokenizer; inspect normalization and special-token behavior, truncation, and prompt templates. Compare quality and latency at matched token budgets, then adjust chunking or model selection if justified.
**Evaluation:** Token-versus-character distinction, slices, quality/cost balance.
**Follow-up:** Why might truncating by characters be unsafe?
**Common mistake:** Assuming equal text length implies equal token cost.

### 12. Decide whether to use an LLM
**Prompt:** A team proposes an LLM for extracting invoice fields. What evidence would justify it over rules or a smaller model?
**Role/domain:** Applied NLP; model selection. **Difficulty:** Hard.
**Expected answer:** Define schema, error costs, volume, privacy, latency, and labeled evaluation first. Compare deterministic parsing and smaller models on representative difficult cases; use an LLM only if it materially improves required quality and can be constrained, validated, and escalated when uncertain.
**Evaluation:** Boundary decision, baseline, operational requirements.
**Follow-up:** How should unsupported or malformed output be handled?
**Common mistake:** Selecting an LLM because the input is text rather than because it meets measurable requirements.

### 13. Protect retrieval-augmented generation from injection
**Prompt:** Retrieved content includes instructions telling the model to reveal secrets. What controls do you add?
**Role/domain:** NLP/LLM security. **Difficulty:** Hard.
**Expected answer:** Treat retrieved text as untrusted data, separate it from system instructions, restrict tools and sensitive context, validate outputs, and test adversarial retrieved passages. Use least privilege and provenance; prompting alone is not a complete security boundary.
**Evaluation:** Trust boundary, defense layers, test strategy.
**Follow-up:** What should happen if evidence conflicts with trusted policy?
**Common mistake:** Assuming retrieval from an internal index is inherently trusted.

### 14. Evaluate answer faithfulness
**Prompt:** A RAG assistant gives correct answers on a small demo. How do you test whether it is grounded in evidence?
**Role/domain:** NLP/LLM evaluation. **Difficulty:** Hard.
**Expected answer:** Build answerable, unanswerable, conflicting, and stale-evidence cases with source references. Score retrieval recall separately from claim support, require citations to entail claims, and measure abstention and unsupported assertions with human review for ambiguous examples.
**Evaluation:** Separate pipeline stages, adversarial cases, evidence criteria.
**Follow-up:** Why is citation presence insufficient?
**Common mistake:** Counting any cited passage as proof that the answer follows from it.

### 15. Plan a domain adaptation evaluation
**Prompt:** A sentiment model is moving from product reviews to short social posts. How do you determine whether adaptation helped?
**Role/domain:** NLP transfer; evaluation. **Difficulty:** Hard.
**Expected answer:** Create representative labeled target-domain splits, define a source baseline and target slices, inspect label-policy differences and class balance, then compare adaptation methods with uncertainty. Preserve a final untouched test set and check calibration, subgroup quality, and temporal robustness.
**Evaluation:** Domain-specific labels, fair baseline, robust holdout.
**Follow-up:** What if source and target label definitions differ?
**Common mistake:** Treating unlabeled vocabulary shift as sufficient proof that fine-tuning is beneficial.
