"""Compare prompt truncation, retrieval, and extractive summarization."""
def allocate(items, budget, strategy, query_terms=()):
    if budget < 0: raise ValueError("budget must be nonnegative")
    if strategy == "truncate": return items[:budget]
    if strategy == "retrieve":
        ranked=sorted(enumerate(items),key=lambda pair:(-sum(t in pair[1].lower().split() for t in query_terms),pair[0]))
        return [x for _,x in ranked[:budget]]
    if strategy == "summarize":
        ranked=sorted(enumerate(items),key=lambda p:(-sum(t in p[1].lower().split() for t in query_terms),p[0]))
        return [" ".join(x.split()[:max(1,budget)]) for _,x in ranked[:budget]]
    raise ValueError("strategy must be truncate, retrieve, or summarize")
if __name__ == "__main__":
    docs=["policy renews annually", "invoice due in thirty days", "support weekdays", "annual renewal can be cancelled"]
    for s in ("truncate","retrieve","summarize"): print(s,allocate(docs,2,s,("annual","renewal")))
