"""Construct interpretable boolean attention masks and report edge connectivity."""
def mask(n, pattern, window=2, global_indices=()):
    if n<0 or window<0: raise ValueError("n and window must be nonnegative")
    out=[]
    for q in range(n):
        row=[]
        for k in range(n):
            if pattern=="dense": yes=True
            elif pattern=="causal": yes=k<=q
            elif pattern=="sliding": yes=q-window<=k<=q
            elif pattern=="global_local": yes=(k in global_indices or q in global_indices or q-window<=k<=q)
            else: raise ValueError("unknown pattern")
            row.append(bool(yes))
        out.append(row)
    return out
def density(m): return sum(map(sum,m))/max(1,len(m)**2)
if __name__ == "__main__":
    for p in ("dense","causal","sliding","global_local"):
        m=mask(8,p,2,(0,)); print(p,density(m),sum(map(sum,m)))
