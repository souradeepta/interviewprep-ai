"""Dependency-light population stability and labeled concept-drift diagnostics."""
from math import log

def histogram(values, edges):
    if len(edges)<2 or sorted(edges)!=list(edges): raise ValueError("edges must be increasing")
    counts=[0]*(len(edges)-1)
    for value in values:
        idx=next((i for i in range(len(edges)-1) if edges[i] <= value < edges[i+1]),None)
        if value==edges[-1]: idx=len(counts)-1
        if idx is None: raise ValueError("value outside supplied edges")
        counts[idx]+=1
    return counts

def psi(reference,current,edges,epsilon=1e-6):
    if not reference or not current: raise ValueError("both samples must be nonempty")
    a=histogram(reference,edges); b=histogram(current,edges)
    pa=[x/len(reference) for x in a]; pb=[x/len(current) for x in b]
    score=sum((y-x)*log((y+epsilon)/(x+epsilon)) for x,y in zip(pa,pb))
    return {"psi":score,"reference_proportions":pa,"current_proportions":pb}

def label_rate_shift(reference_labels,current_labels):
    if not reference_labels or not current_labels: raise ValueError("label samples must be nonempty")
    a=sum(reference_labels)/len(reference_labels); b=sum(current_labels)/len(current_labels)
    return {"reference_rate":a,"current_rate":b,"absolute_shift":abs(b-a)}

if __name__ == "__main__":
    ref=[.1,.2,.3,.4,.5,.6,.7,.8]; cur=[.1,.2,.7,.8,.85,.9,.95,.99]
    print(psi(ref,cur,[0,.25,.5,.75,1]),label_rate_shift([0,0,1,0],[1,1,1,0]))
