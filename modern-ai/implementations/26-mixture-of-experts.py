"""Deterministic top-k MoE routing and capacity accounting (NumPy only)."""
import numpy as np

def route(logits, k=2, capacity_factor=1.0):
    """Return accepted (token, expert, weight) routes and load diagnostics."""
    logits=np.asarray(logits,dtype=float); n,e=logits.shape
    if not 1 <= k <= e or capacity_factor <= 0: raise ValueError("invalid k or capacity")
    z=logits-logits.max(axis=1,keepdims=True); p=np.exp(z); p/=p.sum(axis=1,keepdims=True)
    cap=int(np.ceil(capacity_factor*n*k/e)); counts=np.zeros(e,dtype=int); accepted=[]; dropped=0
    for t in range(n):
        ids=np.argsort(-p[t],kind="stable")[:k]; w=p[t,ids]; w=w/w.sum()
        for expert,weight in zip(ids,w):
            if counts[expert] < cap: accepted.append((t,int(expert),float(weight))); counts[expert]+=1
            else: dropped+=1
    fractions=counts/max(1,len(accepted))
    mean_p=p.mean(axis=0)
    # Switch-style proxy: E * sum(f_i * mean_probability_i); capacity drops reported separately.
    loss=float(e*np.dot(counts/max(1,counts.sum()),mean_p))
    return {"routes":accepted,"counts":counts,"capacity_per_expert":cap,"dropped":dropped,
            "dropped_token_rate":dropped/max(1,n*k),"load_balance_loss":loss,"probabilities":p}

if __name__ == "__main__":
    x=np.array([[4,1,0],[3,2,0],[2,3,0],[1,4,0],[0,1,4],[0,2,3]],float)
    for factor in (0.5,1.0,2.0):
        r=route(x,k=2,capacity_factor=factor); print(factor,r["counts"],r["dropped_token_rate"],round(r["load_balance_loss"],3))
