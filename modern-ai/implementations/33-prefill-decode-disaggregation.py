"""Deterministic FCFS prefill/decode pipeline simulation."""
def simulate(requests):
    pfree=dfree=0.0; rows=[]
    for i,(arrival,prefill,decode) in enumerate(requests):
        ps=max(arrival,pfree); pd=ps+prefill; pfree=pd
        ds=max(pd,dfree); done=ds+decode; dfree=done
        rows.append({"id":i,"arrival":arrival,"first_token":pd,"finish":done,"latency":done-arrival})
    return rows
def percentile(values,p):
    s=sorted(values)
    if not s:return 0
    return s[min(len(s)-1,max(0,int((len(s)-1)*p+0.999999)))]
if __name__ == "__main__":
    r=simulate([(0,5,1),(1,1,4),(2,2,2),(3,1,1)]); lat=[x['latency'] for x in r]
    print(r,"p50",percentile(lat,.5),"p95",percentile(lat,.95))
