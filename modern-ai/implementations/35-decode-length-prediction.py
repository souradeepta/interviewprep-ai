"""Length prediction calibration and tail-reservation diagnostics."""
def isotonic_non_decreasing(predicted, observed):
    pairs=sorted(zip(predicted,observed)); blocks=[]
    for _,y in pairs:
        blocks.append([float(y),1])
        while len(blocks)>1 and blocks[-2][0]/blocks[-2][1] > blocks[-1][0]/blocks[-1][1]:
            b=blocks.pop(); blocks[-1][0]+=b[0]; blocks[-1][1]+=b[1]
    out=[]
    for total,n in blocks: out.extend([total/n]*n)
    return out
def underprediction_rate(predicted,actual): return sum(p<a for p,a in zip(predicted,actual))/max(1,len(actual))
def tail_report(predicted,actual):
    errors=[a-p for p,a in zip(predicted,actual)]; s=sorted(actual)
    p95=s[min(len(s)-1,int(.95*len(s)))] if s else 0
    return {"underprediction_rate":underprediction_rate(predicted,actual),"mean_absolute_error":sum(abs(x) for x in errors)/max(1,len(errors)),"actual_p95":p95,"reserve_p95":p95}
if __name__ == "__main__":
    y=[8,9,10,12,13,18,20,44,51,63]; p=[7,10,9,11,15,15,18,22,27,30]
    print("calibrated",isotonic_non_decreasing(p,y)); print(tail_report(p,y))
