"""Late-fusion baseline with explicit missing-modality handling."""
def fuse(text=None,image=None,audio=None,weights=(.5,.3,.2)):
    values=[text,image,audio]; present=[(i,v) for i,v in enumerate(values) if v is not None]
    if not present: raise ValueError("at least one modality is required")
    total=sum(weights[i] for i,_ in present)
    if total==0: raise ValueError("present modalities have zero total weight")
    return sum(weights[i]*float(v) for i,v in present)/total
def disagreement(a,b,c):
    vals=[x for x in (a,b,c) if x is not None]
    return max(vals)-min(vals) if len(vals)>1 else 0.0
if __name__ == "__main__":
    print("all:",fuse(.8,.4,.9),"image missing:",fuse(.8,None,.9),"spread:",disagreement(.8,None,.9))
