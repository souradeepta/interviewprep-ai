"""Measure toy quantization and teacher-student approximation trade-offs."""
def quantize(values,bits):
    if bits<1 or bits>16: raise ValueError("bits must be in [1,16]")
    x=[float(v) for v in values]; lo=min(x); hi=max(x); levels=2**bits-1
    if hi==lo: return x[:]
    return [round((v-lo)/(hi-lo)*levels)/levels*(hi-lo)+lo for v in x]
def mse(a,b): return sum((x-y)**2 for x,y in zip(a,b))/max(1,len(a))
def compare(weights,student,formats=(8,4,2)):
    return [{"bits":b,"weight_bytes":len(weights)*b/8,"quantization_mse":mse(weights,quantize(weights,b)),
             "combined_mse":mse(weights,quantize(weights,b))+mse(quantize(weights,b),student)} for b in formats]
if __name__ == "__main__": print(compare([-.91,-.3,.2,.55,.98],[-.8,-.25,.14,.48,.82]))
