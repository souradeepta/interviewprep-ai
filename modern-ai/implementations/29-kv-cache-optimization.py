"""KV memory arithmetic and a tiny prefix-sharing cache with safe pinning/LRU."""
from collections import OrderedDict
def cache_bytes(tokens,layers,kv_heads,head_dim,bytes_per_value=2):
    if min(tokens,layers,kv_heads,head_dim,bytes_per_value)<0: raise ValueError("dimensions must be nonnegative")
    return 2*tokens*layers*kv_heads*head_dim*bytes_per_value
class BlockPool:
    def __init__(self, blocks):
        if blocks < 1: raise ValueError("pool needs at least one block")
        self.free=set(range(blocks)); self.prefix=OrderedDict(); self.live=OrderedDict(); self.pins={}
    def _evict_one(self):
        candidate=next(((key,block) for key,block in self.prefix.items() if self.pins.get(block,0)==0),None)
        if candidate is not None:
            key,block=candidate; del self.prefix[key]; self.pins.pop(block,None); self.free.add(block); return key
        raise MemoryError("no free or unpinned cached block")
    def acquire(self, request, prefix_key=None):
        if request in self.live: raise ValueError("request already owns blocks")
        if prefix_key in self.prefix:
            block=self.prefix[prefix_key]; self.prefix.move_to_end(prefix_key)
            self.live[request]=[block]; self.pins[block]=self.pins.get(block,0)+1
            return block,True
        if not self.free:
            self._evict_one()
        b=min(self.free); self.free.remove(b); self.live[request]=[b]
        self.pins[b]=1
        if prefix_key is not None: self.prefix[prefix_key]=b
        return b,False
    def release(self,request):
        blocks=self.live.pop(request)
        for block in blocks:
            self.pins[block]-=1
            if self.pins[block]==0 and block not in self.prefix.values():
                del self.pins[block]; self.free.add(block)
    def evict_prefix(self,key):
        block=self.prefix[key]
        if self.pins.get(block,0): raise ValueError("cannot evict a prefix while requests pin it")
        del self.prefix[key]; self.pins.pop(block,None); self.free.add(block)
if __name__ == "__main__":
    print("GiB for 4096 tokens:",cache_bytes(4096,32,8,128)/(1024**3))
    p=BlockPool(1); print("first and shared:",p.acquire("a","system-v1"),p.acquire("b","system-v1"))
    p.release("a"); p.release("b"); print("evicted key:",p._evict_one()); print("new prefix:",p.acquire("c","system-v2"))
