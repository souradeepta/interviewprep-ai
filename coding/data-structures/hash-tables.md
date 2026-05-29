# Hash Tables

A hash table stores key-value pairs with O(1) average insert, lookup, and delete by mapping keys to array indices via a hash function. Python's `dict`, `set`, `Counter`, `defaultdict`, and `OrderedDict` are all hash-table variants. The majority of "O(n) time, O(n) space" interview solutions use a hash table.

## Core Patterns

| Pattern | When to use | Time | Space |
|---|---|---|---|
| Frequency count (Counter) | Anagrams, top-K, majority element | O(n) | O(n) |
| Two-sum lookup | Find complement in O(1) instead of O(n) | O(n) | O(n) |
| Group by key (defaultdict) | Anagram groups, group intervals | O(n) | O(n) |
| Seen set | Visited nodes, duplicate detection, cycle | O(n) | O(n) |
| LRU cache (OrderedDict) | Evict least-recently-used in O(1) | O(1) per op | O(capacity) |
| Prefix sum map | Subarray sum equals K | O(n) | O(n) |

## Python Implementations

```python
from collections import Counter, defaultdict, OrderedDict
from typing import Optional


# ── Two Sum ────────────────────────────────────────────────────────────────

def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of two numbers that add up to target.

    Store each value's index in a dict; for every element check whether
    its complement (target - val) was already seen.
    Time: O(n)  Space: O(n)
    """
    seen: dict[int, int] = {}  # value → index
    for i, val in enumerate(nums):
        complement = target - val
        if complement in seen:
            return [seen[complement], i]
        seen[val] = i
    return []


# ── Group Anagrams ─────────────────────────────────────────────────────────

def group_anagrams(strs: list[str]) -> list[list[str]]:
    """Group strings that are anagrams of each other.

    Canonical key = sorted string.  All anagrams share the same key.
    Time: O(n * m log m) where m = avg word length.  Space: O(n*m)
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))   # canonical form
        groups[key].append(s)
    return list(groups.values())


# ── Longest Consecutive Sequence ──────────────────────────────────────────

def longest_consecutive(nums: list[int]) -> int:
    """Find length of the longest consecutive integer sequence.

    Convert to set, then start a chain only from sequence beginnings
    (n-1 not in set) to avoid O(n^2) restarts.
    Time: O(n)  Space: O(n)
    """
    num_set: set[int] = set(nums)
    best = 0
    for n in num_set:
        if n - 1 not in num_set:    # start of a new chain
            length = 1
            while n + length in num_set:
                length += 1
            best = max(best, length)
    return best


# ── Subarray Sum Equals K ─────────────────────────────────────────────────

def subarray_sum(nums: list[int], k: int) -> int:
    """Count number of contiguous subarrays with sum exactly k.

    Prefix sum trick: sum[i..j] = prefix[j] - prefix[i-1].
    If prefix[j] - k was seen before, a valid subarray ends at j.
    Time: O(n)  Space: O(n)
    """
    count = 0
    prefix = 0
    prefix_counts: dict[int, int] = defaultdict(int)
    prefix_counts[0] = 1   # empty prefix
    for num in nums:
        prefix += num
        count += prefix_counts[prefix - k]
        prefix_counts[prefix] += 1
    return count


# ── LRU Cache ─────────────────────────────────────────────────────────────

class LRUCache:
    """Least-Recently-Used cache backed by OrderedDict.

    OrderedDict preserves insertion order; move_to_end + popitem(last=False)
    gives O(1) LRU eviction.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        """Return value for key, or -1 if not present.  O(1)"""
        if key not in self._cache:
            return -1
        self._cache.move_to_end(key)   # mark as most recently used
        return self._cache[key]

    def put(self, key: int, value: int) -> None:
        """Insert or update key.  Evict LRU if over capacity.  O(1)"""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)  # evict oldest (front)
```

## Complexity Summary

| Operation | Average | Worst | Notes |
|---|---|---|---|
| Insert (dict/set) | O(1) | O(n) | Worst case: all keys collide |
| Lookup (dict/set) | O(1) | O(n) | Rare with good hash function |
| Delete (dict/set) | O(1) | O(n) | Same as lookup |
| Counter construction | O(n) | O(n) | Always linear |
| LRU get/put | O(1) | O(1) | OrderedDict + move_to_end |
| Subarray sum prefix | O(n) | O(n) | Single pass |

## Interview Recognition Template

- "Find two elements summing to X" → store complements in a dict (two-sum).
- "Count frequencies", "most common", "anagram check" → Counter.
- "Group items by shared property" → defaultdict(list).
- "Visited nodes", "detect duplicates" → set.
- "O(1) eviction by access order" → OrderedDict (LRU cache).
- "Count subarrays with sum K" → prefix sum + hash map storing prefix counts.
- Anagram canonical key: sorted(word) or tuple(Counter(word).items()).

## Worked Examples

### 1. Two Sum

**Problem:** Given an array of integers and a target, return indices of two numbers that add up to target.

```python
def twoSum(nums: list[int], target: int) -> list[int]:
    """One-pass hash map: store value→index as we scan.
    Time: O(n)  Space: O(n)
    """
    seen: dict[int, int] = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
    return []
```

### 49. Group Anagrams

**Problem:** Given an array of strings, group the anagrams together.

```python
from collections import defaultdict as _dd

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    """Sort each word to get a canonical key; group by key.
    Time: O(n * m log m)  Space: O(n * m)
    """
    groups: dict[str, list[str]] = _dd(list)
    for word in strs:
        key = ''.join(sorted(word))
        groups[key].append(word)
    return list(groups.values())
```

### 146. LRU Cache

**Problem:** Design a data structure implementing LRU cache with O(1) get and put.

```python
from collections import OrderedDict as _OD

class LRUCacheSolution:
    """OrderedDict tracks insertion/access order.
    move_to_end marks recent use; popitem(last=False) evicts oldest.
    All operations O(1).
    """

    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.cache: _OD[int, int] = _OD()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```

### 128. Longest Consecutive Sequence

**Problem:** Find the length of the longest consecutive elements sequence in O(n).

```python
def longestConsecutive(nums: list[int]) -> int:
    """Build a set; start chains only where n-1 is absent.
    Time: O(n)  Space: O(n)
    """
    num_set = set(nums)
    best = 0
    for n in num_set:
        if n - 1 not in num_set:    # chain start
            length = 1
            while n + length in num_set:
                length += 1
            best = max(best, length)
    return best
```

### 560. Subarray Sum Equals K

**Problem:** Return the number of contiguous subarrays whose sum equals k.

```python
from collections import defaultdict as _ddict

def subarraySum(nums: list[int], k: int) -> int:
    """Prefix sum + hash map.  prefix[j] - prefix[i] == k iff
    prefix[i] == prefix[j] - k was stored before index j.
    Time: O(n)  Space: O(n)
    """
    count = 0
    prefix = 0
    freq: dict[int, int] = _ddict(int)
    freq[0] = 1
    for num in nums:
        prefix += num
        count += freq[prefix - k]
        freq[prefix] += 1
    return count
```

## Related Topics

- [Arrays and Strings](arrays-strings.md) — [Tries](tries.md) — [Union-Find](union-find.md)
