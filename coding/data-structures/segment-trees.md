# Segment Trees

A segment tree is a binary tree built over an array where each node stores an aggregate (sum, min, max) for a contiguous sub-range. It answers range queries and supports point updates in O(log n) after an O(n) build. With lazy propagation the same structure handles range updates in O(log n). Use a segment tree when you need both mutable data and repeated range queries.

## Core Patterns

| Pattern | When to use | Time | Space |
|---|---|---|---|
| Range sum query + point update | Sum of nums[l..r] after point mutations | O(log n) query/update | O(4n) |
| Range min/max query + point update | Min or max over nums[l..r] | O(log n) query/update | O(4n) |
| Lazy propagation | Range update (add X to nums[l..r]) + range query | O(log n) per op | O(4n) |
| Coordinate compression | Sparse indices → dense segment tree | O(n log n) | O(n) |
| Order statistics | Count elements ≤ k in range | O(log n) | O(n) |

## Python Implementations

```python
from typing import Callable


class SegmentTree:
    """Array-based segment tree for range sum queries with point updates.

    Stores values in a 1-indexed array of size 4*n.
    Node i covers a range; its left child is 2*i, right child is 2*i+1.
    Build: O(n).  Query / Update: O(log n).  Space: O(4n).
    """

    def __init__(self, nums: list[int]) -> None:
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(nums, 1, 0, self.n - 1)

    def _build(
        self, nums: list[int], node: int, start: int, end: int
    ) -> None:
        """Recursively build the tree bottom-up."""
        if start == end:
            self.tree[node] = nums[start]
            return
        mid = (start + end) // 2
        self._build(nums, 2 * node, start, mid)
        self._build(nums, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx: int, val: int) -> None:
        """Set nums[idx] = val and propagate the change up.  O(log n)"""
        self._update(1, 0, self.n - 1, idx, val)

    def _update(
        self, node: int, start: int, end: int, idx: int, val: int
    ) -> None:
        if start == end:
            self.tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, l: int, r: int) -> int:
        """Return sum of nums[l..r] (inclusive, 0-indexed).  O(log n)"""
        return self._query(1, 0, self.n - 1, l, r)

    def _query(
        self, node: int, start: int, end: int, l: int, r: int
    ) -> int:
        if r < start or end < l:     # completely outside
            return 0
        if l <= start and end <= r:  # completely inside
            return self.tree[node]
        mid = (start + end) // 2
        left_sum = self._query(2 * node, start, mid, l, r)
        right_sum = self._query(2 * node + 1, mid + 1, end, l, r)
        return left_sum + right_sum


class SegmentTreeLazy:
    """Segment tree with lazy propagation for range-add + range-sum.

    Lazy array stores pending additions not yet pushed to children.
    Each update/query propagates laziness downward as needed.
    Build: O(n).  Range update / Range query: O(log n).  Space: O(4n).
    """

    def __init__(self, nums: list[int]) -> None:
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if self.n > 0:
            self._build(nums, 1, 0, self.n - 1)

    def _build(
        self, nums: list[int], node: int, start: int, end: int
    ) -> None:
        if start == end:
            self.tree[node] = nums[start]
            return
        mid = (start + end) // 2
        self._build(nums, 2 * node, start, mid)
        self._build(nums, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node: int, start: int, end: int) -> None:
        """Push pending lazy value to children."""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            left, right = 2 * node, 2 * node + 1
            # Update children's sums and accumulate their lazy
            self.tree[left] += self.lazy[node] * (mid - start + 1)
            self.tree[right] += self.lazy[node] * (end - mid)
            self.lazy[left] += self.lazy[node]
            self.lazy[right] += self.lazy[node]
            self.lazy[node] = 0

    def range_update(self, l: int, r: int, val: int) -> None:
        """Add val to all elements nums[l..r].  O(log n)"""
        self._range_update(1, 0, self.n - 1, l, r, val)

    def _range_update(
        self,
        node: int, start: int, end: int,
        l: int, r: int, val: int,
    ) -> None:
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] += val * (end - start + 1)
            self.lazy[node] += val
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_update(2 * node, start, mid, l, r, val)
        self._range_update(2 * node + 1, mid + 1, end, l, r, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, l: int, r: int) -> int:
        """Return sum of nums[l..r] after all range updates.  O(log n)"""
        return self._query(1, 0, self.n - 1, l, r)

    def _query(
        self, node: int, start: int, end: int, l: int, r: int
    ) -> int:
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (
            self._query(2 * node, start, mid, l, r)
            + self._query(2 * node + 1, mid + 1, end, l, r)
        )
```

## Complexity Summary

| Operation | Segment Tree | Prefix Sum (static) | BIT (Fenwick) | Notes |
|---|---|---|---|---|
| Build | O(n) | O(n) | O(n log n) | Segment tree builds bottom-up |
| Point update | O(log n) | O(n) rebuild | O(log n) | Prefix sum requires full rebuild |
| Range query | O(log n) | O(1) | O(log n) | Prefix sum best for static data |
| Range update (lazy) | O(log n) | N/A | O(log n) | Lazy propagation required |
| Space | O(4n) | O(n) | O(n) | 4n is safe upper bound |

## Interview Recognition Template

- Range queries + point/range updates → Segment Tree (or BIT/Fenwick for sums only).
- Static range queries (no updates) → prefix sums (O(1) query, O(n) build).
- Range updates + range queries → Segment Tree with lazy propagation.
- "Count numbers ≤ k in subarray" → coordinate-compressed segment tree.
- Prefer BIT (Fenwick tree) when only prefix sums are needed — simpler constant factor.
- Safe array size for segment tree: allocate 4*n nodes.

## Worked Examples

### 307. Range Sum Query — Mutable

**Problem:** Given an array, support point updates and range sum queries efficiently.

```python
class NumArray:
    """Segment tree for mutable range sum queries.
    update: O(log n).  sumRange: O(log n).
    """

    def __init__(self, nums: list[int]) -> None:
        self._st = SegmentTree(nums)

    def update(self, index: int, val: int) -> None:
        self._st.update(index, val)

    def sumRange(self, left: int, right: int) -> int:
        return self._st.query(left, right)
```

### 315. Count of Smaller Numbers After Self (Segment Tree concept)

**Problem:** For each element, count how many elements to its right are strictly smaller.

```python
def countSmaller(nums: list[int]) -> list[int]:
    """Coordinate-compress then use a segment tree / BIT for counting.
    Process from right to left; query count in [0, rank-1].
    Time: O(n log n)  Space: O(n)
    """
    # Coordinate compress
    sorted_unique = sorted(set(nums))
    rank: dict[int, int] = {v: i for i, v in enumerate(sorted_unique)}
    m = len(sorted_unique)

    # BIT (Fenwick tree) for point update + prefix sum query
    bit = [0] * (m + 1)

    def bit_update(i: int) -> None:
        i += 1   # 1-indexed
        while i <= m:
            bit[i] += 1
            i += i & (-i)

    def bit_query(i: int) -> int:
        i += 1   # 1-indexed
        total = 0
        while i > 0:
            total += bit[i]
            i -= i & (-i)
        return total

    result = []
    for num in reversed(nums):
        r = rank[num]
        count = bit_query(r - 1) if r > 0 else 0
        result.append(count)
        bit_update(r)
    return result[::-1]
```

### 699. Falling Squares

**Problem:** Squares drop sequentially; after each drop return the maximum height of any stack.

```python
def fallingSquares(positions: list[list[int]]) -> list[int]:
    """Coordinate-compress x-ranges; segment tree tracks max height.
    For each square, query max height in its range, then update with new height.
    Time: O(n^2) naive; segment tree brings it to O(n log n).

    Naive O(n^2) version for clarity — replace with segment tree for large n.
    """
    n = len(positions)
    heights = [0] * n  # height of each square after landing

    for i, (left_i, size_i) in enumerate(positions):
        right_i = left_i + size_i
        base = 0
        # Find max height in overlapping range
        for j in range(i):
            left_j, size_j = positions[j]
            right_j = left_j + size_j
            if left_i < right_j and left_j < right_i:  # overlap
                base = max(base, heights[j])
        heights[i] = base + size_i

    # Return running maximum after each drop
    result = []
    cur_max = 0
    for h in heights:
        cur_max = max(cur_max, h)
        result.append(cur_max)
    return result
```

## Related Topics

- [Arrays and Strings](arrays-strings.md) — [Trees](trees.md) — [Heaps](heaps.md)
