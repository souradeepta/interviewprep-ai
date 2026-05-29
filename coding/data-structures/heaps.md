# Heaps

A heap is a complete binary tree stored as an array where the parent is always smaller (min-heap) or larger (max-heap) than its children. Python's `heapq` module implements a min-heap. Heaps are the go-to structure for top-K problems, running medians, and k-way merges. Push and pop are O(log n); finding the min/max is O(1).

## Core Patterns

| Pattern | When to use | Time | Space |
|---|---|---|---|
| Top-K largest | k largest elements from n items | O(n log k) | O(k) |
| Top-K smallest | k smallest elements from n items | O(n log k) | O(k) |
| K-way merge | Merge k sorted arrays/lists | O(N log k) | O(k) |
| Running median | Median of a growing stream of numbers | O(log n) per insert | O(n) |
| Priority queue | Process items in order of priority | O(log n) push/pop | O(n) |
| Kth smallest in matrix | Sorted matrix, find kth element | O(k log k) | O(k) |

## Python Implementations

```python
import heapq
from typing import Optional


# ── Basic heap operations ──────────────────────────────────────────────────

def kth_largest(nums: list[int], k: int) -> int:
    """Find the k-th largest element.

    Maintain a min-heap of size k.  After processing all elements
    the top of the heap is the k-th largest.
    Time: O(n log k)  Space: O(k)
    """
    min_heap: list[int] = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)   # evict smallest
    return min_heap[0]


def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """Return the k most frequent elements.

    Count frequencies, then use a min-heap of size k to track top k.
    Time: O(n log k)  Space: O(n)
    """
    from collections import Counter
    freq = Counter(nums)
    # heap stores (frequency, element); min-heap evicts least frequent
    heap: list[tuple[int, int]] = []
    for elem, cnt in freq.items():
        heapq.heappush(heap, (cnt, elem))
        if len(heap) > k:
            heapq.heappop(heap)
    return [elem for _, elem in heap]


# ── ListNode for k-way merge ───────────────────────────────────────────────

class ListNode:
    """Singly linked list node (reused for merge problem)."""

    def __init__(self, val: int = 0, next: Optional[ListNode] = None):
        self.val = val
        self.next = next

    # Required so heapq can compare ListNode objects
    def __lt__(self, other: ListNode) -> bool:
        return self.val < other.val


def merge_k_sorted_lists(
    lists: list[Optional[ListNode]],
) -> Optional[ListNode]:
    """Merge k sorted linked lists into one sorted list.

    Push the head of each list into a min-heap.  Pop the minimum,
    advance its list, and push the next node.
    Time: O(N log k) where N = total nodes, k = number of lists.
    Space: O(k) heap.
    """
    dummy = ListNode(0)
    cur = dummy
    heap: list[ListNode] = []
    for node in lists:
        if node:
            heapq.heappush(heap, node)
    while heap:
        smallest = heapq.heappop(heap)
        cur.next = smallest
        cur = cur.next        # type: ignore[assignment]
        if smallest.next:
            heapq.heappush(heap, smallest.next)
    return dummy.next


# ── Two-heap running median ────────────────────────────────────────────────

class MedianFinder:
    """Find the median of a data stream in O(log n) per insert.

    Uses two heaps:
    - max_heap (lower half): negate values to simulate max-heap with heapq.
    - min_heap (upper half): standard min-heap.
    Invariant: len(max_heap) == len(min_heap) or len(max_heap) == len(min_heap)+1.
    """

    def __init__(self) -> None:
        self.max_heap: list[int] = []   # negated values (lower half)
        self.min_heap: list[int] = []   # natural values (upper half)

    def addNum(self, num: int) -> None:
        """Insert a number.  Time: O(log n)"""
        # Push to max_heap first (lower half)
        heapq.heappush(self.max_heap, -num)
        # Balance: largest of lower half must be <= smallest of upper half
        if (
            self.min_heap
            and (-self.max_heap[0]) > self.min_heap[0]
        ):
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        # Size balance: max_heap may have at most 1 extra element
        if len(self.max_heap) > len(self.min_heap) + 1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        elif len(self.min_heap) > len(self.max_heap):
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)

    def findMedian(self) -> float:
        """Return current median.  Time: O(1)"""
        if len(self.max_heap) > len(self.min_heap):
            return float(-self.max_heap[0])
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0
```

## Complexity Summary

| Operation | Time | Space | Notes |
|---|---|---|---|
| heappush | O(log n) | — | Sift up |
| heappop | O(log n) | — | Sift down |
| heapq.heapify | O(n) | — | Build from list in-place |
| Peek min | O(1) | — | heap[0] |
| Top-K from n | O(n log k) | O(k) | Maintain size-k heap |
| K-way merge (N total) | O(N log k) | O(k) | Pop + push pattern |
| Running median insert | O(log n) | O(n) | Two-heap structure |

## Interview Recognition Template

- "Top K", "K largest", "K most frequent" → min-heap of size K (heapq).
- "Running median", "find median from stream" → two heaps (max left, min right).
- "Merge K sorted" arrays or lists → heap with (value, list_index, element_index).
- "Smallest sum", "cheapest path (uniform cost)" → priority queue (heapq).
- Python min-heap: negate values for max-heap behavior.
- heapq.nlargest(k, iterable) / nsmallest(k, iterable) are convenient but O(n log k).

## Worked Examples

### 215. Kth Largest Element in an Array

**Problem:** Find the k-th largest element in an unsorted array.

```python
def findKthLargest(nums: list[int], k: int) -> int:
    """Min-heap of size k; top is k-th largest.
    Time: O(n log k)  Space: O(k)
    """
    heap: list[int] = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]
```

### 347. Top K Frequent Elements

**Problem:** Return the k most frequent elements from an integer array.

```python
from collections import Counter as _Counter

def topKFrequent(nums: list[int], k: int) -> list[int]:
    """Frequency count + min-heap of size k.
    Time: O(n log k)  Space: O(n)
    """
    freq = _Counter(nums)
    heap: list[tuple[int, int]] = []
    for elem, cnt in freq.items():
        heapq.heappush(heap, (cnt, elem))
        if len(heap) > k:
            heapq.heappop(heap)
    return [elem for _, elem in heap]
```

### 23. Merge K Sorted Lists

**Problem:** Merge k sorted linked lists and return it as one sorted list.

```python
def mergeKLists(lists: list[Optional[ListNode]]) -> Optional[ListNode]:
    """Min-heap on (value, index, node) tuples to avoid direct comparison.
    Time: O(N log k)  Space: O(k)
    """
    dummy = ListNode(0)
    cur = dummy
    heap: list[tuple[int, int, ListNode]] = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    while heap:
        val, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next              # type: ignore[assignment]
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```

### 295. Find Median from Data Stream

**Problem:** Design a data structure supporting addNum and findMedian.

```python
class MedianFinderSolution:
    """Two-heap approach: max_heap (lower half) + min_heap (upper half).
    addNum: O(log n).  findMedian: O(1).
    """

    def __init__(self) -> None:
        self.lo: list[int] = []   # max-heap via negation
        self.hi: list[int] = []   # min-heap

    def addNum(self, num: int) -> None:
        heapq.heappush(self.lo, -num)          # push to lower half
        # Ensure all of lo <= all of hi
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        # Rebalance: lo may have at most 1 more element
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self) -> float:
        if len(self.lo) > len(self.hi):
            return float(-self.lo[0])
        return (-self.lo[0] + self.hi[0]) / 2.0
```

## Related Topics

- [Trees](trees.md) — [Graphs](graphs.md) — [Arrays and Strings](arrays-strings.md)
