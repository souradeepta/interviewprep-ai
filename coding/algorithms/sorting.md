# Sorting

## TL;DR
Sort algorithms split into comparison-based (QuickSort, MergeSort, HeapSort — all O(n log n)) and non-comparison (CountingSort, RadixSort — O(n) given bounded integers). Choice depends on stability, in-place requirement, and input characteristics. Python's built-in `sorted()` uses TimSort, which is stable and O(n log n) worst-case.

## Core Concepts

**Comparison vs Non-Comparison:**
- Comparison sort: must compare elements; theoretical lower bound O(n log n)
- Non-comparison sort: exploit structure of keys (integers, strings); can beat O(n log n)

**Stability:** stable sort preserves relative order of equal elements (important for multi-key sorts)
- Stable: MergeSort, TimSort, CountingSort
- Unstable: QuickSort, HeapSort

**In-place vs Extra Space:**
- In-place (O(1) extra): QuickSort, HeapSort
- Extra space (O(n)): MergeSort, CountingSort

**Complexity Table:**

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| QuickSort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| MergeSort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| HeapSort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| TimSort | O(n) | O(n log n) | O(n log n) | O(n) | Yes |
| CountingSort | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes |
| BubbleSort | O(n) | O(n²) | O(n²) | O(1) | Yes |

**Patterns:**
- **Need stable sort** → MergeSort or TimSort (Python built-in)
- **Need in-place** → QuickSort (avg) or HeapSort (guaranteed)
- **Small integers bounded range** → CountingSort or RadixSort
- **Nearly sorted input** → TimSort excels (O(n))

## Implementations

```python
from typing import List
import random

# --- QuickSort ---
def quicksort(arr: List[int]) -> List[int]:
    """In-place QuickSort with random pivot to avoid O(n^2) worst case."""
    def _qsort(lo: int, hi: int) -> None:
        if lo >= hi:
            return
        pivot_idx = partition(lo, hi)
        _qsort(lo, pivot_idx - 1)
        _qsort(pivot_idx + 1, hi)

    def partition(lo: int, hi: int) -> int:
        # Random pivot to avoid worst-case on sorted input
        rand_idx = random.randint(lo, hi)
        arr[rand_idx], arr[hi] = arr[hi], arr[rand_idx]
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        return i + 1

    _qsort(0, len(arr) - 1)
    return arr


# --- MergeSort ---
def mergesort(arr: List[int]) -> List[int]:
    """Stable MergeSort. O(n log n) time, O(n) space."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:   # <= keeps stability
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# --- HeapSort ---
def heapsort(arr: List[int]) -> List[int]:
    """In-place HeapSort. O(n log n) always, O(1) space. Not stable."""
    n = len(arr)

    def heapify(n: int, root: int) -> None:
        largest = root
        left, right = 2 * root + 1, 2 * root + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest != root:
            arr[root], arr[largest] = arr[largest], arr[root]
            heapify(n, largest)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(i, 0)
    return arr


# --- CountingSort (for bounded non-negative integers) ---
def counting_sort(arr: List[int], max_val: int) -> List[int]:
    """O(n + k) time and space. Only for integers in [0, max_val]."""
    count = [0] * (max_val + 1)
    for x in arr:
        count[x] += 1
    # Prefix sum → positions
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    output = [0] * len(arr)
    for x in reversed(arr):   # reversed for stability
        count[x] -= 1
        output[count[x]] = x
    return output


# --- Custom comparator (Python idiom) ---
# Sort by multiple keys: primary descending score, secondary ascending name
records = [("Alice", 90), ("Bob", 85), ("Carol", 90)]
records_sorted = sorted(records, key=lambda x: (-x[1], x[0]))
# → [('Alice', 90), ('Carol', 90), ('Bob', 85)]

# --- Dutch National Flag (3-way partition) ---
def dutch_national_flag(nums: List[int]) -> None:
    """Sort array of 0s, 1s, 2s in-place. LeetCode 75."""
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]
            lo += 1; mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1
```

## Key Problems

| Problem | Approach | Time | Space |
|---|---|---|---|
| Sort by multiple keys | `sorted(arr, key=lambda x: (-x[1], x[0]))` | O(n log n) | O(n) |
| Dutch National Flag | 3-way partition (lo/mid/hi pointers) | O(n) | O(1) |
| Sort matrix row by row | `[sorted(row) for row in matrix]` | O(m * n log n) | O(1) |
| Sort linked list | MergeSort (O(1) extra space for linked list merge) | O(n log n) | O(log n) |
| K closest to origin | Partial sort via `heapq.nsmallest` | O(n log k) | O(k) |

## Common Mistakes / Gotchas
- **QuickSort worst case:** sorted or reverse-sorted input → always randomize pivot
- **Stability matters for multi-key:** sorting by column A then column B works only if second sort is stable
- **CountingSort range:** must know max value in advance; negative integers need offset
- **Custom key vs cmp_to_key:** Python `sorted` uses `key=` (one-arg); for pairwise comparisons use `functools.cmp_to_key`
- **Shallow copy:** `sorted()` returns a new list; `list.sort()` is in-place — don't confuse them

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "QuickSort vs MergeSort?" | Quick: in-place, O(n²) worst, cache-friendly. Merge: stable, O(n log n) guaranteed, O(n) space. Use merge for linked lists or stability requirement. |
| "When CountingSort?" | When keys are non-negative integers with bounded range k << n; O(n+k) beats O(n log n). |
| "Python sorted() internals?" | TimSort — hybrid merge+insertion sort. Stable, O(n log n) worst, O(n) for nearly-sorted input. |
| "How to sort descending?" | Pass `reverse=True` or negate key: `key=lambda x: -x`. |

## Practice Resources
- LeetCode: 75 Sort Colors, 56 Merge Intervals, 179 Largest Number, 215 Kth Largest Element
- Classic: implement merge sort on linked list, external sort for large files

## Related Topics
- [Divide & Conquer](divide-conquer.md) — MergeSort and QuickSort are D&C algorithms
- [Heaps](../data-structures/heaps.md) — HeapSort uses max-heap; also for partial sort (kth largest)
- [Binary Search](binary-search.md) — requires sorted input
