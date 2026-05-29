# Divide and Conquer

## TL;DR
Divide and conquer splits a problem into independent subproblems, solves each recursively, then merges results. Three steps: **Divide** (split at midpoint), **Conquer** (recurse on each half), **Combine** (merge results). Master theorem gives complexity. Key insight: unlike DP, subproblems don't overlap — no memoization needed. MergeSort, QuickSelect, and binary search are all divide-and-conquer.

## Core Concepts

**Master Theorem:** For `T(n) = a * T(n/b) + O(n^d)` where a = subproblems, b = size reduction, d = combine cost:

| Condition | Complexity | Example |
|---|---|---|
| `d > log_b(a)` | O(n^d) | Combine dominates |
| `d == log_b(a)` | O(n^d * log n) | Equal contribution |
| `d < log_b(a)` | O(n^log_b(a)) | Recursion dominates |

**Classic examples:**

| Algorithm | a | b | d | Complexity |
|---|---|---|---|---|
| MergeSort | 2 | 2 | 1 | O(n log n) — d=1=log_2(2) |
| Binary Search | 1 | 2 | 0 | O(log n) — d=0 < log_2(1)=0, actually O(log n) |
| Max Subarray D&C | 2 | 2 | 1 | O(n log n) |
| Matrix Multiply (naive) | 8 | 2 | 2 | O(n³) — d=2 < log_2(8)=3 |
| Strassen Matrix | 7 | 2 | 2 | O(n^2.81) — d=2 < log_2(7)=2.81 |

**D&C vs DP:** D&C subproblems are **independent** (no overlap); DP subproblems **overlap** (memoize). MergeSort subproblems don't share elements; Fibonacci subproblems share calls.

**D&C vs Greedy:** D&C explores subproblems recursively and combines; greedy makes one local choice and moves on.

**Recursion tree visualization:**
- Draw tree with branching factor `a`
- Each level costs O(n) total (or whatever the combine step is)
- Tree has log_b(n) levels → multiply for total cost

## Implementations

```python
from typing import List, Optional


# --- MergeSort (canonical D&C) ---
def merge_sort(arr: List[int]) -> List[int]:
    """
    Divide: split at mid. Conquer: sort each half. Combine: merge.
    T(n) = 2T(n/2) + O(n) → O(n log n). Stable. O(n) space.
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left: List[int], right: List[int]) -> List[int]:
    """O(n) merge of two sorted arrays."""
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# --- QuickSelect: Kth Largest Element (LC 215) ---
def find_kth_largest(nums: List[int], k: int) -> int:
    """
    Kth largest = (n-k)th smallest (0-indexed).
    QuickSelect: partition like QuickSort, only recurse on one side.
    O(n) average, O(n²) worst. O(1) extra space (in-place).
    """
    import random
    target_idx = len(nums) - k   # 0-indexed position of kth largest

    def quickselect(lo: int, hi: int) -> int:
        if lo == hi:
            return nums[lo]
        # Random pivot to avoid O(n^2) on sorted input
        pivot_idx = random.randint(lo, hi)
        nums[pivot_idx], nums[hi] = nums[hi], nums[pivot_idx]
        pivot = nums[hi]
        store = lo
        for i in range(lo, hi):
            if nums[i] <= pivot:
                nums[store], nums[i] = nums[i], nums[store]
                store += 1
        nums[store], nums[hi] = nums[hi], nums[store]
        # store is now pivot's final position
        if store == target_idx:
            return nums[store]
        elif store < target_idx:
            return quickselect(store + 1, hi)
        else:
            return quickselect(lo, store - 1)

    return quickselect(0, len(nums) - 1)


# --- Maximum Subarray: D&C version (LC 53) ---
def max_subarray_dc(nums: List[int]) -> int:
    """
    D&C approach: O(n log n) — less optimal than Kadane's O(n) but illustrates pattern.
    Max subarray is either in left half, right half, or crosses mid.
    """
    def helper(lo: int, hi: int) -> int:
        if lo == hi:
            return nums[lo]
        mid = lo + (hi - lo) // 2
        left_max = helper(lo, mid)
        right_max = helper(mid + 1, hi)
        cross_max = cross_sum(lo, mid, hi)
        return max(left_max, right_max, cross_max)

    def cross_sum(lo: int, mid: int, hi: int) -> int:
        """Max sum of subarray crossing the midpoint."""
        # Expand left from mid
        left_sum = float("-inf")
        curr = 0
        for i in range(mid, lo - 1, -1):
            curr += nums[i]
            left_sum = max(left_sum, curr)
        # Expand right from mid+1
        right_sum = float("-inf")
        curr = 0
        for i in range(mid + 1, hi + 1):
            curr += nums[i]
            right_sum = max(right_sum, curr)
        return left_sum + right_sum

    return helper(0, len(nums) - 1)


# --- Search 2D Matrix II (LC 240): D&C elimination ---
def search_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    Each row and column is sorted. Start top-right corner.
    Eliminate row or column at each step. O(m+n).
    (This is greedy/D&C hybrid — classic trick worth knowing.)
    """
    if not matrix or not matrix[0]:
        return False
    row, col = 0, len(matrix[0]) - 1
    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1    # eliminate column (all values in this col are too large)
        else:
            row += 1    # eliminate row (all values in this row are too small)
    return False


# --- Count inversions using MergeSort ---
def count_inversions(arr: List[int]) -> int:
    """
    Count pairs (i,j) where i<j and arr[i]>arr[j].
    D&C: count inversions in left, right, and across the split.
    O(n log n).
    """
    count = [0]   # use list to allow mutation inside nested function

    def merge_count(arr: List[int]) -> List[int]:
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_count(arr[:mid])
        right = merge_count(arr[mid:])
        i = j = 0
        merged = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                # All remaining elements in left are > right[j] → inversions
                count[0] += len(left) - i
                merged.append(right[j]); j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    merge_count(arr[:])  # work on copy
    return count[0]
```

## Key Problems

| Problem | D&C Strategy | Time | Space |
|---|---|---|---|
| MergeSort | Split + sort halves + merge | O(n log n) | O(n) |
| LC 215 Kth Largest | QuickSelect — one-sided partition | O(n) avg | O(1) |
| LC 53 Max Subarray (D&C) | Left/right/cross split | O(n log n) | O(log n) |
| LC 240 Search 2D Matrix II | Eliminate row/col from corner | O(m+n) | O(1) |
| Count inversions | MergeSort + count cross-inversions | O(n log n) | O(n) |
| LC 4 Median of Two Sorted Arrays | Binary search on partition point | O(log min(m,n)) | O(1) |

## Common Mistakes / Gotchas
- **Merge cost forgotten:** in master theorem, d is the exponent of the merge/combine step — don't assume it's 0
- **Not splitting evenly:** off-center splits can cause O(n²) (like QuickSort worst case); use random pivot or always split at mid
- **QuickSelect vs QuickSort:** QuickSelect only recurses on **one** side → O(n) average; QuickSort recurses on both → O(n log n)
- **Cross-subarray in max subarray D&C:** must compute cross-mid case or algorithm is incorrect — it's not just left/right max
- **Master theorem applicability:** requires recurrence of form `T(n) = aT(n/b) + O(n^d)`; doesn't apply directly to uneven splits

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "D&C vs DP?" | D&C: subproblems are independent, no memoization needed, typically O(n log n). DP: subproblems overlap, memoize for O(n²) or O(n). MergeSort is D&C; Fibonacci is DP. |
| "MergeSort complexity?" | T(n) = 2T(n/2) + O(n). Master theorem: a=2, b=2, d=1. d=log_b(a)=1 → O(n log n). |
| "QuickSelect average complexity?" | O(n): each partition step processes O(n) elements and discards half the remaining. Sum: n + n/2 + n/4 + ... = 2n → O(n). |
| "Median of Two Sorted Arrays trick?" | Binary search on the shorter array to find partition point that splits both arrays such that left half ≤ right half. O(log min(m,n)). |

## Practice Resources
- LeetCode: 215, 53, 240, 4, 912 (MergeSort implementation), 23 (Merge K Sorted Lists)
- Classic: implement QuickSelect + MergeSort from scratch; derive complexity via recursion tree

## Related Topics
- [Sorting](sorting.md) — MergeSort and QuickSort are D&C algorithms
- [Binary Search](binary-search.md) — binary search is D&C that discards one subproblem
- [Recursion](recursion.md) — D&C is always recursive; understand recursion tree analysis
- [Dynamic Programming](dynamic-programming.md) — DP when subproblems overlap; D&C when independent
