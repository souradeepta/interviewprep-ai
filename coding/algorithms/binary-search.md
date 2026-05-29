# Binary Search

## TL;DR
Binary search eliminates half the search space each step — O(log n) time. There are three distinct templates: exact match, left boundary (first position where condition is true), and right boundary (last position). The most powerful form is "search on answer": binary search over a range of possible answers instead of array indices, useful whenever you can write a monotonic predicate.

## Core Concepts

**Variants:**
- **Exact search:** find target in sorted array or return -1
- **Left boundary:** first index where `arr[i] >= target` (lower_bound)
- **Right boundary:** last index where `arr[i] <= target` (upper_bound)
- **Search on answer:** `lo`/`hi` are answer values, not indices; binary search over feasible range

**Monotonic predicate:** the key insight for "search on answer". If you can write `def feasible(x) -> bool` where `False...False, True...True` (or vice versa), you can binary search on x.

**Off-by-one rules (memorize once):**
- Inclusive `[lo, hi]`: loop `while lo <= hi`; shrink `lo = mid + 1` or `hi = mid - 1`
- Half-open `[lo, hi)`: loop `while lo < hi`; shrink `lo = mid + 1` or `hi = mid`

**Overflow-safe midpoint:** always `mid = lo + (hi - lo) // 2`, never `(lo + hi) // 2`

## Implementations

```python
from typing import List, Optional

# --- Template 1: Exact search ---
def binary_search(arr: List[int], target: int) -> int:
    """Return index of target, or -1 if not found. O(log n)."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# --- Template 2: Left boundary (first position where arr[i] >= target) ---
def left_bound(arr: List[int], target: int) -> int:
    """
    Return first index i where arr[i] >= target.
    Equivalent to bisect_left. Returns len(arr) if all elements < target.
    """
    lo, hi = 0, len(arr)  # hi = len(arr), not len-1
    while lo < hi:         # strict <
        mid = lo + (hi - lo) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid       # NOT mid-1; keep hi as candidate
    return lo              # lo == hi at termination


# --- Template 3: Right boundary (last index where arr[i] <= target) ---
def right_bound(arr: List[int], target: int) -> int:
    """
    Return last index i where arr[i] <= target.
    Equivalent to bisect_right - 1. Returns -1 if all elements > target.
    """
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1   # lo points one past last valid position


# --- Template 4: Search on answer ---
def search_on_answer(nums: List[int], target_sum: int) -> int:
    """
    Koko Eating Bananas pattern: find minimum speed k such that
    eating all piles in h hours.
    Binary search over possible values of k.
    """
    import math

    def feasible(speed: int) -> bool:
        """Can Koko eat all piles at this speed within h hours?"""
        h = 7  # given hours limit
        return sum(math.ceil(pile / speed) for pile in nums) <= h

    lo, hi = 1, max(nums)    # answer range: [1, max pile size]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid           # mid might be optimal, don't exclude
        else:
            lo = mid + 1       # mid too small, must go higher
    return lo                  # leftmost feasible answer


# --- LeetCode 35: Search Insert Position ---
def search_insert(nums: List[int], target: int) -> int:
    """Return index to insert target to keep array sorted. O(log n)."""
    return left_bound(nums, target)  # reuse left_bound


# --- LeetCode 153: Find Minimum in Rotated Sorted Array ---
def find_min_rotated(nums: List[int]) -> int:
    """Binary search: find pivot where rotation happened. O(log n)."""
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1   # min is in right half
        else:
            hi = mid       # mid might be min, keep it
    return nums[lo]


# --- LeetCode 33: Search in Rotated Sorted Array ---
def search_rotated(nums: List[int], target: int) -> int:
    """O(log n). One half is always sorted — check which, then decide."""
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        # Left half sorted
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        # Right half sorted
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```

## Key Problems

| Problem | Pattern | Time | Space |
|---|---|---|---|
| LC 704 Binary Search | Exact match template | O(log n) | O(1) |
| LC 35 Search Insert Position | Left boundary | O(log n) | O(1) |
| LC 34 First and Last Position | Left + right boundary | O(log n) | O(1) |
| LC 153 Find Min in Rotated | Compare mid to hi | O(log n) | O(1) |
| LC 875 Koko Eating Bananas | Search on answer | O(n log m) | O(1) |
| LC 33 Search Rotated Array | Identify sorted half | O(log n) | O(1) |
| LC 1011 Capacity to Ship | Search on answer (min capacity) | O(n log S) | O(1) |

## Common Mistakes / Gotchas
- **Wrong midpoint:** `(lo + hi) // 2` overflows for large integers in some languages; always use `lo + (hi - lo) // 2`
- **Wrong boundary shrink:** for left-boundary template, `hi = mid` (not `mid-1`) because mid itself could be the answer
- **Loop condition:** `while lo <= hi` for exact search; `while lo < hi` for boundary templates
- **Off-by-one on answer range:** in "search on answer", lo and hi must span the entire feasible range including edge cases
- **Forgetting to check feasibility at lo:** after the loop, verify `feasible(lo)` if there might be no valid answer

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "When to use binary search?" | Sorted array → exact search. Monotonic condition → search on answer. 'At least K' or 'minimum X that satisfies Y' → binary search on X. |
| "Left vs right boundary?" | Left: first True (bisect_left). Right: last True (bisect_right - 1). Use when duplicates exist and you need range. |
| "Search on answer approach?" | Define feasible(x). Identify monotone direction. Set lo/hi to answer range. Binary search until lo==hi. |
| "Time complexity?" | O(log n) for array search. O(n log k) for search-on-answer where k is answer range and each feasibility check costs O(n). |

## Practice Resources
- LeetCode: 704, 35, 34, 153, 33, 875, 1011, 410, 287
- Key insight problems: 875 Koko Eating Bananas, 1011 Ship Packages, 410 Split Array Largest Sum

## Related Topics
- [Sorting](sorting.md) — binary search requires sorted input
- [Two Pointers](two-pointers.md) — often combined for O(n log n) solutions
- [Divide & Conquer](divide-conquer.md) — binary search is divide-and-conquer with one branch discarded
