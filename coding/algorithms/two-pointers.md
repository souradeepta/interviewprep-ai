# Two Pointers

## TL;DR
Two pointers uses two indices moving through a data structure to avoid O(n²) brute force. Three flavors: **opposite ends** (sorted array pair sum, palindrome check), **same direction** (fast/slow for cycle detection, duplicates removal), and **partition pointer** (Dutch flag, quicksort partition). All yield O(n) time instead of O(n²) nested loops.

## Core Concepts

**Opposite ends (converging):**
- Start `left=0`, `right=n-1`; move toward each other
- Works on sorted arrays (or palindromes)
- Decision: if current pair satisfies, record; if sum too small, advance left; if too large, retreat right

**Same direction (fast/slow):**
- `slow` tracks write position or current valid element; `fast` scans ahead
- Removes duplicates in-place, finds cycle in linked list (Floyd's algorithm)
- `fast` may advance 2× per step (cycle detection) or 1× (deduplication)

**Partition pointer:**
- Separate array around a pivot or condition
- `left` pointer at start for elements satisfying condition; scan `right` through array
- Used in QuickSort partition, Dutch National Flag

**When to apply:**
- Sorted array + find pair with target sum
- Remove/overwrite in-place (duplicates, zeros)
- Linked list middle, cycle detection
- Palindrome verification
- Partition array into groups

## Implementations

```python
from typing import List, Optional


# --- Two Sum II (sorted array, opposite ends) ---
def two_sum_sorted(numbers: List[int], target: int) -> List[int]:
    """LC 167. Sorted input → two pointers from ends. O(n) time, O(1) space."""
    left, right = 0, len(numbers) - 1
    while left < right:
        s = numbers[left] + numbers[right]
        if s == target:
            return [left + 1, right + 1]   # 1-indexed
        elif s < target:
            left += 1     # need larger sum → advance left
        else:
            right -= 1    # need smaller sum → retreat right
    return []


# --- 3Sum (sort + two pointers per anchor) ---
def three_sum(nums: List[int]) -> List[List[int]]:
    """LC 15. O(n^2) time, O(1) space (excluding output)."""
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue   # skip duplicate anchors
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1    # skip duplicate left
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1   # skip duplicate right
                left += 1; right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result


# --- Remove Duplicates from Sorted Array (same direction) ---
def remove_duplicates(nums: List[int]) -> int:
    """LC 26. slow: write head. fast: scan for new values. O(n), O(1)."""
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1   # length of unique prefix


# --- Partition (fast/slow same direction) ---
def partition(nums: List[int], pivot: int) -> None:
    """
    Rearrange nums so elements <= pivot come before > pivot.
    Uses write pointer (slow) pattern.
    """
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] <= pivot:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


# --- Container With Most Water (opposite ends, greedy) ---
def max_water(height: List[int]) -> int:
    """LC 11. Move the shorter boundary inward to potentially find larger area. O(n)."""
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        water = min(height[left], height[right]) * (right - left)
        best = max(best, water)
        if height[left] < height[right]:
            left += 1    # shorter side limits volume; advance it
        else:
            right -= 1
    return best


# --- Linked list: fast/slow pointer (Floyd's cycle detection) ---
class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next

def has_cycle(head: Optional[ListNode]) -> bool:
    """LC 141. Fast moves 2x, slow moves 1x. They meet iff cycle exists. O(n), O(1)."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """Find middle node. When fast reaches end, slow is at middle. O(n), O(1)."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

## Key Problems

| Problem | Pattern | Time | Space |
|---|---|---|---|
| LC 167 Two Sum II | Converging opposite ends | O(n) | O(1) |
| LC 15 3Sum | Sort + two-pointer per anchor | O(n²) | O(1) |
| LC 11 Container With Most Water | Converging, move shorter | O(n) | O(1) |
| LC 26 Remove Duplicates | Same direction slow/fast | O(n) | O(1) |
| LC 141 Linked List Cycle | Floyd fast/slow | O(n) | O(1) |
| LC 876 Middle of Linked List | Fast moves 2x | O(n) | O(1) |
| LC 125 Valid Palindrome | Converging, skip non-alnum | O(n) | O(1) |

## Common Mistakes / Gotchas
- **Sorting requirement:** opposite-ends only works on sorted input (or specifically structured like palindromes)
- **Duplicate skipping in 3Sum:** must skip both the outer anchor duplicates AND the inner left/right duplicates after recording a triplet
- **Floyd's cycle — starting point:** both slow and fast must start at head (not slow at head, fast at head.next)
- **Partition pointer direction:** ensure `slow` starts at 0 and swaps with `fast` when condition met; don't increment slow without swapping
- **Off-by-one in remove duplicates:** return `slow + 1`, not `slow`

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "Two pointers vs sliding window?" | Two pointers: often converging from ends, doesn't necessarily maintain a window. Sliding window: left ≤ right always, tracks a contiguous window with a property. |
| "When fast/slow pointers?" | Linked list cycle detection (O(1) space vs hash set). Finding middle of list. Kth from end (advance fast k steps first). |
| "Why does Container With Most Water work?" | When we move the shorter boundary, area can only stay same or increase — the taller boundary can only help. Moving taller would definitely decrease. |
| "3Sum complexity?" | O(n²): n anchors × O(n) two-pointer pass per anchor. Sorting is O(n log n), dominated. |

## Practice Resources
- LeetCode: 167, 15, 11, 26, 141, 876, 125, 42 (Trapping Rain Water)
- Key insight: LC 42 Trapping Rain Water has two-pointer O(1) space solution

## Related Topics
- [Sliding Window](sliding-window.md) — specialized same-direction two-pointer
- [Sorting](sorting.md) — 3Sum and many two-pointer problems require sorted input
- [Linked Lists](../data-structures/linked-lists.md) — fast/slow pointers for cycle and middle
