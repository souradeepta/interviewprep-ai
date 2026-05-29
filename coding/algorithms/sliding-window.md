# Sliding Window

## TL;DR
Sliding window maintains a contiguous subarray/substring between two pointers `left` and `right`, expanding and shrinking to track a property (sum, count, character frequency). Avoids O(n²) brute force by never re-scanning elements. Two forms: **fixed-size** (advance both pointers together) and **variable-size** (expand right greedily, shrink left when constraint violated).

## Core Concepts

**Fixed-size window:**
- Window size `k` is constant
- Add `right` element, remove `left` element, advance both
- Use for: max/min average, sum in window of size k

**Variable-size window (expand/shrink):**
- Expand `right` to include more elements
- Shrink `left` when window becomes invalid
- Use for: longest/shortest subarray satisfying a constraint

**Two-pointer window:**
- Same direction pointers on sorted or structured data
- Often overlaps with two-pointer pattern

**Key insight:** when shrinking, you never need to restart from scratch — elements you've already seen are efficiently removed as `left` advances.

**When to apply:**
- Contiguous subarray/substring with property constraint
- "Longest / shortest" subarray with condition
- "All substrings with at most K distinct chars"
- Fixed-k window average/max/min

## Implementations

```python
from typing import List
from collections import defaultdict

# --- Fixed-size window: maximum average subarray of length k ---
def max_average_fixed(nums: List[int], k: int) -> float:
    """LC 643. O(n) time, O(1) space."""
    # Build first window
    window_sum = sum(nums[:k])
    best = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  # slide: add right, remove left
        best = max(best, window_sum)
    return best / k


# --- Variable-size window: longest substring without repeating characters ---
def longest_no_repeat(s: str) -> int:
    """LC 3. Expand right; shrink left when duplicate found. O(n)."""
    char_index: dict[str, int] = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in char_index and char_index[ch] >= left:
            # Move left past the previous occurrence of ch
            left = char_index[ch] + 1
        char_index[ch] = right
        best = max(best, right - left + 1)
    return best


# --- Variable-size window: minimum window substring (LC 76) ---
def min_window_substring(s: str, t: str) -> str:
    """
    Find shortest window in s containing all chars in t.
    O(n + m) where n=len(s), m=len(t).
    """
    if not t or not s:
        return ""

    need = defaultdict(int)
    for ch in t:
        need[ch] += 1
    required = len(need)   # number of unique chars in t

    have = defaultdict(int)
    formed = 0             # unique chars in window meeting required count
    left = 0
    best_len = float("inf")
    best_left = 0

    for right, ch in enumerate(s):
        have[ch] += 1
        if ch in need and have[ch] == need[ch]:
            formed += 1

        # Try to shrink window while it's valid
        while formed == required:
            # Record best
            if right - left + 1 < best_len:
                best_len = right - left + 1
                best_left = left
            # Remove leftmost char
            left_ch = s[left]
            have[left_ch] -= 1
            if left_ch in need and have[left_ch] < need[left_ch]:
                formed -= 1
            left += 1

    return s[best_left:best_left + best_len] if best_len != float("inf") else ""


# --- Variable-size window: fruit into baskets (LC 904) ---
def total_fruit(fruits: List[int]) -> int:
    """
    Max subarray with at most 2 distinct values. Generic: at most K distinct.
    O(n) time, O(k) space.
    """
    basket: dict[int, int] = defaultdict(int)  # fruit_type -> count
    left = 0
    best = 0
    for right, fruit in enumerate(fruits):
        basket[fruit] += 1
        while len(basket) > 2:           # window has more than 2 distinct
            left_fruit = fruits[left]
            basket[left_fruit] -= 1
            if basket[left_fruit] == 0:
                del basket[left_fruit]
            left += 1
        best = max(best, right - left + 1)
    return best


# --- General template (variable-size) ---
def sliding_window_template(s: str) -> int:
    """
    Template for variable-size sliding window.
    Customize: window state tracking, validity check.
    """
    window: dict[str, int] = defaultdict(int)
    left = 0
    best = 0
    for right in range(len(s)):
        # 1. Expand: add s[right] to window
        window[s[right]] += 1

        # 2. Shrink: while window is invalid, move left
        while not is_valid(window):   # replace with actual condition
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1

        # 3. Update answer (window [left..right] is now valid)
        best = max(best, right - left + 1)
    return best

def is_valid(window: dict) -> bool:
    """Placeholder — replace with actual constraint check."""
    return True
```

## Key Problems

| Problem | Pattern | Time | Space |
|---|---|---|---|
| LC 643 Max Average Subarray I | Fixed window, slide sum | O(n) | O(1) |
| LC 3 Longest No Repeat | Variable window, char index | O(n) | O(charset) |
| LC 76 Min Window Substring | Variable window, frequency match | O(n+m) | O(n+m) |
| LC 904 Fruit Into Baskets | Variable, at most 2 distinct | O(n) | O(1) |
| LC 239 Sliding Window Maximum | Fixed window + deque | O(n) | O(k) |
| LC 424 Longest Repeating Char Replace | Variable window, replace budget | O(n) | O(26) |

## Common Mistakes / Gotchas
- **Shrink vs restart:** never reset `left = 0` — only advance it; restarting kills O(n) guarantee
- **Invalid window check:** shrink `while` not `if` — one shrink step may not fix validity
- **Fixed vs variable confusion:** if size is fixed, don't use `while`; just add and remove one element
- **Frequency counting:** use `collections.defaultdict(int)` or `Counter`; remember to delete keys at 0 if checking `len(dict)` for distinct count
- **Right vs right+1:** window size is `right - left + 1` (both inclusive); be consistent

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "When sliding window?" | Contiguous subarray/substring with some property. Fixed size → maintain running sum. Variable size → expand right, shrink left when violated. |
| "Why O(n) not O(n²)?" | Each element enters the window once (right pointer) and leaves once (left pointer) → 2n total operations. |
| "Fixed vs variable window?" | Fixed: window size k is given, slide both pointers together. Variable: expand right until invalid, then shrink left. |
| "Sliding window vs two-pointer?" | Sliding window: left ≤ right, window is contiguous, track window state. Two-pointer: may start from opposite ends, don't always track a window. |

## Practice Resources
- LeetCode: 3, 76, 239, 567, 643, 904, 424, 1004
- Master min-window-substring (LC 76) — covers the full variable-size template

## Related Topics
- [Two Pointers](two-pointers.md) — sliding window is a specialized two-pointer technique
- [Arrays & Strings](../data-structures/arrays-strings.md) — foundational operations
- [String Patterns](string-patterns.md) — anagram finding uses sliding window + Counter
