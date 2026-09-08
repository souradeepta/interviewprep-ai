# Prefix Sums and Difference Arrays

Prefix sums turn repeated range aggregation into constant-time queries after
linear preprocessing. They are especially useful when a problem asks for many
subarray sums, balances, counts, or “how many values lie in this interval?”

## Core templates

```python
def prefix_sum(nums):
    pref = [0]
    for value in nums:
        pref.append(pref[-1] + value)
    return pref


def range_sum(pref, left, right):
    """Inclusive range sum."""
    return pref[right + 1] - pref[left]


def count_subarrays_with_sum(nums, target):
    seen = {0: 1}
    total = answer = 0
    for value in nums:
        total += value
        answer += seen.get(total - target, 0)
        seen[total] = seen.get(total, 0) + 1
    return answer
```

The hashmap variant works with negative values; a sliding window does not,
because the window sum is not monotonic when values can be negative.

## Difference arrays

For many range updates, store only boundary changes. To add `delta` to every
element in `[left, right]`, do `diff[left] += delta` and
`diff[right + 1] -= delta`, then take one prefix sum at the end. This changes
`m` updates on an array of length `n` from O(mn) to O(m+n).

## Recognition checklist

- “Sum/count every subarray” → prefix sum plus hashmap.
- “Many range additions” → difference array.
- “Longest subarray with condition” → prefix sum plus earliest-index map.
- “2-D rectangle sum” → 2-D prefix table with inclusion-exclusion.
- Values can be negative → do not assume a sliding window works.

Typical practice: LeetCode 303, 304, 325, 437, 525, 560, 974, 1109.
