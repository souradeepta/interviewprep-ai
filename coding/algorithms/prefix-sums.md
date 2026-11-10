# Prefix Sums and Difference Arrays

Prefix sums turn repeated range aggregation into constant-time queries after linear preprocessing. They are especially useful when a problem asks for many subarray sums, balances, counts, or “how many values lie in this interval?” The central move is to represent the cumulative contribution before each position, so a range can be recovered from two boundaries.

## Recognition criteria

Look for many queries over contiguous ranges, a count of subarrays satisfying a sum condition, or many range updates that can be applied together. Prefix sums are strongest when each element contributes additively and the query asks for a group of consecutive values. A prefix table changes repeated O(n) scans into O(1) range queries after O(n) preprocessing. If there are only one or two queries, a scan may be clearer and cheaper.

Use a leading zero so a range beginning at index zero follows the same formula as every other range. For an input `a` of length n, define `P[0] = 0` and `P[i+1] = P[i] + a[i]`. The inclusive sum from `left` to `right` is `P[right+1] - P[left]`. This indexing convention is the most important correctness detail; writing down the formula before coding prevents many off-by-one errors.

```python
def prefix_sum(values):
    pref = [0]
    for value in values:
        pref.append(pref[-1] + value)
    return pref


def range_sum(pref, left, right):
    """Inclusive range sum; caller supplies valid zero-based indices."""
    return pref[right + 1] - pref[left]
```

## Worked trace

For `values = [4, -2, 5, 1]`, the prefix table is `[0, 4, 2, 7, 8]`. The sum from index 1 through 3 is `P[4] - P[1] = 8 - 4 = 4`, equal to `-2 + 5 + 1`. An empty range can be represented by equal boundaries in a half-open API, but the inclusive helper above deliberately accepts only `left <= right`; define empty-range behavior rather than letting indexes accidentally decide it.

## Pattern: count subarrays with a target sum

Two prefix boundaries delimit a subarray. If cumulative sums at boundaries `i` and `j` differ by target k, then `P[j] - P[i] = k`, or `P[i] = P[j] - k`. While scanning the right boundary, store how many earlier prefixes have each value. Add the count for `current - k` before inserting `current`, which ensures the subarray is non-empty. This works with negative values because it uses equality rather than a monotonic window.

```python
def count_subarrays_with_sum(nums, target):
    frequencies = {0: 1}
    total = answer = 0
    for value in nums:
        total += value
        answer += frequencies.get(total - target, 0)
        frequencies[total] = frequencies.get(total, 0) + 1
    return answer
```

## Complete problem 1: range sum queries

**Prompt:** Given a static integer array and q inclusive queries `[left, right]`, return every queried sum. **Approach:** Build one prefix table, validate each query’s bounds, and subtract its two boundary totals. **Example:** For `[3, 0, -4, 8]`, queries `[0,2]` and `[2,3]` return `-1` and `4`. **Correctness:** Every value before `left` appears in both prefixes and cancels; values from `left` through `right` remain exactly once. **Complexity:** O(n+q) time and O(n) extra space. For a single query, a direct scan uses O(1) extra space and may be preferable.

## Complete problem 2: count subarrays with sum k

**Prompt:** Given signed integers and target k, count contiguous non-empty subarrays whose sum is k. **Approach:** Use the frequency-map algorithm above. Test `[1, -1, 0]` with target zero: the valid ranges are `[1,-1]`, `[1,-1,0]`, and `[0]`, so the result is 3. The initial `{0: 1}` handles ranges beginning at index zero. Duplicate prefix totals matter, so store counts rather than only the latest index. **Correctness:** At each right boundary, every earlier prefix with value `current-k` produces one distinct matching subarray. **Complexity:** O(n) expected time and O(n) extra space. For the longest matching subarray, retain the earliest index for each prefix instead of a frequency count.

## Difference arrays for range updates

When m updates each add a value to a long interval, applying every update element-by-element costs O(mn). Instead allocate a difference array of length n+1. For an inclusive update `[l,r] += x`, add x at `diff[l]` and subtract x at `diff[r+1]`; one prefix scan reconstructs the final values in O(n+m). The extra boundary at n lets the right cancellation remain in bounds when r is the last index. If updates are followed by online queries between updates, this static trick is insufficient; a Fenwick tree or segment tree may be appropriate.

Two-dimensional prefix sums use inclusion-exclusion. Let `P[r+1][c+1]` include all cells above and left. A rectangle with inclusive corners `(r1,c1)` and `(r2,c2)` has sum `P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]`. The top-left region was subtracted twice, so add it back. The same pattern supports rectangle counts when each cell is a zero or one.

## Common failure cases and diagnostics

- **Boundary mismatch:** Mixing half-open and inclusive endpoints causes one extra or missing element. Pick one API and test a one-element interval at index zero and at the final index.
- **Insert-before-query:** In target-sum counting, inserting the current prefix before looking up `current-k` counts an empty subarray when k is zero. Query first, then insert; test `[0]` with target zero.
- **Sliding window with negatives:** A window sum ceases to be monotonic when values may be negative. Use prefix differences and a hashmap; test alternating positive and negative values.
- **Wrong update cancellation:** Difference arrays need `r+1` cancellation, including a buffer cell when r is n-1. Test an update covering the entire array and an update covering one element.
- **Integer or domain assumptions:** Python integers do not overflow, but fixed-width implementations can. Use a sufficiently wide type; for averages, remember the query sum may need division only after exact accumulation.
- **Dynamic data:** A prefix table becomes stale when values change. Rebuild for batches of changes or choose a tree structure for interleaved updates and queries.

## Complexity and selection summary

Building a one-dimensional prefix table costs O(n) time and O(n) space; each range query costs O(1). The target-sum count takes O(n) expected time and O(n) space. A difference-array batch with m range updates costs O(n+m) time and O(n) space. A 2-D table costs O(rows×columns) preprocessing and storage with O(1) rectangle queries. Choose prefix methods for static additive aggregates and grouped offline updates; choose a Fenwick or segment tree when the problem interleaves modifications with queries.

## Practice list

Practice recognition and explain the invariant before coding: LeetCode 303 (range sum), 304 (2-D range sum), 325 (maximum subarray length), 437 (path sum III), 525 (contiguous array), 560 (subarray sum equals k), 974 (subarray sums divisible by k), and 1109 (corporate flight bookings). For every problem, test empty input, a one-element input, repeated prefix totals, and negative values where allowed.
