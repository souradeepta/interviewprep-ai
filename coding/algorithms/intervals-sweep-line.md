# Intervals and Sweep Line

Interval questions are usually about ordering events, merging overlap, or allocating a scarce resource. Sort by the left endpoint for coverage and merging; sort start/end events separately when the question asks for peak concurrency or capacity. Before implementation, settle whether endpoints are closed or half-open, because a single comparison changes the answer at touching boundaries.

## Recognition criteria

Look for ranges of time, coordinates, bookings, coverage, or repeated “active at this moment” queries. Merge when the output should be disjoint covered spans. Use a sweep line when the output depends on how many intervals are active at every point, such as minimum rooms or maximum load. For overlap between two already sorted schedules, a two-pointer scan may be simpler than a full event list. If intervals are dynamic, sorting once is not enough; consider an ordered structure or segment tree.

The common merge template sorts by `(start, end)` and compares each next interval with the end of the current merged interval. Its invariant is that the output list is sorted and pairwise disjoint under the chosen overlap convention. If the next start is within the current end, extend the end to the maximum; otherwise emit a new span.

```python
def merge(intervals):
    if not intervals:
        return []
    ordered = sorted((start, end) for start, end in intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:  # closed intervals; touching merges
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

For half-open intervals `[start,end)`, two intervals touching at an endpoint do not overlap; use `start < previous_end`. State whether zero-length intervals are legal and how they behave. Normalize reversed endpoints only when the prompt says input can be reversed; silently swapping them may hide invalid data.

## Worked trace: merge

Input: `[[5,7], [1,3], [2,6], [8,8]]`. Sorting gives `[1,3], [2,6], [5,7], [8,8]`. Start with `[1,3]`; `[2,6]` overlaps, so extend to `[1,6]`; `[5,7]` extends to `[1,7]`; `[8,8]` is separate for closed intervals because 8 is greater than 7. The result is `[[1,7], [8,8]]`. Under a convention that merges touching intervals, the final two spans also merge; that is a policy choice, not a sorting detail.

## Sweep-line event ordering

For peak concurrency, turn every start into +1 and every end into -1, sort events, and track the active count. At equal timestamps, event ordering encodes endpoint semantics. For half-open meetings `[start,end)`, process ends before starts so one meeting can reuse a room at the exact instant another ends. For closed intervals, starts may need to count before ends. A direct event tuple sort should make this explicit instead of relying on incidental tuple order.

## Complete problem 1: minimum meeting rooms

**Prompt:** Given meeting intervals, return the fewest rooms required. **Approach:** For valid half-open meetings, sort all starts and ends separately. Walk starts in order; release every room whose end is `<= start`, then allocate one room for the new meeting. Track the maximum allocated count or use a min-heap of room end times. **Example:** `[[0,30],[5,10],[15,20]]` needs 2 rooms; the first overlaps both short meetings, but those short meetings do not overlap each other. **Correctness:** A new room is necessary exactly when every existing room ends after this meeting starts; the earliest ending room is the best candidate to reuse. **Complexity:** O(n log n) time and O(n) space. For only the peak count, a sorted endpoint sweep uses O(n) extra space.

## Complete problem 2: erase minimum intervals to remove overlap

**Prompt:** Remove the fewest intervals so the remaining intervals do not overlap. **Approach:** Sort by end time and greedily retain the interval with the earliest finishing endpoint that does not overlap the last retained interval. Earlier finishing leaves at least as much room for every future interval, which is an exchange argument for the greedy choice. **Example:** `[[1,2],[2,3],[3,4],[1,3]]` retains the first three under half-open semantics and removes `[1,3]`, so one removal suffices. **Correctness:** Replacing a retained interval with a compatible interval that ends earlier cannot reduce future choices. **Complexity:** O(n log n) time and O(n) or O(1) additional space depending on sorting. Explicitly adapt compatibility for closed endpoints.

## Other useful interval patterns

To insert one interval into a sorted, non-overlapping list, copy spans that end before it, merge every span it overlaps, then copy the remainder. For intersections between two sorted interval lists, emit `[max(starts), min(ends)]` when nonempty and advance the interval with the earlier end. For weighted event sums, maintain a running aggregate rather than only an active count. Coordinate compression is useful when endpoints are huge integers but only their order and ranges matter; retain the mapping back to original coordinates if output requires them.

## Common failure cases and diagnostics

- **Unspecified touching rule:** `[1,2]` and `[2,3]` may merge for closed intervals or remain separate for half-open intervals. Ask about endpoint inclusion, write a two-interval boundary test, then select `<` or `<=` deliberately.
- **Wrong same-time event order:** Starts-before-ends overcounts half-open meeting concurrency. Test one interval ending exactly when another begins and assert the expected room count.
- **Using start-time greedy for removals:** Retaining earliest starts can block many later intervals. Sort by end time and verify the exchange argument against nested intervals.
- **Mutating caller data:** Sorting the input list in place may surprise code that reuses it. Copy when the API promises not to mutate, and test input preservation.
- **Invalid or empty intervals:** Decide whether `[t,t]` is a point, an empty half-open span, or invalid. Filter or reject according to the contract rather than letting event ordering decide.
- **Duplicate intervals and tie instability:** Equal starts or ends need deterministic handling when the payload matters. Sort by all relevant tie-breakers and test duplicate records.
- **Overflow or precision:** In fixed-width languages, `end + 1` can overflow; for real-valued endpoints, exact equality may be unsafe. Use half-open boundaries or domain-specific integer ticks when possible.

## Complexity and selection summary

Sorting dominates most static interval algorithms at O(n log n). Merge uses O(n) output space; an event sweep uses O(n) events; a two-pointer intersection uses O(1) extra space beyond output. Heap scheduling also costs O(n log n), with O(k) heap space for k simultaneous rooms. Use merge for union coverage, sweep for active load, greedy earliest end for interval scheduling, and two pointers for pairwise intersections. Use a dynamic range structure only when updates arrive between queries.

## Practice list

Try LeetCode 56 (merge), 57 (insert), 252 (meeting rooms), 253 (meeting rooms II), 435 (non-overlap), 452 (arrows), 986 (interval intersections), 1094 (car pooling), and 1851 (minimum interval to include each query). For each, write the endpoint convention and tie rule before coding; test touching, nested, duplicate, empty, and one-interval inputs.
