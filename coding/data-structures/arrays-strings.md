# Arrays and Strings

## TL;DR
Fundamental data structures for storing sequences. Arrays enable O(1) random access but O(n) insertion/deletion. Strings are immutable arrays of characters — special rules apply. Master array manipulation (slicing, two-pointers, sliding window) and string transformations (reversal, anagrams, palindromes).

## Core Concepts

**Arrays:** fixed-size or dynamic (resizable).
- **Time:** O(1) access by index; O(n) search; O(n) insert/delete
- **Space:** O(n)

**Strings:** immutable sequences of characters (in Python/Java).
- **Reversal:** O(n) time, O(n) space (need new string)
- **Substring search:** naive O(n·m), KMP O(n+m)

**Patterns:**
- **Two-pointers:** sort or move toward goal from both ends
- **Sliding window:** maintain a window of elements, slide to explore subproblems
- **Prefix/suffix tricks:** precompute arrays to avoid redundant work

## Key Problems

| Problem | Approach | Time | Space |
|---|---|---|---|
| Reverse array in-place | Two-pointers (swap i and n-1-i) | O(n) | O(1) |
| Is anagram | Count char frequencies or sort | O(n) | O(1) (26 chars) |
| Longest substring no repeat | Sliding window + hash map | O(n) | O(min(n, charset)) |
| Median of sorted arrays | Binary search on one array | O(log min(m,n)) | O(1) |
| Container with most water | Two-pointers (move inward) | O(n) | O(1) |

## Common Mistakes / Gotchas
- **Off-by-one errors:** check boundary conditions carefully
- **Modifying while iterating:** can skip elements (use new list or iterate backward)
- **String immutability:** concatenation in loop is O(n²) — use list + join instead
- **Two-pointer direction:** which pointer moves depends on goal (smallest, largest, closest, etc.)

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Array vs linked list?" | Array: O(1) access, O(n) insert. Linked list: O(n) access, O(1) insert (if pointer given). |
| "Sliding window when?" | When searching for contiguous subarray/substring with some property (max sum, min length, etc.). |
| "String reversal approach?" | Convert to list, reverse in-place, convert back. Or two-pointers on original if mutable. |

## Practice Resources
- LeetCode: Two Sum, Best Time to Buy Stock, Container With Most Water, Longest Substring Without Repeating Characters
- GeeksforGeeks: Arrays and Strings tutorials

## Related Topics
- [Trees & Graphs](trees-graphs.md) — [Sorting](../algorithms/sorting.md)
