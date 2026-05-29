# String Patterns

## TL;DR
String algorithms go beyond naive O(n*m) pattern matching. **KMP** finds all pattern occurrences in O(n+m) using a failure function. **Rabin-Karp** uses rolling hash for O(n) average. **Palindrome expansion** finds all palindromic substrings in O(n²) (Manacher's in O(n)). **Sliding window + Counter** finds all anagrams in O(n). These four patterns cover ~90% of string interview problems.

## Core Concepts

**Pattern matching comparison:**

| Algorithm | Preprocessing | Search | Space | When to use |
|---|---|---|---|---|
| Naive | O(1) | O(n*m) | O(1) | Short patterns, simple code |
| KMP | O(m) | O(n) | O(m) | Single pattern, guaranteed O(n) |
| Rabin-Karp | O(m) | O(n) avg | O(1) | Multiple patterns, find duplicates |
| Aho-Corasick | O(sum of patterns) | O(n+k) | O(sum) | Many patterns simultaneously |

**Palindrome approaches:**

| Approach | Time | Space | Notes |
|---|---|---|---|
| Expand around center | O(n²) | O(1) | Simple, handles both odd/even |
| Manacher's algorithm | O(n) | O(n) | Optimal but complex to implement |
| DP table | O(n²) | O(n²) | Good when need all palindromes |

**Anagram detection:**
- Fixed window: slide Counter/array of size len(pattern); compare at each step
- Key insight: two strings are anagrams iff their character counts are equal

**String hashing:**
- Rolling hash: `hash(window) = (hash(window[-1]) - s[left] * base^(k-1)) * base + s[right]`
- Avoids recomputing from scratch; O(1) per slide after O(m) initialization
- Collision risk: use double hashing (two different bases/mods) for correctness guarantee

## Implementations

```python
from typing import List
from collections import Counter


# --- KMP: Knuth-Morris-Pratt ---
def kmp_search(text: str, pattern: str) -> List[int]:
    """
    Find all occurrences of pattern in text. O(n + m) time, O(m) space.
    Failure function: lps[i] = length of longest proper prefix of pattern[:i+1]
    that is also a suffix.
    """
    if not pattern:
        return list(range(len(text) + 1))

    # Build failure function (partial match table)
    m = len(pattern)
    lps = [0] * m
    length = 0   # length of previous longest prefix suffix
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]   # don't increment i
            else:
                lps[i] = 0
                i += 1

    # Search using lps
    results = []
    n = len(text)
    i = j = 0   # i: text index, j: pattern index
    while i < n:
        if text[i] == pattern[j]:
            i += 1; j += 1
        if j == m:
            results.append(i - j)   # found match at i-j
            j = lps[j - 1]          # continue searching
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]      # mismatch after some match
            else:
                i += 1
    return results


# --- Rabin-Karp: rolling hash ---
def rabin_karp(text: str, pattern: str) -> List[int]:
    """
    O(n) average case. Use for multiple pattern matching or duplicate substrings.
    Hash collision possible — verify match when hash matches.
    """
    BASE = 31
    MOD = 10**9 + 7
    n, m = len(text), len(pattern)
    if m > n:
        return []

    def char_val(c: str) -> int:
        return ord(c) - ord('a') + 1

    # Compute hash of pattern and first window
    pattern_hash = 0
    window_hash = 0
    power = 1   # BASE^(m-1)
    for i in range(m):
        pattern_hash = (pattern_hash * BASE + char_val(pattern[i])) % MOD
        window_hash = (window_hash * BASE + char_val(text[i])) % MOD
        if i > 0:
            power = (power * BASE) % MOD

    results = []
    for i in range(n - m + 1):
        if window_hash == pattern_hash and text[i:i+m] == pattern:   # verify on hash match
            results.append(i)
        if i < n - m:
            # Roll hash: remove left char, add right char
            window_hash = (window_hash - char_val(text[i]) * power) % MOD
            window_hash = (window_hash * BASE + char_val(text[i + m])) % MOD
            window_hash = (window_hash + MOD) % MOD   # ensure non-negative
    return results


# --- Palindrome: expand around center ---
def longest_palindromic_substring(s: str) -> str:
    """
    LC 5. Expand from each center (n odd centers + n-1 even centers).
    O(n^2) time, O(1) space.
    """
    if not s:
        return ""
    start, max_len = 0, 1

    def expand(left: int, right: int) -> None:
        nonlocal start, max_len
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > max_len:
                max_len = right - left + 1
                start = left
            left -= 1
            right += 1

    for i in range(len(s)):
        expand(i, i)       # odd-length palindromes
        expand(i, i + 1)   # even-length palindromes

    return s[start:start + max_len]


# --- Anagrams: sliding window + Counter (LC 438) ---
def find_all_anagrams(s: str, p: str) -> List[int]:
    """
    Find all start indices of p's anagrams in s. O(n) time, O(1) space.
    Sliding window of size len(p); use Counter to compare frequencies.
    """
    k = len(p)
    if len(s) < k:
        return []

    need = Counter(p)
    window = Counter(s[:k])
    results = []

    if window == need:
        results.append(0)

    for i in range(k, len(s)):
        # Add right character
        window[s[i]] += 1
        # Remove left character
        left_char = s[i - k]
        window[left_char] -= 1
        if window[left_char] == 0:
            del window[left_char]
        if window == need:
            results.append(i - k + 1)

    return results


# --- LC 187: Repeated DNA Sequences (rolling hash) ---
def find_repeated_dna(s: str) -> List[str]:
    """
    Find all 10-letter substrings that appear more than once.
    Rolling hash over fixed window of size 10. O(n) time, O(n) space.
    """
    if len(s) <= 10:
        return []
    seen = set()
    repeated = set()
    for i in range(len(s) - 9):
        substr = s[i:i+10]
        if substr in seen:
            repeated.add(substr)
        seen.add(substr)
    return list(repeated)
```

## Key Problems

| Problem | Pattern | Time | Space |
|---|---|---|---|
| LC 28 Find Index of First Occurrence | KMP | O(n+m) | O(m) |
| LC 5 Longest Palindromic Substring | Expand around center | O(n²) | O(1) |
| LC 438 Find All Anagrams | Sliding window + Counter | O(n) | O(1) |
| LC 187 Repeated DNA Sequences | Rolling hash / set | O(n) | O(n) |
| LC 76 Min Window Substring | Sliding window + need/have | O(n) | O(n) |
| LC 49 Group Anagrams | Sort key or Counter key | O(n * k log k) | O(n) |
| LC 214 Shortest Palindrome | KMP on s + reverse(s) | O(n) | O(n) |

## Common Mistakes / Gotchas
- **KMP failure function edge case:** when `length != 0` and mismatch, set `length = lps[length-1]` without advancing i — this is the non-obvious step that gives O(n+m)
- **Rabin-Karp collision:** always verify `text[i:i+m] == pattern` when hash matches; hash equality alone is not sufficient
- **Counter comparison cost:** `Counter == Counter` is O(k) where k is alphabet size — for small alphabets (26 letters) this is O(1) effectively
- **Even vs odd palindromes:** expand must handle both `expand(i,i)` and `expand(i,i+1)` — forgetting even case misses "abba" style palindromes
- **Rolling hash negative values:** after subtracting, add MOD before taking modulo: `(x - y + MOD) % MOD`

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "Why KMP instead of naive?" | Naive rescans from scratch after mismatch: O(n*m). KMP precomputes how much of the pattern we can skip using the failure function: O(n+m). Key: failure function encodes longest border of each prefix. |
| "When Rabin-Karp over KMP?" | Rabin-Karp is easier to extend to multiple patterns and duplicate substring detection (rolling hash + set). KMP is single-pattern and has guaranteed O(n+m). |
| "Expand center vs Manacher's?" | Expand center is O(n²) but simple to code in interviews. Manacher's is O(n) but complex — only implement if specifically asked. |
| "Anagram detection trade-offs?" | Sorting: O(k log k) per string. Counter comparison: O(k) per window slide. For sliding window over a long string, Counter wins. |

## Practice Resources
- LeetCode: 28, 5, 438, 76, 49, 187, 214, 459
- Master sliding window + Counter (438) and KMP (28) — they're the most commonly tested

## Related Topics
- [Sliding Window](sliding-window.md) — anagram finding is a fixed-size sliding window problem
- [Arrays & Strings](../data-structures/arrays-strings.md) — string fundamentals
- [Hash Tables](../data-structures/hash-tables.md) — Counter and rolling hash rely on hashing
- [Dynamic Programming](dynamic-programming.md) — palindrome DP (LC 516, 132) uses 2D DP on string
