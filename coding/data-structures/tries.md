# Tries

A trie (prefix tree) is a tree where each node represents a single character and paths from root to a marked node spell a complete word. Tries beat hash sets for prefix queries: `startsWith` is O(m) regardless of vocabulary size. The main use cases are autocomplete, spell-check, and dictionary-backed grid word search (Word Search II).

## Core Patterns

| Pattern | When to use | Time | Space |
|---|---|---|---|
| Insert word | Add word to trie | O(m) | O(m) per word |
| Search exact | Does this exact word exist? | O(m) | O(1) |
| Starts-with prefix | Any word with this prefix? | O(m) | O(1) |
| Autocomplete | All words with a given prefix | O(m + output) | O(output) |
| Word search II | Find all dictionary words in a grid | O(M*N*4^L) | O(W*L) |
| Replace prefix | Replace each word with its shortest root prefix | O(n*m) | O(W*m) |

m = length of query word, W = vocabulary size, L = max word length.

## Python Implementations

```python
from __future__ import annotations
from typing import Optional


class TrieNode:
    """Single node in a trie.

    children: maps character to child TrieNode.
    is_end:   marks that the path to this node forms a complete word.
    """

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False


class Trie:
    """Prefix tree supporting insert, search, and prefix queries.

    All operations are O(m) where m = length of the word/prefix.
    Space: O(ALPHABET_SIZE * m * n) worst case (no shared prefixes).
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word into the trie.  O(m)"""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        """Return True if the exact word exists in the trie.  O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        """Return True if any word in the trie starts with prefix.  O(m)"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Traverse to the node at the end of prefix, or return None."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def get_words_with_prefix(self, prefix: str) -> list[str]:
        """Return all words that start with the given prefix.

        Uses DFS from the node at the end of the prefix.
        Time: O(m + output_words * avg_word_length)
        """
        node = self._find_node(prefix)
        if node is None:
            return []
        result: list[str] = []
        self._collect_words(node, list(prefix), result)
        return result

    def _collect_words(
        self, node: TrieNode, path: list[str], result: list[str]
    ) -> None:
        """DFS helper: accumulate words into result."""
        if node.is_end:
            result.append(''.join(path))
        for ch, child in node.children.items():
            path.append(ch)
            self._collect_words(child, path, result)
            path.pop()


def replace_words(dictionary: list[str], sentence: str) -> str:
    """Replace each word in sentence with its shortest root from dictionary.

    Build trie from roots; for each word in sentence walk trie until
    a root end is found (shortest prefix) or word is not in trie.
    Time: O(D*m + S*m) where D = dict size, S = sentence length, m = max word.
    Space: O(D*m)
    """
    trie = Trie()
    for root in dictionary:
        trie.insert(root)

    def find_root(word: str) -> str:
        node = trie.root
        for i, ch in enumerate(word):
            if ch not in node.children:
                break
            node = node.children[ch]
            if node.is_end:
                return word[:i + 1]   # shortest prefix root found
        return word   # no root matched, keep original

    return ' '.join(find_root(w) for w in sentence.split())
```

## Complexity Summary

| Operation | Time | Space | Notes |
|---|---|---|---|
| Insert word (length m) | O(m) | O(m) new nodes | Shared prefix amortizes |
| Search exact | O(m) | O(1) | Traverse + check is_end |
| Starts-with prefix | O(m) | O(1) | Traverse only |
| Autocomplete | O(m + output) | O(output) | DFS from prefix node |
| Build trie (n words) | O(n * m) | O(n * m) | m = avg word length |
| Word Search II | O(M*N*4^L + W*L) | O(W*L) | Grid + trie |

## Interview Recognition Template

- Prefix matching, autocomplete, "starts with" queries → Trie.
- Shared-prefix compression needed, hash set is too slow for prefix → Trie beats set.
- "Find all words in grid" + given dictionary → Trie + DFS backtracking (Word Search II).
- "Replace each word with its shortest root" → Trie early-termination on is_end.
- Trie vs hash set: hash set gives O(1) exact lookup but O(m * n) for all prefix checks; Trie gives O(m) prefix in one traversal.

## Worked Examples

### 208. Implement Trie (Prefix Tree)

**Problem:** Implement a trie with insert, search, and startsWith methods.

```python
class TrieSolution:
    """Standard trie with dict-based children."""

    def __init__(self) -> None:
        self.root: dict = {}   # nested dicts; '#' marks word end

    def insert(self, word: str) -> None:
        """O(m)"""
        node = self.root
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node['#'] = True   # sentinel for end-of-word

    def search(self, word: str) -> bool:
        """O(m)"""
        node = self.root
        for ch in word:
            if ch not in node:
                return False
            node = node[ch]
        return '#' in node

    def startsWith(self, prefix: str) -> bool:
        """O(m)"""
        node = self.root
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        return True
```

### 212. Word Search II

**Problem:** Given an m×n board and a list of words, return all words that can be found in the board (adjacent cells, no re-use).

```python
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """Build trie from words; DFS on grid using trie to prune paths.
    Time: O(M*N*4^L + W*L)  Space: O(W*L)
    """
    # Build trie
    root: dict = {}
    for word in words:
        node = root
        for ch in word:
            node = node.setdefault(ch, {})
        node['$'] = word   # store word at end node

    rows, cols = len(board), len(board[0])
    found: list[str] = []

    def dfs(r: int, c: int, node: dict) -> None:
        ch = board[r][c]
        if ch not in node:
            return
        nxt = node[ch]
        if '$' in nxt:
            found.append(nxt.pop('$'))  # collect and remove to avoid dups
        board[r][c] = '#'               # mark visited
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                dfs(nr, nc, nxt)
        board[r][c] = ch                # restore

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    return found
```

### 648. Replace Words

**Problem:** Given a dictionary of roots and a sentence, replace each word with its shortest root. If no root matches, keep the original.

```python
def replaceWords(dictionary: list[str], sentence: str) -> str:
    """Trie early-exit on is_end gives shortest root match.
    Time: O(D*m + S*m)  Space: O(D*m)
    """
    # Build trie inline with dict nodes
    trie_root: dict = {}
    for root in dictionary:
        node = trie_root
        for ch in root:
            node = node.setdefault(ch, {})
        node['#'] = True   # root endpoint

    def shortest_root(word: str) -> str:
        node = trie_root
        for i, ch in enumerate(word):
            if '#' in node:
                return word[:i]   # found root, return prefix so far
            if ch not in node:
                return word
            node = node[ch]
        return word[:len(word)] if '#' in node else word

    return ' '.join(shortest_root(w) for w in sentence.split())
```

## Related Topics

- [Trees](trees.md) — [Hash Tables](hash-tables.md) — [Arrays and Strings](arrays-strings.md)
