# Current Interview Topic Gap Review — 2026-09-07

## Evidence reviewed

- Recent Reddit interview/OA discussions emphasize strings and parsing, trees
  and tries, intervals/scheduling, arrays with prefix sums and sliding
  windows, BFS/DFS on graphs and grids, heaps/top-k, binary search, union-find,
  simulation, and edge-case testing.
- Recent LeetCode ML interview reports include probability and rejection
  sampling, embeddings/tokenization/attention/BERT, topic modeling, clustering,
  regularization, cold-start recommendation, ranking, and feature pipelines.
- Current ML-prep catalogs have expanded beyond theory into timed from-scratch
  implementations and company-specific paths covering math foundations,
  PyTorch, training, NLP, attention/transformers, and GPT construction.

Sources:

- [Reddit OA topic analysis](https://www.reddit.com/r/leetcode/comments/1u3fpb4/i_collected_analyzed_300_interview_questions_in/)
- [Recent Reddit Google preparation discussion](https://www.reddit.com/r/leetcode/comments/1w8uzg6/google_interview_in_2_weeks_how_should_i_prepare/)
- [LeetCode LinkedIn ML interview report](https://leetcode.com/discuss/post/7405614/linkedin-software-machine-learn-l9pd/)
- [NeetCode ML practice catalog](https://neetcode.io/practice/machine-learning)
- [Deep-ML practice catalog](https://www.deep-ml.com/)

## Coverage comparison

Already covered: arrays/strings, two pointers, sliding window, graphs,
trees, tries, union-find, binary search, DP, greedy, sorting, bit
manipulation, ML fundamentals, attention, embeddings, recommendation/search
case studies, and ML system design.

High-confidence gaps added in this pass:

1. Prefix sums and difference arrays as a named pattern.
2. Intervals, sweep line, and event ordering as a named pattern.
3. Matrix/grid traversal and in-place grid state transitions.
4. Practical simulation, log parsing, and state-machine questions.

Follow-up gaps to add in the next ML-content pass:

- Rejection sampling and probability simulation.
- Topic modeling and clustering implementation tradeoffs.
- A dedicated recommendation/ranking implementation question, alongside the
  existing system-design case studies.
- Explicit SQL/data manipulation practice for MLE screens.

The first four are implemented as reusable coding-pattern guides because they
also address the practical OA trend: messy input transformation, state
tracking, scheduling, and boundary-heavy updates.
