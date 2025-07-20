s = "bbabcbcab"
n = len(s)

# Initialize DP table
dp = [[0 for _ in range(n)] for _ in range(n)]

# All substrings of length 1 are palindromes
for i in range(n):
    dp[i][i] = 1

# Fill the table for substrings of length 2 to n
for l in range(2, n + 1):  # l = length of substring
    for i in range(n - l + 1):
        j = i + l - 1
        if s[i] == s[j] and l == 2:
            dp[i][j] = 2
        elif s[i] == s[j]:
            dp[i][j] = dp[i + 1][j - 1] + 2
        else:
            dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

# Final result
print("Length of Longest Palindromic Subsequence:", dp[0][n - 1])

'''
You got it, S! Let’s explore a popular 2D dynamic programming variation: **Longest Palindromic Subsequence**. It’s an elegant problem that teaches you how to think in terms of subsequence relationships rather than brute force.

I’ll follow the same structure:
- ✅ Python solution (flat style: no class, no nested def)
- 🧪 Five test cases
- 🔍 Detailed walkthrough of one test case with DP table

---

## 🔁 Problem Summary: Longest Palindromic Subsequence

**Given** a string `s`, find the **length** of the longest subsequence that is also a palindrome.  
Characters in the subsequence must maintain their relative order but don’t have to be contiguous.

---

## 🐍 Python Solution (No class, no nested def)

```python
s = "bbabcbcab"
n = len(s)

# Initialize DP table
dp = [[0 for _ in range(n)] for _ in range(n)]

# All substrings of length 1 are palindromes
for i in range(n):
    dp[i][i] = 1

# Fill the table for substrings of length 2 to n
for l in range(2, n + 1):  # l = length of substring
    for i in range(n - l + 1):
        j = i + l - 1
        if s[i] == s[j] and l == 2:
            dp[i][j] = 2
        elif s[i] == s[j]:
            dp[i][j] = dp[i + 1][j - 1] + 2
        else:
            dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

# Final result
print("Length of Longest Palindromic Subsequence:", dp[0][n - 1])
```

---

## 🧪 Test Cases

| Test Case | Input          | Expected Length | Example Palindromic Subsequence |
|-----------|----------------|-----------------|----------------------------------|
| 1         | `"bbabcbcab"`  | `7`             | `"babcbab"`                     |
| 2         | `"cbbd"`       | `2`             | `"bb"`                          |
| 3         | `"abcd"`       | `1`             | `"a"` (or any single letter)    |
| 4         | `"agbdba"`     | `5`             | `"abdba"`                       |
| 5         | `"aaa"`        | `3`             | `"aaa"`                         |

---

## 🧠 Walkthrough of Test Case 1: `"bbabcbcab"`

Initialize a `dp[n][n]` table where `dp[i][j]` stores the length of the LPS in `s[i..j]`.

1. All diagonal elements `dp[i][i] = 1` → Each char is a palindrome of length 1.
2. We iterate over lengths `l = 2 to n`, filling in combinations.

Key transitions:
- If `s[i] == s[j]`:  
  → we add `2` to `dp[i+1][j-1]`
- Else:  
  → take `max(dp[i+1][j], dp[i][j-1])`

Final DP table (simplified visualization):

| i\j | b | b | a | b | c | b | c | a | b |
|-----|---|---|---|---|---|---|---|---|---|
| 0   | 1 | 2 | 2 | 3 | 3 | 5 | 5 | 5 | 7 |
| 1   |   | 1 | 1 | 2 | 2 | 4 | 4 | 4 | 5 |
| 2   |   |   | 1 | 1 | 1 | 3 | 3 | 3 | 3 |
| 3   |   |   |   | 1 | 1 | 1 | 3 | 3 | 3 |
| 4   |   |   |   |   | 1 | 1 | 2 | 2 | 2 |
| 5   |   |   |   |   |   | 1 | 1 | 1 | 3 |
| 6   |   |   |   |   |   |   | 1 | 1 | 1 |
| 7   |   |   |   |   |   |   |   | 1 | 1 |
| 8   |   |   |   |   |   |   |   |   | 1 |

🎯 Result: `dp[0][8] = 7` → Longest palindromic subsequence is `"babcbab"`.

---

Want me to show the traceback logic that reconstructs the actual subsequence from the table? Or turn this into a HackerRank-style challenge with constraints and sample I/O?

'''