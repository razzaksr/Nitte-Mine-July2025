# Example inputs
text1 = "abcde"
text2 = "ace"

m = len(text1)
n = len(text2)

# Initialize the DP table
dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

# Fill the DP table
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if text1[i - 1] == text2[j - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

# Final result
print("Length of Longest Common Subsequence:", dp[m][n])

'''
Absolutely, S! Let’s break it down like a true DP enthusiast. I’ll start by giving you five test cases, then we’ll walk through one step-by-step using the 2D table so you can see exactly how the algorithm unfolds.

---

### 🧪 Test Cases

| Test Case | `text1`     | `text2`     | Expected LCS Length |
|-----------|-------------|-------------|----------------------|
| 1         | `"abcde"`   | `"ace"`     | `3` (`"ace"`)        |
| 2         | `"abc"`     | `"def"`     | `0` (no common subsequence) |
| 3         | `"aggtab"`  | `"gxtxayb"` | `4` (`"gtab"`)       |
| 4         | `"aaaa"`    | `"aa"`      | `2` (`"aa"`)         |
| 5         | `"abcdef"`  | `"acf"`     | `3` (`"acf"`)        |

---

### 🔍 Walkthrough of Test Case 1
Let’s use `text1 = "abcde"` and `text2 = "ace"`. The goal is to find the length of the longest common subsequence (LCS).

We build a DP table `dp` of size `(m+1) x (n+1)`, where `m = len(text1)` and `n = len(text2)`.

Initial table:
```plaintext
    ""  a  c  e
""  0  0  0  0
a   0
b   0
c   0
d   0
e   0
```

Now fill the table step by step:

1. **i=1, j=1**: `text1[0]=a`, `text2[0]=a` ⇒ match  
   → `dp[1][1] = dp[0][0] + 1 = 1`
2. **i=1, j=2,3**: no match  
   → carry max from left/top: `dp[1][2] = max(dp[0][2], dp[1][1]) = 1`, and `dp[1][3] = 1`
3. **i=2, j=1 to 3**: no match  
   → copy max so far: `dp[2][1] = 1`, `dp[2][2] = 1`, `dp[2][3] = 1`
4. **i=3, j=2**: `text1[2]=c`, `text2[1]=c` ⇒ match  
   → `dp[3][2] = dp[2][1] + 1 = 2`
5. Remaining cells: continue copying max values  
   → `dp[3][3] = 2`, `dp[4][3] = 2`,  
   → `i=5, j=3`: `text1[4]=e`, `text2[2]=e` ⇒ match  
   → `dp[5][3] = dp[4][2] + 1 = 3`

Final table:
```plaintext
    ""  a  c  e
""  0  0  0  0
a   0  1  1  1
b   0  1  1  1
c   0  1  2  2
d   0  1  2  2
e   0  1  2  3
```

🟢 The value in `dp[5][3]` is `3`, which is the length of the LCS `"ace"`.

---

Let me know if you'd like me to convert this into an educational challenge or add a traceback to extract the actual subsequence, not just the length!

'''