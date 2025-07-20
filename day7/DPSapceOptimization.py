def optimizedLcs(text1, text2):
    m = len(text1)
    n = len(text2)

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]

# print("Optimized length",optimizedLcs("abcde","ace"))


def optimizedLps(pattern):
    n = len(pattern)
    dp = [0] * n
    for i in reversed(range(n)):
        new_dp = dp[:]
        new_dp[i] = 1
        for j in range(i + 1, n):
            if pattern[i] == pattern[j]:
                new_dp[j] = dp[j - 1] + 2
            else:
                new_dp[j] = max(dp[j], new_dp[j - 1])
        dp = new_dp
    return dp[n - 1]

print("Optimized Palindromic Trading Pattern Length:", optimizedLps("HLHLHHL"))
'''
Fantastic! Let’s optimize space for each of your three favorite DP applications—**LCS**, **LPS**, and **your symbolic trading sequences**—with clear, procedural Python code and explanations so everything clicks into place. 🧠🔧

---

## 1️⃣ Space Optimization: Longest Common Subsequence (LCS)

Normally uses a 2D table `dp[i][j]`. We’ll shrink that to **two 1D arrays**.

### 🔧 Code (Space Optimized to O(N))

```python
text1 = "abcde"
text2 = "ace"

m = len(text1)
n = len(text2)

prev = [0] * (n + 1)
curr = [0] * (n + 1)

for i in range(1, m + 1):
    for j in range(1, n + 1):
        if text1[i - 1] == text2[j - 1]:
            curr[j] = prev[j - 1] + 1
        else:
            curr[j] = max(prev[j], curr[j - 1])
    prev, curr = curr, [0] * (n + 1)

print("Optimized LCS length:", prev[n])
```

### 🧠 How It Works:
- Reuse `prev` and `curr` to simulate 2D rows.
- Only store the previous row and overwrite the current.
- You drop space from O(m×n) to O(n).

---

Space Optimization: Palindromic Trading Sequences

Let’s squeeze out performance from symbolic patterns like `"HLHLHHL"`.

### 🔧 Code (Flat and space-efficient)

```python
pattern = "HLHLHHL"
n = len(pattern)

dp = [0] * n

for i in reversed(range(n)):
    new_dp = dp[:]
    new_dp[i] = 1
    for j in range(i + 1, n):
        if pattern[i] == pattern[j]:
            new_dp[j] = dp[j - 1] + 2
        else:
            new_dp[j] = max(dp[j], new_dp[j - 1])
    dp = new_dp

print("Optimized Palindromic Trading Pattern Length:", dp[n - 1])
```

### 🧠 Technique:
- Loop from end to start (`i` reversed)
- Use `dp[j - 1]` for diagonal dependency
- Avoid full 2D table with clever array reuse

---

Would you like me to visualize memory savings or benchmark these versions? Or build a module that switches between full and optimized modes depending on input size? You're well on your way to building professional-grade algo engines.

'''