# Example inputs
# weights = [1, 2, 3, 2]
# values =  [10, 20, 30, 40]
weights = [1,2,3]
values = [10,15,40]
capacity = 5

n = len(weights)

# Initialize DP table: (n+1) rows and (capacity+1) columns
dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

# Build the table
for i in range(1, n + 1):
    for w in range(capacity + 1):
        # print(w,"and",(i-1),weights[i-1])
        if weights[i - 1] <= w:
            dp[i][w] = max(
                dp[i - 1][w],                        # exclude item
                dp[i - 1][w - weights[i - 1]] + values[i - 1]  # include item
            )
        else:
            dp[i][w] = dp[i - 1][w]
        print(dp[i][w])

# Final result
print("Maximum value:", dp[n][capacity])

'''
Let’s tackle the classic **0/1 Knapsack problem** in Python—flat style again: no `class`, no nested `def`, just a single execution flow. I’ll start with a clean procedural solution, then give you five varied test cases, and finally walk you through one of them step by step with the DP table filled out.

---

## 🎒 Problem Summary: 0/1 Knapsack

You're given:
- A list of item **weights**
- A list of corresponding item **values**
- A maximum **capacity** of the knapsack

Objective: Maximize the **total value** that fits into the knapsack without exceeding its capacity, assuming **each item is either picked entirely or not at all** (hence "0/1").

---

## 🐍 Python Implementation

```python
# Example inputs
weights = [1, 2, 3, 2]
values =  [10, 20, 30, 40]
capacity = 5

n = len(weights)

# Initialize DP table: (n+1) rows and (capacity+1) columns
dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

# Build the table
for i in range(1, n + 1):
    for w in range(capacity + 1):
        if weights[i - 1] <= w:
            dp[i][w] = max(
                dp[i - 1][w],                        # exclude item
                dp[i - 1][w - weights[i - 1]] + values[i - 1]  # include item
            )
        else:
            dp[i][w] = dp[i - 1][w]

# Final result
print("Maximum value:", dp[n][capacity])
```

---

## 🧪 Test Cases

| Test Case | Weights     | Values      | Capacity | Expected Max Value |
|-----------|-------------|-------------|----------|---------------------|
| 1         | `[1,2,3]`   | `[10,15,40]`| `5`      | `55` (`2nd & 3rd item`) |
| 2         | `[1,3,4,5]` | `[10,40,50,70]` | `8` | `110` (`3rd & 4th item`) |
| 3         | `[1,2,3,2]` | `[10,20,30,40]` | `5` | `70` (`2nd & 4th item`) |
| 4         | `[2,3,4]`   | `[12,18,25]` | `6`     | `30` (`1st & 2nd item`) |
| 5         | `[4,5,1]`   | `[60,50,20]` | `5`     | `70` (`1st or 3rd & 2nd`) |

---

## 🔍 Detailed Walkthrough of Test Case 3
**Inputs**:
- `weights = [1, 2, 3, 2]`
- `values  = [10, 20, 30, 40]`
- `capacity = 5`

### Step-by-step DP Table Construction:

We’ll fill a table `dp[i][w]` that represents the maximum value using first `i` items for capacity `w`.

| i \ w | 0 | 1 | 2 | 3 | 4 | 5 |
|------|---|---|---|---|---|---|
| 0    | 0 | 0 | 0 | 0 | 0 | 0 |
| 1    | 0 |10 |10 |10 |10 |10 |
| 2    | 0 |10 |20 |30 |30 |30 |
| 3    | 0 |10 |20 |30 |40 |50 |
| 4    | 0 |10 |40 |50 |60 |70 |

### How it builds:
- `i=1`: Only item 1, fits into any capacity ≥1 → value 10
- `i=2`: Now item 2 added → for capacity 2+ it fits, gives better value
- `i=3`: Item 3 has weight 3/value 30 → combo starts working
- `i=4`: Item 4 (weight 2/value 40) **really boosts** the combos, especially `dp[4][5] = 70` by combining item 2 and 4.

📦 So, **maximum value = 70** for capacity 5 by picking item 2 (20) + item 4 (40) + item 1 (if capacity allowed—here it's tight).

---

Would you like to extract the actual items selected or visualize the inclusion path in a challenge-style output format?

'''