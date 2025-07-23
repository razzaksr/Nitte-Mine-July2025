# Poisonous Plants

'''
There are a number of plants in a garden. Each of the plants has been treated with some amount of pesticide. After each day, if any plant has more pesticide than the plant on its left, being weaker than the left one, it dies.

You are given the initial values of the pesticide in each of the plants. Determine the number of days after which no plant dies, i.e. the time after which there is no plant with more pesticide content than the plant to its left.

Example

 // pesticide levels

Use a -indexed array. On day , plants  and  die leaving . On day , plant  in  dies leaving . There is no plant with a higher concentration of pesticide than the one to its left, so plants stop dying after day .

Function Description
Complete the function poisonousPlants in the editor below.

poisonousPlants has the following parameter(s):

int p[n]: the pesticide levels in each plant
Returns
- int: the number of days until plants no longer die from pesticide

Input Format

The first line contains an integer , the size of the array .
The next line contains  space-separated integers .

Constraints



Sample Input

7
6 5 8 4 7 10 9
Sample Output

2
Explanation

Initially all plants are alive.

Plants = {(6,1), (5,2), (8,3), (4,4), (7,5), (10,6), (9,7)}

Plants[k] = (i,j) => jth plant has pesticide amount = i.

After the 1st day, 4 plants remain as plants 3, 5, and 6 die.

Plants = {(6,1), (5,2), (4,4), (9,7)}

After the 2nd day, 3 plants survive as plant 7 dies.

Plants = {(6,1), (5,2), (4,4)}

Plants stop dying after the 2nd day.
'''

def poisonousPlants(p):
    stack = []
    days = [0] * len(p)
    max_days = 0

    for i in range(len(p)):
        max_day = 0

        # While the stack is not empty and the current plant's pesticide is <= the previous plant
        while stack and p[i] <= p[stack[-1]]:
            stack.pop()

        if stack:
            days[i] = days[stack[-1]] + 1
        else:
            days[i] = 0

        stack.append(i)
        max_days = max(max_days, days[i])

    return max_days


'''
Here’s the **optimized Python solution** for the **"Poisonous Plants"** problem using a **monotonic stack-based approach** with **O(n)** time complexity.

---

### ✅ Problem Recap

Given an array `p` representing pesticide levels:

* Each day, any plant that has **more pesticide than the plant to its left** dies.
* We return the **number of days** until no more plants die.

---

### ✅ Python Code:

```python

```

---

### 🧪 Test Input and Output:

```python
print(poisonousPlants([6, 5, 8, 4, 7, 10, 9]))  # Output: 2
```

---

### 💡 Explanation:

* We use a `stack` to track plants in decreasing pesticide levels.
* For each plant:

  * If it's greater than the one to its left, it will die.
  * We track the **days** it takes for each plant to die using a separate array.
* `max_days` keeps the **longest chain of deaths**, which is the answer.

---

### ✅ Dry Run on Input: `[6, 5, 8, 4, 7, 10, 9]`

* Day 1: `[6, 5, 4, 9]` survive (plants 8, 7, and 10 die)
* Day 2: `[6, 5, 4]` survive (plant 9 dies)
* No further deaths.

**Output: 2**

---

### ⏱️ Time Complexity:

* **O(n)** — each plant is pushed/popped to the stack once.

Let me know if you want a HackerRank-formatted version or this in a `.py` file!

'''