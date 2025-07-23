from collections import deque

def deckRevealedIncreasing(deck):
    n = len(deck)
    index_queue = deque(range(n))
    result = [0] * n
    
    for card in sorted(deck):
        i = index_queue.popleft()
        result[i] = card
        if index_queue:
            index_queue.append(index_queue.popleft())
    
    return result
print(deckRevealedIncreasing([17,13,11,2,3,5,7]))  # [2, 13, 3, 11, 5, 17, 7]
print(deckRevealedIncreasing([1, 1000]))           # [1, 1000]
print(deckRevealedIncreasing([1]))                 # [1]
print(deckRevealedIncreasing([3,1,4,2]))            # [1, 3, 2, 4]
print(deckRevealedIncreasing([10, 5, 1, 7]))         # [1, 7, 5, 10]
'''
Here is the Python solution using a **queue** (via `collections.deque`) to solve the **"Reveal Cards In Increasing Order"** problem:

---

### ✅ **Python Solution**

```python

```

---

### 🧠 **How It Works:**

1. **Sort the input deck** – this is the order the cards must be revealed in.
2. Create a `deque` of indices `[0, 1, 2, ..., n-1]` to simulate positions.
3. For each card in the sorted deck:

   * Place the card at the current index.
   * Simulate the rotation (put the next top card to the bottom) by moving the next index to the end.

---

### 📌 **Test Cases**

```python

```

---

### 📝 **Explanation of Test Case 1:**

For `deck = [17,13,11,2,3,5,7]`:

* Sorted deck = `[2,3,5,7,11,13,17]`
* Simulation arranges these into indices using queue logic to match the reveal pattern.

---

Let me know if you want the HackerRank-formatted version of this problem with input/output format and constraints.

Great question!

You're getting `[1, 3, 2, 4]` instead of `[1, 4, 2, 3]` because **your expected output is incorrect** — `[1, 3, 2, 4]` **is actually the correct output** according to the logic of the problem “Reveal Cards in Increasing Order.”

Let me walk you through how the algorithm works for the input `[3,1,4,2]`:

---

### ✅ Step-by-step Simulation

**Input deck:** `[3, 1, 4, 2]`

1. **Sort the deck**:
   Sorted deck = `[1, 2, 3, 4]`

2. **Use a queue to simulate indices of the result**:
   We’ll fill positions in an initially empty result list of the same length.

   Initial indices queue: `deque([0, 1, 2, 3])`
   Empty result: `[None, None, None, None]`

3. **Place each sorted card at the correct index following the rules**:

   * Pop index 0 → place 1 → result = `[1, None, None, None]`
     Move index 1 to the back → `deque([2, 3, 1])`

   * Pop index 2 → place 2 → result = `[1, None, 2, None]`
     Move index 3 to the back → `deque([1, 3])`

   * Pop index 1 → place 3 → result = `[1, 3, 2, None]`
     Move index 3 to the back → `deque([3])`

   * Pop index 3 → place 4 → result = `[1, 3, 2, 4]`

---

### ✅ Final Result:

```python
[1, 3, 2, 4]
```

### 🔁 Verification by revealing:

Reveal order:

* Reveal 1 → move 3 to bottom → \[2, 4, 3]
* Reveal 2 → move 4 to bottom → \[3, 4]
* Reveal 3 → move 4 to bottom → \[4]
* Reveal 4 → Done

**Revealed order: \[1, 2, 3, 4]** — which is increasing ✔️

---

### 🔎 Why `[1, 4, 2, 3]` is **not** valid:

Reveal simulation:

* Reveal 1 → move 4 to bottom → \[2, 3, 4]
* Reveal 2 → move 3 to bottom → \[4, 3]
* Reveal 4 → move 3 to bottom → \[3]
* Reveal 3 → Done

Revealed: `[1, 2, 4, 3]` ❌ not sorted.

---

### ✅ Correct Code:

Here it is again for your reference:

```python
from collections import deque

def deckRevealedIncreasing(deck):
    n = len(deck)
    index_queue = deque(range(n))
    result = [0] * n
    for card in sorted(deck):
        i = index_queue.popleft()
        result[i] = card
        if index_queue:
            index_queue.append(index_queue.popleft())
    return result
```

---

Let me know if you want a version that simulates the actual revealing as well!


'''