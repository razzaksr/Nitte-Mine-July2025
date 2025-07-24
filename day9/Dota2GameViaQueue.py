from collections import deque

def predictPartyVictory(senate: str) -> str:
    n, radiant, dire = len(senate), deque(), deque()
    # Initialize queues with the indices of R and D
    for i, s in enumerate(senate):
        if s == 'R': radiant.append(i)
        else: dire.append(i)
    # Simulate the rounds
    while radiant and dire:
        r = radiant.popleft()
        d = dire.popleft()
        # The one with the lower index bans the other
        if r < d:
            # Radiant bans Dire, Radiant goes to the back with updated index
            radiant.append(r + n)
        else:
            # Dire bans Radiant, Dire goes to the back with updated index
            dire.append(d + n)
    return "Radiant" if radiant else "Dire"
print(predictPartyVictory("RD"))       # Radiant
print(predictPartyVictory("RDD"))      # Dire
print(predictPartyVictory("RRDDD"))    # Radiant
print(predictPartyVictory("DRRDRDRD")) # Radiant
print(predictPartyVictory("RDRDRDRD")) # Radiant
'''
Here's a **Python solution using `queue`** (`collections.deque`) for the **Dota2 Senate** problem, as per the constraints and logic described:

---

### ✅ **Python Solution (Using Queue)**

```python

```

---

### 🧪 **Test Cases**

You can run the following test cases to validate:

```python

```

---

### 🧠 Explanation:

* Each senator gets added back to the queue with index incremented by `n` if they survive.
* Whoever appears earlier in the queue (based on original index) gets to act first in the round and ban the opposing senator.
* The simulation continues until one party’s queue becomes empty.

Let me know if you need this as a HackerRank-formatted problem statement too.

You're getting `"Radiant"` instead of `"Dire"` for the input `"RRDDD"` likely because of **a misunderstanding in how the queue order affects the banning rights**. Let’s step through what’s happening and explain why:

---

### ✅ Input: `"RRDDD"`

Initial indices:

* Radiant queue: `[0, 1]`
* Dire queue: `[2, 3, 4]`

We simulate rounds as follows:

---

### 🔁 Round 1:

* Radiant index 0 vs Dire index 2 → 0 < 2 → **R bans D**

  * Radiant appends `0 + 5 = 5` → `[1, 5]`
  * Dire becomes `[3, 4]`

---

### 🔁 Round 2:

* Radiant index 1 vs Dire index 3 → 1 < 3 → **R bans D**

  * Radiant appends `1 + 5 = 6` → `[5, 6]`
  * Dire becomes `[4]`

---

### 🔁 Round 3:

* Radiant index 5 vs Dire index 4 → 5 > 4 → **D bans R**

  * Dire appends `4 + 5 = 9` → `[9]`
  * Radiant becomes `[6]`

---

### 🔁 Round 4:

* Radiant index 6 vs Dire index 9 → 6 < 9 → **R bans D**

  * Radiant appends `6 + 5 = 11` → `[11]`
  * Dire becomes `[]`

---

### ❗ Victory:

* Dire queue is empty → Radiant wins

---

### ✅ So, **the correct winner *is* "Radiant"**, not "Dire"\*\* for input `"RRDDD"`.

### 📌 Why you thought Dire should win:

You may have expected Dire to win because they have more senators (3 D vs 2 R), but **R goes first and acts earlier in the index**, which matters more in this simulation.

The simulation **prioritizes turn order**, not majority.

---

### ✅ Summary:

* This output is **correct**.
* Index order defines who acts first.
* Being smart and strategic = ban as early as possible.

Let me know if you'd like a version that shows step-by-step traces in output.


'''