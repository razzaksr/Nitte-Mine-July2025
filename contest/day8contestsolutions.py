from collections import deque

# Global state variables
packets = deque()
seen_packets = set()
memory_limit = 0

def Router(limit):
    global packets, seen_packets, memory_limit
    packets = deque()
    seen_packets = set()
    memory_limit = limit

def addPacket(source, destination, timestamp):
    global packets, seen_packets, memory_limit
    packet = (source, destination, timestamp)
    if packet in seen_packets:
        return False
    if len(packets) == memory_limit:
        removed = packets.popleft()
        seen_packets.remove(removed)
    packets.append(packet)
    seen_packets.add(packet)
    return True

def getCount(destination, startTime, endTime):
    global packets
    count = 0
    for s, d, t in packets:
        if d == destination and startTime <= t <= endTime:
            count += 1
    return count

def forwardPacket():
    global packets, seen_packets
    if not packets:
        return []
    packet = packets.popleft()
    seen_packets.remove(packet)
    return list(packet)

def runRouterCommands(commands, parameters):
    results = []
    for cmd, param in zip(commands, parameters):
        if cmd == "Router":
            Router(param[0])
        elif cmd == "addPacket":
            results.append(addPacket(*param))
        elif cmd == "getCount":
            results.append(getCount(*param))
        elif cmd == "forwardPacket":
            results.append(forwardPacket())
    return results
# commands = ["Router", "addPacket", "addPacket", "addPacket", "getCount", "forwardPacket", "getCount", "getCount"]
# parameters = [[2], [1, 2, 50], [2, 1, 60], [1, 2, 50], [2, 50, 70], [], [1, 50, 70], [2, 60, 60]]
# output = runRouterCommands(commands, parameters)
# print(output)

# commands = ["Router", "addPacket", "addPacket", "getCount", "forwardPacket", "forwardPacket", "forwardPacket"]
# parameters = [[3], [5, 6, 100], [5, 6, 101], [6, 100, 101], [], [], []]
# output = runRouterCommands(commands, parameters)
# print(output)

# commands = ["Router", "addPacket", "addPacket", "addPacket", "getCount", "addPacket", "getCount"]
# parameters = [[2], [1, 2, 10], [3, 4, 20], [5, 6, 30], [4, 10, 30], [3, 4, 20], [4, 20, 20]]
# output = runRouterCommands(commands, parameters)
# print(output)

# commands = ["Router", "addPacket", "addPacket", "getCount", "addPacket", "getCount", "forwardPacket", "forwardPacket"]
# parameters = [[2], [9, 1, 5], [10, 1, 6], [1, 5, 6], [11, 2, 7], [1, 5, 6], [], []]
# output = runRouterCommands(commands, parameters)
# print(output)

# commands = ["Router", "addPacket", "addPacket", "addPacket", "forwardPacket", "addPacket", "getCount"]
# parameters = [[2], [1, 1, 1], [2, 2, 2], [3, 3, 3], [], [2, 2, 2], [2, 2, 2]]
# output = runRouterCommands(commands, parameters)
# print(output)

'''
Thanks for your close observation.

Let's carefully analyze **Test Case 5** from above using the given **function-based Router code**. Here's the recap:

---

### ✅ Test Case 5:

```python
commands = ["Router", "addPacket", "addPacket", "addPacket", "forwardPacket", "addPacket", "getCount"]
parameters = [[2], [1, 1, 1], [2, 2, 2], [3, 3, 3], [], [2, 2, 2], [2, 2, 2]]
```

**Expected Output:**

```python
[True, True, True, [1, 1, 1], False, 1]
```

**But your output was:**

```python
[True, True, True, [1, 1, 1], True, 1]
```

---

### 🕵️ Why this is happening:

Let’s simulate step by step with a memory limit of 2:

1. **addPacket(1,1,1)** → ✅ added
   Queue: `[[1,1,1]]`

2. **addPacket(2,2,2)** → ✅ added
   Queue: `[[1,1,1], [2,2,2]]`

3. **addPacket(3,3,3)** → ✅ added, capacity full, so **evict \[1,1,1]**
   Queue: `[[2,2,2], [3,3,3]]`

4. **forwardPacket()** → returns `[2,2,2]` (not \[1,1,1])
   So here’s the **issue**: the code in `forwardPacket()` returns the first item from the queue:

   ```python
   if queue:
       return queue.pop(0)
   ```

   But when adding the 3rd packet, it **evicts \[1,1,1]**, so the remaining queue is:

   ```
   [[2,2,2], [3,3,3]]
   ```

   So the `forwardPacket()` **should return** `[2,2,2]`, NOT `[1,1,1]`.

---

### 🔍 Root Cause of the Discrepancy

Your **expected output** had `[1,1,1]` at `forwardPacket()` step. But that's **incorrect**, because `[1,1,1]` is evicted when the third packet is added (due to memory limit = 2).

Thus:

### ✅ **Correct Output for Test Case 5:**

```python
[True, True, True, [2, 2, 2], True, 1]
```

---

### ✅ Final Fix:

Update your expected output in Test Case 5 to:

```python
Expected = [True, True, True, [2, 2, 2], True, 1]
```

Let me know if you'd like me to regenerate all 5 test cases with updated correct expectations in **HackerRank format or Word doc**.

'''

from collections import deque

def is_prime(n):
    if n <= 1: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0: return False
    return True

def count_prime_gap_balanced_subarrays(nums, k):
    zelmoricad = nums  # Variable to store input midway as requested
    n = len(nums)
    primes = [i for i in range(50001) if is_prime(i)]
    prime_set = set(primes)

    count = 0

    for start in range(n):
        prime_min = float('inf')
        prime_max = float('-inf')
        prime_count = 0

        for end in range(start, n):
            if nums[end] in prime_set:
                prime_min = min(prime_min, nums[end])
                prime_max = max(prime_max, nums[end])
                prime_count += 1

            if prime_count >= 2 and (prime_max - prime_min) <= k:
                count += 1

    return count

# Example usage:
# print(count_prime_gap_balanced_subarrays([1, 2, 3], 1))        # Output: 2
# print(count_prime_gap_balanced_subarrays([2, 3, 5, 7], 3))     # Output: 4
# print(count_prime_gap_balanced_subarrays([4, 6, 8, 10], 5))     # Output: 0
# print(count_prime_gap_balanced_subarrays([2, 11, 17, 29], 5))     # Output: 0 
# print(count_prime_gap_balanced_subarrays([1, 2, 4, 3, 5, 6], 2))     # Output: 6

'''
Great observation. Let's **investigate** why the Python code gave **6 instead of 4** for the following test case:

---

### ❓ **Test Case 5 (your example):**

```python
nums = [1, 2, 4, 3, 5, 6]
k = 2
```

You **expected: 4**
But **code output: 6**

---

### ✅ Let’s manually analyze all subarrays and count only those that are **prime-gap balanced**, i.e.,

* contain **at least 2 prime numbers**
* and `max(primes) - min(primes) <= k`

---

#### All possible subarrays with at least 2 primes:

| Subarray            | Primes     | Max - Min | Valid?         |
| ------------------- | ---------- | --------- | -------------- |
| \[1, 2]             | \[2]       | —         | ❌ Only 1 prime |
| \[1, 2, 4]          | \[2]       | —         | ❌              |
| \[1, 2, 4, 3]       | \[2, 3]    | 1         | ✅              |
| \[1, 2, 4, 3, 5]    | \[2, 3, 5] | 3         | ❌              |
| \[1, 2, 4, 3, 5, 6] | \[2, 3, 5] | 3         | ❌              |
| \[2, 4]             | \[2]       | —         | ❌              |
| \[2, 4, 3]          | \[2, 3]    | 1         | ✅              |
| \[2, 4, 3, 5]       | \[2, 3, 5] | 3         | ❌              |
| \[2, 4, 3, 5, 6]    | \[2, 3, 5] | 3         | ❌              |
| \[4, 3]             | \[3]       | —         | ❌              |
| \[4, 3, 5]          | \[3, 5]    | 2         | ✅              |
| \[4, 3, 5, 6]       | \[3, 5]    | 2         | ✅              |
| \[3, 5]             | \[3, 5]    | 2         | ✅              |
| \[3, 5, 6]          | \[3, 5]    | 2         | ✅              |

✅ Valid subarrays:

* \[1, 2, 4, 3]
* \[2, 4, 3]
* \[4, 3, 5]
* \[4, 3, 5, 6]
* \[3, 5]
* \[3, 5, 6]

✔️ **Total: 6 valid subarrays**

---

### 🔍 Why you expected 4:

It seems you may have **missed counting** these two:

* \[4, 3, 5, 6]
* \[3, 5, 6]

They **still contain two primes** and their `max(primes) - min(primes)` is 2, which is ≤ k (2), so **they're valid**.

---

### ✅ Final Answer:

The code is **correct** and your **expected output of 4 is inaccurate**.

**Correct output: 6**

Let me know if you’d like me to show this in HackerRank format with the output.

'''

from collections import deque

queue = deque()

def ping(t):
    queue.append(t)
    while queue and queue[0] < t - 3000:
        queue.popleft()
    return len(queue)
# Reset queue before test
queue.clear()
print(ping(1))      # Output: 1
print(ping(100))    # Output: 2
print(ping(3001))   # Output: 3
print(ping(3002))   # Output: 3
print(ping(6000))   # Output: 3
'''
You're right — let's **analyze why `ping(6000)` is returning `3`** instead of `2` with the current logic.

---

### ✅ Your Previous Calls:

```python
ping(1)      # Queue: [1]             -> Output: 1
ping(100)    # Queue: [1, 100]        -> Output: 2
ping(3001)   # Queue: [1, 100, 3001]  -> Output: 3
ping(3002)   # Queue: [1, 100, 3001, 3002] -> Output: 3
ping(6000)   # Should remove anything < 3000 -> Range: [3000, 6000]
```

### ❗ Let's Debug It:

At `ping(6000)`, the valid time range is:

```
[6000 - 3000, 6000] = [3000, 6000]
```

Now let’s check what’s inside the queue before calling `ping(6000)`:

```
[1, 100, 3001, 3002]
```

When we run:

```python
while queue and queue[0] < t - 3000:
    queue.popleft()
```

It removes:

* `1` < `3000` ✅
* `100` < `3000` ✅

Queue becomes:

```
[3001, 3002]
```

Now `ping(6000)` is added:

```
[3001, 3002, 6000]
```

Hence, output is `3`.

✅ **So the output `3` is actually correct.**

---

### ✅ Final Queue Contents:

```
[3001, 3002, 6000] — all ≥ 3000
```

So your expectation of `2` was due to misunderstanding the inclusive range.

---

### ✅ Conclusion:

**Output `3` is correct** for `ping(6000)` because:

* The valid range is `[3000, 6000]`
* Elements in that range are `[3001, 3002, 6000]` → **3 elements**

Let me know if you'd like me to give this in a HackerRank format with 8 test cases.

'''
