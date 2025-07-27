from collections import defaultdict
# Count subarrays with at most k different integers
def count_subarrays_at_most_k(nums, k):
    count = defaultdict(int)
    left, total = 0, 0
    for right in range(len(nums)):
        count[nums[right]] += 1
        while len(count) > k:
            count[nums[left]] -= 1
            if count[nums[left]] == 0: del count[nums[left]]
            left += 1
        total += right - left + 1
    return total
# Count subarrays with exactly k different integers
def subarrays_with_k_distinct(nums, k):
    return count_subarrays_at_most_k(nums, k) - count_subarrays_at_most_k(nums, k - 1)

print(subarrays_with_k_distinct([1, 2, 1, 2, 3], 2))  # Output: 7
print(subarrays_with_k_distinct([1, 2, 1, 3, 4], 3))  # Output: 3

'''
Great observation, S! You're totally right to question why `[1,2,3]` isn't listed in Example 1. Let’s take a closer look at the rules and walk through why it’s **excluded** even though it looks promising at first glance.

### 🤔 What's the Rule Again?
We need **subarrays** that contain **exactly `k` distinct integers** — no more, no less.

In Example 1:
- `nums = [1,2,1,2,3]`
- `k = 2`
- The subarray `[1,2,3]` contains **3 distinct integers** (1, 2, and 3)

### ❌ Why `[1,2,3]` Is Invalid Here
- The problem defines a "good" subarray as one with **_exactly_** `k` distinct elements.
- Since `k = 2`, any subarray with **more than 2 distinct elements**, like `[1,2,3]`, is **not a valid result**.
- So although `[1,2,3]` is indeed a contiguous subarray, it’s ruled out because it violates the constraint of having **exactly** 2 distinct numbers.

### ✅ Bonus Insight
Here’s a little breakdown to reinforce your understanding:

- `nums = [1,2,1,2,3]`
- `k = 2`

| Subarray       | Distinct Elements | Valid (k = 2)? |
|----------------|-------------------|----------------|
| `[1,2]`        | 2                 | ✅              |
| `[2,1]`        | 2                 | ✅              |
| `[2,3]`        | 2                 | ✅              |
| `[1,2,3]`      | 3                 | ❌              |
| `[1,2,1,2]`    | 2                 | ✅              |

If you'd like, I can help you write a clean and efficient sliding window algorithm to compute the total number of "good subarrays" for any array and `k`. Want to tackle it together?

'''