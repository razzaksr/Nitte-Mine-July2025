from collections import deque

def minSumByAndSubarrays(nums, andValues):
    n = len(nums)
    m = len(andValues)

    # Initialize dp: dp[i] = minimum sum if we consider first i elements of nums
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # Base case: no elements used, cost is 0

    for i in range(m):
        new_dp = [float('inf')] * (n + 1)

        # We use a deque to store tuples (start_index, current_and)
        for j in range(i, n):
            cur_and = nums[j]
            for k in range(j, i - 1, -1):
                cur_and &= nums[k]
                if cur_and < andValues[i]:  # Early exit if AND too small
                    break
                if cur_and == andValues[i]:
                    new_dp[j + 1] = min(new_dp[j + 1], dp[k] + nums[j])
        dp = new_dp

    return dp[n] if dp[n] != float('inf') else -1
# Example 1
print(minSumByAndSubarrays([1,4,3,3,2], [0,3,3,2]))  # Output: 12

# Example 2
print(minSumByAndSubarrays([2,3,5,7,7,7,5], [0,7,5]))  # Output: 17

# Example 3
print(minSumByAndSubarrays([1,2,3,4], [2]))  # Output: -1
