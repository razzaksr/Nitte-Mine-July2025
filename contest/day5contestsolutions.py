# Problem 1: Longest Increasing Subsequence
nums = list(map(int, input().split()))
n = len(nums)

# dp[i] represents length of LIS ending at index i
dp = [1] * n

for i in range(1, n):
    for j in range(i):
        if nums[j] < nums[i]:
            dp[i] = max(dp[i], dp[j] + 1)

print(max(dp))


# Problem 2: Maximum Sum Circular Subarray
nums = list(map(int, input().split()))
n = len(nums)

# Case 1: Maximum subarray is non-circular (standard Kadane's)
max_ending_here = max_so_far = nums[0]
for i in range(1, n):
    max_ending_here = max(nums[i], max_ending_here + nums[i])
    max_so_far = max(max_so_far, max_ending_here)

max_kadane = max_so_far

# Case 2: Maximum subarray is circular
# This equals total_sum - minimum_subarray
min_ending_here = min_so_far = nums[0]
for i in range(1, n):
    min_ending_here = min(nums[i], min_ending_here + nums[i])
    min_so_far = min(min_so_far, min_ending_here)

min_kadane = min_so_far
total_sum = sum(nums)

# If all elements are negative, circular case would be empty
if total_sum == min_kadane:
    print(max_kadane)
else:
    # Return maximum of non-circular and circular cases
    print(max(max_kadane, total_sum - min_kadane))


# Problem 3: Minimum Number of Perfect Squares
n = int(input())

# dp[i] represents minimum squares needed to sum to i
dp = [float('inf')] * (n + 1)
dp[0] = 0

# Pre-compute all perfect squares <= n
squares = []
i = 1
while i * i <= n:
    squares.append(i * i)
    i += 1

# Fill dp array
for i in range(1, n + 1):
    for square in squares:
        if square > i:
            break
        dp[i] = min(dp[i], dp[i - square] + 1)

print(dp[n])