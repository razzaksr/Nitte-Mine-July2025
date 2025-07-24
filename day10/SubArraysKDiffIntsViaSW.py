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

# Example usage
nums1 = [1, 2, 1, 2, 3]
k1 = 2
print(subarrays_with_k_distinct(nums1, k1))  # Output: 7

nums2 = [1, 2, 1, 3, 4]
k2 = 3
print(subarrays_with_k_distinct(nums2, k2))  # Output: 3