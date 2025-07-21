from collections import deque

def minKBitFlips(nums, k):
    n = len(nums)
    flip = 0
    queue = deque()
    res = 0

    for i in range(n):
        # Remove expired flip positions
        if queue and queue[0] + k <= i:
            queue.popleft()
            flip ^= 1  # Flip status toggles when flip range expires

        # If current bit equals flip parity, we need a new flip
        if nums[i] == flip:
            if i + k > n:
                return -1  # Not enough space to flip
            queue.append(i)
            flip ^= 1
            res += 1

    return res
nums = [0,0,0,1,0,1,1,0]
k = 3
print(minKBitFlips(nums, k))
nums = [0,1,0]
k = 1
print(minKBitFlips(nums, k))
nums = [1,1,0]
k = 2
print(minKBitFlips(nums, k))