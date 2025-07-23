# Longest Substring Without Repeating Characters
def length_of_longest_substring(s):
    seen = set()
    left = 0
    max_len = 0

    for right in range(len(s)):
        # Shrink window from the left if character is already seen
        while s[right] in seen:
            seen.remove(s[left])
            left += 1

        seen.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len
# Example usage
# print(length_of_longest_substring("abcabcbb"))  # Output: 3
# print(length_of_longest_substring("bbbbb"))     # Output: 1
# print(length_of_longest_substring("pwwkew"))    # Output: 3
# print(length_of_longest_substring("aaaaaaaabcdefghijk"))    # Output: 11
# print(length_of_longest_substring("xyzxyzxyzabcxyz"))   #Output: 6

from collections import Counter
def check_inclusion(s1, s2):
    if len(s1) > len(s2):
        return False
    target = Counter(s1)
    window = Counter()
    for i in range(len(s2)):
        window[s2[i]] += 1
        # Shrink window once it's size exceeds s1
        if i >= len(s1):
            left_char = s2[i - len(s1)]
            window[left_char] -= 1
            if window[left_char] == 0:
                del window[left_char]
        # Compare window with target
        if window == target:
            return True
    return False

# Example usage
# print(check_inclusion("ab", "eidbaooo"))   # Output: True
# print(check_inclusion("ab", "eidboaoo"))   # Output: False
# s1 = "abc"
# s2 = "x" * 5000 + "cab" + "y" * 5000
# print(check_inclusion(s1, s2))  # Output: True
# s1 = "xyz"
# s2 = "a" * 9997 + "zyx"
# print(check_inclusion(s1, s2))  # Output: True
# s1 = "abcde"
# s2 = "aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz"
# print(check_inclusion(s1, s2))  # Output: False

import math

def gcd(a, b):
    return math.gcd(a, b)

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def find_max_product_equivalent(nums):
    max_len = 0

    for start in range(len(nums)):
        prod = 1
        current_gcd = nums[start]
        current_lcm = nums[start]

        for end in range(start, len(nums)):
            num = nums[end]
            prod *= num
            current_gcd = gcd(current_gcd, num)
            current_lcm = lcm(current_lcm, num)

            if prod == current_gcd * current_lcm:
                window_len = end - start + 1
                if window_len > max_len:
                    max_len = window_len

    return max_len
print(find_max_product_equivalent([1, 2, 1, 2, 1, 1, 1]))          # Output: 5
print(find_max_product_equivalent([2, 3, 4, 5, 6]))                # Output: 3
print(find_max_product_equivalent([1, 2, 3, 1, 4, 5, 1]))          # Output: 5
print(find_max_product_equivalent([1, 1, 1, 1, 1, 10]))            # Output: 6
print(find_max_product_equivalent([5, 7, 11, 13, 1, 1]))           # Output: 6

'''
You're right to be skeptical—let's dig into it with precision.

### 🔍 What's Happening in `[5, 7, 11, 13, 1, 1]`?

```python
print(find_max_product_equivalent([5, 7, 11, 13, 1, 1]))  # Output: ?
```

When we run the current implementation, it checks **all subarrays** for the condition:

**prod(subarray) == gcd(subarray) × lcm(subarray)**

The subarray `[5, 7, 11, 13, 1, 1]` as a whole **does satisfy** the condition:
- GCD = 1  
- LCM = 5005  
- Product = 5005  
✅ So `5005 == 1 × 5005`

Hence, the algorithm reports `6` as the longest valid subarray.

### 🤔 Why Did You Expect `2`?

It seems you expected the longest valid subarray to be `[1, 1]` (which does satisfy the condition too), and **perhaps overlooked** that the full array forms a valid window.

But to be sure—let’s break it down:

| Subarray            | Product | GCD | LCM | GCD × LCM | Valid? |
|---------------------|---------|-----|-----|------------|--------|
| `[5, 7]`            | 35      | 1   | 35  | 35         | ✅     |
| `[5, 7, 11]`        | 385     | 1   | 385 | 385        | ✅     |
| `[5, 7, 11, 13]`    | 5005    | 1   | 5005| 5005       | ✅     |
| `[5, 7, 11, 13, 1]` | 5005    | 1   | 5005| 5005       | ✅     |
| `[5, 7, 11, 13, 1, 1]` | 5005 | 1   | 5005| 5005       | ✅     |

So the **entire array** is the valid product-equivalent subarray. Hence, length = **6**

---

### 🧠 Want to narrow it to the first valid chunk?

If you want the solution to **terminate at the first longest group with distinct values**, or apply other rules (like avoid growing window with repeating ones), I can customize the logic with filters or constraints.

Let me know if you’d like a variation!

'''