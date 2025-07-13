"""
Optimized Python Solutions for String Problems
==============================================
"""

# 1. Longest Substring Without Repeating Characters
def length_of_longest_substring(s):
    """
    Find length of longest substring without repeating characters.
    Time: O(n), Space: O(min(m,n)) where m is charset size
    """
    char_index = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Example usage:
# Input: s = "abcabcbb"
# Output: 3 (substring "abc")
print("Longest Substring:", length_of_longest_substring("abcabcbb"))


# 2. Valid Anagram
def is_anagram(s, t):
    """
    Check if two strings are anagrams.
    Time: O(n), Space: O(1) - assuming only lowercase letters
    """
    if len(s) != len(t):
        return False
    
    char_count = {}
    
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in t:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] == 0:
            del char_count[char]
    
    return len(char_count) == 0

# Example usage:
# Input: s = "anagram", t = "nagaram"
# Output: True
print("Valid Anagram:", is_anagram("anagram", "nagaram"))


# 3. Group Anagrams
def group_anagrams(strs):
    """
    Group strings that are anagrams of each other.
    Time: O(n * m log m), Space: O(n * m)
    where n is number of strings, m is average string length
    """
    anagram_groups = {}
    
    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        if key not in anagram_groups:
            anagram_groups[key] = []
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())

# Example usage:
# Input: strs = ["eat","tea","tan","ate","nat","bat"]
# Output: [["eat","tea","ate"],["tan","nat"],["bat"]]
print("Group Anagrams:", group_anagrams(["eat","tea","tan","ate","nat","bat"]))


# 4. Longest Palindromic Substring
def longest_palindromic_substring(s):
    """
    Find longest palindromic substring.
    Time: O(n²), Space: O(1)
    """
    if not s:
        return ""
    
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = ""
    
    for i in range(len(s)):
        # Odd length palindromes
        palindrome1 = expand_around_center(i, i)
        # Even length palindromes
        palindrome2 = expand_around_center(i, i + 1)
        
        current_longest = palindrome1 if len(palindrome1) > len(palindrome2) else palindrome2
        if len(current_longest) > len(longest):
            longest = current_longest
    
    return longest

# Example usage:
# Input: s = "babad"
# Output: "bab" or "aba"
print("Longest Palindromic Substring:", longest_palindromic_substring("babad"))


# 5. Palindromic Substrings
def count_palindromic_substrings(s):
    """
    Count number of palindromic substrings.
    Time: O(n²), Space: O(1)
    """
    def expand_around_center(left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    total_count = 0
    
    for i in range(len(s)):
        # Odd length palindromes
        total_count += expand_around_center(i, i)
        # Even length palindromes
        total_count += expand_around_center(i, i + 1)
    
    return total_count

# Example usage:
# Input: s = "abc"
# Output: 3 ("a", "b", "c")
print("Palindromic Substrings:", count_palindromic_substrings("abc"))


# 6. Minimum Window Substring
def min_window_substring(s, t):
    """
    Find minimum window substring containing all characters of t.
    Time: O(|s| + |t|), Space: O(|s| + |t|)
    """
    if not s or not t:
        return ""
    
    # Count characters in t
    dict_t = {}
    for char in t:
        dict_t[char] = dict_t.get(char, 0) + 1
    
    required = len(dict_t)
    left = right = 0
    formed = 0
    
    window_counts = {}
    
    # ans = (window length, left, right)
    ans = float('inf'), None, None
    
    while right < len(s):
        # Add character from right to window
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1
        
        # If character frequency matches required count
        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1
        
        # Try to contract window
        while left <= right and formed == required:
            char = s[left]
            
            # Save smallest window
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            
            # Remove from left
            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1
            
            left += 1
        
        right += 1
    
    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]

# Example usage:
# Input: s = "ADOBECODEBANC", t = "ABC"
# Output: "BANC"
print("Minimum Window Substring:", min_window_substring("ADOBECODEBANC", "ABC"))


# 7. Isomorphic Strings
def is_isomorphic(s, t):
    """
    Check if two strings are isomorphic.
    Time: O(n), Space: O(1) - assuming ASCII charset
    """
    if len(s) != len(t):
        return False
    
    s_to_t = {}
    t_to_s = {}
    
    for i in range(len(s)):
        char_s, char_t = s[i], t[i]
        
        if char_s in s_to_t:
            if s_to_t[char_s] != char_t:
                return False
        else:
            s_to_t[char_s] = char_t
        
        if char_t in t_to_s:
            if t_to_s[char_t] != char_s:
                return False
        else:
            t_to_s[char_t] = char_s
    
    return True

# Example usage:
# Input: s = "egg", t = "add"
# Output: True
print("Isomorphic Strings:", is_isomorphic("egg", "add"))


# 8. String Compression
def string_compression(chars):
    """
    Compress string array in-place.
    Time: O(n), Space: O(1)
    """
    write = 0
    i = 0
    
    while i < len(chars):
        char = chars[i]
        count = 1
        
        # Count consecutive characters
        while i + count < len(chars) and chars[i + count] == char:
            count += 1
        
        # Write character
        chars[write] = char
        write += 1
        
        # Write count if > 1
        if count > 1:
            for digit in str(count):
                chars[write] = digit
                write += 1
        
        i += count
    
    return write

# Example usage:
# Input: chars = ["a","a","b","b","c","c","c"]
# Output: 6, chars = ["a","2","b","2","c","3"]
test_chars = ["a","a","b","b","c","c","c"]
length = string_compression(test_chars)
print("String Compression:", length, test_chars[:length])


# 9. Multiply Strings
def multiply_strings(num1, num2):
    """
    Multiply two non-negative integers represented as strings.
    Time: O(m * n), Space: O(m + n)
    """
    if num1 == "0" or num2 == "0":
        return "0"
    
    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    
    # Reverse both numbers
    num1 = num1[::-1]
    num2 = num2[::-1]
    
    # Multiply digit by digit
    for i in range(m):
        for j in range(n):
            result[i + j] += int(num1[i]) * int(num2[j])
            # Handle carry
            result[i + j + 1] += result[i + j] // 10
            result[i + j] %= 10
    
    # Convert to string
    result = result[::-1]
    start = 0
    while start < len(result) and result[start] == 0:
        start += 1
    
    return ''.join(map(str, result[start:]))

# Example usage:
# Input: num1 = "2", num2 = "3"
# Output: "6"
print("Multiply Strings:", multiply_strings("123", "456"))


# 10. Implement strStr()
def str_str(haystack, needle):
    """
    Find first occurrence of needle in haystack.
    Time: O(n * m), Space: O(1)
    """
    if not needle:
        return 0
    
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    
    return -1

# Example usage:
# Input: haystack = "hello", needle = "ll"
# Output: 2
print("strStr():", str_str("hello", "ll"))


# 11. Rabin-Karp Implementation
def rabin_karp(text, pattern):
    """
    String matching using Rabin-Karp algorithm.
    Time: O(n + m) average, O(n * m) worst case
    Space: O(1)
    """
    if not pattern:
        return 0
    
    n, m = len(text), len(pattern)
    if m > n:
        return -1
    
    # Hash parameters
    base = 256
    mod = 10**9 + 7
    
    # Calculate hash of pattern and first window
    pattern_hash = 0
    window_hash = 0
    h = 1
    
    # h = base^(m-1) % mod
    for i in range(m - 1):
        h = (h * base) % mod
    
    # Calculate hash of pattern and first window
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % mod
        window_hash = (base * window_hash + ord(text[i])) % mod
    
    # Slide the pattern over text
    for i in range(n - m + 1):
        # Check if hashes match
        if pattern_hash == window_hash:
            # Check character by character
            if text[i:i + m] == pattern:
                return i
        
        # Calculate hash for next window
        if i < n - m:
            window_hash = (base * (window_hash - ord(text[i]) * h) + ord(text[i + m])) % mod
            if window_hash < 0:
                window_hash += mod
    
    return -1

# Example usage:
# Input: text = "ABABDABACDABABCABCABCABCABC", pattern = "ABABCABCABCABC"
# Output: 15
print("Rabin-Karp:", rabin_karp("ABABDABACDABABCABCABCABCABC", "ABABCABCABCABC"))


# 12. KMP Algorithm
def kmp_search(text, pattern):
    """
    String matching using KMP algorithm.
    Time: O(n + m), Space: O(m)
    """
    if not pattern:
        return 0
    
    def compute_lps(pattern):
        """Compute Longest Proper Prefix which is also Suffix array"""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    n, m = len(text), len(pattern)
    if m > n:
        return -1
    
    lps = compute_lps(pattern)
    
    i = j = 0  # i for text, j for pattern
    
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            return i - j
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return -1

# Example usage:
# Input: text = "ABABDABACDABABCABCABCABCABC", pattern = "ABABCABCABCABC"
# Output: 15
print("KMP Search:", kmp_search("ABABDABACDABABCABCABCABCABC", "ABABCABCABCABC"))


# 13. Roman to Integer
def roman_to_int(s):
    """
    Convert Roman numeral to integer.
    Time: O(n), Space: O(1)
    """
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    total = 0
    prev_value = 0
    
    for char in reversed(s):
        value = roman_values[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    
    return total

# Example usage:
# Input: s = "MCMXC"
# Output: 1990
print("Roman to Integer:", roman_to_int("MCMXC"))


# 14. Integer to Roman
def int_to_roman(num):
    """
    Convert integer to Roman numeral.
    Time: O(1), Space: O(1)
    """
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    
    result = ""
    
    for i in range(len(values)):
        count = num // values[i]
        if count:
            result += symbols[i] * count
            num -= values[i] * count
    
    return result

# Example usage:
# Input: num = 1994
# Output: "MCMXCIV"
print("Integer to Roman:", int_to_roman(1994))


# 15. Decode Ways
def decode_ways(s):
    """
    Count number of ways to decode a digit string.
    Time: O(n), Space: O(1)
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    prev2 = prev1 = 1
    
    for i in range(1, n):
        current = 0
        
        # Single digit
        if s[i] != '0':
            current += prev1
        
        # Two digits
        two_digit = int(s[i-1:i+1])
        if 10 <= two_digit <= 26:
            current += prev2
        
        prev2, prev1 = prev1, current
    
    return prev1

# Example usage:
# Input: s = "226"
# Output: 3 ("2-2-6", "22-6", "2-26")
print("Decode Ways:", decode_ways("226"))


# 16. Zigzag Conversion
def zigzag_conversion(s, num_rows):
    """
    Convert string to zigzag pattern.
    Time: O(n), Space: O(n)
    """
    if num_rows == 1 or num_rows >= len(s):
        return s
    
    rows = [''] * num_rows
    current_row = 0
    going_down = False
    
    for char in s:
        rows[current_row] += char
        
        if current_row == 0 or current_row == num_rows - 1:
            going_down = not going_down
        
        current_row += 1 if going_down else -1
    
    return ''.join(rows)

# Example usage:
# Input: s = "PAYPALISHIRING", numRows = 3
# Output: "PAHNAPLSIIGYIR"
print("Zigzag Conversion:", zigzag_conversion("PAYPALISHIRING", 3))


# 17. Reverse Words in a String
def reverse_words(s):
    """
    Reverse words in a string.
    Time: O(n), Space: O(n)
    """
    # Split by whitespace and filter empty strings
    words = s.split()
    # Reverse and join
    return ' '.join(reversed(words))

# Example usage:
# Input: s = "  hello world  "
# Output: "world hello"
print("Reverse Words:", reverse_words("  hello world  "))


# 18. Longest Common Prefix
def longest_common_prefix(strs):
    """
    Find longest common prefix among array of strings.
    Time: O(S) where S is sum of all characters, Space: O(1)
    """
    if not strs:
        return ""
    
    # Find minimum length
    min_len = min(len(s) for s in strs)
    
    for i in range(min_len):
        char = strs[0][i]
        for s in strs[1:]:
            if s[i] != char:
                return strs[0][:i]
    
    return strs[0][:min_len]

# Example usage:
# Input: strs = ["flower","flow","flight"]
# Output: "fl"
print("Longest Common Prefix:", longest_common_prefix(["flower","flow","flight"]))


# 19. Check if One String is Rotation of Another
def is_rotation(s1, s2):
    """
    Check if s2 is a rotation of s1.
    Time: O(n), Space: O(n)
    """
    if len(s1) != len(s2):
        return False
    
    # s2 should be substring of s1 + s1
    return s2 in s1 + s1

# Example usage:
# Input: s1 = "abcde", s2 = "cdeab"
# Output: True
print("Is Rotation:", is_rotation("abcde", "cdeab"))


# 20. Count and Say
def count_and_say(n):
    """
    Generate nth term of count-and-say sequence.
    Time: O(2^n), Space: O(2^n)
    """
    if n == 1:
        return "1"
    
    def get_next_sequence(s):
        result = ""
        i = 0
        
        while i < len(s):
            count = 1
            char = s[i]
            
            # Count consecutive characters
            while i + count < len(s) and s[i + count] == char:
                count += 1
            
            result += str(count) + char
            i += count
        
        return result
    
    current = "1"
    for _ in range(n - 1):
        current = get_next_sequence(current)
    
    return current

# Example usage:
# Input: n = 4
# Output: "1211"
# Sequence: "1" -> "11" -> "21" -> "1211"
print("Count and Say:", count_and_say(4))


# Test all solutions
if __name__ == "__main__":
    print("\n=== String Problems Solutions ===")
    
    # Additional test cases
    print("\nAdditional Test Cases:")
    print("Longest Substring (pwwkew):", length_of_longest_substring("pwwkew"))
    print("Group Anagrams (empty):", group_anagrams([""]))
    print("Palindromic Substring (aaa):", longest_palindromic_substring("aaa"))
    print("Min Window (a, aa):", min_window_substring("a", "aa"))
    print("Multiply (999, 999):", multiply_strings("999", "999"))
    print("Roman to Int (LVIII):", roman_to_int("LVIII"))
    print("Int to Roman (58):", int_to_roman(58))
    print("Decode Ways (12):", decode_ways("12"))
    print("Count and Say (1):", count_and_say(1))
    
    print("\n=== All Solutions Optimized ===")
    print("Each solution uses the most efficient algorithm for its problem class.")
