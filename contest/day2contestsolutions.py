# Problem 1: Roman Numeral Formatter

def int_to_roman(num):
    """
    Convert integer to Roman numeral using greedy approach.
    Time Complexity: O(1) - constant number of operations
    Space Complexity: O(1) - constant space
    """
    # Define values and corresponding Roman numerals in descending order
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    numerals = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    
    result = ""
    
    for i in range(len(values)):
        count = num // values[i]
        if count:
            result += numerals[i] * count
            num -= values[i] * count
    
    return result


# Problem 2: Isomorphic Identifier Mapping

def are_isomorphic(s, t):
    """
    Check if two strings are isomorphic using bidirectional mapping.
    Time Complexity: O(n) where n is length of strings
    Space Complexity: O(1) - at most 26 characters in each dictionary
    """
    if len(s) != len(t):
        return False
    
    # Create mapping dictionaries for both directions
    s_to_t = {}
    t_to_s = {}
    
    for i in range(len(s)):
        char_s = s[i]
        char_t = t[i]
        
        # Check if mapping from s to t is consistent
        if char_s in s_to_t:
            if s_to_t[char_s] != char_t:
                return False
        else:
            s_to_t[char_s] = char_t
        
        # Check if mapping from t to s is consistent
        if char_t in t_to_s:
            if t_to_s[char_t] != char_s:
                return False
        else:
            t_to_s[char_t] = char_s
    
    return True


# Problem 3: IPv4 Address Validator

import re

def is_valid_ipv4(ip):
    """
    Validate IPv4 address using regex.
    Time Complexity: O(1) - constant time regex matching
    Space Complexity: O(1) - constant space
    """
    # Regex pattern for valid IPv4 address
    # ^: start of string
    # (?:...): non-capturing group
    # 25[0-5]: 250-255
    # 2[0-4][0-9]: 200-249
    # 1[0-9][0-9]: 100-199
    # [1-9][0-9]: 10-99
    # [0-9]: 0-9
    # \.: literal dot
    # {3}: exactly 3 times for first three octets
    # $: end of string
    
    pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$'
    
    return bool(re.match(pattern, ip))

def is_valid_ipv4_alternative(ip):
    """
    Alternative validation without complex regex - splits and validates each part.
    Time Complexity: O(1) - constant time operations
    Space Complexity: O(1) - constant space
    """
    parts = ip.split('.')
    
    # Must have exactly 4 parts
    if len(parts) != 4:
        return False
    
    for part in parts:
        # Check if part is empty or not a digit
        if not part or not part.isdigit():
            return False
        
        # Check for leading zeros (except for "0")
        if len(part) > 1 and part[0] == '0':
            return False
        
        # Check if number is in valid range
        num = int(part)
        if num < 0 or num > 255:
            return False
    
    return True


# Test functions for all problems

def test_roman_numeral():
    """Test Roman numeral formatter"""
    print("Testing Roman Numeral Formatter:")
    
    test_cases = [1994, 58, 4, 9, 3999, 2024, 1, 1000, 944, 1666]
    expected = ["MCMXCIV", "LVIII", "IV", "IX", "MMMCMXCIX", "MMXXIV", "I", "M", "CMXLIV", "MDCLXVI"]
    
    for i, test in enumerate(test_cases):
        result = int_to_roman(test)
        print(f"Test {i+1}: {result} - {'PASS' if result == expected[i] else 'FAIL'}")

def test_isomorphic():
    """Test isomorphic identifier mapping"""
    print("\nTesting Isomorphic Identifier Mapping:")
    
    test_cases = [
        ("egg", "add"),
        ("foo", "bar"),
        ("paper", "title"),
        ("badc", "baba"),
        ("abc", "def"),
        ("aab", "xyz"),
        ("ab", "aa"),
        ("turtle", "tletur"),
        ("abcd", "efgh"),
        ("abab", "cdcd")
    ]
    
    expected = [True, False, True, False, True, False, False, False, True, True]
    
    for i, (s, t) in enumerate(test_cases):
        result = are_isomorphic(s, t)
        print(f"Test {i+1}: {result} - {'PASS' if result == expected[i] else 'FAIL'}")

def test_ipv4_validator():
    """Test IPv4 address validator"""
    print("\nTesting IPv4 Address Validator:")
    
    test_cases = [
        "192.168.1.1",
        "255.255.255.255",
        "0.0.0.0",
        "256.100.100.100",
        "192.168.1",
        "192.168.1.01",
        "192.168.1.1.1",
        "abc.def.gha.bcd",
        "172.16.254.1",
        "1.2.3.256"
    ]
    
    expected = [True, True, True, False, False, False, False, False, True, False]
    
    print("Using regex approach:")
    for i, test in enumerate(test_cases):
        result = is_valid_ipv4(test)
        print(f"Test {i+1}: {result} - {'PASS' if result == expected[i] else 'FAIL'}")
    
    print("\nUsing alternative approach:")
    for i, test in enumerate(test_cases):
        result = is_valid_ipv4_alternative(test)
        print(f"Test {i+1}: {result} - {'PASS' if result == expected[i] else 'FAIL'}")

# Run all tests
if __name__ == "__main__":
    test_roman_numeral()
    test_isomorphic()
    test_ipv4_validator()
