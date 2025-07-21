from collections import deque, Counter

def firstNonRepeating(stream):
    freq = Counter()
    q = deque()
    result = []

    for ch in stream:
        freq[ch] += 1
        q.append(ch)

        # Remove characters from the front of the queue if they're repeating
        while q and freq[q[0]] > 1:
            q.popleft()

        if q:
            result.append(q[0])
        else:
            result.append('#')  # No non-repeating character

    return ''.join(result)
stream = "aabc"
print(firstNonRepeating(stream))  # Output: "a#bb"
