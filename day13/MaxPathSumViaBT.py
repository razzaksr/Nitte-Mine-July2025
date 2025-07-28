class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.max_sum = float('-inf')

        def max_gain(node):
            if not node:
                return 0

            # Recursively get max gain from left and right
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)

            # Current path sum through the node
            current_sum = node.val + left_gain + right_gain

            # Update global max sum
            self.max_sum = max(self.max_sum, current_sum)

            # Return max gain if continue the same path
            return node.val + max(left_gain, right_gain)

        max_gain(root)
        return self.max_sum
# Step 1: Build your binary tree
root = TreeNode(-10)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

# Step 2: Create the Solution object
solver = Solution()

# Step 3: Call maxPathSum and print the result
result = solver.maxPathSum(root)
print("Maximum Path Sum:", result)

'''
         5
        / \
      4   8
     /   / \
   11  13  4
  /  \       \
 7    2       1
'''
root = TreeNode(5)
root.left = TreeNode(4)
root.right = TreeNode(8)

root.left.left = TreeNode(11)
root.left.left.left = TreeNode(7)
root.left.left.right = TreeNode(2)

root.right.left = TreeNode(13)
root.right.right = TreeNode(4)
root.right.right.right = TreeNode(1)
solver = Solution()
result = solver.maxPathSum(root)
print("Maximum Path Sum:", result)

'''
Here's a dry run table walking through the logic of `maxPathSum` for your binary tree:

### 🌳 Tree Structure

```
        -10
        /  \
       9   20
           / \
         15   7
```

This function explores all possible paths in the tree and tracks the path with the highest sum—where a path can begin and end at any node.

---

### 🧮 Dry Run Table: `max_gain(node)` and `current_sum`

| Node | Left Gain | Right Gain | `current_sum = node.val + left + right` | Update `max_sum`? | Return Value |
|------|-----------|------------|------------------------------------------|-------------------|--------------|
| 9    | 0         | 0          | 9                                        | ✅ Yes → 9        | 9            |
| 15   | 0         | 0          | 15                                       | ✅ Yes → 15       | 15           |
| 7    | 0         | 0          | 7                                        | —                 | 7            |
| 20   | 15        | 7          | 42 (= 20+15+7)                           | ✅ Yes → 42       | 35 (= 20+max)|
| -10  | 9         | 35         | 34 (= -10+9+35)                          | —                 | 25 (= -10+max)|

✅ Final result: `max_sum = 42`

---

🔍 **Key Insight**  
Although the root’s `current_sum` is 34, the highest path (15 → 20 → 7) gives us 42. That’s why the result isn’t necessarily rooted—it’s about finding the most valuable path **anywhere** in the tree.

---

Want to tackle variants like “return the actual path” or “exclude negative values”? There’s a whole spectrum of fun challenges we could explore from this one pattern alone. Let’s unlock it!

Let’s walk through the dry run of your test case for the `maxPathSum` function. This one's a fantastic example with multiple potential high-value paths!

---

### 🌳 Tree Structure

```
            5
          /   \
         4     8
        /     / \
      11    13  4
     /  \         \
    7    2         1
```

---

### 🧮 Dry Run Table: `max_gain(node)` + Path Calculations

| Node | Left Gain | Right Gain | `current_sum = node.val + left + right` | Updates `max_sum`? | Returns |
|------|-----------|------------|------------------------------------------|---------------------|----------|
| 7    | 0         | 0          | 7                                        | ✅ Yes → 7          | 7        |
| 2    | 0         | 0          | 2                                        | —                   | 2        |
| 11   | 7         | 2          | 20 (= 11+7+2)                            | ✅ Yes → 20         | 18 (=11+max)|
| 4    | 18        | 0          | 22 (= 4+18+0)                            | ✅ Yes → 22         | 22       |
| 13   | 0         | 0          | 13                                       | —                   | 13       |
| 1    | 0         | 0          | 1                                        | —                   | 1        |
| 4 (right) | 0    | 1          | 5 (= 4+0+1)                              | —                   | 5        |
| 8    | 13        | 5          | 26 (= 8+13+5)                            | ✅ Yes → 26         | 21 (=8+max)|
| 5    | 22        | 21         | 48 (= 5+22+21)                          | ✅ Yes → 48         | 27 (=5+max)|

---

### 🏁 Final Result:

```python
Maximum Path Sum: 48
```

🔥 That path is: `7 → 11 → 4 → 5 → 8 → 13` (or including 2 instead of 7, still leads to the same `22` from left)

---

Would you like to extend this into a challenge that returns the actual path nodes too? Or explore a version that excludes negative nodes? This setup is ripe for some advanced variants. Let’s flex your HackerRank instincts!

'''