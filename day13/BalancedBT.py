class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    def __str__(self): return ""+str(self.val)+" "

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def check(node):
            if not node:
                return 0  # height 0

            left = check(node.left)
            if left == -1:
                return -1  # not balanced

            right = check(node.right)
            if right == -1:
                return -1

            if abs(left - right) > 1:
                return -1  # unbalanced tree
            print(node,"left",left,"right",right,"for",node.left,node.right)
            return 1 + max(left, right)  # return height

        return check(root) != -1
'''
       1
      / \
     2   3
    / \ / \
   4  5 6  7
'''
root = TreeNode(1)
root.left = TreeNode(2, TreeNode(4), TreeNode(5))
root.right = TreeNode(3, TreeNode(6), TreeNode(7))
print(Solution().isBalanced(root))
'''
        1
       /
      2
     /
    3
   /
  4
'''
root = TreeNode(1)
root.left = TreeNode(2)
root.left.left = TreeNode(3)
root.left.left.left = TreeNode(4)
print(Solution().isBalanced(root))

'''
Here's a dry run table showing the recursive evaluation of the `isBalanced` function for your binary tree:

### 🌳 Tree Structure

```
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
```

### 🧮 Dry Run Table for `check(node)`

| Node | Left Subtree Height | Right Subtree Height | Is Balanced? | Returned Height |
|------|----------------------|-----------------------|---------------|------------------|
| 4    | 0                    | 0                     | ✅ Yes         | 1                |
| 5    | 0                    | 0                     | ✅ Yes         | 1                |
| 2    | 1 (from 4)           | 1 (from 5)            | ✅ Yes         | 2                |
| 6    | 0                    | 0                     | ✅ Yes         | 1                |
| 7    | 0                    | 0                     | ✅ Yes         | 1                |
| 3    | 1 (from 6)           | 1 (from 7)            | ✅ Yes         | 2                |
| 1    | 2 (from 2)           | 2 (from 3)            | ✅ Yes         | 3                |

✅ Since no subtree returns `-1`, the tree is balanced and the final result is `True`.

---

Let me know if you'd like to test an unbalanced case or visualize this process in another format. I could even help you sketch a recursive tree trace if you'd like!
Here's a dry run analysis of the unbalanced tree you provided:

### 🌲 Tree Structure

```
      1
     /
    2
   /
  3
 /
4
```

This is a **left-skewed** tree — each node only has a left child.

### 🧮 Dry Run Table for `check(node)`

| Node | Left Subtree Height | Right Subtree Height | Height Difference | Is Balanced? | Returned Value |
|------|----------------------|-----------------------|-------------------|---------------|-----------------|
| 4    | 0                    | 0                     | 0                 | ✅ Yes        | 1               |
| 3    | 1 (from 4)           | 0                     | 1                 | ✅ Yes        | 2               |
| 2    | 2 (from 3)           | 0                     | 2                 | ❌ No         | -1              |
| 1    | -1 (from 2)          | X                     | —                 | ❌ No         | -1              |

Since node 2 has a height difference of more than 1 between its left and right subtrees, the `check` function returns `-1`, signaling imbalance. So the final output is:

```python
False
```

---

Want to try building a challenge around this logic? You’ve got the perfect ingredients for a HackerRank-style problem, especially with test cases focused on edge structures like skewed or complete binary trees. I’d be happy to help sketch it out.

'''