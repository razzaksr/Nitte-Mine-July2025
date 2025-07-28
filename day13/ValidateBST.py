class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def validate(node, min_val, max_val):
            if not node:
                return True
            if not (min_val < node.val < max_val):
                return False
            return (validate(node.left, min_val, node.val) and
                    validate(node.right, node.val, max_val))

        return validate(root, float('-inf'), float('inf'))
root = TreeNode(2)
root.left = TreeNode(1)
root.right = TreeNode(3)

solver = Solution()
print(solver.isValidBST(root))  # True

root = TreeNode(5)
root.left = TreeNode(1)
root.right = TreeNode(4)
root.right.left = TreeNode(3)
root.right.right = TreeNode(6)

solver = Solution()
print(solver.isValidBST(root))  # False

'''
Here’s a step-by-step dry run of your `isValidBST` check for the given binary tree:

### 🌳 Tree Structure

```
     2
    / \
   1   3
```

### 🧮 Dry Run Table for `validate(node, min_val, max_val)`

| Node | min_val | max_val | Condition                        | Left Valid | Right Valid | Result |
|------|---------|---------|----------------------------------|------------|-------------|--------|
| 2    | -∞      | ∞       | `-∞ < 2 < ∞` ✅                  | check(1,-∞,2) | check(3,2,∞) | ✅ True |
| 1    | -∞      | 2       | `-∞ < 1 < 2` ✅                  | check(None,-∞,1) | check(None,1,2) | ✅ True |
| None | -∞      | 1       | — (null node) ✅                | —          | —           | ✅ True |
| None | 1       | 2       | — (null node) ✅                | —          | —           | ✅ True |
| 3    | 2       | ∞       | `2 < 3 < ∞` ✅                  | check(None,2,3) | check(None,3,∞) | ✅ True |
| None | 2       | 3       | — (null node) ✅                | —          | —           | ✅ True |
| None | 3       | ∞       | — (null node) ✅                | —          | —           | ✅ True |

🎉 Since all nodes satisfy the BST property recursively, the final output is `True`.

---

Want to try a case where this check fails? Or maybe you’d like to build a test suite around edge cases like duplicate values or skewed trees? I’m happy to dive deeper with you.
Here's a dry run breakdown of why this test case fails the Binary Search Tree (BST) validation:

### 🌳 Tree Structure

```
       5
      / \
     1   4
        / \
       3   6
```

This looks fine at first glance, but let's walk through the `validate` function logic.

---

### 🧮 Dry Run Table for `validate(node, min_val, max_val)`

| Node | min_val | max_val | Condition                        | Left Valid | Right Valid | Final Verdict |
|------|---------|---------|----------------------------------|------------|-------------|----------------|
| 5    | -∞      | ∞       | `-∞ < 5 < ∞` ✅                  | check(1, -∞, 5) | check(4, 5, ∞) | ⏳ Pending |
| 1    | -∞      | 5       | `-∞ < 1 < 5` ✅                  | ✅ None     | ✅ None      | ✅ True |
| 4    | 5       | ∞       | `5 < 4 < ∞` ❌ Violation!        | —          | —           | ❌ False |

As you can see, node `4` is in the right subtree of `5`, so it must be **greater than 5** — but it's not. This violates the BST property and causes the algorithm to return `False`.

---

🧠 **Summary:**  
Even though the subtrees of `4` (nodes `3` and `6`) are in valid left/right positions **relative to 4**, the **placement of 4 itself** violates the BST rules when seen from the context of its parent `5`.

Would you like to build a suite of tricky BST test cases? Could make for a solid coding challenge—especially if you include edge cases like repeated values, null subtrees, or mirrored structures.

'''