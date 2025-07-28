import heapq
from collections import defaultdict
class Solution:
    def findCheapestPrice(self, n, flights, src, dst, k):
        graph = defaultdict(list)
        for u, v, w in flights: graph[u].append((v, w))
        # Priority queue: (price, city, stops)
        heap = [(0, src, 0)]
        while heap:
            price, city, stops = heapq.heappop(heap)
            if city == dst: return price
            if stops <= k:
                for nei, p in graph[city]:
                    heapq.heappush(heap, (price + p, nei, stops + 1))
        return -1
n = 4
flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
src = 0
dst = 3
k = 1
sol = Solution()
print(sol.findCheapestPrice(n, flights, src, dst, k))  # Output: 700
'''
Here’s a dry run of your `findCheapestPrice` implementation with the given graph and constraints:

---

### ✈️ Flight Graph Representation

From the list of flights:

```
0 → 1 (100)
1 → 2 (100), 3 (600)
2 → 0 (100), 3 (200)
```

---

### 🧮 Dry Run Table – Dijkstra-style Traversal with Stop Constraint

| Heap State          | Popped Node (price, city, stops) | Action Taken                                                                                   | Updated Heap                                   |
|---------------------|----------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------|
| `[(0, 0, 0)]`        | `(0, 0, 0)`                      | City 0 not dst; stops ≤ k → explore 0’s neighbors: 1 → push (100, 1, 1)                        | `[(100, 1, 1)]`                                |
| `[(100, 1, 1)]`      | `(100, 1, 1)`                    | City 1 not dst; stops ≤ k → explore: 2 → (200, 2, 2), 3 → (700, 3, 2); stops now exceed k     | `[(200, 2, 2), (700, 3, 2)]`                   |
| `[(200, 2, 2), (700, 3, 2)]` | `(200, 2, 2)`            | Stops = 2 > k → no further exploration                                                         | `[(700, 3, 2)]`                                |
| `[(700, 3, 2)]`      | `(700, 3, 2)`                    | Reached destination! ✅ Return price                                                           | —                                              |

✅ **Final Output:** `700` — via path `0 → 1 → 3` with total cost `100 + 600`

---

### 💡 Summary

- Stops limit includes *up to* `k = 1` edges between origin and destination.
- Though the cheaper path exists (0 → 1 → 2 → 3 = `100+100+200 = 400`), it takes **2 stops**, violating the constraint.
- So the algorithm correctly returns `700`.

---

Want to build a test set for increasing values of `k` or explore a variation that tracks the path itself? This pattern is rich with competitive challenge potential!

'''