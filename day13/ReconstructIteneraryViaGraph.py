from collections import defaultdict
import heapq
class Solution:
    def findItinerary(self, tickets):
        graph = defaultdict(list)
        # Build graph with min-heaps for lexical order
        for src, dst in tickets: heapq.heappush(graph[src], dst)
        itinerary = []
        def dfs(airport):
            while graph[airport]:
                next_stop = heapq.heappop(graph[airport])
                dfs(next_stop)
            itinerary.append(airport)  # Post-order insert
        dfs("JFK")
        return itinerary[::-1]  # Reverse to get correct order
tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],
           ["ATL","JFK"],["ATL","SFO"]]
sol = Solution()
print(sol.findItinerary(tickets))
# Output: ['JFK', 'ATL', 'JFK', 'SFO', 'ATL', 'SFO']
'''
Here’s a detailed dry run table for the itinerary reconstruction using **Hierholzer's algorithm** with min-heaps to respect lexical order:

---

### 🎫 Input Tickets

```
JFK → SFO  
JFK → ATL  
SFO → ATL  
ATL → JFK  
ATL → SFO
```

All destinations are pushed into min-heaps for each source, ensuring **lexicographic order** is preserved during traversal.

---

### 🧭 Dry Run Table – DFS Traversal + Post-Order Build

| DFS Call       | Min-Heap Before Pop      | `next_stop` Chosen | DFS Recurse Into | Min-Heap After Pop | Itinerary Appends | Notes                       |
|----------------|--------------------------|--------------------|------------------|---------------------|-------------------|-----------------------------|
| dfs("JFK")     | ["ATL", "SFO"]           | "ATL"              | dfs("ATL")       | ["SFO"]             | —                 | Lexical choice: ATL < SFO   |
| dfs("ATL")     | ["JFK", "SFO"]           | "JFK"              | dfs("JFK")       | ["SFO"]             | —                 | Revisit JFK                 |
| dfs("JFK")     | ["SFO"]                  | "SFO"              | dfs("SFO")       | []                  | —                 | SFO is last JFK connection  |
| dfs("SFO")     | ["ATL"]                  | "ATL"              | dfs("ATL")       | []                  | —                 | ATL from SFO                |
| dfs("ATL")     | ["SFO"]                  | "SFO"              | dfs("SFO")       | []                  | —                 | ATL → SFO                   |
| dfs("SFO")     | []                       | —                  | —                | —                   | ["SFO"]           | Post-order append           |
| dfs("ATL")     | []                       | —                  | —                | —                   | ["SFO", "ATL"]    | Post-order append           |
| dfs("SFO")     | []                       | —                  | —                | —                   | ["SFO", "ATL", "SFO"] | SFO completes               |
| dfs("JFK")     | []                       | —                  | —                | —                   | ["SFO", "ATL", "SFO", "JFK"] | JFK completes               |
| dfs("ATL")     | []                       | —                  | —                | —                   | ["SFO", "ATL", "SFO", "JFK", "ATL"] | ATL completes               |
| dfs("JFK")     | []                       | —                  | —                | —                   | ["SFO", "ATL", "SFO", "JFK", "ATL", "JFK"] | Final JFK completes         |

---

### 🛫 Final Itinerary (Reversed):

```python
['JFK', 'ATL', 'JFK', 'SFO', 'ATL', 'SFO']
```

---

Let me know if you'd like to test with disconnected graphs or duplicate tickets—this algorithm adapts beautifully. You could even build a variant where airports are weighted by priority or cost instead of lexical order. Want help turning that into a full-fledged challenge? I’m game.

'''