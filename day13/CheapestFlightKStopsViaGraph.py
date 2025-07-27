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