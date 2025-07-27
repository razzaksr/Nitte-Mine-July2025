class Node:
    def __init__(self,data):
        self.data = data
        self.next = None

class Links:
    def __init__(self):
        self.head = None
    def insertIntoLink(self,val):
        node = Node(val)
        if not self.head:
            self.head = node
            # print("insertion @ beginning")
        else:
            temp = self.head
            while temp.next:
                temp = temp.next
            temp.next = node
            # print("insertion @ last")
    def showOff(self,given=None):
        if not given: iter = self.head
        else: iter = given
        while iter:
            print(iter.data,end= "->" if iter.next else " ")
            iter=iter.next
        print()
    def findMin(self):
        finalMin = float('inf')
        iter = self.head
        while iter:
            finalMin=min(finalMin,iter.data)
            iter=iter.next
        return finalMin
    def findMax(self):
        finalMax = float('-inf')
        iter = self.head
        while iter:
            finalMax=max(finalMax,iter.data)
            iter=iter.next
        return finalMax
    def findOdds(self):
        odds = []
        iter = self.head
        while iter:
            if iter.data%2!=0: odds.append(iter.data)
            iter=iter.next
        return odds
    def findEvens(self):
        evens = []
        iter = self.head
        while iter:
            if iter.data%2==0: evens.append(iter.data)
            iter=iter.next
        return evens
    def findMidViaBruteForce(self):
        count = 0
        iter = self.head
        while iter:
            count+=1
            iter=iter.next
        count//=2
        iter = self.head
        for _ in range(count):
            iter=iter.next
        print(iter.data,"@",count)
    def findMid(self):
        extends = self.head
        close = self.head
        while extends and extends.next:
            close = close.next
            extends = extends.next.next
        return close.data if close else None
    def readAt(self,pos):
        iter = self.head
        for _ in range(1,pos):
            iter=iter.next
        print(iter.data,"@",pos)
    def readBetween(self,start,end):
        iter = self.head
        index = 0
        while iter:
            if start<=index<=end:
                print(iter.data,"@",index)
            iter=iter.next
            index+=1
    
    # Convert given array values into linked list
    def arrToLinked(self,arr):
        if not arr: return None
        temp = Node(arr[0])
        t = temp
        for each in arr[1:]:
            t.next = Node(each)
            t = t.next
        return temp
    
    def mergeLeftRight(self,left,right):
        select = Node(0)
        cursor = select
        while left and right:
            if left.data < right.data:
                cursor.next = left
                left = left.next
            else:
                cursor.next = right
                right = right.next
            cursor = cursor.next
        cursor.next = left if left else right
        return select.next
    def mergeCumulative(self,whole):
        if not whole: return None
        while len(whole) > 1:
            merged = []
            for index in range(0, len(whole), 2):
                left = whole[index]
                right = whole[index+1] if index+1 < len(whole) else None
                merged.append(self.mergeLeftRight(left, right))
            whole = merged
        return whole[0]
        
links = Links()
links.showOff()
constructed = []
constructed.append(links.arrToLinked([1,4,5]))
constructed.append(links.arrToLinked([1,3,4]))
constructed.append(links.arrToLinked([2,6]))
# links.showOff(links.mergeLeftRight(constructed[0],constructed[1]))
links.showOff(links.mergeCumulative(constructed))

# basic executions
# links.insertIntoLink(22)
# links.insertIntoLink(9)
# links.insertIntoLink(11)
# links.insertIntoLink(4)
# links.insertIntoLink(32)
# links.insertIntoLink(5)
# links.showOff()
# print(links.findMin())
# print(links.findMax())
# print("Odds ",links.findOdds())
# print("Evens ",links.findEvens())
# links.findMidViaBruteForce()
# print(links.findMid())
# links.readAt(5)
# links.readBetween(2,5)

'''
1. find min
2. find max
3. find mid
4. find odds
5. find evens
'''


# merge k sorted
