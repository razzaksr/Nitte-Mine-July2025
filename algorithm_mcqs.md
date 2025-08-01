# Algorithm Concepts - Multiple Choice Questions

## Question 1: Two Sum Algorithm
What is the time complexity of the hash table approach used in the Two Sum algorithm?

A) O(n²)  
B) O(n log n)  
C) O(n)  
D) O(1)

**Answer: C) O(n)**  
*Explanation: The hash table approach iterates through the array once, and hash table operations (insert/lookup) are O(1) on average.*

---

## Question 2: Finding Second Maximum
In the second maximum finding algorithm, what happens when all elements in the array are equal?

A) The algorithm throws an error  
B) It returns the same element as both first and second maximum  
C) It returns null  
D) It returns the first element as second maximum

**Answer: B) It returns the same element as both first and second maximum**  
*Explanation: When all elements are equal, the algorithm will identify the same value as both first and second maximum since no truly different second maximum exists.*

---

## Question 3: Dutch National Flag Algorithm
The Dutch National Flag algorithm is primarily used for:

A) Sorting any array in ascending order  
B) Partitioning an array with three distinct values  
C) Finding the median of an array  
D) Reversing an array

**Answer: B) Partitioning an array with three distinct values**  
*Explanation: The Dutch National Flag algorithm efficiently partitions an array containing only three distinct values (typically 0, 1, 2) in a single pass.*

---

## Question 4: Moving Zeros Algorithm
What is the key insight behind the "moving zeros to end" algorithm?

A) Use extra space to store non-zero elements  
B) Maintain a pointer for the next position of non-zero elements  
C) Sort the entire array first  
D) Count zeros and append them later

**Answer: B) Maintain a pointer for the next position of non-zero elements**  
*Explanation: The algorithm uses a "valid" pointer to track where the next non-zero element should be placed, ensuring all non-zeros are moved to the front.*

---

## Question 5: Kadane's Algorithm
Kadane's algorithm is used to solve which classic problem?

A) Finding the longest common subsequence  
B) Maximum sum of contiguous subarray  
C) Shortest path in a graph  
D) Binary search optimization

**Answer: B) Maximum sum of contiguous subarray**  
*Explanation: Kadane's algorithm efficiently finds the maximum sum of any contiguous subarray in O(n) time.*

---

## Question 6: Kadane's Algorithm - Key Concept
What is the core principle behind Kadane's algorithm?

A) Always include the current element in the subarray  
B) Reset the running sum when it becomes negative  
C) Start fresh from current element if previous sum plus current is less than current alone  
D) Both B and C are correct

**Answer: D) Both B and C are correct**  
*Explanation: Kadane's algorithm resets (starts fresh) when the accumulated sum becomes negative, or equivalently, when adding the current element to the previous sum gives less than the current element alone.*

---

## Question 7: Pivot Index Problem
In the pivot index algorithm, what condition must be satisfied at the pivot position?

A) Left sum = Right sum  
B) Left sum + Right sum = Total sum  
C) Left sum = Total sum - Left sum - Current element  
D) Both A and C are correct

**Answer: D) Both A and C are correct**  
*Explanation: At pivot index, left sum equals right sum, which mathematically translates to: left_sum = total_sum - left_sum - current_element.*

---

## Question 8: Stock Market Analysis
The stock buy-sell algorithm aims to find:

A) The day with maximum price  
B) The day with minimum price  
C) Maximum profit from one buy-sell transaction  
D) All profitable transactions

**Answer: C) Maximum profit from one buy-sell transaction**  
*Explanation: The algorithm finds the maximum profit possible from buying on one day and selling on another day (single transaction).*

---

## Question 9: Stock Algorithm Strategy
What is the key strategy in the stock buy-sell algorithm?

A) Always buy at the first day  
B) Keep track of minimum price seen so far and calculate profit at each step  
C) Find the maximum price first  
D) Sort all prices first

**Answer: B) Keep track of minimum price seen so far and calculate profit at each step**  
*Explanation: The algorithm maintains the minimum buying price encountered so far and calculates potential profit at each price point.*

---

## Question 10: Interval Merging
In the insert new interval algorithm, what is the correct order of operations?

A) Merge first, then add non-overlapping intervals  
B) Add all intervals first, then merge  
C) Add non-overlapping intervals before new interval, merge overlapping ones, then add remaining  
D) Sort all intervals first

**Answer: C) Add non-overlapping intervals before new interval, merge overlapping ones, then add remaining**  
*Explanation: The algorithm processes intervals in three phases: add non-overlapping intervals that come before the new interval, merge all overlapping intervals with the new one, and finally add remaining non-overlapping intervals.*

---

## Scoring Guide:
- **8-10 correct**: Excellent understanding of algorithmic concepts
- **6-7 correct**: Good grasp with minor gaps
- **4-5 correct**: Basic understanding, needs more practice
- **Below 4**: Requires significant review of concepts