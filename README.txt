A Comparison of Different Search Algorithms for the 8-Puzzle Problem

Implementation of 6 different search algorithms, namely: A-Star Search (A*), Bidirectional Search (Bidirectional), Dijkstra’s Algorithm (Dijkstra), Depth-First Search (DFS), Iterative-Deepening Depth-First Search (IDDFS), and Breadth First Search (BFS). 

In order to experiment with the algorithms, we present the time and space complexity of the individual algorithms along with their respective benchmarked running times for completing the respective search algorithms. Additionally, we compare the algorithms by determining how many moves are required to get to the goal state along with reporting the number of expanded nodes (nodes popped off the queue). Another aspect we report is the optimality and completeness of the algorithms. Lastly, we compare and discuss our results and the particular benefits and drawbacks from the pseudocodes of our proposed algorithms.

Installations:

Pip install numpy

Usage: 

python3 PuzzleProblem.py [Puzzle Instance Number (1-20)] [Search Algorithm Name (dfs, iddfs, dijkstra, astar, bidirectional, bfs)]
