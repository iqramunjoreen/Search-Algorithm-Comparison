"""
    CMPT 417 Final Project -- A Comparison of Different Search Algorithms
    for the 8-Puzzle Problem
"""

import numpy as np
import time
import sys
from search import *
from heapq import heappush, heappop

"""A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state."""
class Node_():
    def __init__(self, state, parent=None, action=None, depth=0, actionCost=0, pathCost=0):
        self.state = state #The state for a respective node
        self.parent = parent #The Parent Node of the search tree
        self.action = action #Actions available at that specific node (up,down,left,right in the 8-puzzle case)
        self.depth = depth #The depth of a specific node in the tree
        self.actionCost = actionCost #The cost to move to a successor state from the given node -- g(n)
        self.pathCost = pathCost #The sum of all g(n) -- cost from root to that specific node
        self.actionUp = None #Action for successor node to move up 
        self.actionDown = None #Action for successor node to move down 
        self.actionLeft = None #Action for successor node to move left
        self.actionRight = None #Action for successor node to move right
        

    """ 
        Functions representing possible actions for each state 
    """
    def ActionUp(self): #Function that checks validity of moving tile down 
        blankIndex=[i[0] for i in np.where(self.state==0)] #Determine the index where the blank tile is located 
        if (blankIndex[0] == 2): #Determine if moving up is valid, if not return False
            return False
        else:
            tileBelow = self.state[blankIndex[0]+1,blankIndex[1]] #Determine the value of the tile below the blank tile 
            stateCopy = self.state.copy() #Copy the current configuration of the state to determine possible moves and for swapping indices purposes
            stateCopy[blankIndex[0],blankIndex[1]] = tileBelow #Set the value of blank tile to the value of the tile below the blank tile
            stateCopy[blankIndex[0]+1,blankIndex[1]] = 0 #Set the value of the tile below the previously blank tile to the blank tile 
            return stateCopy,tileBelow #Return the configuration of the new state and the value of the tile below the blank tile 

    def ActionDown(self): #Function that checks validity of moving tile down 
        blankIndex=[i[0] for i in np.where(self.state==0)] #Determine the index where the blank tile is located 
        if (blankIndex[0] == 0): #Determine if moving down is valid, if not return False
            return False
        else:
            tileAbove = self.state[blankIndex[0]-1,blankIndex[1]] #Determine the value of the tile above the blank tile
            stateCopy = self.state.copy() #Copy the current configuration of the state to determine possible moves and for swapping indices purposes
            stateCopy[blankIndex[0],blankIndex[1]] = tileAbove #Set the value of blank tile to the value of the tile above the blank tile
            stateCopy[blankIndex[0]-1,blankIndex[1]] = 0 #Set the value of the tile above the previously blank tile to the blank tile  
            return stateCopy,tileAbove #Return the configuration of the new state and the value of the tile above the blank tile 

    def ActionLeft(self): #Function that checks validity of moving tile left
        blankIndex=[i[0] for i in np.where(self.state==0)] #Determine the index where the blank tile is located
        if (blankIndex[1] == 2): #Determine if moving left is valid, if not return False
            return False
        else:
            tileRight = self.state[blankIndex[0],blankIndex[1]+1] #Determine the value of the tile to the right of the blank tile
            stateCopy = self.state.copy() #Copy the current configuration of the state to determine possible moves and for swapping indices purposes
            stateCopy[blankIndex[0],blankIndex[1]] = tileRight #Set the value of blank tile to the value of the tile to the right of the blank tile
            stateCopy[blankIndex[0],blankIndex[1]+1] = 0 #Set the value of the tile to the right of the previously blank tile to the blank tile  
            return stateCopy,tileRight #Return the configuration of the new state and the value of the tile the right of the blank tile 

    def ActionRight(self):  #Function that checks validity of moving tile right
        blankIndex=[i[0] for i in np.where(self.state==0)] #Determine the index where the blank tile is located
        if (blankIndex[1] == 0): #Determine if moving right is valid, if not return False
            return False
        else:
            tileLeft = self.state[blankIndex[0],blankIndex[1]-1] #Determine the value of the tile to the left of the blank tile
            stateCopy = self.state.copy() #Copy the current configuration of the state to determine possible moves and for swapping indices purposes
            stateCopy[blankIndex[0],blankIndex[1]] = tileLeft #Set the value of blank tile to the value of the tile to the left of the blank tile
            stateCopy[blankIndex[0],blankIndex[1]-1] = 0 #Set the value of the tile to the left of the previously blank tile to the blank tile  
            return stateCopy,tileLeft #Return the configuration of the new state and the value of the tile to the left of the blank tile 
        
    """ 
        Function that prints out the path to reach a given state  
    """
    def DisplayResult(self): #Function that backtracks the path from goal to root and outputs relevant information 
        StateList = [self.state]
        ActionList = [self.action]
        DepthList = [self.depth]
        ActionCostList = [self.actionCost]
        PathCostList = [self.pathCost]
        
        while (self.parent): #While tracing back to root of search tree, append all relevant information to their respective stacks
            self = self.parent
            StateList.append(self.state)
            ActionList.append(self.action)
            DepthList.append(self.depth)
            ActionCostList.append(self.actionCost)
            PathCostList.append(self.pathCost)

        numberMoves = 0
        while (StateList):
            StateList.pop()
            #print('action=',ActionList.pop(),', depth=',str(DepthList.pop()), ', action cost=',str(ActionCostList.pop()),', path cost=', str(PathCostList.pop()),'\n')
            numberMoves += 1
    
    def DepthFirstSearch(self, goalState):
        startTime = time.time() #Start timer for executing the algorithm 

        Stack = [self] #Set the frontier implemented as a Stack (LIFO)
        NumberRemovedNodes = 0 #Variable to hold number of nodes popped off stack
        MaxNodesInStack = 1 #Variable to hold maximum number of nodes in stack during execution

        DepthList = [0] #List to hold depth of node in the search tree 
        PathCostList = [0] #List to hold path cost to node in the search tree 
        explored = set() #Contains explored states 

        while (Stack): #While the stack is not empty, iterate 
            if (len(Stack) > MaxNodesInStack): #Check if the length of the stack is > than max number of nodes in the stack to determine max number of nodes in stack at one time
                MaxNodesInStack = len(Stack)

            currNode = Stack.pop(0) #Pop node from stack 
            NumberRemovedNodes =  NumberRemovedNodes + 1 #Increment the number of nodes popped from the stack 
            NodeDepth = DepthList.pop(0) #Pop from stack to reveal the level of the node in the tree
            NodePathCost = PathCostList.pop(0) #Pop from stack to reveal the cost to reaching node in the treeselect and remove the path cost for reaching current node
            explored.add(tuple(currNode.state.reshape(1,9)[0])) #Append current state to explored
            
            if (np.array_equal(currNode.state,goalState)): #Goal Test condition, if found, backtrack and print results
                currNode.DisplayResult() 
                print("Goal has been found!")
                print('Time performance:',NumberRemovedNodes,'nodes popped off the stack.')
                print('Space performance:', MaxNodesInStack,'nodes in the stack at its max.')
                print('Time spent: %0.2fs' % (time.time()-startTime))
                return True
            else:    
                if (currNode.ActionDown()): #Check if moving tile down is permissible 
                    stateCopy,tileAbove = currNode.ActionDown()
                    if tuple(stateCopy.reshape(1,9)[0]) not in explored: #If node is not labelled as discovered 
                        #Label V as discovered and set current node to new state
                        currNode.actionDown = Node_(state=stateCopy,parent=currNode,action='down',depth=NodeDepth+1,actionCost=tileAbove,pathCost=NodePathCost+tileAbove)
                        Stack.insert(0,currNode.actionDown) #Push state to stack
                        DepthList.insert(0,NodeDepth+1) #Push depth for that node to depth stack
                        PathCostList.insert(0,NodePathCost+tileAbove) #Push cost to reach that node from initial state to path cost stack

                if (currNode.ActionRight()): #Check if moving left tile to right is permissible 
                    stateCopy,tileLeft = currNode.ActionRight()
                    if tuple(stateCopy.reshape(1,9)[0]) not in explored: #If node is not labelled as discovered 
                        #Label V as discovered and set current node to new state
                        currNode.actionRight = Node_(state=stateCopy,parent=currNode,action='right',depth=NodeDepth+1,actionCost=tileLeft,pathCost=NodePathCost+tileLeft)
                        Stack.insert(0,currNode.actionRight) #Push state to stack
                        DepthList.insert(0,NodeDepth+1) #Push depth for that node to depth stack
                        PathCostList.insert(0,NodePathCost+tileLeft) #Push cost to reach that node from initial state to path cost stack

                if (currNode.ActionUp()): #Check if moving lower tile up is permissible 
                    stateCopy,tileBelow = currNode.ActionUp()
                    if tuple(stateCopy.reshape(1,9)[0]) not in explored: #If node is not labelled as discovered 
                        #Label V as discovered and set current node to new state
                        currNode.actionUp = Node_(state=stateCopy, parent=currNode,action='up',depth=NodeDepth+1,actionCost=tileBelow,pathCost=NodePathCost+tileBelow)
                        Stack.insert(0,currNode.actionUp) #Push state to stack
                        DepthList.insert(0,NodeDepth+1) #Push depth for that node to depth stack
                        PathCostList.insert(0,NodePathCost+tileBelow) #Push cost to reach that node from initial state to path cost stack

                if (currNode.ActionLeft()): #Check if moving right tile to the left is permissible 
                    stateCopy,tileRight = currNode.ActionLeft()
                    if tuple(stateCopy.reshape(1,9)[0]) not in explored: #If node is not labelled as discovered 
                        #Label V as discovered and set current node to new state
                        currNode.actionLeft = Node_(state=stateCopy,parent=currNode,action='left',depth=NodeDepth+1,actionCost=tileRight,pathCost=NodePathCost+tileRight)
                        Stack.insert(0,currNode.actionLeft) #Push state to stack
                        DepthList.insert(0,NodeDepth+1) #Push depth for that node to depth stack
                        PathCostList.insert(0,NodePathCost+tileRight) #Push cost to reach that node from initial state to path cost stack


                        
    def IterativeDeepeningDFS(self, goalState):
        startTime = time.time() #Start timer for executing the algorithm 

        NumberRemovedNodes = 0 #Variable to hold number of nodes popped off stack
        MaxNodesInStack = 1 #Variable to hold maximum number of nodes in stack during execution

        #DEPTH-LIMITED SEARCH -- CONSTRAINED TO 50 LEVEL DEEP DFS 
        for DepthLimit in range(50):
            Stack = [self] #Set the frontier implemented as a Stack (LIFO)
            DepthList = [0] #List to hold depth of node in the search tree 
            PathCostList = [0] #List to hold path cost to node in the search tree 
            explored = set() #Contains explored states 

            while (Stack):
                if len(Stack) > MaxNodesInStack: #Check if the length of the stack is > than max number of nodes in the stack to determine max number of nodes in stack at one time
                    MaxNodesInStack = len(Stack)

                currNode = Stack.pop(0) #Pop node from stack 
                NumberRemovedNodes = NumberRemovedNodes + 1 #Increment the number of nodes popped from the stack 
                NodeDepth = DepthList.pop(0) #Pop from stack to reveal the level of the node in the tree
                NodePathCost = PathCostList.pop(0) #Pop from stack to reveal the cost to reaching node in the treeselect and remove the path cost for reaching current node
                explored.add(tuple(currNode.state.reshape(1,9)[0])) #Append current state to explored

                if (np.array_equal(currNode.state,goalState)): #Goal Test condition, if found, backtrack and print results
                    currNode.DisplayResult()
                    print('Time performance:',str(NumberRemovedNodes),'nodes popped off the stack.')
                    print('Space performance:', str(MaxNodesInStack),'nodes in the stack at its max.')
                    print('Time spent: %0.2fs' % (time.time()-startTime))
                    return True

                else:              
                    if (NodeDepth < DepthLimit): #As long as depth of current node is less than the constraint set
                        if (currNode.ActionDown()): #Check if moving tile down is permissible 
                            stateCopy,tileAbove = currNode.ActionDown()
                            if tuple(stateCopy.reshape(1,9)[0]) not in explored: #If node is not labelled as discovered 
                                #Label V as discovered and set current node to new state
                                currNode.actionDown = Node_(state=stateCopy,parent=currNode,action='down',depth=NodeDepth+1,actionCost=tileAbove,pathCost=NodePathCost+tileAbove)
                                Stack.insert(0,currNode.actionDown)
                                DepthList.insert(0,NodeDepth+1)
                                PathCostList.insert(0,NodePathCost+tileAbove)

                        if currNode.ActionRight(): #Check if moving left tile to right is permissible 
                            stateCopy,tileLeft = currNode.ActionRight()
                            if tuple(stateCopy.reshape(1,9)[0]) not in explored: #If node is not labelled as discovered 
                                #Label V as discovered and set current node to new state
                                currNode.actionRight = Node_(state=stateCopy,parent=currNode,action='right',depth=NodeDepth+1,actionCost=tileLeft,pathCost=NodePathCost+tileLeft)
                                Stack.insert(0,currNode.actionRight)
                                DepthList.insert(0,NodeDepth+1)
                                PathCostList.insert(0,NodePathCost+tileLeft)

                        if currNode.ActionUp(): #Check if moving lower tile up is permissible 
                            stateCopy,tileBelow = currNode.ActionUp()
                            if tuple(stateCopy.reshape(1,9)[0]) not in explored: #If node is not labelled as discovered 
                                #Label V as discovered and set current node to new state
                                currNode.actionUp = Node_(state=stateCopy,parent=currNode,action='up',depth=NodeDepth+1,actionCost=tileBelow,pathCost=NodePathCost+tileBelow)
                                Stack.insert(0,currNode.actionUp)
                                DepthList.insert(0,NodeDepth+1)
                                PathCostList.insert(0,NodePathCost+tileBelow)

                        if currNode.ActionLeft(): #Check if moving right tile to the left is permissible 
                            stateCopy,tileRight = currNode.ActionLeft()
                            if tuple(stateCopy.reshape(1,9)[0]) not in explored: #If node is not labelled as discovered 
                                #Label V as discovered and set current node to new state
                                currNode.action = Node_(state=stateCopy,parent=currNode,action='left',depth=NodeDepth+1, actionCost=tileRight,pathCost=NodePathCost+tileRight)
                                Stack.insert(0,currNode.action)
                                DepthList.insert(0,NodeDepth+1)
                                PathCostList.insert(0,NodePathCost+tileRight)


##########################
## DIJKSTRA'S ALGORITHM ##
##########################

def dijkstra(problem, display=False):

    """
    This function takes an 8-puzzle as input and returns the goal node
    after finding the shortest path to it. All paths in an 8-puzzle
    weigh the same so there is no need for a priority queue
    """
    
    node = Node(problem.initial)
    count = 0
    if problem.goal_test(node.state):
        if display:
            print("0 paths have been expanded and 0 paths remain in the frontier")
        print("Nodes removed:\t" , 0)
        return node

    explored = set()
    frontier = deque([node])
    i=1
    while frontier:
        node = frontier.pop()
        count+=1

        if count>10000:
            print("10,000 nodes were removed but no solution")
            print("This puzzle is unsolvable using Dijkstra's algorithm on this device")
            print("")
            return None
        
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                
                if problem.goal_test(child.state):
                    if display:
                        print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
                    print("Nodes removed:\t" , count)
                    return child
                frontier.append(child)
        i=i+1
    return None

################
## A* SEARCH  ##
################

    """
    This function takes an 8-puzzle as input and returns the goal node
    after finding the shortest path to it. It considers f to be the cost
    to get to a node + the number of misplaced tiles in that node.
    """

def astar(problem, h, display=False):
    h = memoize(h or problem.h, 'h')
    f = lambda n: n.path_cost + h(n)
    f = memoize(f, 'f')
    count = 0
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        count +=1

        if count>20000:
            print("20,000 nodes were removed but no solution")
            print("This puzzle is unsolvable using A* search on this device")
            print("")
            return None
        
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            print("Nodes removed:\t" , count)
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


def h(node):

        return sum(abs((val-1)%3 - i%3) + abs((val-1)//3 - i//3) for i, val in enumerate(node.state) if val)



######################
## helper functions ##
######################


def display(state):
    """ This function takes an 8-puzzle state as input and
    prints a neat and readable representation of it """

    state_ = list(state)
    zero_index = state_.index(0)
    state_[zero_index] = "*"
    print("")
    print(state_[0], state_[1], state_[2])
    print(state_[3], state_[4], state_[5])
    print(state_[6], state_[7], state_[8])
    print("")



def main():
    print("----------------------------------Welcome to the CMPT 417 8 Puzzle Solver."
          "Below are a list of Puzzle Instances you can choose to solve!"
          "----------------------------------\n")
    print("01. Initial State: [8,6,7,2,5,4,3,0,1] with Goal State: [1,2,3,4,5,6,7,8,0]")
    print("02. Initial State: [6,4,7,8,5,0,3,2,1] with Goal State: [1,2,3,4,5,6,7,8,0]")
    print("03. Initial State: [4,1,2,0,8,7,6,3,5] with Goal State: [1,2,3,4,5,6,7,8,0]")
    print("04. Initial State: [1,6,2,5,7,3,0,4,8] with Goal State: [1,2,3,4,5,6,7,8,0]")
    print("05. Initial State: [8,0,6,5,4,7,2,3,1] with Goal State: [0,1,2,3,4,5,6,7,8]")
    print("06. Initial State: [6,4,1,3,0,2,7,5,8] with Goal State: [0,1,2,3,4,5,6,7,8]")
    print("07. Initial State: [1,5,8,3,2,7,0,6,4] with Goal State: [0,1,2,3,4,5,6,7,8]")
    print("08. Initial State: [3,2,8,4,5,1,6,7,0] with Goal State: [0,1,2,3,4,5,6,7,8]")
    print("09. Initial State: [0,3,5,4,2,8,6,1,7] with Goal State: [0,1,2,3,4,5,6,7,8]")
    print("10. Initial State: [7,2,5,3,1,0,6,4,8] with Goal State: [0,1,2,3,4,5,6,7,8]")
    print("11. Initial State: [1,2,3,8,0,4,7,6,5] with Goal State: [1,3,4,8,6,2,7,0,5]")
    print("12. Initial State: [1,2,3,8,0,4,7,6,5] with Goal State: [2,8,1,0,4,3,7,6,5]")
    print("13. Initial State: [1,2,3,8,0,4,7,6,5] with Goal State: [2,8,1,4,6,3,0,7,5]")
    print("14. Initial State: [1,3,4,8,0,5,7,2,6] with Goal State: [1,2,3,8,0,4,7,6,5]")
    print("15. Initial State: [2,3,1,7,0,8,6,5,4] with Goal State: [1,2,3,8,0,4,7,6,5]")
    print("16. Initial State: [2,3,1,8,0,4,7,6,5] with Goal State: [1,2,3,8,0,4,7,6,5]")
    print("17. Initial State: [1,2,3,8,0,4,7,6,5] with Goal State: [2,3,1,8,0,4,7,6,5]")
    print("18. Initial State: [2,8,3,1,0,4,7,6,5] with Goal State: [1,2,3,8,0,4,7,6,5]")
    print("19. Initial State: [8,7,6,1,0,5,2,3,4] with Goal State: [1,2,3,8,0,4,7,6,5]")
    print("20. Initial State: [1,2,3,8,0,4,7,6,5] with Goal State: [5,6,7,4,0,8,3,2,1]\n")

    initialState = None
    goalState = None
    """ 
        8-Puzzle Initial Test Instances
    """
    test1 = np.array([8,6,7,2,5,4,3,0,1]).reshape(3,3)
    test2 = np.array([6,4,7,8,5,0,3,2,1]).reshape(3,3)
    test3 = np.array([4,1,2,0,8,7,6,3,5]).reshape(3,3)
    test4 = np.array([1,6,2,5,7,3,0,4,8]).reshape(3,3)
    test5 = np.array([8,0,6,5,4,7,2,3,1]).reshape(3,3)
    test6 = np.array([6,4,1,3,0,2,7,5,8]).reshape(3,3)
    test7 = np.array([1,5,8,3,2,7,0,6,4]).reshape(3,3)
    test8 = np.array([3,2,8,4,5,1,6,7,0]).reshape(3,3)
    test9 = np.array([0,3,5,4,2,8,6,1,7]).reshape(3,3)
    test10 = np.array([7,2,5,3,1,0,6,4,8]).reshape(3,3)
    test11 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3) 
    test12 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3) 
    test13 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3) 
    test14 = np.array([1,3,4,8,0,5,7,2,6]).reshape(3,3) 
    test15 = np.array([2,3,1,7,0,8,6,5,4]).reshape(3,3) 
    test16 = np.array([2,3,1,8,0,4,7,6,5]).reshape(3,3) 
    test17 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3) 
    test18 = np.array([2,8,3,1,0,4,7,6,5]).reshape(3,3) 
    test19 = np.array([8,7,6,1,0,5,2,3,4]).reshape(3,3) 
    test20 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3) 

    """ 
        8-Puzzle Initial Test Instances
    """
    goal1 = np.array([1,2,3,4,5,6,7,8,0]).reshape(3,3)
    goal2 = np.array([1,2,3,4,5,6,7,8,0]).reshape(3,3)
    goal3 = np.array([1,2,3,4,5,6,7,8,0]).reshape(3,3)
    goal4 = np.array([1,2,3,4,5,6,7,8,0]).reshape(3,3)
    goal5 = np.array([0,1,2,3,4,5,6,7,8]).reshape(3,3)
    goal6 = np.array([0,1,2,3,4,5,6,7,8]).reshape(3,3)
    goal7 = np.array([0,1,2,3,4,5,6,7,8]).reshape(3,3)
    goal8 = np.array([0,1,2,3,4,5,6,7,8]).reshape(3,3)
    goal9 = np.array([0,1,2,3,4,5,6,7,8]).reshape(3,3)
    goal10 = np.array([0,1,2,3,4,5,6,7,8]).reshape(3,3)
    goal11 = np.array([1,3,4,8,6,2,7,0,5]).reshape(3,3)
    goal12 = np.array([2,8,1,0,4,3,7,6,5]).reshape(3,3)
    goal13 = np.array([2,8,1,4,6,3,0,7,5]).reshape(3,3)
    goal14 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3)
    goal15 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3)
    goal16 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3)
    goal17 = np.array([2,3,1,8,0,4,7,6,5]).reshape(3,3)
    goal18 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3)
    goal19 = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3)
    goal20 = np.array([5,6,7,4,0,8,3,2,1]).reshape(3,3)

    #Get the Number of Argument passed in the command line
    numArgs = len(sys.argv)

     #Ensure the input user gave is valid
    if (numArgs == 1):
        print("Usage: 'python3 PuzzleProblem.py [Puzzle Instance Number (1-20)]"
              "[Search Algorithm Name (dfs, iddfs, dijkstra, astar, bfs, bidirectional)]")
        return

    instance = int(sys.argv[1])
    algorithm = sys.argv[2]
    

    if instance == 1:
        initialState = test1
        goalState = goal1
    elif instance == 2: 
        initialState = test2
        goalState = goal2
    elif instance == 3:
        initialState = test3
        goalState = goal3
    elif instance == 4: 
        initialState = test4
        goalState = goal4
    elif instance == 5: 
        initialState = test5
        goalState = goal5
    elif instance == 6: 
        initialState = test6
        goalState = goal6
    elif instance == 7: 
        initialState = test7
        goalState = goal7
    elif instance == 8: 
        initialState = test8
        goalState = goal8
    elif instance == 9: 
        initialState = test9
        goalState = goal9
    elif instance == 10: 
        initialState = test10
        goalState = goal10
    elif instance == 11: 
        initialState = test11
        goalState = goal11
    elif instance == 12: 
        initialState = test12
        goalState = goal12
    elif instance == 13: 
        initialState = test13
        goalState = goal13
    elif instance == 14: 
        initialState = test14
        goalState = goal14
    elif instance == 15: 
        initialState = test15
        goalState = goal15
    elif instance == 16:
        initialState = test16
        goalState = goal16
    elif instance == 17: 
        initialState = test17
        goalState = goal17
    elif instance == 18: 
        initialState = test18
        goalState = goal18
    elif instance == 19: 
        initialState = test19
        goalState = goal19
    elif instance == 20: 
        initialState = test20
        goalState = goal20
    else:
        print("Error: You did not enter a value between 1-20.")
        exit()

    node = Node_(state=initialState,parent=None,action=None,depth=0,actionCost=0,pathCost=0)

    if algorithm == "dfs":
        print("Solving the following 8 puzzle instance with DFS:")
        print("Initial State")
        print ('|', initialState[0][0],'|', initialState[0][1],'|', initialState[0][2], '|')
        print ('|', initialState[1][0],'|', initialState[1][1],'|', initialState[1][2], '|')
        print ('|', initialState[2][0],'|', initialState[2][1],'|', initialState[2][2], '|')
    
        print("\nGoal State")
        print ('|', goalState[0][0],'|', goalState[0][1],'|', goalState[0][2], '|')
        print ('|', goalState[1][0],'|', goalState[1][1],'|', goalState[1][2], '|')
        print ('|', goalState[2][0],'|', goalState[2][1],'|', goalState[2][2], '|')
        dfs = node.DepthFirstSearch(goalState)
        print("Finished!")
    elif algorithm == "iddfs":
        print("Solving the following 8 puzzle instance with IDDFS:")
        print("Initial State")
        print ('|', initialState[0][0],'|', initialState[0][1],'|', initialState[0][2], '|')
        print ('|', initialState[1][0],'|', initialState[1][1],'|', initialState[1][2], '|')
        print ('|', initialState[2][0],'|', initialState[2][1],'|', initialState[2][2], '|')
    
        print("\nGoal State")
        print ('|', goalState[0][0],'|', goalState[0][1],'|', goalState[0][2], '|')
        print ('|', goalState[1][0],'|', goalState[1][1],'|', goalState[1][2], '|')
        print ('|', goalState[2][0],'|', goalState[2][1],'|', goalState[2][2], '|')
        iddfs = node.IterativeDeepeningDFS(goalState)
        print("Finished!")


    #iqra
    if algorithm == "dijkstra" or algorithm == "astar" or algorithm == "bfs" or algorithm == "bidirectional":

        test1 = (8,6,7,2,5,4,3,0,1)
        test2 = (6,4,7,8,5,0,3,2,1)
        test3 = (4,1,2,0,8,7,6,3,5)
        test4 = (1,6,2,5,7,3,0,4,8)
        test5 = (8,0,6,5,4,7,2,3,1)
        test6 = (6,4,1,3,0,2,7,5,8)
        test7 = (1,5,8,3,2,7,0,6,4)
        test8 = (3,2,8,4,5,1,6,7,0)
        test9 = (0,3,5,4,2,8,6,1,7)
        test10 = (7,2,5,3,1,0,6,4,8)
        test11 = (1,2,3,8,0,4,7,6,5) 
        test12 = (1,2,3,8,0,4,7,6,5) 
        test13 = (1,2,3,8,0,4,7,6,5) 
        test14 = (1,3,4,8,0,5,7,2,6) 
        test15 = (2,3,1,7,0,8,6,5,4) 
        test16 = (2,3,1,8,0,4,7,6,5)
        test17 = (1,2,3,8,0,4,7,6,5) 
        test18 = (2,8,3,1,0,4,7,6,5) 
        test19 = (8,7,6,1,0,5,2,3,4) 
        test20 = (1,2,3,8,0,4,7,6,5)

        goal1 = (1,2,3,4,5,6,7,8,0)
        goal2 = (1,2,3,4,5,6,7,8,0)
        goal3 = (1,2,3,4,5,6,7,8,0)
        goal4 = (1,2,3,4,5,6,7,8,0)
        goal5 = (0,1,2,3,4,5,6,7,8)
        goal6 = (0,1,2,3,4,5,6,7,8)
        goal7 = (0,1,2,3,4,5,6,7,8)
        goal8 = (0,1,2,3,4,5,6,7,8)
        goal9 = (0,1,2,3,4,5,6,7,8)
        goal10 = (0,1,2,3,4,5,6,7,8)
        goal11 = (1,3,4,8,6,2,7,0,5)
        goal12 = (2,8,1,0,4,3,7,6,5)
        goal13 = (2,8,1,4,6,3,0,7,5)
        goal14 = (1,2,3,8,0,4,7,6,5)
        goal15 = (1,2,3,8,0,4,7,6,5)
        goal16 = (1,2,3,8,0,4,7,6,5)
        goal17 = (2,3,1,8,0,4,7,6,5)
        goal18 = (1,2,3,8,0,4,7,6,5)
        goal19 = (1,2,3,8,0,4,7,6,5)
        goal20 = (5,6,7,4,0,8,3,2,1)

        if instance == 1:
            new_state = tuple(test1)
            goal_state = tuple(goal1)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 2:
            new_state = tuple(test2)
            goal_state = tuple(goal2)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 3:
            new_state = tuple(test3)
            goal_state = tuple(goal3)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 4:
            new_state = tuple(test4)
            goal_state = tuple(goal4)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 5:
            new_state = tuple(test5)
            goal_state = tuple(goal5)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 6:
            new_state = tuple(test6)
            goal_state = tuple(goal6)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 7:
            new_state = tuple(test7)
            goal_state = tuple(goal7)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 8:
            new_state = tuple(test8)
            goal_state = tuple(goal8)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 9:
            new_state = tuple(test9)
            goal_state = tuple(goal9)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 10:
            new_state = tuple(test10)
            goal_state = tuple(goal10)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 11:
            new_state = tuple(test11)
            goal_state = tuple(goal11)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 12:
            new_state = tuple(test12)
            goal_state = tuple(goal12)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 13:
            new_state = tuple(test13)
            goal_state = tuple(goal13)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 14:
            new_state = tuple(test14)
            goal_state = tuple(goal14)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 15:
            new_state = tuple(test15)
            goal_state = tuple(goal15)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 16:
            new_state = tuple(test16)
            goal_state = tuple(goal16)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 17:
            new_state = tuple(test17)
            goal_state = tuple(goal17)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 18:
            new_state = tuple(test18)
            goal_state = tuple(goal18)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 19:
            new_state = tuple(test19)
            goal_state = tuple(goal19)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        elif instance == 20:
            new_state = tuple(test20)
            goal_state = tuple(goal20)
            new_puzzle = EightPuzzle(Problem(new_state), goal_state)
            new_puzzle.initial = new_state
        

        if algorithm == "dijkstra":
            print("Solving the following 8 puzzle instance with Dijkstra's Algorithm:\n")
            print("Initial State")
            display(new_puzzle.initial)
            print("Goal State")
            display(new_puzzle.goal)
            start_time = time.time()
            x = dijkstra(new_puzzle, display)
            elapsed_time = time.time() - start_time
            print("Time taken:\t", elapsed_time, " seconds")
            steps = 0
            if x != None:
                for i in x.path():
                    steps += 1
                print("Steps:\t\t", steps-1)
            print("")
            print("Finished!\n")

        if algorithm == "astar":
            print("Solving the following 8 puzzle instance with A* Search:\n")
            print("Initial State")
            display(new_puzzle.initial)
            print("Goal State")
            display(new_puzzle.goal)
            start_time = time.time()
            x = astar(new_puzzle, h, display)
            elapsed_time = time.time() - start_time
            print("Time taken:\t", elapsed_time, " seconds")
            steps = 0
            if x != None:
                for i in x.path():
                    steps += 1
                print("Steps:\t\t", steps-1)
            print("")
            print("Finished!\n")

        if algorithm == "bfs":
            print("Solving the following 8 puzzle instance with Breadth First Search:\n")
            print("Initial State")
            display(new_puzzle.initial)
            print("Goal State")
            display(new_puzzle.goal)
            start_time = time.time()
            x = breadth_first_graph_search(new_puzzle)
            elapsed_time = time.time() - start_time
            print("Time taken:\t", elapsed_time, " seconds")
            steps = 0
            if x != None:
                for i in x.path():
                    steps += 1
                print("Steps:\t\t", steps-1)
            print("")
            print("Finished!\n")

        if algorithm == "bidirectional":
            print("Solving the following 8 puzzle instance with Bidirectional Search:\n")
            print("Initial State")
            display(new_puzzle.initial)
            print("Goal State")
            display(new_puzzle.goal)
            start_time = time.time()
            x = bidirectional_search(new_puzzle)
            elapsed_time = time.time() - start_time
            print("Time taken:\t", elapsed_time, " seconds")
            steps = 0
            print("")
            print("Finished!\n")

            
            
 
        



if __name__ == "__main__":
    main()



