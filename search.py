# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import *

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    REVERSE_PUSH = False

    @staticmethod
    def reverse_push():
        SearchProblem.REVERSE_PUSH = not SearchProblem.REVERSE_PUSH

    @staticmethod
    def print_push():
        print(SearchProblem.REVERSE_PUSH)

    @staticmethod
    def get_push():
        return SearchProblem.REVERSE_PUSH

    def get_expanded(self):
        return self.__expanded

    def inc_expanded(self):
        self.__expanded+=1

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() 
    node = problem.getStartState()
    if(problem.isGoalState(node)):
        return []
    frontier = Stack()
    explored = []
    parents = {}
    frontier.push(node)

    while(True):
        if(frontier.isEmpty()):
            return []
        node = frontier.pop()
        explored.append(node)

        if(problem.isGoalState(node)):
            break

        successors = problem.getSuccessors(node)

        for (child,action,_) in successors:
            if not ((child in explored)):
                parents[child] = (node,action)
                frontier.push(child)

    actions = []
    while(node!=start):
        (par,act) = parents[node]
        actions.insert(0,act)
        node = par
    return actions

def printpq(frontier):
    temp = Queue()
    while(not frontier.isEmpty()):
        k = frontier.pop()
        temp.push(k)
        
    while(not temp.isEmpty()):
        k = temp.pop()
        print(k)
        frontier.push(k)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState() 
    node = problem.getStartState()
    if(problem.isGoalState(node)):
        return []
    frontier = Queue()
    frontier_list = []
    explored = []
    parents = {}
    frontier.push(node)
    frontier_list.append(node)
    
    while(True):
        if(frontier.isEmpty()):
            return []
        node = frontier.pop()
        frontier_list.remove(node)
        explored.append(node)

        if(problem.isGoalState(node)):
            break
        
        successors = problem.getSuccessors(node)
        for (child,action,_) in successors:
            if not ((child in explored) or (child in frontier_list)):
                parents[child] = (node,action)
                frontier.push(child)
                frontier_list.append(child)

    actions = []
    while(node!=start):
        (par,act) = parents[node]
        actions.insert(0,act)
        node = par
    return actions

def checkinlistwithhigherpriority(child,cost,frontier_list):
    for (x,y) in frontier_list:
        if x==child and y>cost:
            frontier_list.remove((x,y))
            return True
    return False

def checkinlist(child,frontier_list):
    for (x,_) in frontier_list:
        if(x==child):
            return True
    return False

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    node = problem.getStartState()
    frontier = PriorityQueue()
    explored = []
    path = {}
    front = {}
    frontier.push(problem.getStartState(),0)
    front[problem.getStartState()] = 0
    while(not frontier.isEmpty()):
        node = frontier.pop()
        parent_cost = front[node]
        front.pop(node)
        explored.append(node)
        if(problem.isGoalState(node)):
            break
        succ= problem.getSuccessors(node)
        for (state, action , child_cost) in succ:
            if not (state in explored or state in front):
                frontier.push(state,parent_cost+child_cost)
                front[state] = parent_cost+child_cost
                path[state]=(node, action)
            elif state in front and front[state]>parent_cost+child_cost:
                frontier.update(state,parent_cost+child_cost)
                front[state] = parent_cost+child_cost
                path[state]=(node, action)
    action = []
    start = problem.getStartState() 
    while(node!=start):
        (par,act) = path[node]
        action.insert(0,act)
        node = par
    return action

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    node = problem.getStartState()
    frontier = PriorityQueue()
    explored = []
    path = {}
    front = {}
    frontier.push(problem.getStartState(),0+heuristic(problem.getStartState(),problem))
    front[problem.getStartState()] = 0
    while(not frontier.isEmpty()):
        node = frontier.pop()
        parent_cost = front[node]
        front.pop(node)
        explored.append(node)
        if(problem.isGoalState(node)):
            break
        succ= problem.getSuccessors(node)
        for (state, action , child_cost) in succ:
            if not (state in explored or state in front):
                frontier.push(state,parent_cost+child_cost+heuristic(state,problem))
                front[state] = parent_cost+child_cost
                path[state]=(node, action)
            elif state in front and front[state]>parent_cost+child_cost:
                frontier.update(state,parent_cost+child_cost+heuristic(state,problem))
                front[state] = parent_cost+child_cost
                path[state]=(node, action)
    action = []
    start = problem.getStartState() 
    while(node!=start):
        (par,act) = path[node]
        action.insert(0,act)
        node = par
    return action

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
reverse_push=SearchProblem.reverse_push
print_push=SearchProblem.print_push
