# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        pellets = successorGameState.getCapsules()
        foodPosition = newFood.asList()
        foodCount = len(foodPosition)
        closestDistance = 1e6
        longestDistance = 0

        for i in range(foodCount):
            for j in range(i+1,foodCount,1):
                distance = manhattanDistance(foodPosition[i],foodPosition[j])
                if distance > longestDistance:
                    longestDistance = distance

        for i in range(foodCount):
            distance = manhattanDistance(foodPosition[i],newPos) + foodCount*100 + longestDistance
            if distance < closestDistance:
                closestDistance = distance

        if foodCount == 0:
            closestDistance = 0
        score = -closestDistance

        closestDistance = 1e6
        closestPos = (0,0)
        closestScareTime = 0
        flag = True
        for i in range(len(newGhostStates)):
            ghostPos = successorGameState.getGhostPosition(i+1)
            distance = manhattanDistance(newPos,ghostPos)
            if newScaredTimes[i] > 0:
                score += 1/float(distance)*100
            if distance < closestDistance:
                closestDistance = distance
                closestPos = ghostPos
                closestScareTime = newScaredTimes[i]
            if(newScaredTimes[i]>0):
                if manhattanDistance(ghostPos,newPos) <= 2:
                    return 1e8
                flag = False
            if manhattanDistance(newPos,ghostPos)<=1:
                score -= 1e6

        if manhattanDistance(newPos,closestPos)<=1 and closestScareTime==0:
            score -= 1e6

        if (newScaredTimes[0] - closestDistance) > 0:
            score+=1e6
        score -= 10000*len(pellets)

        for p in pellets:
            distance = manhattanDistance(newPos,p)
            score += 1/float(distance)*100
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def Max_Value(depth, gameState):
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            return max(Min_Value(1,depth,gameState.generateSuccessor(0,action)) for action in gameState.getLegalActions(0))

        def Min_Value(agent, depth, gameState):
            if agent==0:
                return Max_Value(depth+1,gameState)
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            return min(Min_Value((agent+1)%gameState.getNumAgents(),depth,gameState.generateSuccessor(agent,action)) for action in gameState.getLegalActions(agent))
            
        final_action = None
        value = None
        for action in gameState.getLegalActions(0):
            val = Min_Value(1,0,gameState.generateSuccessor(0,action))
            if value == None or val > value:
                value = val
                final_action = action
        return final_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def Max_Value(depth, gameState, alpha, beta):
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = float('-inf')
            for action in gameState.getLegalActions(0):
                val = Min_Value(1,depth,gameState.generateSuccessor(0,action), alpha,beta)
                value = max(value,val)
                if value > beta:
                    return value
                alpha = max(alpha,value)
            return value

        def Min_Value(agent, depth, gameState, alpha, beta):
            if agent==0:
                return Max_Value(depth+1,gameState, alpha, beta)
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = float('inf')
            for action in gameState.getLegalActions(agent):
                val = Min_Value((agent+1)%gameState.getNumAgents(),depth,gameState.generateSuccessor(agent,action), alpha,beta)
                value = min(value,val)
                if value < alpha:
                    return value
                beta = min(beta,value)
            return value
            
        alpha = float('-inf')
        beta = float('inf')
        final_action = None
        value = float('-inf')
        for action in gameState.getLegalActions(0):
            val = Min_Value(1,0,gameState.generateSuccessor(0,action),alpha,beta)
            if value == float('-inf') or val > value:
                value = val
                final_action = action
            if value > beta:
                return value
            alpha = max(alpha, value)
        return final_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def Max_Value(depth, gameState):
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            return max(Exp_Value(1,depth,gameState.generateSuccessor(0,action)) for action in gameState.getLegalActions(0))

        def Exp_Value(agent, depth, gameState):
            if agent==0:
                return Max_Value(depth+1,gameState)
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = 0
            length = float(len(gameState.getLegalActions(agent)))
            for action in gameState.getLegalActions(agent):
                p = 1.0/length
                v += p * Exp_Value((agent+1)%gameState.getNumAgents(),depth,gameState.generateSuccessor(agent,action))
            return v
            
        final_action = None
        value = None
        for action in gameState.getLegalActions(0):
            val = Exp_Value(1,0,gameState.generateSuccessor(0,action))
            if value == None or val > value:
                value = val
                final_action = action
        return final_action

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"    
    if currentGameState.isLose():
        return -1e6
    if currentGameState.isWin():
        return 1e6 

    ghosts = currentGameState.getGhostStates()
    pacman_pos = currentGameState.getPacmanPosition()
    food_pos = currentGameState.getFood().asList()
    pellets = currentGameState.getCapsules()

    score_from_ghost = 0
    for ghost in ghosts:
        distance = manhattanDistance(pacman_pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            score_from_ghost += max(8 - distance, 0)**2
        else:
            score_from_ghost -= max(7 - distance, 0)**2

    score_from_suicide = 0
    for ghost in ghosts:
        dist = manhattanDistance(pacman_pos, ghost.getPosition())
        if dist<=1:
            score_from_suicide = -1e6

    score_from_food = -1e6
    for food in food_pos:
        distance = 1.0 / float(manhattanDistance(pacman_pos, food))
        score_from_food = max(score_from_food,distance)
    if len(food_pos)==0:
        score_from_food = 0

    score_from_capsule = -1e6
    for pellet in pellets:
        distance = 1.0 / float(manhattanDistance(pacman_pos, pellet))
        score_from_capsule = max(score_from_capsule,distance)
    if len(pellets) == 0:
        score_from_capsule=0

    return currentGameState.getScore() + score_from_ghost + score_from_food + 100.0 * score_from_capsule + score_from_suicide


# Abbreviation
better = betterEvaluationFunction
