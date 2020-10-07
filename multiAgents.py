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
import random
import util

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        return self.minimaxDecision(gameState, 0)[1]  # using the minimax algorithm to get the best action in the current state
        # util.raiseNotDefined()

    def minimaxDecision(self, state, depth):
        """
        The minimax decision algorithm
        :param state: The current state 
        :param depth: The current depth of the minimax tree
        :return: The utility value for the action and the action itself
        """
        numAgents = state.getNumAgents()

        # Terminal test
        if depth == self.depth * numAgents or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        # Since the tree can have several levels of min, we have to get the correct agentIndex
        agentIndex = depth % numAgents

        if agentIndex == 0:
            return self.maxValue(state, agentIndex, depth)  # PacMan's turn
        else:
            return self.minValue(state, agentIndex, depth)  # Ghost's turn

    def maxValue(self, state, agentIndex, depth):
        """
        Calculates the utility value of the best action for PacMan
        :param state: The current state
        :param agentIndex: Which player we are searching for
        :param depth: The current depth
        Returns the max value along with the action
        """
        legalActions = state.getLegalActions(agentIndex)

        # Terminal test
        if not legalActions:
            return self.evaluationFunction(state), None

        value = float("-inf")
        bestAction = None

        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            result = self.minimaxDecision(successor, depth + 1)[0]

            # Updating the utility value and the action if the successor's state utility value is better for PacMan
            if result > value:
                value = result
                bestAction = action

        return value, bestAction

    def minValue(self, state, agentIndex, depth):
        """
        Calculates the utility value of the worst action for PacMan
        :param state: The current state
        :param agentIndex: Which player we are searching for
        :param depth: The current depth
        Returns the min value along with the action
        """
        legalActions = state.getLegalActions(agentIndex)

        if not legalActions:
            return self.evaluationFunction(state), None

        value = float("inf")
        bestAction = None

        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            result = self.minimaxDecision(successor, depth + 1)[0]

            # Updating the utility value and the action if the successor's state utility value is worse for PacMan.
            if result < value:
                value = result
                bestAction = action

        return value, bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphaBetaSearch(gameState, 0)[1]
        # util.raiseNotDefined()

    def alphaBetaSearch(self, state, depth, alpha=float("-inf"), beta=float("inf")):
        """
        The minimax decision algorithm with alpha-beta pruning
        :param state: The current state 
        :param depth: The current depth of the minimax tree
        :param alpha: The alpha value for pruning
        :param beta: The beta value for pruning
        :return: The utility value for the action and the action itself
        """
        numAgents = state.getNumAgents()

        if depth == self.depth * numAgents or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        agentIndex = depth % numAgents

        if agentIndex == 0:
            return self.maxValue(state, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(state, agentIndex, depth, alpha, beta)

    def maxValue(self, state, agentIndex, depth, alpha, beta):
        """
        Calculates the utility value of the best action for PacMan
        :param state: The current state
        :param agentIndex: Which player we are searching for
        :param depth: The current depth
        :param alpha: The best utility value for max
        :param beta: The best utility value for min
        Returns the max value along with the action
        """
        legalActions = state.getLegalActions(agentIndex)

        if not legalActions:
            return self.evaluationFunction(state), None

        value = float("-inf")
        bestAction = None

        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            result = self.alphaBetaSearch(successor, depth + 1, alpha, beta)[0]
            if result > value:
                value = result
                bestAction = action

            # Pruning
            if value > beta:
                return value, bestAction

            # Updating alpha
            if value > alpha:
                alpha = value

        return value, bestAction

    def minValue(self, state, agentIndex, depth, alpha, beta):
        """
        Calculates the utility value of the worst action for PacMan
        :param state: The current state
        :param agentIndex: Which player we are searching for
        :param depth: The current depth
        :param alpha: The best utility value for max
        :param beta: The best utility value for min
        Returns the min value along with the action
        """
        legalActions = state.getLegalActions(agentIndex)

        if not legalActions:
            return self.evaluationFunction(state), None

        value = float("inf")
        bestAction = None

        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            result = self.alphaBetaSearch(successor, depth + 1, alpha, beta)[0]
            if result < value:
                value = result
                bestAction = action

            # Pruning
            if value < alpha:
                return value, bestAction

            # Updating beta
            if value < beta:
                beta = value

        return value, bestAction


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
