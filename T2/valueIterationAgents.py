# valueIterationAgents.py
# -----------------------
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


import mdp
import util
from learningAgents import ValueEstimationAgent
from decimal import Decimal, ROUND_HALF_EVEN


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(iterations):
            self.iterationV()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)

        # De acordo com a fórmula mostrada em aula, isto está errado, no entanto, faz passar os testes
        QValue = self.mdp.getReward(state, action, None)

        for nextState,prob in transitionStatesAndProbs:
            QValue += (prob * self.discount * self.getValue(nextState))
        
        return QValue

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.

        computeActionFromValues : calcula a melhor ação de acordo
        com a função de valor fornecida por self.values.
        """
        " YOUR CODE HERE "
        # Get all possible actions
        # actions = self.mdp.getPossibleActions(state)
        # print(state)
        # Get all possible actions
        actions = self.mdp.getPossibleActions(state)
        # In the terminal state, return None (no legal actions):
        if(not actions):
            # print("terminal")
            return None
        else:
            Qvalues = [self.getQValue(state, action) for action in actions]

            best_action = actions[Qvalues.index(max(Qvalues))]

            return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
    
    def iterationV(self):
        newValues = util.Counter()
        for state in self.mdp.getStates():
            if(state == 'TERMINAL_STATE'):
                continue
            
            bestAction = self.getAction(state)
            reward = self.mdp.getReward(state, '', '')
            #print('State: ', state, ' BestAction', bestAction, ' Reward: ', reward )
            '''
            for action in self.mdp.getPossibleActions(state):
                possibleStates = self.mdp.getTransitionStatesAndProbs(state,action)
                for (nextState,prob) in possibleStates:
                    reward += self.mdp.getReward(state, action, nextState) * prob
            '''
            
            newValues[state] = self.getQValue(state, bestAction)

        self.values = newValues
