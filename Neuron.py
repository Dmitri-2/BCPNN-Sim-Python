#!/usr/bin/env python3
import numpy as np
import math
import random

class Neuron:

    ## Only takes the number of inputs it should expect
    def __init__(self, numOfInputs, initialProbability):
        self.value = 0 # e.g "activity"
        self.bias = 0
        self.probability = initialProbability
        self.previousProbability = 0
        self.connectedProbabilities = []
        self.beta = 0
        self.numOfInputs = numOfInputs
        self.input = 0
        self.connections = np.array([])
        self.isWinningNode = False
        self.connectionWeights = np.array([])
        self.weights = np.random.uniform(low=-0.25, high=0.25, size=(numOfInputs))
        self.tau = 0.9
        # print("Initialized prop to: "+str(initialProbability))

    def isActivated(self):
        return True if(self.value > 0.5) else False

    def reinitializeWeightMatrix(self):
        self.weights = np.random.uniform(low=-0.25, high=0.25, size=(self.numOfInputs))

    def reinitializeConWeightMatrix(self):
        ## Reinitialize the connection weight matrix
        self.connectionWeights = np.random.uniform(low=-0.25, high=0.25, size=(len(self.connections)))

        ## Reinitialize the connected prob array
        self.connectedProbabilities =  np.repeat(0.95, len(self.connections))

    def addConnectionFrom(self, otherNode):
        self.connections = np.append(self.connections, otherNode)
        self.reinitializeConWeightMatrix()

    def addConnectionList(self, connectionList):
        self.connections = np.concatenate([self.connections, connectionList])
        self.reinitializeConWeightMatrix()



    # Function that calculates the INITIAL activation of the neuron
    # Inputs:
    #     - input values
    #     - weights
    # Output:
    #     - node activation value
    def calculate(self, input):
        self.input = input

        # ## Equation 1 - update probability for self
        # changeInProb = (input - self.probability) / self.tau
        # self.probability += changeInProb

        # Calculate the node activation (should get a single value)
        # Following formula #5 from Lansner paper
        activation = sum(np.dot(input, self.weights)) + self.bias

        ## Take average of activation
        self.value = activation



    # Function that augments the node's activation by computing the connection weights
    # Inputs:
    #     - node activations
    # Output:
    #     - new node activation for self
    def calculateConnectedNodes(self):

        # print("Weights are: ")
        # print(self.weights)

        if (self.probability <= 0):
            self.probability = 0.00000000000000000000000001

        # Equation 3 (Lansner)
        self.bias = math.log(self.probability, 10)

        # For each connected node, multiply the other node's value by a internally stored weight
        # print("Initial value: "+str(self.value))
        for index, node in enumerate(self.connections):

            ## Update the co-activation probability - EQUATION #2
            newConnProb = ((self.input*node.input) - self.connectedProbabilities[index])
            if (newConnProb == 0):
                newConnProb = 0.00000000000000000000000001

            if (newConnProb > 1e20):
                newConnProb = 1e20

            self.connectedProbabilities[index] += newConnProb / self.tau

            self.value = 0

            # Equation 5 - taking sum of unit's activity * connected weights
            self.value += node.value * self.connectionWeights[index]

        # Equation 5 (support value being calculated with bias)
        self.value += self.bias

        # Taking sigmoid - otherwise value accelerates away
        self.value = self.sigmoidOfValue(self.value)


    def sigmoidOfValue(self, value):
        sigmoid = 1 / (1 + math.exp(-value))
        return sigmoid

    # Function to update the weights that the node has control over
    # Inputs:
    #     - self weights
    #     - input to node
    # Output:
    #     - new node weights for self
    def updateWeights(self, target):

        # if self.isWinningNode == False:
        #     self.probability = 0
        #     return

        tau = 2

        # Equation 1
        ## Calculate own
        self.previousProbability = self.probability
        changeInProb = (target - self.probability)/tau
        self.probability = self.probability + changeInProb

        # print("My prob: " + str(self.probability) + " | change: " + str(changeInProb)+ " | old: " + str(self.previousProbability ))

        if(self.probability <= 0):
            self.probability = 0.01
            # print("Probability was less than 0 - "+str(self.probability))

        # Doing logs in base 10
        for index, weight in enumerate(self.weights):
            # New weight value = log( input - e^(-1 / tau) * (input - a ^ old weight value)

            # Verify the connected probability is not 0
            if (self.connectedProbabilities[index] <= 0):
                self.connectedProbabilities[index] = 0.00000000000000000000000001

            # Weight update rule
            self.weights[index] = math.log(self.connectedProbabilities[index]/(self.probability * self.connections[index].probability), 10)



    # Update the interconnected nodes
    def updateConnectedProbabilities(self):

        for index, connNode in enumerate(self.connections):
            if(connNode.value > 0.5 and self.value > 0.5):
                self.connectedProbabilities[index] *= 1.3




    def increaseConnectionWeights(self, percent):
        # Check if both nodes are active at the same time
        for index, node in enumerate(self.connections):
            # Implementing hebbian learning - if
            # both nodes active at the same time - strengthen the connection

            if(self.isWinningNode and node.isWinningNode):
                self.connectionWeights[index] += abs(self.connectionWeights[index] * percent)


    def decreaseConnectionWeights(self, percent):

        # Check if both nodes are active at the same time
        for index, node in enumerate(self.connections):
            if (self.isWinningNode or node.isWinningNode):
                self.connectionWeights[index] -= abs(self.connectionWeights[index] * percent)


