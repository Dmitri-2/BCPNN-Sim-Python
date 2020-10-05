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

    def increaseWeights(self, percent):
        self.weights += abs(self.weights * percent)

    def decreaseWeights(self, percent):
        self.weights -= abs(self.weights * percent)


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
        #
        # self.probability += changeInProb

        # Calculate the node activation (should get a single value)
        # Following formula #5 from Lansner paper
        # activation = sum(np.dot(input, self.weights))/len(self.weights)
        activation = sum(np.dot(input, self.weights)) + self.bias

        ## Take average of activation
        self.value = activation





    ## Think of this as EQUATION 5

    # Function that augments the node's activation by computing the connection weights
    # Inputs:
    #     - node activations
    # Output:
    #     - new node activation for self
    def calculateConnectedNodes(self):

        if (self.probability <= 0):
            self.probability = 0.00000000000000000000000001

        # Equation 3 (Lansner)
        self.bias = math.log(self.probability, 10)

        # For each connected node, multiple the other node's value by a internally stored weight
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

        ## ASK HERE
        ## Taking sigmoid ??? Otherwise accelerates away
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

        # Equation 1
        ## Calculate own
        self.previousProbability = self.probability
        changeInProb = (target - self.probability)/self.tau
        self.probability = self.probability + changeInProb

        # print("My prob: " + str(self.probability) + " | change: " + str(changeInProb)+ " | old: " + str(self.previousProbability ))

        if(self.probability <= 0):
            self.probability = 0.00000000000000000000000001

        # Doing logs in base 10
        for index, weight in enumerate(self.weights):
            # New weight value = log( input - e^(-1 / tau) * (input - a ^ old weight value)

            # Verify the connected probability is not 0
            if (self.connectedProbabilities[index] <= 0):
                self.connectedProbabilities[index] = 0.00000000000000000000000001

            # print(self.connectedProbabilities[index])

            # New weight update rule
            # print("Updating weight from: " + str(self.weights[index]))
            # print("Taking log of:" +str(self.connectedProbabilities[index]/(self.probability * self.connections[index].probability)))
            #
            self.weights[index] = math.log(self.connectedProbabilities[index]/(self.probability * self.connections[index].probability), 10)
            # print("To : " + str(self.weights[index]))



    # Update the interconnected nodes
    def updateConnectedProbabilities(self):

        for index, connNode in enumerate(self.connections):
            if(connNode.value > 0.5 and self.value > 0.5):
                self.connectedProbabilities[index] *= 1.3




    def increaseConnectionWeights(self, percent):

        # Check if both nodes are active at the same time
        for index, node in enumerate(self.connections):
            # Implimenting hebbian learning - if
            # both nodes active at the same time - strengthen the connection
            # print("both nodes - values " + str(self.isWinningNode) + " - " + str(node.isWinningNode))
            if(self.isWinningNode or node.isWinningNode):
                self.connectionWeights[index] += abs(self.connectionWeights[index] * percent)
                # print("both nodes activated !!!!")


    def decreaseConnectionWeights(self, percent):

        # Check if both nodes are active at the same time
        for index, node in enumerate(self.connections):
            # Implimenting hebbian learning - if
            # both nodes active at the same time - strengthen the connection
            # print("both nodes - values " + str(self.isWinningNode) + " - " + str(node.isWinningNode))
            if (self.isWinningNode or node.isWinningNode):
                self.connectionWeights[index] -= abs(self.connectionWeights[index] * percent)



if __name__ == "__main__":
    print("Starting")

    ## Initialize the nodes
    x = Neuron(3, 0.5)
    y = Neuron(3, 0.5)

    # Step 1
    ## Calculate the initial node values
    x.calculate(np.array([2,2,2]))
    y.calculate(np.array([2,2,2]))

    ## Add a connection from y to x
    x.addConnectionFrom(y)

    # Step 2
    ## Recalculate x's value with the new connection
    x.calculateConnectedNodes()

    print("Final X Value: "+str(x.value))

    ## Weight update check
    print(x.weights)
    x.increaseWeights(0.85)
    print(x.weights)
    x.decreaseWeights(0.15)
    print(x.weights)

    print("---- Connected Weights ---- ")
    print(x.connectionWeights)
    x.updateConnectionWeights(0.5)
    print(x.connectionWeights)

    ## Add a connection from y to x
    x.addConnectionFrom(y)

    # Step 2
    ## Recalculate x's value with the new connection
    x.calculateConnectedNodes()

    print("X Value AFTER UPDATES: "+str(x.value))

    print("Done")
