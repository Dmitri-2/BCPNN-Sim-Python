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
        self.beta = 0
        self.numOfInputs = numOfInputs
        self.inputs = []
        self.connections = np.array([])
        self.isWinningNode = False
        self.connectionWeights = np.array([])
        self.weights = np.random.uniform(low=-0.25, high=0.25, size=(numOfInputs))

    def isActivated(self):
        return True if(self.value > 0.5) else False

    def reinitializeWeightMatrix(self):
        self.weights = np.random.uniform(low=-0.25, high=0.25, size=(self.numOfInputs))

    def reinitializeConWeightMatrix(self):
        self.connectionWeights = np.random.uniform(low=-0.25, high=0.25, size=(len(self.connections)))

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


    # Function that calculates the activation of the neuron
    # Inputs:
    #     - input values
    #     - weights
    # Output:
    #     - node activation value

    def calculate(self, inputs):

        self.inputs = inputs

        # Equation 1
        # Following formula #1 from the Lansner paper
        self.beta = math.log(self.probability, 10)

        # Calculate the node activation (should get a single value)
        # Following formula #5 from Lansner paper
        activation = np.dot(inputs, self.weights) + self.beta
        self.value = activation


    # Function that augments the node's activation by computing the connection weights
    # Inputs:
    #     - node activations
    # Output:
    #     - new node activation for self
    def calculateConnectedNodes(self):
        # For each connected node, multiple the other node's value by a internally stored weight
        # print("Initial value: "+str(self.value))
        for index, node in enumerate(self.connections):
            self.value += node.value * self.connectionWeights[index]


    # Function to update the weights that the node has control over
    # Inputs:
    #     - self weights
    #     - input to node
    # Output:
    #     - new node weights for self
    def updateWeights(self, target):
        # Tau
        tau = 1.5
        # Doing logs in base 10
        for index, weight in enumerate(self.weights):
            # New weight value = log( input - e^(-1 / tau) * (input - a ^ old weight value)
            # Where a is a the log base - I chose 10
            a = 10

            input = target
            # input = np.mean(self.inputs)

            #using target as input??

            self.weights[index] = math.log((input - (math.exp(-1/tau)*(input-a**self.weights[index]))), a)
            # print("New weight: "+str(self.weights[index]))




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
