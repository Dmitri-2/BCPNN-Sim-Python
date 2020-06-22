#!/usr/bin/env python3
import numpy as np
import math
import random

class Neuron:

    ## Only takes the number of inputs it should expect
    def __init__(self, numOfInputs):
        self.value = 0
        self.numOfInputs = numOfInputs
        self.inputs = 0
        self.winningNodeIndex = 0  # Index of node that won the round
        self.connections = np.array([])
        self.connectionWeights = np.array([])
        self.weights = np.random.uniform(low=-0.25, high=0.25, size=(numOfInputs))

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

    def calculate(self, inputs):
        # Calculate the node activation (should get a single value)
        activation = np.dot(inputs, self.weights)
        self.value = activation

    ## Certified
    def calculateConnectedNodes(self):
        # For each connected node, multiple the other node's value by a internally stored weight
        # print("Initial value: "+str(self.value))
        for index, node in enumerate(self.connections):
            self.value += node.value * self.connectionWeights[index]



if __name__ == "__main__":
    print("Starting")

    ## Initialize the nodes
    x = Neuron(3)
    y = Neuron(3)

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

    ## Add a connection from y to x
    x.addConnectionFrom(y)

    # Step 2
    ## Recalculate x's value with the new connection
    x.calculateConnectedNodes()

    print("X Value AFTER UPDATES: "+str(x.value))

    print("Done")
