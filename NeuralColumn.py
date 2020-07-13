#!/usr/bin/env python3

import Neuron
import numpy as np
import math
import random

### Idea for weight updates:
#  1. For this first basic test (classification)
#     The output of the majority of columns gets the vote.
#     We should switch to doing a per-column result when classifying patterns and other
#
#  In any case, weight updates should go like this:
#
#  1. Mark whole columns as correct/incorrect, based on their winning value - say over/under
#  2. Column then goes back and adjusts weight based on the node that
#     won and those that lost the WTA process
#  3. Winner gets their weights decreased by 8%
#     Everyone else gets their weights increased by 2%
#
#     Could also do - Nodes that overshot get their weights reduced by 10%
#



class NeuralColumn: 
    def __init__(self, numNeurons, colNumber, numOfInputs):
        self.number = colNumber
        self.test = []
        self.winningNodeIndex = 0
        self.neurons = np.array([])
        self.neuronConnectionProbabilities = np.array([])
        self.numNeuronsWithBias = numNeurons # + 1

        print("Initializing column: "+str(colNumber))

        # Generate all the neurons in the column
        for num in range(self.numNeuronsWithBias):
            self.neurons = np.append(self.neurons, Neuron.Neuron(numOfInputs, float(1)/float(self.numNeuronsWithBias)))

        # Fully connect the column
        for neuron in self.neurons:
            neuron.addConnectionList(self.neurons)



    # Run the calculations
    def setInputs(self, inputs):
        for neuron in self.neurons:
            neuron.setValue(inputs)

    def calculateInitialColValues(self, inputs):

        ## Normalize the inputs

        # Don't do this - use constraints instead (for weights)

        inputs = self.sigmoidOfList(inputs)

        ## Step 1 - Compute all neuron values
        for neuron in self.neurons:
            neuron.calculate(inputs)



    def calculateColumnOutputWithConnections(self):

        ## Step 2 - Compute connection effects
        for neuron in self.neurons:
            neuron.calculateConnectedNodes()

        ## Step 3 - Return the max node value (WTA)
        winNeuron = max(self.neurons, key=lambda neuron: abs(neuron.value))

        ## Record index of winning node
        self.winningNodeIndex = np.where(self.neurons == winNeuron)[0][0] #self.neurons.index(winNeuron)

        self.neurons[self.winningNodeIndex].isWinningNode = True

        # winNeuron.value = self.sigmoidOfValue(winNeuron.value)
        # print ("returning - "+str((winNeuron.value)))
        return winNeuron.value


    def sigmoidOfValue(self, value):
        sigmoid = 1 / (1+math.exp(-value))
        return sigmoid

    def sigmoidOfList(self, list):
        counter = 0
        newList = []
        for value in list:
            newList.append(float(value)/float(30.0))
            counter += 1
        return newList

    def addConnection(self, number, otherNode):
        self.neurons[number].addConnection(otherNode)
        # self.crossColumnConnections.append([column, node])
        print("Adding connection from "+str(self.number)+" to "+str(column.number))

    def addRandomConnectionFrom(self, srcColumn):
        randomNumSelf = random.randrange(0, len(self.neurons))
        randomNumOther = random.randrange(0, len(srcColumn.neurons))
        print("Adding random connection to node #"+str(randomNumSelf))
        self.neurons[randomNumSelf].addConnectionFrom(srcColumn.neurons[randomNumOther])
        # self.crossColumnConnections.append(srcNode)

    # Update weights based on if we were correct or not
    # If correct - do nothing
    # def updateWeights(self, wasCorrect, direction):

    def updateWeights(self, target):

        # print("winning node was: "+str(self.winningNodeIndex))
        for index, neuron in enumerate(self.neurons):

            # Reset the winning node flag
            neuron.isWinningNode = False

            # Update weights based on Lansner formula
            neuron.updateWeights(target)




            # Old weight update algorithm

            # if(index == self.winningNodeIndex):
            #     if(direction == "undershot"):
            #         neuron.decreaseWeights(0.02) # decrease by 2%
            #         neuron.decreaseConnectionWeights(0.01)
            #     elif (direction == "overshot"):
            #         neuron.decreaseWeights(0.04) # decrease by 5%
            #         neuron.decreaseConnectionWeights(0.02)
            #         # print("Decreasing node #"+str(index))
            #
            # else:
            #     if (direction == "undershot"):
            #         neuron.increaseWeights(0.015) # increase by 1.5%
            #         neuron.increaseConnectionWeights(0.03)
            #     elif (direction == "overshot"):
            #         neuron.decreaseWeights(0.015) # decrease by 1.5%
            #         neuron.decreaseConnectionWeights(0.02)
                    # print("Increasing node #"+str(index))


if __name__ == "__main__":
    print("Starting")

    ## Initialize the sample columns
    column1 = NeuralColumn(5, 3)
    column2 = NeuralColumn(5, 1)

    ## Initialize the inputs
    inputs = np.array([15,25,5,30,20])

    # print(column1.neurons)
    for n in column1.neurons:
        print(n.weights)
    # weights = np.random.random_sample((5, 5))

    ## Add a random connection between the two columns
    column1.addRandomConnectionFrom(column2)

    column1.calculateInitialColValues(inputs)
    column2.calculateInitialColValues(inputs)

    ## Calculate the two WTA values from the columns
    print(column1.calculateColumnOutputWithConnections())
    print(column2.calculateColumnOutputWithConnections())

    ## Do a weight update
    print("Updated weights")

    column1.updateWeights(False, "undershot")
    for n in column1.neurons:
        print(n.weights)


