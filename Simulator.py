#!/usr/bin/env python3

import NeuralColumn
import numpy as np
import random
import csv

class Simulator:

    def __init__(self):
        self.value = 0
        self.columns = []
        self.partitionedList = []
        self.targets = []
        self.inputArray = [12, 19, 5]
        self.inputVectors = []
        self.testVectors = []

        self.numberOfColumns = 10

        ## Prepare the input data
        self.readDataIn()

        ## Set the columns up
        self.setUpColumns(self.numberOfColumns)

    def readDataIn(self):

        with open('trainVectors.txt') as file:
            reader = csv.reader(file)
            inputData = list(reader)

        counter = 0
        for row in inputData:
            counter += 1
            if(counter > 5):
                break

            inputVector = np.array(row).astype(np.int)
            self.inputVectors.append(np.array(row).astype(np.int))

            # Change 2 values
            testVector = inputVector

            randInt1 = random.randint(0, len(inputVector)-1)
            randInt2 = random.randint(0, len(inputVector)-1)
            testVector[randInt1] = (testVector[randInt1]+1) % 2
            testVector[randInt2] = (testVector[randInt2]+1) % 2

            self.testVectors.append(testVector)


    def splitArray(self, lst, parts):
        x = []
        for i in range(0, len(lst), parts):
            x.append(lst[i:i + parts])
        return x


    def setUpColumns(self, numberOfColumns):
        ## Setup Columns
        for num in range(numberOfColumns):
            self.columns = np.append(self.columns, NeuralColumn.NeuralColumn(64, num, len(self.inputVectors[0])))

        ## Connect nodes across columns
        for num in range(self.numberOfColumns-1):
            ## In %30 of cases, make connection across columns
            for i in range(15):
                if (random.randint(0, 10) < 10):
                    # Connect a random node in the other column
                    self.columns[num].addRandomConnectionFrom(self.columns[random.randint(0, self.numberOfColumns-1)])


    ## Run the simulation
    def runSimulation(self):

        epochs = 0

        while(epochs < 25):
            for index, inputVector in enumerate(self.inputVectors):

                # Increment epoch
                epochs += 1

                for i in range(10):
                    outputVector = []

                    # Run the initial input through the columns
                    for column in self.columns:
                        column.calculateInitialColValues(inputVector)

                    # Re-run the computation now taking into account node connections
                    # to generate the final output value
                    for column in self.columns:
                        outputVector.append(column.calculateColumnOutputWithConnections())

                    # Round outputs to integers
                    outputVector = map(lambda x: 0 if int(round(x)) <= 0 else 1, outputVector)

                    numCorrect = 0

                    print("Input  = "+str(inputVector))
                    print("Output = "+str(outputVector))

                    ## Update weights for any incorrect columns
                    for index, column in enumerate(self.columns):

                        expected = inputVector[index]
                        actual = outputVector[index]

                        # If it was correct/incorrect - update weights accordingly
                        wasCorrect = (expected == actual)

                        ## Only applicable to incorrect results
                        # resultType = "overshot"
                        #
                        # if(expected == 1 and actual == 0):
                        #     resultType = "undershot"
                        # elif (expected == 0 and actual == 1):
                        #     resultType = "overshot"

                        if(wasCorrect):
                            numCorrect += 1
                        else:
                            column.updateWeights(expected)


                    print(str(i)+" Total correct: "+str(numCorrect)+" / "+str(len(outputVector)))

                    if(numCorrect == len(outputVector)):
                        break





if __name__ == "__main__":
    print("Starting")

    sim = Simulator()

    sim.runSimulation()



