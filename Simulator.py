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
        self.inputArray = [12, 19, 10, 25, 22, 10, 18, 14, 3, 22, 30, 7, 29, 23, 11, 11, 23, 24, 3, 8, 29, 4, 8, 27, 23, 16, 27, 22, 7, 25, 24, 15, 22, 7, 24, 1, 12, 18, 20, 19, 21, 28, 14, 9, 28, 23, 29, 8, 3, 7]
        self.inputVectors = []
        self.testVectors = []

        ## Prepare the input data
        self.readDataIn()

        ## Set the columns up
        self.setUpColumns(10)

    def readDataIn(self):

        with open('trainVectors.txt') as file:
            reader = csv.reader(file)
            inputData = list(reader)

        for row in inputData:
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

    def prepareInputData(self):
        self.partitionedList = self.splitArray(self.inputArray, 10)

        print(str(len(self.inputArray)))
        print(str(self.partitionedList))

        ## Check if list sum >
        for list in self.partitionedList:
            self.targets.append(1 if sum(list) > 71 else 0)

    def setUpColumns(self, numberOfColumns):
        ## Setup Columns
        for num in range(numberOfColumns):
            self.columns = np.append(self.columns, NeuralColumn.NeuralColumn(28, num, len(self.inputVectors[0])))

        ## Connect nodes across columns
        for num in range(5):
            ## In %30 of cases, make connection across columns
            if (random.randint(0, 10) < 10):
                # Connect a random node in the other column
                self.columns[num].addRandomConnectionFrom(self.columns[random.randint(0, 4)])


    ## Run the simulation
    def runSimulation(self):

        epochs = 0

        while(epochs < 200):
            for index, inputVector in enumerate(self.inputVectors):

                # Increment epoch
                epochs += 1

                outputVector = []

                # Run the initial input through the columns
                for column in self.columns:
                    column.calculateInitialColValues(inputVector)

                # Re-run the computation now taking into account node connections
                # to generate the final output value
                for column in self.columns:
                    outputVector.append(column.calculateColumnOutputWithConnections())

                # Round outputs to integers
                outputVector = map(lambda x: int(round(x)), outputVector)

                numCorrect = 0

                ## Update weights for any incorrect columns
                for index, column in enumerate(self.columns):

                    expected = inputVector[index]
                    actual = outputVector[index]

                    # If it was correct/incorrect - update weights accordingly
                    wasCorrect = (expected == actual)

                    ## Only applicable to incorrect results
                    resultType = "overshot"

                    if(expected == 1 and actual == 0):
                        resultType = "undershot"
                    elif (expected == 0 and actual == 1):
                        resultType = "overshot"

                    if(wasCorrect):
                        numCorrect += 1
                    column.updateWeights(wasCorrect, resultType)


                print("Total correct: "+str(numCorrect)+" / "+str(len(outputVector)))





if __name__ == "__main__":
    print("Starting")

    sim = Simulator()

    sim.runSimulation()



