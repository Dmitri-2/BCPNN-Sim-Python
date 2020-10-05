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

        with open('trainVectors100.txt') as file:
            reader = csv.reader(file)
            inputData = list(reader)

        counter = 0
        for row in inputData:
            counter += 1
            if(counter > 3):
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
            self.columns = np.append(self.columns, NeuralColumn.NeuralColumn(10, num, 10)) ## specifying 10 inputs per col

        ## Connect nodes across columns

        for num in range(self.numberOfColumns-1):
            ## In %30 of cases, make connection across columns
            for i in range(15):
                if (random.randint(0, 10) < 10):
                    # Connect a random node in the other column
                    self.columns[num].addRandomConnectionFrom(self.columns[random.randint(0, self.numberOfColumns-1)])


    ## Run the simulation
    def trainNetwork(self):

        epochs = 0

        while(epochs < 25):
            for index, inputVector in enumerate(self.inputVectors):

                # Increment epoch
                epochs += 1

                # Clamp each input for x cycles
                for i in range(5):
                    outputVector = self.runVectorThroughNetwork(inputVector)

                    ## Update weights for any incorrect columns
                    for index, column in enumerate(self.columns):

                        endingIndex = (index+1)*10      # 10, 20,  30,  40,  50
                        startingIndex = endingIndex-10  # 0,  10,  20,  30,  40


                        for index in range(startingIndex, endingIndex):
                            expected = inputVector[index]

                            # Update weights in any case
                            column.updateWeights(expected)





###### TESTING THE RESULTS
    ## Run the simulation
    def runEvaluation(self):

        for index1, inputVector in enumerate(self.inputVectors):
            outputVector = self.runVectorThroughNetwork(inputVector)

            numCorrect = 0

            # print("Input  = "+str(inputVector))
            # print("Output = "+str(np.array(outputVector)))

            ## Evaluate how many we got correct for the vector
            for index2, column in enumerate(self.columns):

                endingIndex = (index2+1)*10      # 10, 20,  30,  40,  50
                startingIndex = endingIndex-10  # 0,  10,  20,  30,  40


                for index3 in range(startingIndex, endingIndex):
                    expected = inputVector[index3]
                    actual = outputVector[index3]

                    # If it was correct/incorrect - update weights accordingly
                    wasCorrect = (expected == actual)
                    numCorrect += 1 if wasCorrect else 0

            print(str(index1)+" Total correct: "+str(numCorrect)+" / "+str(len(outputVector)))



    def runVectorThroughNetwork(self, inputVector):
        outputVector = []
        # Run the initial input through the columns
        for column in self.columns:
            column.calculateInitialColValues(inputVector)

        # Re-run the computation now taking into account node connections
        # to generate the final output value
        for column in self.columns:
            output = column.calculateColumnOutputWithConnections()
            outputVector.extend(output)

        # Round outputs to integers
        outputVector = map(lambda x: 0 if x <= 0.5 else 1, outputVector)
        return outputVector


if __name__ == "__main__":
    print("Starting")

    sim = Simulator()

    sim.trainNetwork()
    sim.runEvaluation()



