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
        self.inputVectors = []
        self.testVectors = []

        self.numberOfColumns = 10
        self.debug = True

        ## Prepare the input data
        self.readDataIn()
        self.readTestDataIn()

        ## Set the columns up
        self.setUpColumns(self.numberOfColumns)

    def readDataIn(self):

        with open('trainVectors100.txt') as file:
            reader = csv.reader(file)
            inputData = list(reader)

        counter = 0
        for row in inputData:
            counter += 1
            if(counter > 2):
                break

            inputVector = np.array(row).astype(np.int)
            self.inputVectors.append(np.array(row).astype(np.int))

            # # Change 2 values
            # testVector = inputVector
            #
            # randInt1 = random.randint(0, len(inputVector)-1)
            # randInt2 = random.randint(0, len(inputVector)-1)
            # testVector[randInt1] = (testVector[randInt1]+1) % 2
            # testVector[randInt2] = (testVector[randInt2]+1) % 2
            #
            # self.testVectors.append(testVector)

    def readTestDataIn(self):

        with open('trainVector100-test1.txt') as file:
            reader = csv.reader(file)
            inputData = list(reader)

        counter = 0
        for row in inputData:
            counter += 1
            if(counter > 2):
                break

            # testVector = np.array(row).astype(np.int)
            self.testVectors.append(np.array(row).astype(np.int))

        print("READ IN TEST VECTORS")
        print(self.testVectors)



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
            for i in range(150):
                    # Connect a random node in the other column
                    self.columns[num].addRandomConnectionFrom(self.columns[random.randint(0, self.numberOfColumns-1)])


    ## Run the simulation
    def trainNetwork(self):

        epochs = 0

        while(epochs < 12):
            print("Running Epoch - "+str(epochs))

            # Increment epoch
            epochs += 1

            for index, inputVector in enumerate(self.inputVectors):
                print("Trainning with Vector #"+str(index))

                # Clamp each input for x cycles
                for i in range(35):
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
    # Run the simulator without updating weights, track accuracy
    def runEvaluation(self):

        for index1, inputVector in enumerate(self.inputVectors):
            outputVector = self.runVectorThroughNetwork(inputVector)

            numCorrect = 0


            print("Input  = ")
            print(list(inputVector))
            print("Output = ")
            print(outputVector)



            # print("-------------- COL PROB ------------------\n\n\n")
            #
            # for index5, column in enumerate(self.columns):
            #     print("COLUMN PROBABILITY "+str(index5))
            #     column.printProbability()
            #
            # print("-------------- COL PROB END ------------------\n\n\n")


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

            testOutput = self.runVectorThroughNetwork(self.testVectors[index1])

            numCorrect = 0

            ## Evaluate how many we got correct for the vector
            for index2, column in enumerate(self.columns):

                endingIndex = (index2 + 1) * 10  # 10, 20,  30,  40,  50
                startingIndex = endingIndex - 10  # 0,  10,  20,  30,  40

                for index3 in range(startingIndex, endingIndex):
                    expected = inputVector[index3]
                    actual = testOutput[index3]

                    # If it was correct/incorrect - update weights accordingly
                    wasCorrect = (expected == actual)
                    numCorrect += 1 if wasCorrect else 0


            print("Test input = ")
            print(list(self.testVectors[index1]))
            print("Test output = ")
            print(testOutput)

            print(str(index1) + " Total correct: " + str(numCorrect) + " / " + str(len(testOutput)))


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

        if self.debug:
            print(outputVector)

        # Round outputs to integers
        outputVector = map(lambda x: 0 if x <= -0.5 else 1, outputVector)
        return outputVector


if __name__ == "__main__":
    print("Starting")

    sim = Simulator()

    sim.trainNetwork()
    sim.runEvaluation()



