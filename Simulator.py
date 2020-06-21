#!/usr/bin/env python3

import NeuralColumn
import numpy as np
import random

class Simulator:

    def __init__(self):
        self.value = 0
        self.columns = []
        self.partitionedList = []
        self.targets = []
        self.inputArray = [12, 19, 10, 25, 22, 10, 18, 14, 3, 22, 30, 7, 29, 23, 11, 11, 23, 24, 3, 8, 29, 4, 8, 27, 23, 16, 27, 22, 7, 25, 24, 15, 22, 7, 24, 1, 12, 18, 20, 19, 21, 28, 14, 9, 28, 23, 29, 8, 3, 7]

        ## Prepare the input data
        self.prepareInputData()

        ## Set the columns up
        self.setUpColumns()

    def splitArray(self, lst, parts):
        x = []
        for i in range(0, len(lst), parts):
            x.append(lst[i:i + parts])
        return x

    def prepareInputData(self):
        self.partitionedList = self.splitArray(self.inputArray, 5)

        print(str(len(self.inputArray)))
        print(str(self.partitionedList))

        ## Check if list sum >
        for list in self.partitionedList:
            # print(sum(list))
            self.targets.append(1 if sum(list) > 71 else 0)

    def setUpColumns(self):
        ## Setup Columns
        for num in range(5):
            self.columns = np.append(self.columns, NeuralColumn.NeuralColumn(5, num))

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
            for num in range(10):
                # x = input("Proceed?")

                ## Increment epoch
                epochs += 1

                # Get output for all columns
                # print("Input = " + str(self.partitionedList[num]))
                # print("Sum = " + str(sum(self.partitionedList[num])))
                # print("Target = " + str(self.targets[num]))

                outputs = []

                # Run the initial input through the columns
                for column in self.columns:
                    column.calculateInitialColValues(self.partitionedList[num])

                # Re-run the computation now taking into account node connections
                # to generate the final output value
                for column in self.columns:
                    outputs.append(column.calculateColumnOutputWithConnections())

                ## STOPPED HERE - WRITING THE WEIGHT UPDATE FOR THE COLUMNS
                # print("Outputs are: ")
                # print(outputs)

                numCorrect = 0

                ## Update weights for any incorrect columns
                for index, column in enumerate(self.columns):
                    # If it was correct/incorrect - update weights accordingly
                    wasCorrect = (outputs[index] > 0.5 and self.targets[index] == 1) or (outputs[index] < 0.5 and self.targets[index] == 0)
                    # resultType = "overshot" if (outputs[index] - 0.5) > 0 and self.targets[index] == 0 else "undershot"

                    ## Only applicable to incorrect results
                    resultType = "overshot"
                    diffAmount = outputs[index] - 0.5
                    # if(diffAmount > 0 and self.targets[index] == 1):
                    #     resultType = "overshot"
                    ## only incorrect situations are where target / amount disagree

                    if(diffAmount > 0 and self.targets[index] == 0):
                        resultType = "overshot"
                    elif (diffAmount < 0 and self.targets[index] == 0):
                        resultType = "undershot"
                    elif (diffAmount > 0 and self.targets[index] == 1):
                        resultType = "overshot"
                    elif (diffAmount < 0 and self.targets[index] == 1):
                        resultType = "undershot"

                    # print("output = "+str(outputs[index]))
                    # print("Target was: "+str(self.targets[index]))
                    # print("Was correct?: "+str(wasCorrect) + " result type: "+resultType)

                    if(wasCorrect):
                        numCorrect += 1
                    column.updateWeights(wasCorrect, resultType)
                    # print("Updating column #"+str(index)+" - it was: "+ "correct" if outputs[index] > 0.45 else "wrong")


                print("Total correct: "+str(numCorrect)+" / "+str(len(outputs)))



if __name__ == "__main__":
    print("Starting")

    sim = Simulator()

    sim.runSimulation()



