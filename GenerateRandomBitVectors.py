import numpy as np
import random

nodesInColumn = 10
columns = 100
numberOfVectors = 15
allVectors = []

file = open("trainVectors1000.txt", "w")

# For the amount of vectors we want to generate
for k in range(0, numberOfVectors):
    colVector = np.array([])
    # Loop through each "column"
    for i in range(0, columns):
        # Loop through and generate each column vector
        nodes = np.random.randint(1, size=nodesInColumn)
        nodes[random.randint(0, 9)] = 1
        print nodes
        colVector = np.append(colVector, nodes).astype(np.int) #.append(nodes)


    allVectors.append(colVector.tolist())

    # Write out each vector to file
    for index, int in enumerate(colVector):
        if(index != 0):
            file.write(",")
        file.write(str(int))

    file.write("\n")

file.close()