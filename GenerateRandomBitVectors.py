import numpy as np
import random

vectorLength = 10
numberOfVectors = 100
allVectors = []

file = open("trainVectors.txt", "w")

for i in range(0, numberOfVectors):
    vector = np.random.randint(1, size=vectorLength)
    vector[random.randint(0, 9)] = 1
    allVectors.append(vector)

    # Write out each vector to file
    for index, int in enumerate(vector):
        if(index != 0):
            file.write(",")
        file.write(str(int))

    file.write("\n")

file.close()