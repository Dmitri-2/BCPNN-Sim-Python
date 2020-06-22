import numpy as np

vectorLength = 10
numberOfVectors = 100
allVectors = []

file = open("trainVectors.txt", "w")

for i in range(0, numberOfVectors):
    vector = np.random.randint(2, size=vectorLength)
    allVectors.append(vector)

    # Write out each vector to file
    for index, int in enumerate(vector):
        if(index != 0):
            file.write(",")
        file.write(str(int))

    file.write("\n")

file.close()