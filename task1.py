# Python 3.9

import numpy as np

matrix = np.genfromtxt("task1_input.csv", delimiter="\t", skip_header=True)
unitMatrix = np.eye(*matrix.shape)
resultMatrix = matrix * unitMatrix
np.savetxt("task1_output.csv", resultMatrix, delimiter="|")
