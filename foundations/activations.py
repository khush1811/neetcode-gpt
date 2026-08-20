import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        ls = []
        for elem in z:
            elem = 1 / (1+np.exp(-elem))
            ls.append(round(elem,5))
        return np.array(ls)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        ls = []
        for elem in z:
            if elem <= 0:
                ls.append(0.0)
            else:
                ls.append(round(elem,5))
        return np.array(ls)
