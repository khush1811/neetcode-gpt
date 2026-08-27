import numpy as np
from numpy.typing import NDArray
import math


class Solution:
    epsilon = 1e-7
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        total = 0
        for i in range(len(y_true)):
            total += y_true[i]*math.log(y_pred[i]+self.epsilon)+(1-y_true[i])*(math.log(1-y_pred[i]+self.epsilon))
        return round(-total/len(y_true),4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        m = len(y_true[0])
        total=0
        for i in range(n):
            total1 = 0
            for j in range(m):
                total1 += y_true[i][j]*math.log(y_pred[i][j]+self.epsilon)
            total += total1
        return round(-total/n,4)




