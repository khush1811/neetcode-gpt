import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        if training:
            mu_b = np.mean(x,axis=0)
            variance_b = np.var(x,axis=0)
            x_cap = (x-mu_b)/np.sqrt(variance_b+eps)
            y = gamma*x_cap+beta
            running_mean = running_mean*(1-momentum) + momentum*mu_b
            running_var = running_var*(1-momentum) + momentum*variance_b
        else:
            x_cap = (x-running_mean)/np.sqrt(running_var+eps)
            y = gamma*x_cap+beta
        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )

