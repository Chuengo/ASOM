import numpy as np


class BlackBox:
    def __init__(self):
        pass
    
    def sample_y(self, x):
        return peaks(x)
    
    def sample_t(self, x):
        return feas(x)


def peaks(x):
    term1 = 3 * (1 - x[:, 0]) ** 2 * np.exp(-(x[:, 0] ** 2) - (x[:, 1] + 1) ** 2)
    term2 = - 10 * (x[:, 0] / 5 - x[:, 0] ** 3 - x[:, 1] ** 5) * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2)
    term3 = - 1 / 3 * np.exp(-(x[:, 0] + 1) ** 2 - x[:, 1] ** 2)
    y = sum([term1, term2, term3])
    return y.reshape(-1, 1)


def feas(x):
    t = np.ones(len(x))
    for i in range(x.shape[0]):
        if ( x[i, 0] ** 2 + x[i, 1] ** 2 > 4 ):
            t[i] = 0
    return t.reshape(-1, 1)
