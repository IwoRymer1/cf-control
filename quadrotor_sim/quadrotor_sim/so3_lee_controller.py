import numpy as np


class LeeSE3Controller:
    def __init__(self, model):
        self.m = model.m
        self.g = model.g
        self.J = model.I

        # tunable params
        self.kx = np.diag([6, 6, 8])
        self.kv = np.diag([4, 4, 5])

        self.kR = np.diag([8, 8, 1])
        self.kOmega = np.diag([0.15, 0.15, 0.1])
