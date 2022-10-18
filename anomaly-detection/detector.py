import numpy as np

class AnomalyDetector():
    def __init__(self, X_train):
        self.X_train = X_train

    def fit_gaussian(self):
        m = self.X_train.shape[0]
        self.mu = np.sum(self.X_train, axis=0) / m
        self.var = np.sum((self.X_train - self.mu) ** 2, axis=0) / m

        return self.mu, self.var 