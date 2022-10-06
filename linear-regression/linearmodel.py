import numpy as np
class LinearModel():
    def __init__(self):
        self.w = np.zeros(1)
        self.b = 0


    def fit(self, X, Y):
        self.w = np.random.randint(low=-100, high=100, size=X.shape)
        self.b = np.random.randint(low=-100, high=100, size=1)
        

    def predict(self, x):
        return self.w @ x + self.b

    
    