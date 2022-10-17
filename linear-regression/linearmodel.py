import numpy as np

class LinearModel():
    def __init__(self):
        self.w = np.zeros(1)
        self.b = 0


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.w = np.zeros_like(X)
        self.b = 0.
        

    def predict(self, x):
        return np.dot(self.w, x) + self.b


    def computer_cost(self, x):
        
        return np.sum( np.dot(self.w, x) )


    def zscore_normalization(self, x):
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        x_norm = (x - mu)/sigma

        return (x_norm, mu, sigma)


    def mean_normalization(self, x):
        mu = np.mean(x, axis=0)
        max_ = np.max(x)
        min_ = np.min(x)
        x_norm = (x - mu) / (max_ - min_)

        return (x_norm, mu, max_, min_) 
    
    