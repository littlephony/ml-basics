import numpy as np
import matplotlib.pyplot as plt
import detector as dect

def split_dataset(X):
    """
    Split the given dataset X into x_train, x_cv, and x_test
    (training set, cross validation set, and test set)
    in a 60% - 20% - 20% proportion
    """

    size = X.shape[0]

    # indices of the rightmost (not including) elements for each subset
    train_idx, cv_idx, test_idx = int(0.6 * size), int(0.8 * size), size
    X_train, X_cv, X_test = X[:train_idx], X[train_idx:cv_idx], X[cv_idx:]

    return (X_train, X_cv, X_test)


def get_gaussian_probability(x, mu, var):
    '''
    Compute Gaussian probability of an event x
    given mean mu and standard deviation var
    
    '''

    return np.exp(-(x - mu) ** 2 / (2 * var ** 2)) / (np.sqrt(2 * np.pi) * var)

sample_size, mean, std = 10000, 15, 3.

np.random.seed(0)

# Generate the training set (60% of sample_size)
X_train = np.random.normal(loc=mean, scale=std, size=(int(sample_size * 0.6), 2))

# Setup pyplot
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].axis([0, 30, 0, 30])

# Plot the training set
axes[0].scatter(x=X_train[:, 0], y=X_train[:, 1], c='b', label="x")
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Training set scatter plot')

# Fit Gaussian distribution to the training set
detector = dect.AnomalyDetector(X_train)
mu, var = detector.fit_gaussian()

# Plot the contour plot of the fitted distribution
x, y = np.linspace(0, 30, 100), np.linspace(0, 30, 100)
z = np.multiply.outer(get_gaussian_probability(x, mu[0], var[0]), get_gaussian_probability(y, mu[1], var[1]))

levels = [0.2, 0.4, 0.6, 0.8, 1.0]
axes[1].contourf(x, y, z, levels=20, cmap='RdGy_r')
axes[0].contour(x, y, z, colors='k')
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
axes[1].set_title("Gaussian distribution contour plot")

plt.show()