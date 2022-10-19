from re import M
import numpy as np

def gaussian(x, mu, var):
    '''
    Compute Gaussian probability of event x
    given mean mu and standard deviation var
    '''
    return np.exp(-(x - mu) ** 2 / (2 * var ** 2)) / (np.sqrt(2 * np.pi) * var)

def fit_gaussian(X):
    '''
    Compute mean and variance of a given set
    to fit a Gaussian distribution
    '''
    m = X.shape[0]
    mu = np.sum(X, axis=0) / m
    var = np.sum((X - mu) ** 2, axis=0) / m
    return (mu, var)

def get_cv_set(size, dim, mu, var):
    '''
    Generate a cross validation set with a number size
    of normal elements and ~1% anomalous elements, given dimension, 
    mean and variance of a data set
    '''

    normal_max_size, anomalous_max_size = size, int(0.01 * size)

    mean_mu = np.mean(mu)

    X_cv = np.zeros(shape=(normal_max_size + anomalous_max_size, dim))
    y_cv = np.zeros(shape=(normal_max_size + anomalous_max_size,))
    X_cv_anomalous = np.zeros(shape=(anomalous_max_size, dim))

    normal_count, anomalous_count = 0, 0
    while normal_count < normal_max_size or anomalous_count < anomalous_max_size:
        point = np.random.normal(loc=mu, scale=var, size=(dim,))

        dist = np.sum((point - mu) ** 2)
        idx = normal_count + anomalous_count

        if dist <= (2 * mean_mu / 3) ** 2 and normal_count < normal_max_size:
            X_cv[idx] = point
            y_cv[idx] = 0
            normal_count += 1

        if dist > (2 * mean_mu / 3) ** 2 and anomalous_count < anomalous_max_size:
            X_cv[idx] = point
            X_cv_anomalous[anomalous_count] = point
            y_cv[idx] = 1
            anomalous_count += 1

    return (X_cv, y_cv, X_cv_anomalous)

def get_test_set(size, dim, mu, var):
    '''
    Generate a test set with a number size of normal
    elements and ~1% anomalous elements, given dimension, 
    mean and variance of a data set
    '''
    normal_max_size, anomalous_max_size = size, int(0.01 * size)

    mean_mu = np.mean(mu)

    X_test = np.zeros(shape=(normal_max_size + anomalous_max_size, dim))

    normal_count, anomalous_count = 0, 0
    while normal_count < normal_max_size or anomalous_count < anomalous_max_size:
        point = np.random.normal(loc=mu, scale=var, size=(dim,))

        dist = np.sum((point - mu) ** 2)
        idx = normal_count + anomalous_count

        if dist <= (2 * mean_mu / 3) ** 2 and normal_count < normal_max_size:
            X_test[idx] = point
            normal_count += 1

        if dist > (2 * mean_mu / 3) ** 2 and anomalous_count < anomalous_max_size:
            X_test[idx] = point
            anomalous_count += 1

    return X_test

def tune_epsilon(X, mu, var, y):
    '''
    Tune parameter ε by calculating F1-score

    prec = tp / (tp + fp)

    rec = tp / (tp + fn)

    F1 = 2 * prec * rec / (prec + rec)
    '''
    probabilities = gaussian(X[:,0], mu[0], var[0])
    for i in range(1, X.shape[1]):
        probabilities = probabilities * gaussian(X[:,i], mu[i], var[i])

    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step_size = (max(probabilities) - min(probabilities)) / 1000
    for epsilon in np.arange(min(probabilities), max(probabilities), step_size):
        predictions = (probabilities < epsilon)

        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        f1 = 2 * prec * rec / (prec + rec)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon
    
def predict(X, mu, var, epsilon):
    '''
    Predict normal anomalous and normal examples 
    for given distribution and epsilon
    '''
    probabilities = gaussian(X[:,0], mu[0], var[0])

    for i in range(1, X.shape[1]):
        probabilities = probabilities * gaussian(X[:,i], mu[i], var[i])

    return (probabilities < epsilon)