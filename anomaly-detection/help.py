import numpy as np



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
