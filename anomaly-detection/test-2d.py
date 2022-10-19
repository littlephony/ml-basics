import numpy as np
import matplotlib.pyplot as plt
import detector as dt

def axis_settings(ax, axis, xlabel, ylabel, title):
    '''
    Set up a given axis
    '''
    ax.axis(axis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


sample_size, mean, std = 10000, 15, 3.

np.random.seed(0)

# Generate the training set (60% of sample_size)
X_train = np.random.normal(loc=mean, scale=std, size=(int(sample_size * 0.6), 2))

# Setup pyplot axes
fig, axes = plt.subplots(nrows=2, ncols=2)
axis_train, axis_contour, axis_cv, axis_test = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

axis_settings(axis_train, [-2, 32, -2, 32], 'x1', 'x2', 'Training set')
axis_settings(axis_contour, [-2, 32, -2, 32], 'x1', 'x2', 'Gaussian distribution contour plot')
axis_settings(axis_cv, [-2, 32, -2, 32], 'x1', 'x2', 'Cross validation set')
axis_settings(axis_test, [-2, 32, -2, 32], 'x1', 'x2', 'Test set')

# Plot the training set
axis_train.scatter(x=X_train[:, 0], y=X_train[:, 1], c='b', marker="x")

# Fit Gaussian distribution to the training set
mu_fit, var_fit = dt.fit_gaussian(X_train)

# Plot the contour plot of the fitted distribution
x, y = np.linspace(-2, 32, 100), np.linspace(-2, 32, 100)
z = np.multiply.outer(dt.gaussian(x, mu_fit[0], var_fit[0]), dt.gaussian(y, mu_fit[1], var_fit[1]))

axis_contour.contourf(x, y, z, levels=20, cmap='RdGy_r')
axis_train.contour(x, y, z, colors='k')

# Generate and plot CV set
X_cv, y_cv, X_cv_anomalous = dt.get_cv_set(2000, 2, mu_fit, var_fit)
    
axis_cv.scatter(X_cv[:,0], X_cv[:,1], c='g', marker="*")
axis_cv.scatter(X_cv_anomalous[:,0], X_cv_anomalous[:,1], c='r', marker="*")
axis_cv.contour(x, y, z, colors='k')

#Tune epsilon
best_epsilon = dt.tune_epsilon(X_cv, mu_fit, var_fit, y_cv)

# Generate test set
X_test = dt.get_test_set(2000, 2, mu_fit, var_fit)

# Predict for test set
test_predictions = dt.predict(X_test, mu_fit, var_fit, best_epsilon)
X_test_anomalous = X_test[test_predictions]

axis_test.scatter(X_test[:,0], X_test[:,1], c='g', marker="*")
axis_test.scatter(X_test_anomalous[:,0], X_test_anomalous[:,1], c='r', marker="*")
axis_test.contour(x, y, z, colors='k')

plt.show()