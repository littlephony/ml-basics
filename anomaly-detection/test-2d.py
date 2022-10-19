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


def gaussian_probability(x, mu, var):
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
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0][0].axis([-2, 32, -2, 32])

# Plot the training set
axes[0][0].scatter(x=X_train[:, 0], y=X_train[:, 1], c='b', label="x")
axes[0][0].set_xlabel('x1')
axes[0][0].set_ylabel('x2')
axes[0][0].set_title('Training set scatter plot')

# Fit Gaussian distribution to the training set
detector = dect.AnomalyDetector(X_train)
mu, var = detector.fit_gaussian()

# Plot the contour plot of the fitted distribution
x, y = np.linspace(-2, 32, 100), np.linspace(-2, 32, 100)
z = np.multiply.outer(gaussian_probability(x, mu[0], var[0]), gaussian_probability(y, mu[1], var[1]))

axes[0][1].contourf(x, y, z, levels=20, cmap='RdGy_r')
axes[0][0].contour(x, y, z, colors='k')
axes[0][1].set_xlabel("x1")
axes[0][1].set_ylabel("x2")
axes[0][1].set_title("Gaussian distribution contour plot")

'''
Generate cross validation set. We need ~2000
non-anomalous examples and ~10 anomalous samples.
Give each example a label 0 (normal) or 1 (anomalous)
'''
X_cv = np.zeros(shape=(2010, 2))
y_cv = np.zeros(shape=(2010,))
X_cv_anomalous = np.zeros(shape=(10, 2))

normal_count, anomalous_count = 0, 0
while normal_count < 2000 or anomalous_count < 10:
    x_r = np.random.normal(loc=mean, scale=std)
    y_r = np.random.normal(loc=mean, scale=std)

    dist = (x_r - mean) ** 2 + (y_r - mean) ** 2 
    idx = normal_count + anomalous_count
    if dist <= 100 and normal_count < 2000:
        X_cv[idx] = np.array([x_r, y_r])
        y_cv[idx] = 0
        normal_count += 1

    if dist > 100 and anomalous_count < 10:
        X_cv[idx] = np.array([x_r, y_r])
        X_cv_anomalous[anomalous_count] = np.array([x_r, y_r])
        y_cv[idx] = 1
        anomalous_count += 1
    
axes[1][0].axis([-2, 32, -2, 32])
axes[1][0].scatter(X_cv[:,0], X_cv[:,1], c='g', marker="*")
axes[1][0].scatter(X_cv_anomalous[:,0], X_cv_anomalous[:,1], c='r', marker="*")
axes[1][0].set_xlabel("x1")
axes[1][0].set_ylabel("x2")
axes[1][0].set_title("Cross validation set")
axes[1][0].contour(x, y, z, colors='k')

'''
Tune parameter ε by calculating F1-score

prec = tp / (tp + fp)

rec = tp / (tp + fn)

F1 = 2 * prec * rec / (prec + rec)
'''

p1 = gaussian_probability(X_cv[:,0], mu[0], var[0])
p2 = gaussian_probability(X_cv[:,1], mu[1], var[1])
prob_val = p1 * p2
print(prob_val.shape)

best_epsilon = 0
best_f1 = 0
f1 = 0

step_size = (max(prob_val) - min(prob_val)) / 1000
for epsilon in np.arange(min(prob_val), max(prob_val), step_size):
    predictions = (prob_val < epsilon)

    tp = np.sum((predictions == 1) & (y_cv == 1))
    fp = np.sum((predictions == 1) & (y_cv == 0))
    fn = np.sum((predictions == 0) & (y_cv == 1))

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    f1 = 2 * prec * rec / (prec + rec)

    if f1 > best_f1:
        best_f1 = f1
        best_epsilon = epsilon

'''
Generate cross validation set. We need ~2000
non-anomalous examples and ~10 anomalous samples.
Give each example a label 0 (normal) or 1 (anomalous)
'''

X_test = np.zeros(shape=(2010, 2))
normal_count, anomalous_count = 0, 0

while normal_count < 2000 or anomalous_count < 10:
    x_r = np.random.normal(loc=mean, scale=std)
    y_r = np.random.normal(loc=mean, scale=std)

    dist = (x_r - mean) ** 2 + (y_r - mean) ** 2 
    idx = normal_count + anomalous_count

    if dist <= 100 and normal_count < 2000:
        X_test[idx] = np.array([x_r, y_r])
        normal_count += 1

    if dist > 100 and anomalous_count < 10:
        X_test[idx] = np.array([x_r, y_r])
        anomalous_count += 1


test_prob = gaussian_probability(X_test[:,0], mu[0], var[0]) * gaussian_probability(X_test[:,1], mu[1], var[1])
test_predictions = (test_prob < best_epsilon)
X_test_anomalous = X_test[test_predictions] 

axes[1][1].axis([-2, 32, -2, 32])
axes[1][1].scatter(X_test[:,0], X_test[:,1], c='g', marker="*")
axes[1][1].scatter(X_test_anomalous[:,0], X_test_anomalous[:,1], c='r', marker="*")
axes[1][1].set_xlabel("x1")
axes[1][1].set_ylabel("x2")
axes[1][1].set_title("Test set")
axes[1][1].contour(x, y, z, colors='k')

plt.show()