import numpy as np
import matplotlib.pyplot as plt
import help

sample_size, mean, std = 1000, 0., 10.

np.random.seed(0)

X_data = np.random.normal(loc=mean, scale=std, size=(sample_size, 2))

X_train, X_cv, X_test = help.split_dataset(X_data)

print(f"First 5 elements of the training set: \n {X_train[:5]}")
print(f"First 5 elements of the cross validation set: \n {X_train[:5]}")
print(f"First 5 elements of the test set: \n {X_train[:5]}")

fig, axes = plt.subplots(nrows=2, ncols=2)
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
#   ax = fig.add_axes((left, bottom, width, height))

axes[0][0].scatter(x=X_train[:, 0], y=X_train[:, 1], c='b')
axes[0][0].set_xlabel('x1')
axes[0][0].set_ylabel('x2')
axes[0][0].set_title('Training set scatter plot')
plt.show()


