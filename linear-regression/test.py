import numpy as np
import linearmodel

sample_size = 12
X_train = np.random.randint(low=-1028, high=1027, size=sample_size)
Y_train = np.random.randint(low=-1028, high=1027, size=sample_size)

regr = linearmodel.LinearModel()
regr.fit(X_train, Y_train)

X_predict = np.random.randint(low=-1028, high=1027, size=sample_size)

Y_predict = regr.predict(X_predict)

print(Y_predict)
