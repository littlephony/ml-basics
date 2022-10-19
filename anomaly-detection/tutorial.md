# Anomaly detection algorithm

The basic outline of the algorithm:

1. Fit Gaussian (normal) distribution to an unlabled training set

2. Use a labled cross validation set with a small number of anomalous examples to tune parameter ε 

3. Test the algorithm on a labled test set with a small number of anomalous examples
