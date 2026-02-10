from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(n_samples=2000, centers=4, cluster_std=0.60, random_state=0)
#y -> cluster number

# Cluster the data using OPTICS
optics = OPTICS(min_samples=50, xi=.05)
optics.fit(X)

# Plot the results
labels = optics.labels_
plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='turbo')
plt.show()