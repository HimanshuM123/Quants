import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Generate the data
X = np.random.randn(500,2)

# Estimate the bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)

# Initialize the Mean-Shift algorithm
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# Train the model
ms.fit(X)

# Visualize the results
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

# Plot the data points and the centroids
plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='summer')
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='*',
s=200, c='r')
plt.show()