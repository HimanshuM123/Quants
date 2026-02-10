import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Create moon data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.15, min_samples=5)
labels = dbscan.fit_predict(X)

# Define 4 colors (extra labels will reuse colors)
colors = ['red', 'blue', 'green', 'orange']

# Map cluster labels to colors
unique_labels = np.unique(labels)
color_map = {}

for i, label in enumerate(unique_labels):
    if label == -1:
        color_map[label] = 'black'  # noise
    else:
        color_map[label] = colors[i % 4]

# Plot
for label in unique_labels:
    plt.scatter(
        X[labels == label, 0],
        X[labels == label, 1],
        c=color_map[label],
        label=f'Cluster {label}'
    )

plt.legend()
plt.show()
