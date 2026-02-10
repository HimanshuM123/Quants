import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import summer
from sklearn.cluster import KMeans

X=np.random.rand(300,2)


plt.figure(figsize=(7.5,3.5))
plt.scatter(X[:,0],X[:,1],s=20, cmap='summer')
plt.show()

kmeans = KMeans(n_clusters=3, max_iter=100)
kmeans.fit(X)

plt.figure(figsize=(7.5, 3.5))
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, s=20, cmap='summer')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
marker='x', c='r', s=50, alpha=0.9)
plt.show()