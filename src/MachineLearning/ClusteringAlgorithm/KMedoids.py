from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=500, centers=3, random_state=42)

kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(X)

