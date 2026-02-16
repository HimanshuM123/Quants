import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def euclidean_distance(point1,point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def knn_predict(training_data, training_labels,test_point,k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i], training_data[i]))  # keep the point too
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]  # first k neighbors
    k_nearest_labels = [label for _, label, _ in k_nearest]
    #Take the first k elements from the sorted list → distances[:k]
    # Extract only the labels (ignore the distances)
    # k_nearest_labels now contains the labels of the k nearest neighbors
    # distances[:3] = [(0.5, 'A'), (0.7, 'A'), (1.2, 'B')]
    # k_nearest_labels = ['A', 'A', 'B']

    prediction = Counter(k_nearest_labels).most_common(1)[0][0]
    return prediction, k_nearest  # return both


training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [4, 5]
k = 3

prediction, k_nearest = knn_predict(training_data, training_labels, test_point, k)
print("Predicted label:", prediction)


# --- Plot ---
plt.figure(figsize=(6,6))

# Plot training points
for point, label in zip(training_data, training_labels):
    color = 'blue' if label=='A' else 'red'
    plt.scatter(point[0], point[1], c=color, s=100, label=f'Train {label}' if f'Train {label}' not in plt.gca().get_legend_handles_labels()[1] else "")

# Plot test point
plt.scatter(test_point[0], test_point[1], c='green', s=150, marker='*', label='Test Point')

# Draw circles to nearest neighbors
for dist, label, point in k_nearest:
    circle_color = 'blue' if label=='A' else 'red'
    circle = plt.Circle(point, dist, color=circle_color, fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)

plt.title(f"KNN Prediction (k={k}): {prediction}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.legend()
plt.show()