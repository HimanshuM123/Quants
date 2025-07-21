import numpy as np
import matplotlib.pyplot as plt

n = np.random.standard_normal((1000,2))

plt.figure(figsize=(7,5))
plt.scatter(n[:,0], n[:,1], marker='o')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
plt.show()


