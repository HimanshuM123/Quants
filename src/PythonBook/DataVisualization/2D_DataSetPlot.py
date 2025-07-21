import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(2000)
y = np.random.standard_normal((20,2)).cumsum(axis=0)

print(y)

plt.plot(y, lw=1.5)
plt.plot(y, 'ro') # dots
plt.xlabel('index')
plt.ylabel('value')
plt.show()