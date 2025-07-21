import numpy as np
import matplotlib.pyplot as plt

n = np.random.standard_normal((1000,2))
plt.figure(figsize=(7,5))
plt.hist(n , label=['1st','2nd'], bins=25)
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Histogram')
plt.show()