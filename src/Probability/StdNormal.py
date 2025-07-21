import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt



sample_size=500
rn1=npr.standard_normal(sample_size)

fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize=(7,7))

ax1.hist(rn1,bins=25)
ax1.set_title('Standard Normal')
ax1.set_ylabel('frequency')
ax1.grid(True)
plt.show()