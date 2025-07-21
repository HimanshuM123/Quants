import numpy.random as npr
import matplotlib.pyplot as plt

sample_size=500
rn=npr.poisson(lam=1.0,size=sample_size)

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(7,7))

ax.hist(rn,bins=25)
ax.set_title('Poisson')
ax.grid(True)
plt.show()