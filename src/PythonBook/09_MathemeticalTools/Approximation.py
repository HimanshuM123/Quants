import numpy as np
import matplotlib.pyplot as plt

def f(x):
    val = np.sin(x)+ 0.5 * x
    return val

x = np.linspace(-2 * np.pi , 2 * np.pi,50)
plt.plot(x, f(x), 'b')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.show()