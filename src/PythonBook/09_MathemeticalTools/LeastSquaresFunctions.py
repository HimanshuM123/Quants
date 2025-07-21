from cProfile import label

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    r = np.sin(x)+0.5*x
    return r


x = np.linspace(-2 * np.pi, 2 * np.pi,50)
matrix = np.zeros((3+1, len(x)))
matrix[3,:]= x**3
matrix[2,:]= x**2
matrix[1,:]=x
matrix[0,:]=1
reg = np.linalg.lstsq(matrix.T, f(x))[0]
print(reg)

ry = np.dot(reg,matrix)
plt.plot(x,f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')

plt.show()


