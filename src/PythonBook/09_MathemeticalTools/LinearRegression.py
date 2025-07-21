import numpy as np
import matplotlib.pyplot as plt



def f(x):
    r = np.sin(x)+0.5*x
    return r


x = np.linspace(-2 * np.pi, 2 * np.pi,50)
reg = np.polyfit(x, f(x), deg=1)
# reg = np.polyfit(x, f(x), deg=5)
ry= np.polyval(reg,x)
plt.plot(x , f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()




