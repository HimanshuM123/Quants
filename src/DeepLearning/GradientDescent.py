import numpy as np

# Define the gradient function for f(x) = x^2
def gradient(x):
    return 2 * x

# Gradient descent optimization function
def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector

# Initial point
start = 5.0
# Learning rate
learn_rate = 0.1
# Number of iterations
n_iter = 50
# Tolerance for convergence
tolerance = 1e-6

# Gradient descent optimization
result = gradient_descent(gradient, start, learn_rate, n_iter, tolerance)
print(result)