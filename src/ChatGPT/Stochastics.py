import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100        # Initial stock price
mu = 0.1        # Expected return (10%)
sigma = 0.2     # Volatility (20%)
T = 1.0         # Time horizon (1 year)
N = 252         # Number of time steps (daily)
M = 5           # Number of simulation paths

dt = T / N
t = np.linspace(0, T, N + 1)

# Initialize paths: M rows, N+1 columns
S = np.zeros((M, N + 1))
S[:, 0] = S0

# Simulate M paths
for i in range(M):
    for j in range(1, N + 1):
        Z = np.random.normal()
        S[i, j] = S[i, j-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * Z * np.sqrt(dt))

# Plot the paths
plt.figure(figsize=(10, 6))
for i in range(M):
    plt.plot(t, S[i], lw=1.5, alpha=0.8)

plt.title("Simulated Stock Price Paths (Geometric Brownian Motion)")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()
