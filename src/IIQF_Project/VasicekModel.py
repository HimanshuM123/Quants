import numpy as np
import matplotlib.pyplot as plt

# Vasicek model parameters
a = 0.1        # speed of mean reversion
b = 0.05       # long-term mean
sigma = 0.02   # volatility
r0 = 0.03      # initial interest rate
T = 10         # total time in years
dt = 0.01      # time step
N = int(T / dt)
t = np.linspace(0, T, N)

# Simulate path
r = np.zeros(N)
r[0] = r0
for i in range(1, N):
    dr = a * (b - r[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    r[i] = r[i-1] + dr

# Plot
plt.plot(t, r)
plt.title("Vasicek Interest Rate Simulation")
plt.xlabel("Time (Years)")
plt.ylabel("Short Rate")
plt.grid(True)
plt.show()
