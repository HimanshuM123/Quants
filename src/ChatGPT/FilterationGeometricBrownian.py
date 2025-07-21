#dSt=μStdt+σStdWt
# •  St = Stock price at time t
# •  μ = Drift (expected return)
# •  σ = Volatility
# •  Wt = Standard Brownian motion


import numpy as np
import matplotlib.pyplot as plt

# Parameters
np.random.seed(42)
T = 1.0         # Time horizon (1 year)
N = 1000        # Number of time steps
dt = T / N      # Time step size
t = np.linspace(0, T, N)  # Time grid

# GBM parameters
S0 = 100       # Initial stock price
mu = 0.1       # Expected return (drift)
sigma = 0.2    # Volatility

# Simulate Brownian Motion
dW = np.sqrt(dt) * np.random.randn(N)  # Brownian increments
W = np.cumsum(dW)  # Standard Brownian motion

# Simulate GBM using the exact solution: S_t = S0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)
S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

# Plot Filtration Over Time
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(1, 5):  # Show filtration at different times
    ax.plot(t[:i * 250], S[:i * 250], label=f"Filtration at t={round(i*T/4, 2)}")

ax.set_title("Filtration in Geometric Brownian Motion (Stock Prices)")
ax.set_xlabel("Time (t)")
ax.set_ylabel("Stock Price (S_t)")
ax.legend()
plt.show()
