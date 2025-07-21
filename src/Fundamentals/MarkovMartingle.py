import numpy as np
import matplotlib.pyplot as plt

# Ensure reproducibility
np.random.seed(42)

# Simulation Parameters
n_steps = 100        # Number of time steps
n_paths = 5          # Number of sample paths

# ---------------------------------------
# 1. 📈 Martingale Simulation (Fair Game)
# ---------------------------------------
# A martingale is a fair game where the expected future value is equal to the current value.
# We'll simulate it using a simple coin toss game (+1 for heads, -1 for tails)

martingale_paths = np.zeros((n_paths, n_steps))

for i in range(n_paths):
    steps = np.random.choice([-1, 1], size=n_steps)      # Fair coin: +1 or -1
    martingale_paths[i] = np.cumsum(steps)               # Cumulative sum (starts from 0)

# ---------------------------------------
# 2. 📉 Markov Process Simulation (GBM)
# ---------------------------------------
# Geometric Brownian Motion (GBM) is a Markov process — future depends only on the present state.
# It is commonly used to model stock prices.

# Parameters for GBM
mu = 0.1             # Drift (expected return)
sigma = 0.2          # Volatility
S0 = 100             # Initial stock price
dt = 1 / n_steps     # Time increment

markov_paths = np.zeros((n_paths, n_steps))

for i in range(n_paths):
    prices = [S0]
    for _ in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
        # GBM formula: S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*dW)
        St = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        prices.append(St)
    markov_paths[i] = prices

# ---------------------------------------
# 📊 Plotting the Results
# ---------------------------------------

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: Martingale Paths
for path in martingale_paths:
    axs[0].plot(path, alpha=0.8)
axs[0].axhline(0, color='black', linestyle='--', linewidth=1)
axs[0].set_title("Martingale Process (Fair Coin Toss Game)")
axs[0].set_ylabel("Cumulative Winnings ($)")
axs[0].grid(True)

# Plot 2: Markov Paths (GBM)
for path in markov_paths:
    axs[1].plot(path, alpha=0.8)
axs[1].set_title("Markov Process (Geometric Brownian Motion)")
axs[1].set_ylabel("Stock Price")
axs[1].set_xlabel("Time Steps")
axs[1].grid(True)

plt.tight_layout()
plt.show()
