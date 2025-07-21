import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_steps = 100
n_paths = 10
mu = 0.1
sigma = 0.2
S0 = 100
dt = 1 / n_steps
time = np.linspace(0, 1, n_steps)

# Simulate GBM paths
gbm_paths = np.zeros((n_paths, n_steps))
for i in range(n_paths):
    prices = [S0]
    for _ in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        St = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        prices.append(St)
    gbm_paths[i] = prices

# Expected value curve
expected_curve = S0 * np.exp(mu * time)

# Average and Std Dev across all paths
average_path = gbm_paths.mean(axis=0)
std_dev_path = gbm_paths.std(axis=0)
upper_band = average_path + std_dev_path
lower_band = average_path - std_dev_path

# Plotting
plt.figure(figsize=(12, 6))
for i in range(n_paths):
    plt.plot(time, gbm_paths[i], alpha=0.4, label=f"Path {i+1}" if i == 0 else "")
plt.plot(time, expected_curve, 'k--', linewidth=2.5, label="Expected Value (E[S(t)])")
plt.plot(time, average_path, 'r-', linewidth=2, label="Average of Simulated Paths")
plt.fill_between(time, lower_band, upper_band, color='red', alpha=0.2, label="±1 Std Dev")

plt.title("GBM: Simulated Paths, Expected Value, and Confidence Band")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
