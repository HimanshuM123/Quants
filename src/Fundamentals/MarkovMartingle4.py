import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_steps = 100
n_paths = 10

# Simulate Martingale Paths (Fair Coin Toss Game)
martingale_paths = np.zeros((n_paths, n_steps))
for i in range(n_paths):
    steps = np.random.choice([-1, 1], size=n_steps)  # +1 or -1 with equal probability
    martingale_paths[i] = np.cumsum(steps)

# Compute expected value and standard deviation at each step
expected_value = martingale_paths.mean(axis=0)
std_dev = martingale_paths.std(axis=0)
upper_band = expected_value + std_dev
lower_band = expected_value - std_dev
time_steps = np.arange(n_steps)

# Plotting the results
plt.figure(figsize=(12, 6))
for i in range(n_paths):
    plt.plot(time_steps, martingale_paths[i], alpha=0.4, label=f"Path {i+1}" if i == 0 else "")
plt.plot(time_steps, expected_value, 'r-', linewidth=2, label="Average of Paths (Expected Value)")
plt.fill_between(time_steps, lower_band, upper_band, color='red', alpha=0.2, label="±1 Std Dev")

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Martingale Process: Simulated Paths with Expected Value and Confidence Band")
plt.xlabel("Time Steps")
plt.ylabel("Cumulative Winnings")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
