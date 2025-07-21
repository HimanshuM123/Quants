import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_steps = 100
n_paths = 5
mu = 0.1
sigma = 0.2
S0 = 100
dt = 1 / n_steps

# -------------------------------
# 1. Generate Martingale Paths
# -------------------------------
martingale_paths = np.zeros((n_paths, n_steps))
for i in range(n_paths):
    steps = np.random.choice([-1, 1], size=n_steps)  # fair coin toss
    martingale_paths[i] = np.cumsum(steps)  # cumulative sum

# -------------------------------
# 2. Generate GBM (Markov) Paths
# -------------------------------
markov_paths = np.zeros((n_paths, n_steps))
for i in range(n_paths):
    prices = [S0]
    for _ in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        St = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        prices.append(St)
    markov_paths[i] = prices

# -------------------------------
# 3. Setup the Animation Figure
# -------------------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
lines_martingale = [axs[0].plot([], [], label=f'Path {i+1}')[0] for i in range(n_paths)]
lines_markov = [axs[1].plot([], [], label=f'Path {i+1}')[0] for i in range(n_paths)]

axs[0].set_xlim(0, n_steps)
axs[0].set_ylim(-30, 30)
axs[0].set_title("Martingale Process (Fair Coin Toss)")
axs[0].set_ylabel("Cumulative Winnings")
axs[0].grid(True)

axs[1].set_xlim(0, n_steps)
axs[1].set_ylim(80, 150)
axs[1].set_title("Markov Process (Geometric Brownian Motion)")
axs[1].set_ylabel("Stock Price")
axs[1].set_xlabel("Time Steps")
axs[1].grid(True)

# -------------------------------
# 4. Define Animation Update Function
# -------------------------------
def update(frame):
    for i in range(n_paths):
        lines_martingale[i].set_data(range(frame), martingale_paths[i, :frame])
        lines_markov[i].set_data(range(frame), markov_paths[i, :frame])
    return lines_martingale + lines_markov

# -------------------------------
# 5. Run and Save Animation
# -------------------------------
ani = FuncAnimation(fig, update, frames=n_steps, interval=100, blit=True)

# 💾 Save as GIF (requires Pillow)
ani.save("martingale_vs_markov.gif", writer=PillowWriter(fps=10))

# 💾 To save as MP4 (requires ffmpeg installed):
# ani.save("martingale_vs_markov.mp4", writer="ffmpeg", fps=10)

print("✅ Animation saved as 'martingale_vs_markov.gif'")
