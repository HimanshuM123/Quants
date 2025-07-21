# Generate a Brownian motion (Wiener process).
# Define the filtration Ft  as the information up to time
# Plot progressive knowledge of Wt as time unfolds.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
np.random.seed(42)
T = 1.0        # Time horizon
N = 1000       # Number of time steps
dt = T / N     # Time step size
t = np.linspace(0, T, N)  # Time grid

# Simulate Brownian motion W_t
dW = np.sqrt(dt) * np.random.randn(N)  # Brownian increments
W = np.cumsum(dW)  # Standard Brownian motion

# Visualization of Filtration (Progressive Information)
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(1, 5):  # Show filtration at different time snapshots
    ax.plot(t[:i * 250], W[:i * 250], label=f"Filtration at t={round(i*T/4, 2)}")

ax.set_title("Filtration in Brownian Motion")
ax.set_xlabel("Time (t)")
ax.set_ylabel("W_t (Brownian Motion)")
ax.legend()
plt.show()



# Explanation
# ✅ Brownian Motion Wt  – Simulated using random normal increments.
# ✅ Filtration Ft– Progressively reveals information at different time steps.
# ✅ Visualization – Shows how at each time t, only past values are known.

# Key Insights t=0, no information is available.
# At t=0.25,0.5,0.75,1 we progressively gain more history of Wt
# This mimics real-world financial markets, where traders make decisions based on past and current data.

