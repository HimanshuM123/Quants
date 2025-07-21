import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
K = 100       # Strike price
T = 1         # Time to maturity
r = 0.05      # Risk-free rate
sigma = 0.2   # Volatility

# Range of stock prices
S = np.linspace(50, 150, 200)

# Compute d1 and N(d1)
d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
print("d1->",d1)
Nd1 = norm.cdf(d1)
print("Nd1->",Nd1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S, Nd1, label=r'$N(d_1)$ (Delta)', color='blue', linewidth=2)
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(K, color='red', linestyle='--', label='Strike Price (K)')

plt.title("Intuition Behind $N(d_1)$: Delta of a European Call Option")
plt.xlabel("Stock Price $S$")
plt.ylabel(r'$N(d_1)$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
