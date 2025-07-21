import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def gamma_bs(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# Parameters
K = 100       # Strike price
T = 1         # Time to maturity (1 year)
r = 0.05      # Risk-free rate (5%)
sigma = 0.2   # Volatility (20%)

# Range of underlying prices
S_values = np.linspace(50, 150, 500)

# Calculate Gamma for each underlying price
gamma_values = gamma_bs(S_values, K, T, r, sigma)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(S_values, gamma_values, label='Gamma')
plt.xlabel('Underlying Price (S)')
plt.ylabel('Gamma')
plt.title('Gamma of a European Call Option vs Underlying Price')
plt.grid(True)
plt.legend()
plt.show()
