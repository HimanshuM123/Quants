import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def rho_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * T * np.exp(-r * T) * norm.cdf(d2)

# Parameters
K = 100
T = 1
r = 0.05
sigma = 0.2
S_range = np.linspace(50, 150, 500)
rho_values = rho_call(S_range, K, T, r, sigma)

plt.figure(figsize=(10, 6))
plt.plot(S_range, rho_values, label='Call Rho', color='darkgreen')
plt.axvline(K, linestyle='--', color='gray', label='Strike Price')
plt.xlabel("Stock Price (S)")
plt.ylabel("Rho")
plt.title("Call Option Rho vs Stock Price")
plt.grid(True)
plt.legend()
plt.show()
