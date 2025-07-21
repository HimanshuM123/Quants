import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def theta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 + term2

# Parameters
K = 100
T = 0.5   # 6 months
r = 0.05
sigma = 0.2

S_range = np.linspace(50, 150, 500)
theta_values = theta_call(S_range, K, T, r, sigma)

plt.figure(figsize=(10, 6))
plt.plot(S_range, theta_values, label='Call Theta', color='brown')
plt.axvline(K, linestyle='--', color='gray', label='Strike Price')
plt.xlabel("Stock Price (S)")
plt.ylabel("Theta")
plt.title("Theta vs Stock Price (Call Option)")
plt.grid(True)
plt.legend()
plt.show()
