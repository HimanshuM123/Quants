import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def vega_bs(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# Parameters
K = 100       # Strike Price
T = 1         # Time to Maturity
r = 0.05      # Risk-Free Rate
sigma = 0.2   # Volatility

S_range = np.linspace(50, 150, 500)
vega_values = vega_bs(S_range, K, T, r, sigma)

plt.figure(figsize=(10, 6))
plt.plot(S_range, vega_values, label='Vega', color='purple')
plt.axvline(K, color='red', linestyle='--', label='Strike Price')
plt.title('Vega vs Stock Price')
plt.xlabel('Stock Price (S)')
plt.ylabel('Vega')
plt.grid(True)
plt.legend()
plt.show()
