import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Download real stock data (e.g., Apple - AAPL)
ticker = 'AAPL'
data = yf.download(ticker, start='2023-01-01', end='2024-01-01')
real_prices = data['Close']
real_dates = real_prices.index
real_log_returns = np.log(real_prices / real_prices.shift(1)).dropna()

# Step 2: Estimate μ (drift) and σ (volatility)
mu = real_log_returns.mean() * 252        # Annualized drift
sigma = real_log_returns.std() * np.sqrt(252)  # Annualized volatility
# print(f"Estimated μ = {mu:.4f}, σ = {sigma:.4f}")

# Step 3: Simulate GBM path using same time frame
S0 = real_prices.iloc[0]
T = 1  # 1 year
N = len(real_prices)
dt = T / N
t = np.linspace(0, T, N)
simulated_prices = [S0]

for i in range(1, N):
    Z = np.random.normal()
    St = simulated_prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * Z * np.sqrt(dt))
    simulated_prices.append(St)

# Step 4: Plot Real vs Simulated Path
plt.figure(figsize=(12, 6))
plt.plot(real_dates, real_prices.values, label='Real AAPL Price', linewidth=2)
plt.plot(real_dates, simulated_prices, label='Simulated GBM Path', linestyle='--', linewidth=2)
plt.title(f"AAPL: Real vs Simulated Price Path (GBM)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
