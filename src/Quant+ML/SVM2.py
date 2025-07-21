import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Black-Scholes Pricing Function (for labels)
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# Step 2: Simulate Option Pricing Data
np.random.seed(42)
n_samples = 1000
S = np.random.uniform(90, 110, n_samples)         # Spot price
K = np.random.uniform(90, 110, n_samples)         # Strike price
T = np.random.uniform(0.01, 1.0, n_samples)        # Time to maturity in years
sigma = np.random.uniform(0.1, 0.5, n_samples)     # Implied volatility
r = 0.01                                           # Risk-free rate

X = np.column_stack((S, K, T, sigma))
y = black_scholes_call(S, K, T, r, sigma)

# Step 3: Prepare SVR Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

svr = SVR(kernel='rbf', C=100, epsilon=0.5)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

# Step 4: Plot Predictions
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='SVR Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Fit')
plt.xlabel("Actual Option Price (BSM)")
plt.ylabel("Predicted Option Price (SVR)")
plt.title("SVR vs Black-Scholes Option Prices")
plt.legend()
plt.grid(True)
plt.show()
