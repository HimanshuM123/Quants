# Use Case:
# We’ll simulate Geometric Brownian Motion (GBM) to model stock prices and then use Linear Regression
# (a simple ML model) to predict future stock prices based on the historical data.
# This is a simplified example of how stochastic processes and machine learning can be used together in finance.

# Steps:
# Simulate Stock Price using GBM.
# Train a Linear Regression Model to predict future stock prices based on past values.
# Evaluate the model's performance.

# Geometric Brownian Motion (GBM):

# We simulate stock prices using the GBM formula: St=S0⋅exp((μ-1/2σ^2)t+σWt)
# Parameters:
# μ is the drift (expected return), and
# σ is the volatility.
# We use Brownian motion increments to simulate the random fluctuations

# Linear Regression:
#
# Input: Previous day's stock price.
# Output: The stock price at the next time step.
# Model: We train a Linear Regression model to predict the future price based on the lagged price (previous price).
# Evaluation:
#
# We compute the Mean Squared Error (MSE) to evaluate the accuracy of our predictions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Parameters for GBM
S0 = 100        # Initial stock price
mu = 0.05       # Drift (expected return)
sigma = 0.2     # Volatility
T = 1           # Time horizon (1 year)
N = 1000        # Number of time steps
dt = T / N      # Time step size
t = np.linspace(0, T, N)  # Time grid

# Simulate Geometric Brownian Motion
dW = np.sqrt(dt) * np.random.randn(N)  # Brownian increments
print("dW ",dW)
W = np.cumsum(dW)  # Standard Brownian motion
S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)  # Stock price evolution using GBM

# Create a DataFrame for the simulated stock price data
df = pd.DataFrame({'Time': t, 'StockPrice': S})

# Plot the simulated stock price
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Stock Price (Simulated)')
plt.title('Simulated Stock Price Using Geometric Brownian Motion')
plt.xlabel('Time (t)')
plt.ylabel('Stock Price (S_t)')
plt.legend()
plt.show()

# Prepare data for machine learning model (use past prices to predict future)
X = df['StockPrice'].shift(1).dropna().values.reshape(-1, 1)  # Lag 1 for previous price
y = df['StockPrice'][1:].values  # Next time step's price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict stock prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Plot the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Prices', linewidth=2)
plt.title('Stock Price Prediction Using Linear Regression')
plt.xlabel('Previous Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.show()
