#Pattern Recognition in Technical Trading using KNN

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Step 1: Download data
data = yf.download("AAPL", start="2017-01-01", end="2023-01-01")
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Step 2: Define pattern window and forecast horizon
N = 5   # Look-back pattern length (e.g., last 5 days)
M = 3   # Look-ahead return (e.g., return over next 3 days)

# Step 3: Create pattern matrix
patterns = []
targets = []
print("Length of data ",len(data))
for i in range(N, len(data) - M):
    pattern = data['Return'].iloc[i - N:i].values
    future_return = data['Close'].iloc[i + M] / data['Close'].iloc[i] - 1
    patterns.append(pattern)
    targets.append(future_return)

X = np.array(patterns)
y = np.array(targets)

# Step 4: Train KNN regressor
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X, y)

# Step 5: Use latest pattern to predict forward return
latest_pattern = data['Return'].iloc[-N:].values.reshape(1, -1)
predicted_return = knn.predict(latest_pattern)[0]

# print(f"Predicted 3-day forward return based on similar patterns: {predicted_return:.2%}")

# Step 6: Visualize similar historical patterns
distances, indices = knn.kneighbors(latest_pattern)
plt.figure(figsize=(10, 6))
for idx in indices[0]:
    past_idx = idx + N
    pattern_prices = data['Close'].iloc[past_idx - N:past_idx + M].values
    base_price = pattern_prices[N-1]
    normalized = pattern_prices / base_price
    plt.plot(range(-N+1, M+1), normalized, alpha=0.5)
plt.axvline(0, color='black', linestyle='--')
plt.title("Top K Similar Historical Patterns to the Current One")
plt.xlabel("Days (0 = Today)")
plt.ylabel("Normalized Price Movement")
plt.grid(True)
plt.show()


#
# We encode each N-day pattern as a vector of returns.
#
# The target is the % change in price M days later.
#
# KNN finds the top K most similar historical return patterns.
#
# It averages their future returns to forecast the current outcome.
#
# We plot the matched patterns to visualize what historically happened in similar situations.