import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Simulate historical options data (1000 rows)
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'Delta': np.random.uniform(0.1, 0.9, n),
    'Gamma': np.random.uniform(0.01, 0.1, n),
    'Vega': np.random.uniform(0.1, 0.6, n),
    'Theta': np.random.uniform(-1.0, -0.01, n),
    'Rho': np.random.uniform(0.01, 0.2, n),
    'IV': np.random.uniform(0.1, 0.6, n),
    'Moneyness': np.random.uniform(0.8, 1.2, n),  # Spot / Strike
    'TimeToMaturity': np.random.uniform(5, 180, n),
    'OptionPrice': np.random.uniform(1.0, 25.0, n)  # Target: Historical premium
})

# Step 2: Define features and target
features = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'IV', 'Moneyness', 'TimeToMaturity']
X = data[features]
y = data['OptionPrice']

# Step 3: Train a KNN Regressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

# Step 4: Predict price of a new option
new_option = pd.DataFrame({
    'Delta': [0.5],
    'Gamma': [0.05],
    'Vega': [0.3],
    'Theta': [-0.2],
    'Rho': [0.1],
    'IV': [0.25],
    'Moneyness': [1.05],
    'TimeToMaturity': [30]
})

predicted_price = knn.predict(new_option)[0]
print(f"📈 Predicted Option Price (KNN): ${predicted_price:.2f}")

# Step 5: Visualize K nearest neighbors
distances, indices = knn.kneighbors(new_option)
neighbors = X_train.iloc[indices[0]]
neighbors['Price'] = y_train.iloc[indices[0]].values

plt.figure(figsize=(10,6))
plt.bar(range(1, 11), neighbors['Price'], color='skyblue')
plt.axhline(predicted_price, color='red', linestyle='--', label='Predicted Price')
plt.title("Option Prices of Nearest Neighbors")
plt.xlabel("Neighbor Index")
plt.ylabel("Option Price")
plt.legend()
plt.grid(True)
plt.show()


# Predict the price of a new option by finding similar historical options based on:
#
# Greeks: Delta, Gamma, Theta, Vega, Rho
#
# Market conditions: Implied volatility, spot price, time to maturity
#
# Then, use KNN regression to estimate the option premium.

# Options with similar Greeks behave similarly in response to market changes.
#
# Matching on Greeks allows pricing without assuming Black-Scholes or other models.
#
# Especially useful for exotic options, illiquid markets, or missing model assumptions.