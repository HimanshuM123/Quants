#Predict Stock Direction Using KNN Classifier
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Download data
data = yf.download("AAPL", start="2019-01-01", end="2023-12-31")
data['Return'] = data['Close'].pct_change()

# Step 2: Feature Engineering
data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)
data['Volatility'] = data['Return'].rolling(5).std()
data['Momentum'] = data['Close'] - data['Close'].rolling(10).mean()

# RSI Calculation
delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Step 3: Target Variable (Next Day Up = 1, Down = 0)
data['Target'] = (data['Return'].shift(-1) > 0).astype(int)

# Drop NA
data.dropna(inplace=True)

# Step 4: Train-Test Split
features = ['Lag1', 'Lag2', 'Volatility', 'Momentum', 'RSI']
X = data[features]
y = data['Target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Step 5: Train KNN model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Step 6: Predict and Evaluate
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize Predictions
plt.figure(figsize=(14, 6))
plt.plot(data.index[-len(y_test):], y_test.values, label='Actual Direction', color='red', alpha=0.5)
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Direction', color='blue', alpha=0.6)
plt.title("KNN Predicted vs Actual Stock Movement (1=Up, 0=Down)")
plt.xlabel("Date")
plt.ylabel("Direction")
plt.legend()
plt.grid(True)
plt.show()
