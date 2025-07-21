
#Logistic Regression to Predict Market Regime Changes using Stochastic Indicators
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Download S&P 500 data
data = yf.download("^GSPC", start="2018-01-01", end="2023-01-01")
data['Return'] = data['Close'].pct_change()

# Step 2: Calculate stochastic features
data['Volatility'] = data['Return'].rolling(10).std()
data['Sharpe'] = data['Return'].rolling(10).mean() / data['Volatility']
data['Drift'] = data['Return'].rolling(10).mean()
data['Momentum'] = data['Close'] - data['Close'].rolling(10).mean()

# Step 3: Create binary target (Bull = 1 if next 5-day return > 0)
data['Future_Return'] = data['Close'].pct_change(periods=5).shift(-5)
data['Regime'] = (data['Future_Return'] > 0).astype(int)

# Step 4: Clean data
data = data.dropna()

# Step 5: Prepare training data
X = data[['Volatility', 'Sharpe', 'Drift', 'Momentum']]
y = data['Regime']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

# Step 6: Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Plot predicted regime
data_test = data.iloc[-len(X_test):].copy()
data_test['Predicted_Regime'] = y_pred
data_test['Close'].plot(figsize=(14,6), label='S&P 500 Price', color='black')
plt.scatter(data_test.index, data_test['Close'],
            c=data_test['Predicted_Regime'], cmap='coolwarm', label='Predicted Regime')
plt.title('Predicted Market Regimes (Red = Bear, Blue = Bull)')
plt.legend()
plt.grid(True)
plt.show()

# Use stochastic calculus-derived indicators (like volatility, moving standard deviation, returns) to predict market regime:
#
# Regime 1 (Bull): Market expected to go up
#
# Regime 0 (Bear): Market expected to go down
#
# This approximates a binary classification of states modeled in regime-switching models or Hidden Markov Models — but here, we use logistic regression.