#🔥 Python Code: Logistic Regression Backtest


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Download stock data
data = yf.download("TSLA", start="2020-01-01", end="2023-01-01")
data['Return'] = data['Close'].pct_change()

# Step 2: Create target: 1 if next day return > 0 else 0
data['Target'] = (data['Return'].shift(-1) > 0).astype(int)

# Step 3: Features - lagged returns and momentum
data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)
data['Momentum'] = data['Close'] - data['Close'].shift(5)

data.dropna(inplace=True)

# Step 4: Features and target
X = data[['Lag1', 'Lag2', 'Momentum']]
y = data['Target']

# Step 5: Train-test split (no shuffle for time series)
split_index = int(len(data)*0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
dates_test = data.index[split_index:]

# Step 6: Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Predict probabilities on test set
probs = model.predict_proba(X_test)[:, 1]

# Step 8: Generate trading signals
# Buy if predicted prob > 0.5, else no position (0)
signals = (probs > 0.5).astype(int)

# Step 9: Calculate strategy returns
# Strategy return = position yesterday * actual return today
# Shift signals by 1 day to avoid lookahead bias
strategy_returns = signals.shift(1) * data['Return'].iloc[split_index:]

# Step 10: Calculate cumulative returns
cumulative_strategy = (1 + strategy_returns).cumprod() - 1
cumulative_buy_and_hold = (1 + data['Return'].iloc[split_index:]).cumprod() - 1

# Step 11: Performance metrics
def sharpe_ratio(returns, freq=252):
    return (returns.mean() / returns.std()) * np.sqrt(freq)

sharpe = sharpe_ratio(strategy_returns.dropna())
print(f"Strategy Sharpe Ratio: {sharpe:.2f}")

# Step 12: Plot cumulative returns
plt.figure(figsize=(12,6))
plt.plot(cumulative_strategy, label='Logistic Regression Strategy')
plt.plot(cumulative_buy_and_hold, label='Buy and Hold')
plt.title('Cumulative Returns: Strategy vs Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()


#  Explanation:
# We use lagged returns and momentum as features.
#
# Logistic regression predicts the probability of next-day positive return.
#
# We go long if predicted probability > 0.5; otherwise, stay flat.
#
# Returns are calculated based on the actual next-day return weighted by the position.
#
# We compare our strategy to a simple buy-and-hold benchmark.
#
# We compute Sharpe ratio to measure risk-adjusted return.