import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Download stock and market data
stock = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
market = yf.download("^GSPC", start="2020-01-01", end="2023-01-01")  # S&P 500

print(stock.columns)

# Calculate daily returns
stock['Return'] = stock['Close'].pct_change()
market['Market_Return'] = market['Close'].pct_change()

# Merge and clean
df = pd.concat([stock['Return'], market['Market_Return']], axis=1).dropna()
df.columns = ['Stock_Return', 'Market_Return']

# Regression (CAPM)
X = sm.add_constant(df['Market_Return'])
y = df['Stock_Return']
model = sm.OLS(y, X).fit()

# Show results
print(model.summary())

# Plot regression line
plt.scatter(df['Market_Return'], df['Stock_Return'], alpha=0.5)
plt.plot(df['Market_Return'], model.predict(X), color='red')
plt.xlabel('Market Return')
plt.ylabel('Stock Return')
plt.title('CAPM Regression: AAPL vs S&P 500')
plt.grid(True)
plt.show()
