import numpy as np

# Example returns
market_returns = np.array([0.02, 0.01, -0.01, 0.03, 0.02])
stock_returns = np.array([0.025, 0.015, -0.02, 0.04, 0.03])

# Calculate beta
cov = np.cov(stock_returns, market_returns)[0, 1]
var = np.var(market_returns)
beta = cov / var
print(f"Beta: {beta:.2f}")


# Beta (β) is a measure of a stock’s systematic risk — how much its returns move
# relative to the overall market. It tells you how sensitive a stock is to market movements.