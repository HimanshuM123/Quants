import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime

def download_data(stock, start_date, end_date):
    # Download stock data
    ticker = yf.download(stock, start=start_date, end=end_date)

    # Ensure 'Close' column exists
    if 'Close' not in ticker:
        print(f"Warning: No 'Close' data found for {stock}")
        return pd.DataFrame()

    return ticker[['Close']].rename(columns={'Close': stock})

# Calculate 1-day Value at Risk
def calculate_var(position, c, mu, sigma):
    var = position * (mu - sigma * norm.ppf(1-c))
    return var
#$25409.15 --> we can loose at most this amount tommorow

# this is how we calculate the VaR for any days in future
def calculate_var_nthDay(position, c, mu, sigma,n):
    var2 = position * (mu * n - sigma * np.sqrt(n) * norm.ppf(1 - c))
    return var2

if __name__ == '__main__':
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2018, 1, 1)

    stock_symbol = 'C'  # Citigroup
    stock_data = download_data(stock_symbol, start, end)

    if not stock_data.empty:
        # Calculate log returns
        stock_data['returns'] = np.log(stock_data[stock_symbol] / stock_data[stock_symbol].shift(1))
        stock_data = stock_data.dropna()

        print(stock_data.head())

        # Investment
        S = 1e6  # $1,000,000
        # Confidence level
        c = 0.95
        # Assume daily returns are normally distributed
        mu = stock_data['returns'].mean()
        sigma = stock_data['returns'].std()

        var = calculate_var(S, c, mu, sigma)
        print('Value at Risk (1-day, 95%% confidence): $%0.2f' % var)

        var_n=calculate_var_nthDay(S,c,mu,sigma,10)
        print('Value at Risk (10-day, 95%% confidence): $%0.2f' % var_n)
