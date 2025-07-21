import numpy as np
import pandas as pd
import yfinance as yf
import datetime


def download_data(stock, start_date, end_date):
    ticker = yf.download(stock, start=start_date, end=end_date)

    # Ensure the column exists
    if 'Close' not in ticker:
        print(f"Warning: No 'Adj Close' data found for {stock}")
        return pd.DataFrame()

    return ticker[['Close']]


class ValueAtRiskMonteCarlo:
    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S  # Investment amount
        self.mu = mu  # Mean of returns
        self.sigma = sigma  # Standard deviation of returns
        self.c = c  # Confidence level
        self.n = n  # Time horizon in days
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.iterations])

        # Simulated future stock price after n days
        stock_price = self.S * np.exp(
            self.n * (self.mu - 0.5 * self.sigma ** 2) +
            self.sigma * np.sqrt(self.n) * rand
        )

        # Sort the simulated prices
        stock_price = np.sort(stock_price)

        # Determine the percentile loss value
        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        # Value at Risk is initial investment minus the simulated lower bound
        return self.S - percentile


if __name__ == '__main__':
    S = 1e6  # Investment amount ($1,000,000)
    c = 0.95  # Confidence level
    n = 1  # 1 day
    iterations = 100_000

    start_date = datetime.datetime(2014, 1, 1)
    end_date = datetime.datetime(2017, 10, 15)

    # Download data
    citi = download_data('C', start_date, end_date)

    if not citi.empty:
        citi['returns'] = citi['Close'].pct_change()
        citi = citi.dropna()  # Drop first row with NaN return

        mu = citi['returns'].mean()
        sigma = citi['returns'].std()

        model = ValueAtRiskMonteCarlo(S, mu, sigma, c, n, iterations)
        var = model.simulation()
        print('Value at Risk with Monte Carlo simulation: $%.2f' % var)
    else:
        print("Data download failed.")
