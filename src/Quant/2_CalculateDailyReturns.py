import pandas as pd
import yfinance as yf



data = yf.download("AAPL",start="2020-01-01", end="2024-12-31")

data['Return'] = data["Close"].pct_change()

print(data[["Close","Return"]].dropna().head())