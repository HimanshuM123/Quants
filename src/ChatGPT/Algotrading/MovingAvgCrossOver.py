import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download data
symbol = 'AAPL'
df = yf.download(symbol, start='2022-01-01', end='2023-12-31')

# Step 2: Calculate Moving Averages
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# Step 3: Generate Buy/Sell Signals
df['Signal'] = 0
df.loc[df['SMA_10'] > df['SMA_50'], 'Signal'] = 1  # Buy
df.loc[df['SMA_10'] < df['SMA_50'], 'Signal'] = -1 # Sell

# Step 4: Plot
plt.figure(figsize=(14,7))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['SMA_10'], label='SMA 10', alpha=0.75)
plt.plot(df['SMA_50'], label='SMA 50', alpha=0.75)
plt.title(f"{symbol} - Moving Average Crossover")
plt.legend()
plt.grid()
plt.show()
