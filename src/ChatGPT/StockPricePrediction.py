import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Download historical stock data
ticker = "AAPL"
df = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Prepare data (using closing prices)
data = df["Close"].values.reshape(-1, 1)

# Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i])
    y.append(data_scaled[i])

X, y = np.array(X), np.array(y)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, batch_size=32, epochs=10)

# Predict next day's price
next_day = model.predict(X[-1].reshape(1, 60, 1))
print("Predicted Next Day Price:", scaler.inverse_transform(next_day))
