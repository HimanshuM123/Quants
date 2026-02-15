#💻 Python Example: Predict Stock Movement Using SVM

import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np

# Step 1: Load stock data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
data['Return'] = data['Close'].pct_change()
print('Return',data['Return'])
data['Lag1'] = data['Return'].shift(1) #shifts values in a Series (or DataFrame) by n time steps,
print('Lag1',data['Lag1'])
data['Lag2'] = data['Return'].shift(2)
print('Lag2',data['Lag2'])
#Lag1 → Yesterday’s return
#Lag2 → Return from 2 days ago
data['Volatility'] = data['Return'].rolling(5).std() #On Day 5, Volatility = std of returns from Day 1 to Day 5
data['Direction'] = (data['Return'].shift(-1) > 0).astype(int)
data.dropna(inplace=True)

#hyperplane
# w1*Lag1+w2*Lag2+w3*Volatility+b=0

# Step 2: Feature matrix & target
X = data[['Lag1', 'Lag2', 'Volatility']]
y = data['Direction']

# Step 3: Standardize features
#This code is used to standardize your feature data X so that each feature has:
# Mean = 0
# Standard deviation = 1
# This process is also called Z-score normalization.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.3)

# Step 5: Train SVM classifier
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Step 6: Predict & evaluate
y_pred = model.predict(X_test)
print("Predicted value",y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred))
