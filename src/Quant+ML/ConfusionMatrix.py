from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load stock data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
data['Return'] = data['Close'].pct_change()
data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)
data['Volatility'] = data['Return'].rolling(5).std()
data['Direction'] = (data['Return'].shift(-1) > 0).astype(int)
data.dropna(inplace=True)

# Features and target
X = data[['Lag1', 'Lag2', 'Volatility']]
y = data['Direction']

# Standardize and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.3)

# Train SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix – SVM for Stock Direction")
plt.show()


# Suppose your output is:
#
# Predicted   ↓    Down   Up
# Actual ↓  Down   80     20
#            Up     30     70
# 80 stocks that actually went down were correctly predicted.
#
# 70 stocks that went up were correctly predicted.
#
# 30 missed opportunities (predicted down, went up).
#
# 20 false positives (predicted up, but fell).