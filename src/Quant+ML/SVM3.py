import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01", progress=False)
data['Return'] = data['Close'].pct_change()
data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)
data['Direction'] = (data['Return'].shift(-1) > 0).astype(int)
data.dropna(inplace=True)

# 2. Use only Lag1 and Lag2 for visualization
X = data[['Lag1', 'Lag2']]
y = data['Direction']

# 3. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_scaled, y)

# 5. Create mesh grid
h = 0.005
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 6. Predict over mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 7. Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("SVM Decision Boundary (Lag1 vs Lag2)")
plt.xlabel("Lag1 (Standardized)")
plt.ylabel("Lag2 (Standardized)")
plt.grid(True)
plt.tight_layout()
plt.show()
