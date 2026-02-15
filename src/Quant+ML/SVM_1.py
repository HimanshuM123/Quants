# 💻 Predict Stock Movement Using SVM + Plot Decision Boundary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# Step 1: Download stock data
# -----------------------------
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# -----------------------------
# Step 2: Feature Engineering
# -----------------------------
data['Return'] = data['Close'].pct_change()

data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)

# Target variable (1 = Up tomorrow, 0 = Down tomorrow)
data['Direction'] = (data['Return'].shift(-1) > 0).astype(int)

data.dropna(inplace=True)

# -----------------------------
# Step 3: Select ONLY 2 features (for visualization)
# -----------------------------
X = data[['Lag1', 'Lag2']]
y = data['Direction']

# -----------------------------
# Step 4: Train/Test Split (Time-series safe)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.3
)

# -----------------------------
# Step 5: Standardization
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 6: Train SVM (Linear Kernel for clear boundary)
# -----------------------------
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# -----------------------------
# Step 7: Prediction & Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Step 8: Plot Data + Decision Boundary
# -----------------------------
plt.figure(figsize=(8,6))

# Plot training data
plt.scatter(X_train[y_train==0][:,0],
            X_train[y_train==0][:,1],
            color='red', label='Down')

plt.scatter(X_train[y_train==1][:,0],
            X_train[y_train==1][:,1],
            color='blue', label='Up')

# Create mesh grid
x_min, x_max = X_train[:,0].min() - 1, X_train[:,0].max() + 1
y_min, y_max = X_train[:,1].min() - 1, X_train[:,1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

# Plot support vectors
plt.scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1],
            s=100, facecolors='none',
            edgecolors='black')

plt.xlabel("Lag1 (Scaled)")
plt.ylabel("Lag2 (Scaled)")
plt.title("SVM Hyperplane Separation")
plt.legend()
plt.show()
