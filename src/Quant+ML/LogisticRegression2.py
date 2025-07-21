import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Download stock data
data = yf.download("TSLA", start="2020-01-01", end="2023-01-01")
data['Return'] = data['Close'].pct_change()
#.pct_change() function calculates the percentage change between the current and a prior element (by default, the previous one).

# Step 2: Create target (1 if next day return > 0, else 0)
data['Target'] = (data['Return'].shift(-1) > 0).astype(int)

# Step 3: Feature engineering
data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)
data['Momentum'] = data['Close'] - data['Close'].shift(5)

# Drop NaNs
data.dropna(inplace=True)

# Step 4: Define features and target
X = data[['Lag1', 'Lag2', 'Momentum']]
y = data['Target']

# Step 5: Split data (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 6: Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Predict probabilities and classes
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Step 8: Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Plot predicted probabilities and actual classes
plt.figure(figsize=(14,6))
plt.plot(y_prob, label='Predicted Probability (Price Up)', color='blue')
plt.scatter(np.arange(len(y_test)), y_test, color='red', alpha=0.3, label='Actual Direction (1=Up, 0=Down)')
plt.title('Logistic Regression: Predicted Probability vs Actual Next Day Direction')
plt.xlabel('Test Set Time Index')
plt.ylabel('Probability / Actual Direction')
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Plot ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
