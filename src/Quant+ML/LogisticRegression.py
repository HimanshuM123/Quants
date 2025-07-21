#using logistic regression to predict loan defaults using a real-world-style dataset.

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# Simulated sample data (you can replace with real data)
data = pd.DataFrame({
    'loan_amount': [5000, 10000, 15000, 20000, 3000, 12000, 18000],
    'term_months': [36, 60, 36, 60, 36, 60, 60],
    'credit_score': [700, 650, 620, 590, 710, 640, 600],
    'income': [55000, 48000, 42000, 36000, 62000, 40000, 30000],
    'late_payments': [0, 1, 2, 3, 0, 2, 4],
    'defaulted': [0, 0, 1, 1, 0, 1, 1]
})

# Features and target
X = data[['loan_amount', 'term_months', 'credit_score', 'income', 'late_payments']]
y = data['defaulted']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities and class
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))


#🔍 Interpret Model Coefficients

coeffs = pd.Series(np.exp(model.coef_[0]), index=X.columns)
print("\nOdds Ratios:\n", coeffs)


#ROC Curve

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
