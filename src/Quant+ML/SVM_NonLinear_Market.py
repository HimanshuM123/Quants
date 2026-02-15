import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
import yfinance as yf


# 1️⃣ Fetch Real Market Data (SPY ETF)
data = yf.download("SPY",start="2020-01-01", end="2024-01-01")

# 2️⃣ Feature Engineering
data['Return'] = data['Close'].pct_change()
data['Lag1']=data['Return'].shift(1)
data['Lag2']=data['Return'].shift(2)
data['Volatility']=data['Return'].rolling(5).std()

# RSI calculation
delta = data['Close'].diff()
gain = delta.where(delta >0,0).rolling(14).mean()
loss= -delta.where(delta <0,0).rolling(14).mean()
rs = gain/loss
data['RSI'] = 100 - (100 / (1 + rs)) #This converts RS into a value between 0 and 100.
# What RSI Means
# Above 70 → Overbought (price may pull back)
# Below 30 → Oversold (price may bounce)
# Around 50 → Neutral

# Moving Average Signal
data['MA_10']= data['Close'].rolling(10).mean()
data['MA_30']= data['Close'].rolling(30).mean()
data['MA_Signal'] = (data['MA_10'] > data['MA_30']).astype(int)

# Target: Next day direction
data['Direction'] = (data['Return'].shift(-1) > 0).astype(int)
data.dropna(inplace=True)

# 3️⃣ Define Features & Target
features=['Lag1','Lag2','Volatility','RSI','MA_Signal']
X = data[features]
y=data['Direction']

# 4️⃣ Train-Test Split (Time-Series Style)
split = int(len(data)*0.7)
X_train, X_test= X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5️⃣ Scale Features
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

# 6️⃣ Non-Linear SVM (RBF)
model =SVC(kernel='rbf', C=10, gamma=0.5 , probability=True)
model.fit(X_train_scaled, y_train)

# 7️⃣ Predictions
y_pred = model.predict(X_test_scaled)
y_score = model.predict_proba(X_test_scaled)[:,1]

# 8️⃣ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12,5))
# ROC
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'r--')
plt.title("ROC Curve - RBF SVM on SPY")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

# Confusion Matrix
plt.subplot(1,2,2)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()

print("AUC:", roc_auc)














