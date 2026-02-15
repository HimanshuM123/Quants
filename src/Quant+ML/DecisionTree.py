import yfinance  as yf
import pandas as pd
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
import matplotlib.pyplot as plt
from sympy import false


data = yf.download("AAPL" , start="2020-01-01", end="2023-01-01")

data['Return'] = data['Close'].pct_change()
data['Lag1']= data['Return'].shift(1)
data['Lag2']= data['Return'].shift(2)
data['Volatility']= data['Return'].rolling(5).std() #Take 5 consecutive rows at a time, move forward one row at a time, and apply a calculation to each group.
data['Direction'] = (data['Return'].shift(-1) > 0).astype(int)  # Target: 1 if up, 0 if down (shift(-1) → looks one day into the future)
data.dropna(inplace=True)
print(data)

# Step 2: Features & Target

X = data[['Lag1','Lag2', 'Volatility']]
y=data['Direction']


# Step 3: Train-test split

X_train , X_test,y_train, y_test = train_test_split(X,y,shuffle=False, test_size=0.2)

# Step 4: Train decision tree

tree = DecisionTreeClassifier(max_depth=4,random_state=42)
tree.fit(X,y)
y_pred =  tree.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Plot the tree
plt.figure(figsize=(12,8))
plot_tree(tree, feature_names=X.columns, class_names=["Down", "Up"], filled=True)
plt.title("Decision Tree for AAPL Price Movement")
plt.show()