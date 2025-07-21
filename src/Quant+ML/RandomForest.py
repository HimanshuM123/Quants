import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01", progress=False)
if data.empty or 'Close' not in data.columns:
    raise ValueError("Failed to load stock data. Check ticker or internet connection.")


# Feature engineering
data['Return'] = data['Close'].pct_change() #Return_t =(Close_t - Close_t-1)/Close_t-1
data['Lag1'] = data['Return'].shift(1) #yesterday’s return.
data['Lag2'] = data['Return'].shift(2) #return from 2 days ago.

data['Volatility'] = data['Return'].rolling(5).std() #Calculates 5-day rolling standard deviation of returns. / Measures recent market volatility.
data['Direction'] = (data['Return'].shift(-1) > 0).astype(int)
# Target variable: whether price goes up the next day.
# 1 = price up, 0 = price down.
# shift(-1) moves future return to current row, enabling supervised learning.

# Drop missing values
#Drops rows with NaNs due to shifting/rolling.
data.dropna(inplace=True)

# Define features and target
# X = feature matrix
# y = target (0/1 = down/up)
X = data[['Lag1', 'Lag2', 'Volatility']]
y = data['Direction']


# Check shape
print("X shape:", X.shape, "| y shape:", y.shape)

# Split data
# Splits data chronologically (important for time series).
# 70% for training, 30% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Train model
# 100 trees (n_estimators)
# Trees limited to depth 5 (simpler model)
# Predicts Direction (0 or 1)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))


# Step 6: Feature importance
importances = rf.feature_importances_
# feat_names = [str(col) for col in X.columns]  # flatten any MultiIndex to strings
feat_names = X.columns.to_flat_index().astype(str)

# Create a clean DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': feat_names,
    'Importance': importances
})

# Plot with Seaborn
plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='Blues_d')
plt.title("Feature Importance in Stock Movement Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.grid(True)
plt.show()


##########################################################

# ✅ Title:
# “Feature Importance in Stock Movement Prediction”
# You trained a Random Forest Classifier to predict whether AAPL stock price will go up or down the next day (binary classification: 1 or 0), based on:
#
# Lag1: return from yesterday
#
# Lag2: return from 2 days ago
#
# Volatility: 5-day rolling standard deviation (price fluctuations)
#
# The plot tells you how much each of these features contributed to the model's decisions.

#
# 📐 Part 2: What Is “Feature Importance”?
# In Decision Trees and Random Forests:
# Feature importance quantifies:
# How useful a feature is in reducing uncertainty (a.k.a. impurity) when building decision trees.
# For each feature:
# When the model splits on that feature, how much does it improve prediction quality?
# The more it improves, the higher its importance.
#
# Technically:
# It’s based on the mean decrease in impurity (MDI) across all trees:
#
# Calculated as the total reduction in Gini impurity or entropy that feature brings, weighted by the number of samples it affects.


# 📊 Part 3: What the Numbers Mean
# Let’s say the bar chart showed:
# Feature	Importance Value
# Volatility	0.34
# Lag1	0.33
# Lag2	0.33
#
# These numbers always add up to 1.0 (or 100% if scaled to percent).
#
# Interpretation:
# Volatility = 0.34 → ~34% of the model’s predictive power came from detecting recent market volatility.
#
# Lag1 and Lag2 = ~0.33 each → each lagged return contributed about 33% of predictive importance.
#
# All 3 are almost equally important, but Volatility edges slightly ahead, meaning the model relies on recent price swings slightly more than simple past direction.

#
# 📈 Why This Matters for Trading:
# Let’s tie this to financial intuition:
#
# Feature|	What It Tells Us                            |	Why It May Help Predict Direction
# Lag1	|Was yesterday's return positive or negative?	 |   Trend-following or mean-reverting behavior
# Lag2	|What happened 2 days ago?	                       |  Captures short-term patterns or reversals
# Volatility|	Was there a recent spike in price movements?|	High volatility often precedes reversals or breakout moves
#
# By measuring how “important” each is, we can:
# Interpret what the model is learning
# Avoid over-relying on any weak or misleading signals
# Guide future feature engineering (e.g., add Lag3, Momentum, RSI, etc.)


#  Visual Takeaway from the Graph:
# All three features are informative.
# Volatility has a slightly higher influence, suggesting price swings matter more than just yesterday's return.
# You’ve built an interpretable ML model that gives insight into market behavior.
