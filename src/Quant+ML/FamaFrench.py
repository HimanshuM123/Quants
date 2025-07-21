import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Get stock data (e.g., AAPL)
aapl = yf.download("AAPL", start="2020-01-01", end="2023-01-01", progress=False)
aapl['AAPL_Return'] = aapl['Adj Close'].pct_change()

# Step 2: Load Fama-French factors (daily)
ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
ff = pd.read_csv(ff_url, skiprows=4)
ff = ff[ff.columns[:4]]  # [Date, Mkt-RF, SMB, HML, RF]
ff.columns = ['Date', 'Mkt_RF', 'SMB', 'HML', 'RF']
ff = ff[ff['Date'].str.isnumeric()]
ff['Date'] = pd.to_datetime(ff['Date'], format='%Y%m%d')
ff.set_index('Date', inplace=True)
ff = ff.astype(float) / 100  # Convert to decimal returns

# Step 3: Merge datasets
aapl = aapl[['AAPL_Return']].dropna()
df = pd.merge(aapl, ff, left_index=True, right_index=True)
df['Excess_AAPL'] = df['AAPL_Return'] - df['RF']

# Step 4: Run regression
X = df[['Mkt_RF', 'SMB', 'HML']]
X = sm.add_constant(X)
y = df['Excess_AAPL']
model = sm.OLS(y, X).fit()

# Step 5: Results
print(model.summary())

# Plot: Actual vs Fitted
df['Fitted'] = model.predict(X)
plt.scatter(df['Excess_AAPL'], df['Fitted'], alpha=0.3)
plt.plot([-0.05, 0.05], [-0.05, 0.05], color='red')
plt.title('Actual vs Fitted Excess Returns (Fama-French 3-Factor)')
plt.xlabel('Actual Excess Return')
plt.ylabel('Fitted Return')
plt.grid(True)
plt.show()
