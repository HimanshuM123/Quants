import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calibrate_vasicek_from_url(csv_url):
    # Step 1: Load Fed Funds Rate data
    df = pd.read_csv(csv_url)
    df.dropna(inplace=True)

    # Assume column names: DATE and FEDFUNDS or similar
    if 'FEDFUNDS' not in df.columns:
        print("CSV must contain 'FEDFUNDS' column.")
        return

    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df.sort_values('observation_date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 2: Calculate r_t and r_{t+1}
    rates = df['FEDFUNDS'].values
    r_t = rates[:-1]
    r_t1 = rates[1:]
    delta_r = r_t1 - r_t

    # Step 3: Estimate a and b using linear regression
    X = r_t.reshape(-1, 1)
    y = delta_r

    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    delta_t = 1  # Assuming monthly steps

    # Vasicek Parameters
    a = -slope / delta_t
    b = intercept / (a * delta_t)

    print(f"Estimated a (mean reversion speed): {a:.4f}")
    print(f"Estimated b (long-term mean rate): {b:.4f}")

    # Step 4: Plot the Fed Funds Rate
    plt.figure(figsize=(10, 4))
    plt.plot(df['observation_date'], df['FEDFUNDS'], label='Fed Funds Rate')
    plt.title("Federal Funds Rate Over Time")
    plt.xlabel("observation_date")
    plt.ylabel("Interest Rate (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 5: Plot delta_r vs r_t with regression line (slope visualization)
    plt.figure(figsize=(8, 5))
    plt.scatter(r_t, delta_r, alpha=0.5, label='Δr vs r(t)')
    r_t_range = np.linspace(min(r_t), max(r_t), 100)
    delta_r_pred = slope * r_t_range + intercept
    plt.plot(r_t_range, delta_r_pred, color='red', label=f'Regression Line\nΔr = {slope:.4f}·r + {intercept:.4f}')
    plt.title("Regression Line Showing Slope in Vasicek Calibration")
    plt.xlabel("Interest Rate r(t)")
    plt.ylabel("Δr = r(t+1) - r(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
csv_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS'
calibrate_vasicek_from_url(csv_url)
