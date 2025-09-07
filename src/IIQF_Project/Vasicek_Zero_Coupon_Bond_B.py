#Use the above model to calculate the analytical price of a 5-year Zero-coupon (with a
# face value of $1) bond.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calibrate_vasicek_and_price_bond(csv_url, T=5, sigma=0.015):
    # Step 1: Load Fed Funds Rate data
    df = pd.read_csv(csv_url)
    df.dropna(inplace=True)

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

    delta_t = 1  # monthly data, use delta_t = 1
    a = -slope / delta_t
    b = intercept / (a * delta_t)
    r0 = rates[-1]  # last observed rate

    # Print calibrated values
    print(f"Estimated a (mean reversion speed): {a:.4f}")
    print(f"Estimated b (long-term mean rate): {b:.4f}")
    print(f"Using last observed short rate r0 = {r0:.4f}")
    print(f"Assumed volatility sigma = {sigma:.4f}")

    # Step 4: Compute bond price
    B = (1 - np.exp(-a * T)) / a
    A = np.exp(
        (B - T) * (a * b - sigma**2 / (2 * a**2)) - (sigma**2 * B**2) / (4 * a)
    )
    P = A * np.exp(-B * r0)

    print(f"\nPrice of 5-year zero-coupon bond (Face = $1): ${P:.4f}")

    # Optional: Plot Fed Funds Rate
    plt.figure(figsize=(10, 5))
    plt.plot(df['observation_date'], df['FEDFUNDS'], label='Fed Funds Rate')
    plt.title("Federal Funds Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Interest Rate (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
csv_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS'
calibrate_vasicek_and_price_bond(csv_url)
