import numpy as np
import pandas as pd

def vasicek_zero_price(r0, a, b, sigma, t):
    """Return P(0,t) under Vasicek and also A,B if needed."""
    B = (1 - np.exp(-a * t)) / a
    A = np.exp((B - t) * (a * b - (sigma**2) / (2 * a**2)) - (sigma**2 * B**2) / (4 * a))
    return A * np.exp(-B * r0)

# Parameters (example)
a = 0.0099
b = 0.050269
sigma = 0.015
r0 = 0.04
N = 5         # 5-year swap
delta = 1.0   # annual payments

# Build zero prices for years 1..N
times = np.arange(1, N+1) * delta
zeros = [vasicek_zero_price(r0, a, b, sigma, t) for t in times]

# Swap (par) fixed rate formula for annual payments:
Z_sum = sum(zeros)
Z_N = zeros[-1]
swap_rate = (1 - Z_N) / Z_sum

# Display
df = pd.DataFrame({'Year': times, 'ZeroPrice': zeros})
print(df.to_string(index=False, float_format='{:0.6f}'.format))
print(f"\n5-year par swap fixed rate (annual) = {swap_rate:.8f} ({swap_rate*100:.4f}%)")
