import numpy as np

# Two return series
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Compute Pearson correlation
correlation = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {correlation:.2f}")
# Output: Correlation: 1.00
