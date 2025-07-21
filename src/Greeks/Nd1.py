import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# d1 range
d1 = np.linspace(-4, 4, 500)

# N(d1) and N'(d1)
cdf = norm.cdf(d1)
pdf = norm.pdf(d1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(d1, cdf, label="N(d₁) - CDF", color='blue')
plt.plot(d1, pdf, label="N′(d₁) - PDF", color='green')
plt.axvline(0, linestyle='--', color='gray', label='d₁ = 0')
plt.title("Comparison: N(d₁) vs N′(d₁)")
plt.xlabel("d₁")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
