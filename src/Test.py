import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf

aapl = yf.download('AAPL' , start='2004-01-01')

#print(aapl)

#plt.show()

aapl['Open'].plot()
plt.show()

