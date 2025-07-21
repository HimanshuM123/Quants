import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('placement.csv')

print(df.head())

df = df.iloc[:,1:]

print(df.head())

plt.scatter(df['cgpa'], df['iq'], c=df['placement'])
plt.show()