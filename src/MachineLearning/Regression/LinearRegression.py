import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Generating random datasets

np.random.seed(42)
X= np.random.rand(50,1)*100 #Generates random numbers from a uniform distribution between 0 and 1
                            #Shape is (50, 1) → 50 rows, 1 column

Y = 3.5 * X + np.random.randn(50, 1) * 20
#Generates random numbers from a standard normal distribution
#Mean = 0, standard deviation = 1

model = LinearRegression()
model.fit(X,Y)

Y_pred =model.predict(X)


plt.figure(figsize=(8,6))
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression on Random Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()


