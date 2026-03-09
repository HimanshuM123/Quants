#Regression problem
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from sklearn.metrics import r2_score

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import  matplotlib.pyplot as plt


df=pd.read_csv('admission_data.csv')

print(df.head())
print(df.info())

# df.drop(columns=['Serial No.'], inplace=True)
# print(df.head())

X=df.iloc[:,0:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2 , random_state=1)
print(X_train)

scalar = MinMaxScaler()
X_train_scaled = scalar.fit_transform((X_train))
X_test_scaled = scalar.fit_transform((X_test))

print(X_train_scaled)

model = Sequential()
model.add(Dense(7,activation='relu', input_dim=7))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='linear'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='Adam')
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

y_pred= model.predict(X_test_scaled)
print(r2_score(y_test,y_pred))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

