import tensorflow

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# print(X_train.shape)
# print(X_train[0])
# print(y_train)

X_train = X_train / 255
X_test = X_test / 255

print(X_train[0])

model= Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
history =model.fit(X_train,y_train,epochs=10,validation_split=0.2)

y_prob =model.predict(X_test)
print(y_prob)

y_pred=y_prob.argmax(axis=1)
print(y_pred)

print(accuracy_score(y_test,y_pred))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.title("Model Loss")
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.title("Model Accuracy")
plt.show()

plt.imshow(X_test[1])
plt.show()

predicted_val=model.predict(X_test[1].reshape(1,28,28)).argmax(axis=1)
print(predicted_val)


