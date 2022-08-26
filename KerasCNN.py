import keras
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten

model = Sequential()

# input_shape tells the computer how much data to allocate for the NN, (3,3) is the kernal size
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(10, activation='softmax'))

# helps visualization
model.summary()

# Keras Sequential needs to be compiled to train it and calculate the loss
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta())
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# fit method - train with a data set (x_train) and a set of labels (y_test)/output data.
# Here the inputs for the fit method are the training data, the training targets (another name for labels),
# the batch size, and the number of epochs, or times the entire data set will be tested.
model.fit(x_train, y_train, epochs=5, batch_size=64)

img = x_train[2].reshape(28, 28)
plt.imshow(img)
# plt.show(img)
img = img.reshape(-1, 28, 28, 1)
out = model.predict(img)  # y_test/result/output

print("My guess is: " + str(np.argmax(out)))
