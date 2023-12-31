import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
np.random.seed(123)
import keras.backend as K
K.set_image_data_format("channels_first")
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import livelossplot
plot_losses = livelossplot.PlotLossesKeras()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
model = Sequential()
model.add(Convolution2D(64, kernel_size = 3, activation='relu',padding = 'same' ,input_shape=(1,28,28)))
model.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=500, epochs=50, callbacks = [plot_losses], verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)