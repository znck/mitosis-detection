#!/usr/bin/env python
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import SGD


def model1():
    dnn = Sequential()
    dnn.add(Convolution2D(16, 2, 2, input_shape=(3, 101, 101)))
    dnn.add(MaxPooling2D())
    dnn.add(Activation('relu'))
    dnn.add(Convolution2D(16, 3, 3))
    dnn.add(MaxPooling2D())
    dnn.add(Activation('relu'))
    dnn.add(Convolution2D(16, 3, 3))
    dnn.add(MaxPooling2D())
    dnn.add(Activation('relu'))
    dnn.add(Convolution2D(16, 2, 2))
    dnn.add(MaxPooling2D())
    dnn.add(Activation('relu'))
    dnn.add(Convolution2D(16, 2, 2))
    dnn.add(MaxPooling2D())
    dnn.add(Activation('relu'))
    dnn.add(Flatten())
    dnn.add(Dense(100))
    dnn.add(Dense(2))
    dnn.add(Activation('softmax'))
    dnn.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return dnn


def model2():
    dnn = Sequential()
    dnn.add(Convolution2D(16, 4, 4, input_shape=(3, 101, 101)))
    dnn.add(MaxPooling2D())
    dnn.add(Activation('relu'))
    dnn.add(Convolution2D(16, 4, 4))
    dnn.add(MaxPooling2D())
    dnn.add(Activation('relu'))
    dnn.add(Convolution2D(16, 4, 4))
    dnn.add(MaxPooling2D())
    dnn.add(Activation('relu'))
    dnn.add(Convolution2D(16, 3, 3))
    dnn.add(MaxPooling2D())
    dnn.add(Activation('relu'))
    dnn.add(Flatten())
    dnn.add(Dense(100))
    dnn.add(Dense(2))
    dnn.compile(loss='binary_crossentropy', optimizer='sgd')

    return dnn


def model_base():
    nn = Sequential()
    nn.add(Convolution2D(4, 4, 4, input_shape=(3, 101, 101)))
    nn.add(MaxPooling2D())
    nn.add(Activation('relu'))
    nn.add(Convolution2D(8, 3, 3))
    nn.add(MaxPooling2D())
    nn.add(Activation('relu'))
    nn.add(Convolution2D(12, 2, 2))
    nn.add(MaxPooling2D())
    nn.add(Activation('relu'))
    nn.add(Flatten())
    nn.add(Dense(200))
    nn.add(Dense(100))
    nn.add(Dense(2))
    nn.add(Activation('softmax'))
    nn.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')

    return nn


if __name__ == '__main__':
    from runner import main

    main()
