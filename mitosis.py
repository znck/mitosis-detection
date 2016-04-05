#!/usr/bin/env python
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adamax
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializations import get_fans
import keras.backend as K
import numpy as np

from layers import MyConvolution2D


def custom_initialization(shape, name=None):
    fan_in, fan_out = get_fans(shape)
    loc = (fan_in + fan_out) / 2.
    scale = (fan_in + fan_out) / 2.
    return K.variable(np.random.normal(loc, scale, shape) * (1. / np.sqrt(fan_in / 2)), name=name)


def model_1(lr=.001, rho=.9, epsilon=1.0e-6):
    dnn = Sequential()
    dnn.add(BatchNormalization(input_shape=(3, 101, 101)))
    dnn.add(Convolution2D(16, 2, 2, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Convolution2D(16, 3, 3, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Convolution2D(16, 3, 3, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Convolution2D(16, 2, 2, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Convolution2D(16, 2, 2, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Flatten())
    dnn.add(Dense(100))
    dnn.add(Dense(2))
    dnn.add(Activation('softmax'))
    dnn.compile(loss='binary_crossentropy', optimizer=Adamax(lr=lr))

    return dnn


def model_2(lr=.001, rho=.9, epsilon=1.0e-6):
    dnn = Sequential()
    dnn.add(BatchNormalization(input_shape=(3, 101, 101)))
    dnn.add(Convolution2D(16, 4, 4, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Convolution2D(16, 4, 4, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Convolution2D(16, 4, 4, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Convolution2D(16, 3, 3, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Flatten())
    dnn.add(Dense(100))
    dnn.add(Dense(2))
    dnn.add(Activation('softmax'))
    dnn.compile(loss='binary_crossentropy', optimizer=Adamax(lr=lr))

    return dnn


def model_base(lr=.001, rho=.9, epsilon=1.0e-6):
    nn = Sequential()
    nn.add(BatchNormalization(input_shape=(3, 101, 101)))
    nn.add(MyConvolution2D(16, 6, 6, init='he_normal'))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(MaxPooling2D())
    #nn.add(BatchNormalization())
    nn.add(MyConvolution2D(16, 5, 5, init='he_normal'))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(MaxPooling2D())
    nn.add(MyConvolution2D(16, 3, 3, init='he_normal'))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(MaxPooling2D())
    #nn.add(MyConvolution2D(16,4,4, init='he_normal'))
    #nn.add(LeakyReLU(alpha=0.01))
    #nn.add(MaxPooling2D())
    nn.add(Flatten())
    nn.add(Dense(200))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(Dense(100))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(Dense(2))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(Activation('softmax'))
    nn.compile(loss='binary_crossentropy', optimizer=Adamax(lr=lr))

    return nn


if __name__ == '__main__':
    from runner import main

    main()
