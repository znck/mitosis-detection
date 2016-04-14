from keras.layers import BatchNormalization, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import Adamax


def model_1(lr=0.002):
    dnn = Sequential()
    dnn.add(Convolution2D(16, 2, 2, init='he_normal', input_shape=(3, 101, 101)))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Dropout(0.25))
    dnn.add(Convolution2D(16, 3, 3, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Convolution2D(16, 3, 3, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Dropout(0.25))
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


def model_2(lr=0.002):
    dnn = Sequential()
    dnn.add(Convolution2D(16, 4, 4, init='he_normal', input_shape=(3, 101, 101)))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Convolution2D(16, 4, 4, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Dropout(0.25))
    dnn.add(Convolution2D(16, 4, 4, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Dropout(0.25))
    dnn.add(Convolution2D(16, 3, 3, init='he_normal'))
    dnn.add(MaxPooling2D())
    dnn.add(LeakyReLU(alpha=.01))
    dnn.add(Flatten())
    dnn.add(Dense(100))
    dnn.add(Dense(2))
    dnn.add(Activation('softmax'))
    dnn.compile(loss='binary_crossentropy', optimizer=Adamax(lr=lr))

    return dnn


def model_base(lr=0.002):
    nn = Sequential()
    nn.add(Convolution2D(16, 4, 4, input_shape=(3, 101, 101), init='he_normal'))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(MaxPooling2D())
    nn.add(Dropout(0.25))
    nn.add(Convolution2D(32, 4, 4, init='he_normal'))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(Dropout(0.25))
    nn.add(MaxPooling2D())
    nn.add(Convolution2D(64, 2, 2, init='he_normal'))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(MaxPooling2D())
    nn.add(Flatten())
    nn.add(Dense(200))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(Dense(100))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(Dense(2))
    nn.add(LeakyReLU(alpha=.01))
    nn.add(Activation('softmax'))
    nn.compile(loss='binary_crossentropy', optimizer=Adamax(lr=lr), class_mode='binary')

    return nn
