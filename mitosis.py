#! /usr/bin/env python
import os
import sys

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.models import Sequential

from dataset import read_all_files, Frame


def model1():
    dnn = Sequential()
    dnn.add(Convolution2D(16, 2, 2, input_shape=(3, 101, 101)))
    dnn.add(MaxPooling2D())
    dnn.add(Convolution2D(16, 3, 3))
    dnn.add(MaxPooling2D())
    dnn.add(Convolution2D(16, 3, 3))
    dnn.add(MaxPooling2D())
    dnn.add(Convolution2D(16, 2, 2))
    dnn.add(MaxPooling2D())
    dnn.add(Convolution2D(16, 2, 2))
    dnn.add(MaxPooling2D())
    dnn.add(Flatten())
    dnn.add(Dense(100))
    dnn.add(Dense(2))
    dnn.compile(loss='binary_crossentropy', optimizer='sgd')

    return dnn


def model2():
    dnn = Sequential()
    dnn.add(Convolution2D(16, 4, 4, input_shape=(3, 101, 101)))
    dnn.add(MaxPooling2D())
    dnn.add(Convolution2D(16, 4, 4))
    dnn.add(MaxPooling2D())
    dnn.add(Convolution2D(16, 4, 4))
    dnn.add(MaxPooling2D())
    dnn.add(Convolution2D(16, 3, 3))
    dnn.add(MaxPooling2D())
    dnn.add(Flatten())
    dnn.add(Dense(100))
    dnn.add(Dense(2))
    dnn.compile(loss='binary_crossentropy', optimizer='sgd')

    return dnn


def model_base():
    nn = Sequential()
    nn.add(Convolution2D(10, 3, 3, input_shape=(3, 101, 101)))
    nn.add(Activation('relu'))
    nn.add(Convolution2D(10, 3, 3))
    nn.add(Activation('relu'))
    nn.add(Flatten())
    nn.add(Dense(100))
    nn.add(Dense(1))
    nn.compile(loss='binary_crossentropy', optimizer='sgd')

    return nn


# noinspection PyProtectedMember
def produce_probability_map(path):
    files = read_all_files(path)
    files = files[:2]
    name = os.path.basename(path) + '.model'
    model = model_base()
    epoch = 0
    b = 0
    batch_size = 100
    for f in files:
        frame = Frame(f[0], f[1])
        batches = frame.batches(batch_size)
        epoch += 1
        for batch in batches:
            b += 1
            print "File: %d Batch: %d" % (epoch, b)
            model.train_on_batch(batch[0], batch[1], accuracy=True)
    model.save_weights(name, True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        produce_probability_map(sys.argv[1])
    else:
        print 'usage: %s <path>' % sys.argv[0]
