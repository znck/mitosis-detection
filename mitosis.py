#! /usr/bin/env python
import os
import sys

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.models import Sequential

#from dataset import read_all_files, Frame
import numpy as np
from prepare import JsonIterator


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


batch1 = JsonIterator('/media/ankur/Seagate Backup Plus Drive/mark-4/training_aperio/positives.json')
pos = []
y_pos = []
i = 0
for patch, prob in batch1:
    i += 1
    pos.append(patch)
    y_pos.append(prob)
    if i >= 500:
        break

batch2 = JsonIterator('/media/ankur/Seagate Backup Plus Drive/mark-4/training_aperio/negatives.json')
neg = []
y_neg = []
i = 0
max_patches = 500
for patch, prob in batch2:
    i += 1
    neg.append(patch)
    y_neg.append(prob)
    if i >= max_patches:
        break

(pos.append(it) for it in neg)
(y_pos.append(it) for it in y_neg)
print len(pos)
itr = np.asarray(range(len(pos)))
np.random.shuffle(itr)

x = []
y = []
for i in itr:
    x.append(pos[i])
    y.append(y_pos[i])


nn1 = model_base()
y =np.asarray(y)
x = np.asarray(x)
print x.shape, y.shape
nn1.fit(x, y, nb_epoch=10, batch_size=100, verbose=1)

#np.random.shuffle(data)
#x,y = data

if __name__ == '__main__':
    if len(sys.argv) > 1:
        produce_probability_map(sys.argv[1])
    else:
        print 'usage: %s <path>' % sys.argv[0]
