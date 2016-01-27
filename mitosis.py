#! /usr/bin/env python
import glob
import os
import traceback
import argparse
import signal

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.models import Sequential

from prepare import JsonIterator, RandomSampler, BatchGenerator, TT

import theano


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


def _task_train_filter(arguments):
    if arguments.disable_optimisation:
        print TT.WARNING + "> Theano Config :: optimizer=None, exception_verbosity=high", TT.END
        theano.config.optimizer = 'None'
        theano.config.exception_verbosity = 'high'

    print TT.INFO + "> Training non competent pixel filter", TT.END
    path = arguments.path
    assert os.path.exists(path), path + " does not exists"
    path = os.path.abspath(path)

    assert len(glob.glob(os.path.join(path, '*/mitosis'))), "No valid mitosis dataset provided."

    load_path = os.path.join(path, 'weights.npy')
    if arguments.model:
        load_path = os.path.abspath(arguments.model)

    # Init Random Sampler
    dataset = RandomSampler(path, verbose=arguments.verbose)

    positive, n_positive = dataset.positive()
    sample, n_sample = dataset.sample(n_positive)

    if arguments.verbose:
        print TT.INFO + "> Compiling model...", TT.END
    model = model_base()

    if os.path.exists(load_path):
        print TT.INFO, "Loading model from %s" % load_path, TT.END
        model.load_weights(load_path)
    n_epoch = arguments.epoch

    def save_weights():
        model.save_weights(load_path)
        exit(0)

    signal.signal(signal.SIGINT, save_weights)

    val_split = .3
    if not arguments.validation:
        val_split = .0

    for epoch in xrange(n_epoch):
        print TT.SUCCESS + "> Epoch %d or %d" % (epoch + 1, n_epoch), TT.END
        for X_train, Y_train in BatchGenerator(JsonIterator(positive), n_positive, JsonIterator(sample), n_sample,
                                               arguments.batch):
            model.fit(X_train, Y_train, batch_size=arguments.mini_batch, nb_epoch=1, shuffle=True, validation_split=val_split)
        sample, n_sample = dataset.sample(n_positive)

    model.save_weights(load_path)


def _parse_args():
    stub = argparse.ArgumentParser(description="Mitosis Detection Task Runner")
    stub.add_argument("task", help="Run task. (train-filter, train, test, predict)",
                      choices=['train-filter', 'train', 'test', 'predict'], metavar="task")
    stub.add_argument("path", type=str, help="Directory containing mitosis images", metavar="path")
    stub.add_argument("--epoch", type=int, help="Number of epochs. (Default: 1)", default=1)
    stub.add_argument("--batch", type=int, help="Size of batch fits in memory. (Default: 1000)", default=1000)
    stub.add_argument("--mini-batch", type=int, help="Size of training batch. (Default: 50)", default=50)
    stub.add_argument("-v", action="store_true", help="Increase verbosity. (Default: Disabled)", default=False,
                      dest='verbose')
    stub.add_argument("--model", type=str, help="Saved model weights. (Default: ${path}/weights.npy)")
    stub.add_argument("--no-validate", action='store_false', help="Disable validation. (Default: Enabled)",
                      default=True,
                      dest='validation')
    stub.add_argument("--no-optimisation", action='store_true',
                      help="Disable theano optimisations. (Default: Disabled)", default=False,
                      dest="disable_optimisation")

    return stub


if __name__ == '__main__':
    parser = _parse_args()
    args = parser.parse_args()

    try:
        if args.task == 'train-filter':
            _task_train_filter(args)
        else:
            parser.print_help()
            exit()
    except AssertionError as e:
        print TT.WARNING + e.message + TT.END
        if args.verbose:
            print TT.DANGER + traceback.format_exc() + TT.END
    finally:
        print '..'
