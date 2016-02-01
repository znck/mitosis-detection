#! /usr/bin/env python
import glob
import os
import traceback
from random import randint

import argparse
import signal

import time

from keras.callbacks import Callback
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import array_to_img

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid import AxesGrid

from prepare import JsonIterator, RandomSampler, BatchGenerator, TT

import theano


class VisHistory(Callback):
    def __init__(self, layer=(0,)):
        super(VisHistory, self).__init__()
        self.nb_batches = 0
        self.epoch_loss = 0
        self.epoch_acc = 0
        self.train_length = 0
        self.val_split = 1.
        self.layer = layer
        self.losses = {
            'val_acc': [],
            'val_loss': [],
            'loss': [],
            'acc': []
        }
        self.img_to_visualize = 0
        # plt.figure(1)
        self.f, self.axarr = plt.subplots(2, 3, sharex=True)
        self.axarr[0][0].set_title('Loss')
        self.axarr[1][0].set_title('Accuracy')
        self.axarr[0][1].set_title('Test image')
        self.axarr[1][1].set_title('Convolutions Layer 1')
        self.axarr[0][2].set_title('Convolutions Layer 2')
        self.axarr[1][2].set_title('Convolutions Layer 3')
        self.test_image = None
        self.e = 0

    def show_convolutions(self):
        if len(self.layer):
            pos = [235, 233, 236]
            for i in range(0, 4):
                if i >= len(self.layer):
                    break
                self.visualize_layer(self.f, self.layer[i], pos[i])
            layer = len(self.model.layers) - 1
        convout1_f = theano.function([self.model.get_input(train=False)],
                                     self.model.layers[layer].get_output(train=False))
        output = convout1_f(self.model.training_data[0][self.img_to_visualize: self.img_to_visualize + 1])
        print output.shape
        self.axarr[1][2].set_title('Convolutions Layer 3 (P = %f)' % output[0][0])

    def visualize_layer(self, figure, layer, pos=224):
        convout1_f = theano.function([self.model.get_input(train=False)],
                                     self.model.layers[layer].get_output(train=False))
        convolutions = convout1_f(self.model.training_data[0][self.img_to_visualize: self.img_to_visualize + 1])
        nb_filters = convolutions[0].shape[0]
        from math import sqrt, ceil
        width = int(ceil(sqrt(nb_filters)))
        grid = AxesGrid(figure, pos,
                        nrows_ncols=(width, width),
                        axes_pad=0.05,
                        label_mode="1",
                        )
        grid.axes_llc.set_xticks([])
        grid.axes_llc.set_yticks([])
        for i, convolution in enumerate(convolutions[0]):
            grid[i].imshow(convolution, cmap=cm.Greys_r)
            plt.draw()

    def _set_params(self, params):
        super(VisHistory, self)._set_params(params)
        self.train_length = int(params.get('nb_sample'))
        self.nb_batches = int(self.train_length / int(params.get('batch_size')))

    def on_train_begin(self, logs=None):
        # self.img_to_visualize = randint(0, self.train_length - 1)
        if self.test_image is None:
            self.test_image = array_to_img(self.model.training_data[0][self.img_to_visualize])
        image1 = self.test_image
        # plt.figure(1)
        self.axarr[0][1].imshow(image1)
        plt.ion()
        plt.show()

    def on_train_end(self, logs=None):
        plt.ioff()
        # pass

    def on_epoch_begin(self, epoch, logs=None):
        # self.epoch_loss = 0
        # self.epoch_acc = 0
        pass

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = dict()
        self.epoch_loss += float(logs.get('loss'))
        self.epoch_acc += float(logs.get('acc'))

    def on_epoch_end(self, epoch, logs):
        self.epoch_loss = self.epoch_loss / float(self.train_length) * float(self.nb_batches)
        self.epoch_acc = self.epoch_acc / float(self.train_length) * float(self.nb_batches)
        self.losses['val_acc'].append(logs['val_acc'])
        self.losses['val_loss'].append(logs['val_loss'])
        self.losses['loss'].append(self.epoch_loss)
        self.losses['acc'].append(self.epoch_acc)
        # if self.e % 5 == 0:
        self.show_convolutions()
        self.e += 1
        # plt.figure(1)
        vl, = self.axarr[0][0].plot(self.losses['val_loss'], label='val_loss', color='r', linewidth=2.0)
        ll, = self.axarr[0][0].plot(self.losses['loss'], label='loss', color='b')
        va, = self.axarr[1][0].plot(self.losses['val_acc'], label='val_acc', color='r', linewidth=2.0)
        aa, = self.axarr[1][0].plot(self.losses['acc'], label='acc', color='b')
        # self.axarr[0][0].legend(handles=[vl, ll], loc=1)
        # self.axarr[1][0].legend(handles=[va, aa], loc=4)
        plt.pause(0.001)
        plt.draw()


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
    nn.add(Convolution2D(4, 4, 4, input_shape=(3, 101, 101)))
    nn.add(Activation('tanh'))
    nn.add(Convolution2D(8, 3, 3))
    nn.add(Activation('tanh'))
    nn.add(Convolution2D(12, 2, 2))
    nn.add(Activation('tanh'))
    nn.add(Flatten())
    nn.add(Dense(200))
    nn.add(Dense(100))
    nn.add(Dense(2))
    nn.add(Activation('softmax'))
    nn.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.0003), class_mode='binary')

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

    if arguments.verbose:
        print TT.INFO + "> Compiling model...", TT.END
    model = model_base()

    if os.path.exists(load_path):
        print TT.SUCCESS + "> Loading model from %s" % load_path, TT.END
        model.load_weights(load_path)
    n_epoch = arguments.epoch

    def save_weights(_1, _2):
        dying_path = load_path + '.dying.npy'
        print TT.DANGER + 'Program Terminated. Saving progressing in %s' % dying_path, TT.END
        model.save_weights(dying_path, True)
        exit(0)

    signal.signal(signal.SIGINT, save_weights)

    val_split = .2
    if not arguments.validation:
        val_split = .0

    train_start = time.time()
    callbacks = []
    if arguments.visualize:
        vis = VisHistory((1, 3, 5))
        callbacks.append(vis)
    for epoch in xrange(n_epoch):
        epoch_start = time.time()
        print TT.INFO + "> Epoch %d of %d" % (epoch + 1, n_epoch), TT.END
        sample, n_sample = dataset.sample(n_positive)
        batch = BatchGenerator(JsonIterator(positive), n_positive, JsonIterator(sample), n_sample, arguments.batch)
        for X_train, Y_train in batch:
            model.fit(X_train, Y_train, batch_size=arguments.mini_batch, nb_epoch=1, shuffle=True,
                      validation_split=val_split, show_accuracy=True, callbacks=callbacks)
        model.save_weights(load_path, True)
        print TT.SUCCESS + "> Epoch %d of %d took %.2f seconds." % (
            epoch + 1, n_epoch, time.time() - epoch_start), TT.END
    print TT.SUCCESS + "> Training finished. Time take: %.2f seconds." % (time.time() - train_start), TT.END


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
    stub.add_argument("--visualize", action='store_true', help="Disable validation. (Default: Disabled)",
                      default=False,
                      dest='visualize')
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
