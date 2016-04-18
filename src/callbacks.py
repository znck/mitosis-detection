import datetime
import os
import random
import time

import numpy
from keras.callbacks import Callback

from utilities import TT


def get_file_name():
    """
    :return: Create a filename with current timestamp.
    """
    d = datetime.datetime.now()
    return 'logs/log_%04d-%02d-%02d_%02d-%02d_%d.txt' % (d.year, d.month, d.day, d.hour, d.minute, random.randint(0, 10000))


class LearnLog(Callback):
    """
    Record loss history.
    """
    def __init__(self, name, path):
        super(LearnLog, self).__init__()
        self.log_file = os.path.join(path, name+'.txt')
        self.weights_file = os.path.join(path, name+'.%d.weights.npy')
        self.last_loss = None
        d = datetime.datetime.now()
        with open(self.log_file, 'a') as fd:
            fd.write("# Learn Log: %s Model\n" % name)
            fd.write("# Timestamp: %04d-%02d-%02d %02d:%02d\n" % (d.year, d.month, d.day, d.hour, d.minute))
        self.loss = 0.0
        self.epoch = 1

    def on_dataset_epoch_begin(self, epoch, logs={}):
        self.loss = 0.0
        self.epoch = epoch
        self.start = time.time()

    def on_dataset_epoch_end(self, epoch, logs={}):
        numpy.savetxt(open(self.log_file, 'a'), [[self.epoch, self.loss]], fmt="%g")
        if self.last_loss is None or self.last_loss > self.loss:
            filename = self.weights_file % epoch
            TT.debug("Saving weights to", filename)
            self.model.save_weights(filename)
        self.last_loss = self.loss

    def on_batch_end(self, batch, logs={}):
        self.loss += logs.get('loss')
