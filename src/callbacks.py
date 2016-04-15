import datetime
import numpy
import time

from keras.callbacks import Callback


def get_file_name():
    """
    :return: Create a filename with current timestamp.
    """
    d = datetime.datetime.now()
    return 'logs/log_%04d-%02d-%02d_%02d-%02d_%f.txt' % (d.year, d.month, d.day, d.hour, d.minute, time.clock())


class LearnLog(Callback):
    """
    Record loss history.
    """
    def __init__(self, name):
        super(LearnLog, self).__init__()
        self.log_file = get_file_name()
        open(self.log_file, 'w').write("# Learn Log: %s Model\n" % name)
        self.loss = None
        self.batch = 1
        self.epoch = 1

    def on_batch_end(self, batch, logs={}):
        self.loss = logs.get('loss')

    def on_train_end(self, logs={}):
        numpy.savetxt(open(self.log_file, 'a'), [[self.epoch, self.batch, self.loss]], fmt="%g")
