import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.layers import Convolution2D, Dense
from math import sqrt, ceil


class PrintGradients(Callback):
    def on_epoch_end(self, epoch, logs=None):
        for update, _ in self.model.optimizer.updates:
            print update.get_value()


class PlotLoss(Callback):
    def __init__(self):
        super(PlotLoss, self).__init__()
        plt.ion()
        plt.show()
        self.loss = 0
        self.data = []
        self.val = []

    def on_batch_end(self, batch, logs=None):
        self.loss = logs['loss']

    def on_epoch_end(self, batch, logs=None):
        self.data.append(self.loss)
        self.val.append(logs['val_loss'])
        if len(self.data) > 100:
            self.data.pop(0)
            self.val.pop(0)
        plt.clf()
        plt.xlim([0, 101])
        plt.plot(range(len(self.data)), self.data, '-ro')
        plt.plot(range(len(self.val)), self.val, '-g^')
        plt.draw()
        plt.pause(0.0001)


class VisualizeWeights(Callback):
    def __init__(self):
        super(VisualizeWeights, self).__init__()
        self.flat = None
        self.epoch = 0
        self.all = None

    def on_train_end(self, logs={}):
        plt.ion()
        plt.show()

    def on_train_begin(self, logs={}):
        plt.ioff()

    def on_epoch_begin(self, epoch, logs=None):
        i = 0
        n_plots = 0
        map_name = 'gray'
        if self.epoch % 2 == 1:
            map_name = 'hot'
        self.epoch += 1
        if self.flat is None:
            for layer in self.model.layers:
                if isinstance(layer, Convolution2D):
                    data = layer.W.get_value()
                    n_plots += data.shape[0]
                if isinstance(layer, Dense):
                    n_plots += 1
            n_p = int(ceil(sqrt(n_plots)))
            _, axes = plt.subplots(n_p, n_p)
            self.flat = axes.flatten()
        flat = self.flat
        for layer in self.model.layers:
            if isinstance(layer, Convolution2D):
                data = layer.W.get_value()
                n_filters = data.shape[0]
                for j in xrange(n_filters):
                    im = data[j, :, :, :].flatten()
                    im = self.np2im(im)
                    flat[i].imshow(im, cmap=plt.get_cmap(map_name), interpolation='none')
                    i += 1
            elif isinstance(layer, Dense):
                data = layer.W.get_value()
                im = self.np2im(data)
                flat[i].clear()
                flat[i].imshow(im, cmap=plt.get_cmap(map_name), interpolation='none')
                i += 1
        plt.setp([a.get_xticklabels() for a in flat], visible=False)
        plt.setp([a.get_yticklabels() for a in flat], visible=False)
        plt.title("ITER %d" % self.epoch)
        plt.draw()
        plt.pause(0.0001)
        # this = np.asarray(this, dtype=np.float32)
        # if not (self.all is None):
        #     print self.all - this
        # self.all = this

    def np2im(self, im, reshape=True):
        imin = np.min(im)
        imax = np.max(im)
        r = abs(imax - imin)
        im -= imin
        im *= 255. / r
        if reshape:
            width = int(sqrt(im.size))
            while len(im) % width != 0:
                width -= 1
            im = im.reshape((width, -1))
        return im
