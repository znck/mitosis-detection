import theano
from keras.callbacks import Callback
from keras.layers import Convolution2D


class PrintGradients(Callback):
    def on_batch_end(self, batch, logs={}):
        for layer in self.model.layers:
            if isinstance(layer, Convolution2D):
                print theano.pp(layer.W)
