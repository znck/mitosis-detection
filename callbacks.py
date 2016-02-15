from keras.callbacks import Callback

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
        # self.f, self.axarr = plt.subplots(2, 3, sharex=True)
        # self.axarr[0][0].set_title('Loss')
        # self.axarr[1][0].set_title('Accuracy')
        # self.axarr[0][1].set_title('Test image')
        # self.axarr[1][1].set_title('Convolutions Layer 1')
        # self.axarr[0][2].set_title('Convolutions Layer 2')
        # self.axarr[1][2].set_title('Convolutions Layer 3')
        self.test_image = None
        self.e = 0

    def show_convolutions(self):
        if len(self.layer):
            pos = [235, 233, 236]
            for i in range(0, 4):
                if i >= len(self.layer):
                    break
                self.visualize_layer(None, self.layer[i], pos[i])
            layer = len(self.model.layers) - 1
        convout1_f = theano.function([self.model.get_input(train=False)],
                                     self.model.layers[layer].get_output(train=False))
        output = convout1_f(self.model.training_data[0][self.img_to_visualize: self.img_to_visualize + 1])
        print output.shape
        print 'Convolutions Layer 3 (P = %f)' % output[0][0]

    def visualize_layer(self, figure, layer, pos=224):
        convout1_f = theano.function([self.model.get_input(train=False)],
                                     self.model.layers[layer].get_output(train=False))
        convolutions = convout1_f(self.model.training_data[0][self.img_to_visualize: self.img_to_visualize + 1])
        nb_filters = convolutions[0].shape[0]
        from math import sqrt, ceil
        width = int(ceil(sqrt(nb_filters)))
        # grid = AxesGrid(figure, pos,
        #                 nrows_ncols=(width, width),
        #                 axes_pad=0.05,
        #                 label_mode="1",
        #                 )
        # grid.axes_llc.set_xticks([])
        # grid.axes_llc.set_yticks([])
        # for i, convolution in enumerate(convolutions[0]):
        # grid[i].imshow(convolution, cmap=cm.Greys_r)
        # plt.draw()

    def _set_params(self, params):
        super(VisHistory, self)._set_params(params)
        self.train_length = int(params.get('nb_sample'))
        self.nb_batches = int(self.train_length / int(params.get('batch_size')))

    def on_train_begin(self, logs=None):
        # self.img_to_visualize = randint(0, self.train_length - 1)
        if self.test_image is None:
            self.test_image = array_to_img(self.model.training_data[0][self.img_to_visualize])
            # self.axarr[0][1].set_title('Test image %f' % self.model.training_data[1][0][0])
        # image1 = self.test_image
        # self.axarr[0][1].imshow(image1)
        # plt.ion()
        # plt.show()

    def on_train_end(self, logs=None):
        # plt.ioff()
        pass

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
        # vl, = self.axarr[0][0].plot(self.losses['val_loss'], label='val_loss', color='r', linewidth=2.0)
        # ll, = self.axarr[0][0].plot(self.losses['loss'], label='loss', color='b')
        # va, = self.axarr[1][0].plot(self.losses['val_acc'], label='val_acc', color='r', linewidth=2.0)
        # aa, = self.axarr[1][0].plot(self.losses['acc'], label='acc', color='b')
        # self.axarr[0][0].legend(handles=[vl, ll], loc=1)
        # self.axarr[1][0].legend(handles=[va, aa], loc=4)
        # plt.pause(0.001)
        # plt.draw()
