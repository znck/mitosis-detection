import argparse
import os
import time

import numpy

from callbacks import LearnLog
from iterators import Dataset, BatchGenerator, ImageIterator
from utilities import TT, np_append, change_ext


def task_train_filter(args):
    ff, mapper = getattr(__import__('dataset'), args.dataset)()
    dataset = Dataset(root_path=args.path, verbose=args.verbose, name='base-model',
                      mapper=mapper, filename_filter=ff, rotation=False)
    dataset_batches = BatchGenerator(dataset, args.batch)
    from mitosis import model_base
    model = model_base(args.lr)
    model_saved_weights_path = os.path.join(args.path, 'base-model.weights.npy')
    if os.path.exists(model_saved_weights_path):
        TT.info("Loading weights from %s" % model_saved_weights_path)
        model.load_weights(model_saved_weights_path)
    train_start = time.time()
    log = LearnLog("Base")
    for epoch in xrange(args.epoch):
        TT.debug(epoch + 1, "of", args.epoch, "epochs")
        log.epoch = epoch + 1
        log.batch = 0
        for x, y in dataset_batches:
            log.batch += 1
            model.fit(x, y, batch_size=args.mini_batch, nb_epoch=1, validation_split=.1,
                      callbacks=[log], show_accuracy=True, shuffle=True)
        TT.info("Saving weights to %s" % model_saved_weights_path)
        model.save_weights(model_saved_weights_path, overwrite=True)
    TT.success("Training finished in %.2f hours." % ((time.time() - train_start) / 3600.))


def task_train_cnn(args):
    ff, mapper = getattr(__import__('dataset'), args.dataset)()
    dataset = Dataset(root_path=args.path, verbose=args.verbose, name='cnn',
                      mapper=mapper, filename_filter=ff, ratio=9)
    dataset_batches = BatchGenerator(dataset, args.batch)
    from mitosis import model_base, model_1, model_2

    model = model_base(lr=0)
    model1 = model_1(args.lr)
    model2 = model_2(args.lr)
    model_saved_weights_path = os.path.join(args.path, 'base-model.weights.npy')
    model1_saved_weights_path = os.path.join(args.path, 'model1.weights.npy')
    model2_saved_weights_path = os.path.join(args.path, 'model2.weights.npy')
    if os.path.exists(model_saved_weights_path):
        TT.info("Loading weights from %s" % model_saved_weights_path)
        model.load_weights(model_saved_weights_path)
    if os.path.exists(model1_saved_weights_path):
        TT.info("Loading weights from %s" % model1_saved_weights_path)
        model1.load_weights(model1_saved_weights_path)
    if os.path.exists(model2_saved_weights_path):
        TT.info("Loading weights from %s" % model2_saved_weights_path)
        model2.load_weights(model2_saved_weights_path)
    train_start = time.time()
    log1 = LearnLog("DNN 1")
    log2 = LearnLog("DNN 2")
    for epoch in xrange(args.epoch):
        TT.debug(epoch + 1, "of", args.epoch, "epochs")
        log1.epoch = log2.epoch = epoch + 1
        log1.batch = log2.batch = 0
        for x, y in dataset_batches:
            log1.batch += 1
            log2.batch += 1
            outputs = model.predict(x, batch_size=args.mini_batch, verbose=args.verbose)
            # Multiply each window with it's prediction and then pass it to the next layer
            x_new = []
            y_new = []
            for i in range(len(outputs)):
                if outputs[i][0] > .3:
                    x_new.append(x[i])
                    y_new.append(y[i])
            TT.debug("Model 1 on epoch %d" % (epoch + 1))
            model1.fit(numpy.asarray(x_new), numpy.asarray(y_new), batch_size=args.mini_batch, nb_epoch=1, validation_split=.1,
                       callbacks=[log1], show_accuracy=True, shuffle=True)
            # TT.debug("Model 2 on epoch %d" % (epoch + 1))
            # model2.fit(x_new, y_new, batch_size=args.mini_batch, nb_epoch=1, validation_split=.1,
            #            callbacks=[log2], show_accuracy=True, shuffle=True)
        TT.info("Saving weights to %s" % model_saved_weights_path)
        model.save_weights(model_saved_weights_path, overwrite=True)
        model1.save_weights(model1_saved_weights_path, overwrite=True)
        # model2.save_weights(model2_saved_weights_path, overwrite=True)
    TT.success("Training finished in %.2f hours." % ((time.time() - train_start) / 3600.))


def task_test_filter(args):
    dataset = ImageIterator(args.input, args.output)
    dataset_batches = BatchGenerator(dataset, args.batch)
    from mitosis import model_base
    model = model_base(args.lr)
    model_saved_weights_path = os.path.join(args.path, 'base-model.weights.npy')
    TT.info("Loading weights from %s" % model_saved_weights_path)
    model.load_weights(model_saved_weights_path)
    test_start = time.time()
    out = None
    for x, y in dataset_batches:
        tmp = model.predict(x, args.mini_batch, args.verbose)
        out = np_append(out, tmp)
    out = numpy.reshape(out[:, 0], dataset.image_size)
    numpy.save(change_ext(args.input, 'predicted.npy'), out)
    numpy.save(change_ext(args.input, 'expected.npy'), dataset.output)
    TT.success("Testing finished in %.2f minutes." % ((time.time() - test_start) / 60.))


def task_test_cnn(args):
    dataset = ImageIterator(args.input, args.output)
    dataset_batches = BatchGenerator(dataset, args.batch)
    from mitosis import model_base, model_1, model_2
    model = model_base(0)
    model1 = model_1(0)
    model2 = model_2(0)
    model_saved_weights_path = os.path.join(args.path, 'base-model.weights.npy')
    model1_saved_weights_path = os.path.join(args.path, 'model1.weights.npy')
    model2_saved_weights_path = os.path.join(args.path, 'model2.weights.npy')
    TT.info("Loading weights from %s" % model_saved_weights_path)
    model.load_weights(model_saved_weights_path)
    TT.info("Loading weights from %s" % model1_saved_weights_path)
    model1.load_weights(model1_saved_weights_path)
    TT.info("Loading weights from %s" % model2_saved_weights_path)
    model2.load_weights(model2_saved_weights_path)
    test_start = time.time()
    out = out1 = out2 = None
    for x, y in dataset_batches:
        tmp = model.predict(x, args.mini_batch, args.verbose)
        out = np_append(out, tmp)
        x_new = []
        indices = []
        for i in range(len(tmp)):
            if tmp[i][0] > .3:
                x_new.append(x[i])
                indices.append(i)
        tmp1 = model.predict(x_new, args.mini_batch, args.verbose)
        local = numpy.zeros(tmp.shape)
        local[indices] = tmp1
        out1 = np_append(out1, local)
        tmp1 = model.predict(x_new, args.mini_batch, args.verbose)
        local = numpy.zeros(tmp.shape)
        local[indices] = tmp1
        out2 = np_append(out2, local)
    out = numpy.reshape(out[:, 0], dataset.image_size)
    out1 = numpy.reshape(out1[:, 0], dataset.image_size)
    out2 = numpy.reshape(out2[:, 0], dataset.image_size)
    numpy.save(change_ext(args.input, 'predicted.npy'), out)
    numpy.save(change_ext(args.input, 'model1.predicted.npy'), out1)
    numpy.save(change_ext(args.input, 'model2.predicted.npy'), out2)
    numpy.save(change_ext(args.input, 'expected.npy'), dataset.output)
    TT.success("Testing finished in %.2f minutes." % ((time.time() - test_start) / 60.))


def parse_args():
    parser = argparse.ArgumentParser(description="Mitosis Detection Task Runner")
    parser.add_argument("task", help="Run task. (train-filter, train-cnn, test-filter, test-cnn)",
                        choices=['train-filter', 'test-cnn', 'train-cnn', 'test-filter'],
                        metavar="task")
    parser.add_argument("path", type=str, help="Directory containing mitosis images", metavar="path")
    parser.add_argument("--epoch", type=int, help="Number of epochs. (Default: 10)", default=10)
    parser.add_argument("--batch", type=int, help="Size of batch fits in memory. (Default: 3000)", default=3000)
    parser.add_argument("--mini-batch", type=int, help="Size of training batch. (Default: 100)", default=100)
    parser.add_argument("--lr", type=float, help="Learning Rate. (Default: .002)", default=.002)
    parser.add_argument("--output", type=str, help="output. (Default: None)", default=None)
    parser.add_argument("--input", type=str, help="input. (Default: None)", default=None)
    parser.add_argument("-v", action="store_true", help="Increase verbosity. (Default: Disabled)", default=False,
                        dest='verbose')
    parser.add_argument("--dataset", type=str, help="Dataset type: icpr2012", default='icpr2012')

    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    TT.verbose = args.verbose
    if args.task == 'train-filter':
        TT.debug("Running: Task Train Filter")
        task_train_filter(args)
    if args.task == 'train-cnn':
        TT.debug("Running: Task Train CNN")
        task_train_cnn(args)
    elif args.task == 'test-filter':
        TT.debug("Running: Task Test Filter")
        task_test_filter(args)
    elif args.task == 'test-cnn':
        TT.debug("Running: Task Test CNN")
        task_test_cnn(args)
    else:
        parser.print_help()
        exit(0)


if __name__ == '__main__':
    main()
