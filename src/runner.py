import argparse
import os
import time

import numpy

from iterators import Dataset, BatchGenerator, ImageIterator
from utilities import TT, np_append, change_ext

callbacks = []


def task_train_filter(args):
    ff, mapper = getattr(__import__('dataset'), args.dataset)()
    dataset = Dataset(root_path=args.path, verbose=args.verbose, name='base-model',
                      mapper=mapper, filename_filter=ff)
    dataset_batches = BatchGenerator(dataset, args.batch)
    from mitosis import model_base
    model = model_base(args.lr)
    model_saved_weights_path = os.path.join(args.path, 'base-model.weights.npy')
    if os.path.exists(model_saved_weights_path):
        TT.info("Loading weights from %s" % model_saved_weights_path)
        model.load_weights(model_saved_weights_path)
    train_start = time.time()
    for epoch in xrange(args.epoch):
        TT.debug(epoch + 1, "of", args.epoch, "epochs")
        for x, y in dataset_batches:
            model.fit(x, y, batch_size=args.mini_batch, nb_epoch=1, validation_split=.1,
                      callbacks=callbacks, show_accuracy=True, shuffle=True)
        TT.info("Saving weights to %s" % model_saved_weights_path)
        model.save_weights(model_saved_weights_path, overwrite=True)
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
    import scipy.misc
    scipy.misc.imsave(change_ext(args.input, 'predicted.tiff'), out)
    scipy.misc.imsave(change_ext(args.input, 'expected.tiff'), dataset.output)


def parse_args():
    parser = argparse.ArgumentParser(description="Mitosis Detection Task Runner")
    parser.add_argument("task", help="Run task. (train-filter, train, test, predict)",
                        choices=['train-filter', 'train', 'test', 'predict', 'train-cnn', 'test-filter'], metavar="task")
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
    elif args.task == 'test-filter':
        TT.debug("Running: Task Test Filter")
        task_test_filter(args)
    else:
        parser.print_help()
        exit(0)


if __name__ == '__main__':
    main()
