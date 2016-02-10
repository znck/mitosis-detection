#! /usr/bin/env python
import argparse
import glob
import os
import signal
import time
import traceback

from callbacks import VisHistory
from helpers import JsonIterator, RandomSampler, TT, DatasetGenerator, BatchGenerator


def _task_train_filter(arguments):
    TT.imp("> Training non competent pixel filter")
    path = arguments.path
    assert os.path.exists(path), path + " does not exists"
    path = os.path.abspath(path)
    assert len(glob.glob(os.path.join(path, '*/mitosis'))), "No valid mitosis dataset provided."

    # Init Random Sampler
    dataset = RandomSampler(path, verbose=arguments.verbose)
    positive, n_positive = dataset.positive()

    if arguments.verbose:
        TT.info("> Compiling model...")
    from mitosis import model_base
    model = model_base()

    load_path = os.path.join(path, 'weights.npy')
    if arguments.model:
        load_path = os.path.abspath(arguments.model)
    if os.path.exists(load_path):
        TT.success("> Loading model from %s" % load_path)
        model.load_weights(load_path)
    n_epoch = arguments.epoch

    def save_weights(_1, _2):
        dying_path = load_path + '.dying.npy'
        TT.danger('\r\nProgram Terminated. Saving progressing in %s' % dying_path)
        model.save_weights(dying_path, True)
        exit(0)
        return _1, _2

    signal.signal(signal.SIGINT, save_weights)

    val_split = .1
    if not arguments.validation:
        val_split = .0

    train_start = time.time()
    callbacks = []
    if arguments.visualize:
        vis = VisHistory((1, 3, 5))
        callbacks.append(vis)
    for epoch in xrange(n_epoch):
        epoch_start = time.time()
        sample, n_sample = dataset.sample(n_positive)
        TT.info("> Training on sample dataset %d of %d" % (epoch + 1, n_epoch))
        batches = BatchGenerator(JsonIterator(positive), n_positive, JsonIterator(sample), n_sample, arguments.batch)
        for (x, y) in batches:
            model.fit(x, y, batch_size=arguments.mini_batch, nb_epoch=n_epoch, validation_split=val_split,
                      show_accuracy=True, callbacks=callbacks)
        model.save_weights(load_path, True)
        TT.success(
            "> Finished sample dataset %d of %d took %.2f minutes." % (
                epoch + 1, n_epoch, (time.time() - epoch_start) / 60.))
    TT.success("> Training finished. Time take: %.2f hours." % ((time.time() - train_start) / 3600.))


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


def _theano_optimisation(arguments):
    if arguments.disable_optimisation:
        print TT.WARNING + "> Theano Config :: optimizer=None, exception_verbosity=high", TT.END
        from theano import config
        config.optimizer = 'None'
        config.exception_verbosity = 'high'


def main():
    parser = _parse_args()
    args = parser.parse_args()

    _theano_optimisation(arguments=args)
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


if __name__ == '__main__':
    main()
