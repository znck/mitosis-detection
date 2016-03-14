import json
import os
import random
import warnings

import cv2
import numpy as np
from keras.utils.generic_utils import Progbar


class TT:
    def __init__(self):
        pass

    HEADER = '\033[95m'
    INFO = '\033[94m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    DANGER = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def info(*args):
        print TT.INFO + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def success(*args):
        print TT.SUCCESS + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def danger(*args):
        print TT.DANGER + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def warn(*args):
        print TT.WARNING + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def imp(*args):
        print TT.HEADER + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def ul(*args):
        print TT.UNDERLINE + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def b(*args):
        print TT.BOLD + ' '.join(map(str, args)) + TT.END


class RandomSampler(object):
    def __init__(self, path, size=(1539, 1376), patch=(101, 101), verbose=True):
        """
        Samples all the positive pixels or given no. of random pixels from all the images in a path.
        """
        self.__count = None
        self.__positives = None
        self._path = path
        self._files = _read_all_files(path)
        self._size = size
        self._patch = patch
        self._eff_x = size[0] - patch[0] + 1
        self._eff_y = size[1] - patch[1] + 1
        self._pad_x = patch[0] / 2
        self._pad_y = patch[1] / 2
        self._pixels_per_image = np.prod(np.asarray(size) - np.asarray(patch) + (1, 1))
        self._i = 0
        self._batch = 1
        self._n = self._pixels_per_image * len(self._files)
        self.verbose = False
        self._positives = self.positive_sorted()
        self.verbose = verbose

    def set_batch_size(self, batch_size=100):
        self._batch = batch_size
        return self

    def sample(self, batch_size=100):
        self._batch = batch_size
        if self.verbose:
            TT.info("> Creating a random dataset...")
        indices = [int(random.uniform(0, self._n)) for i in xrange(0, self._batch)]
        sampled = {}
        bar = Progbar(len(indices))
        i = 0
        count = 0
        if self.verbose:
            bar.update(i)
        for index in indices:
            file_id = index / self._pixels_per_image
            filename, _ = self._files[file_id]
            if filename not in sampled:
                sampled[filename] = []
            pixel_index = index % self._pixels_per_image
            x = (pixel_index / self._eff_x) + self._pad_x
            y = (pixel_index % self._eff_y) + self._pad_y
            p = 0.0
            if filename in self._positives and (x * self._size[0] + y) in self._positives[filename]:
                p = float(self._positives[filename][x * self._size[0] + y])
            sampled[filename].append((x, y, (p, 1. - p)))
            count += 1
            if self.verbose:
                i += 1
                bar.update(i)

        return sampled, count

    def positive_sorted(self):
        if os.path.exists(os.path.join(self._path, 'positives_sorted.json')):
            return json.load(open(os.path.join(self._path, 'positives_sorted.json')))

        normal, _, expanded = self.positive(True)

        if False:
            json.dump(normal, open(os.path.join(self._path, 'positives.json'), 'w'))
            json.dump(expanded, open(os.path.join(self._path, 'positives_sorted.json'), 'w'))

        return expanded

    def positive(self, expand=False, n_pix=float('inf')):
        if self.__positives:
            return self.__positives, self.__count
        files = _read_all_files(self._path)

        if self.verbose:
            print TT.INFO + "> Collecting positive samples from dataset...", TT.END

        if self.verbose:
            print "%d files" % len(files)

        bar = Progbar(len(files))
        index = 0
        count = 0
        if self.verbose is 1:
            bar.update(index)
        expanded = {}
        normal = {}
        for f, _ in files:
            v = csv2np(_)
            expanded[f] = {}
            normal[f] = []
            for pos in v:
                (x, y, p) = pos
                x = int(x)
                y = int(y)
                p = float(p)
                for i in xrange(-10, 10):
                    for j in xrange(-10, 10):
                        _x = x + i
                        _y = y + j
                        expanded[f][_x * 1539 + _y] = p
                        normal[f].append((_x, _y, (p, 1. - p)))
                        count += 1
                        if count >= n_pix:
                            break
                    if count >= n_pix:
                        break
                if count >= n_pix:
                    break
            if count >= n_pix:
                break
            index += 1
            if self.verbose is 1:
                bar.update(index)
        self.__positives = normal
        self.__count = count
        if expand:
            return normal, count, expanded
        return normal, count


def _read_all_files(path):
    # Prepare list of directories
    directories = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            directories.append(os.path.join(path, name))
    files = []
    for directory in directories:
        for name in os.listdir(os.path.join(directory, 'frames/x40/')):
            files.append(
                (
                    os.path.join(directory, 'frames/x40/' + name),
                    os.path.join(directory, 'mitosis/' + name.replace('.tiff', '_mitosis.csv'))
                )
            )

    return files


class DatasetGenerator(object):
    def __init__(self, positive, n_positive, sample, n_sample, batch_size, verbose=True):
        """
        :type positive: JsonIterator
        :type n_positive: int
        :type sample: JsonIterator
        :type n_sample: int
        :type batch_size: int
        """
        self.verbose = verbose
        assert batch_size % 2 == 0, "Batch size should be even."
        self.sample = sample
        self.positive = positive
        n_total = n_positive + n_sample
        self.n = int(n_total / batch_size)
        self.i = 0
        self.batch_size = batch_size

    def flow(self):
        for i in xrange(self.n):
            if self.verbose:
                print TT.WARNING + "> Creating batch %d of %d" % (self.i, self.n), TT.END

            x = []
            y = []
            count = 0
            for patch, target in self.positive:
                x.append(patch)
                y.append(target)
                count += 1
                if count == self.batch_size / 2:
                    break
            for patch, target in self.sample:
                x.append(patch)
                y.append(target)
                count += 1
                if count == self.batch_size:
                    break
            yield np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


class BatchGenerator(object):
    def __init__(self, positive, n_positive, sample, n_sample, batch_size, verbose=True):
        """
        :type positive: JsonIterator
        :type n_positive: int
        :type sample: JsonIterator
        :type n_sample: int
        :type batch_size: int
        """
        self.verbose = verbose
        assert batch_size % 2 == 0, "Batch size should be even."
        self.sample = sample
        self.positive = positive
        n_total = n_positive + n_sample
        self.n = int(n_total / batch_size)
        self.i = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self):
        if self.i < self.n:
            self.i += 1
            if self.verbose:
                TT.warn("> Creating batch %d of %d" % (self.i, self.n))

            x = []
            y = []
            data_x = None
            data_y = None
            count = 0
            for patch, target in self.positive:
                x.append(patch.ravel())
                y.append(target)
                count += 1
                if len(x) == 1000:
                    if data_x is None:
                        data_x = np.asanyarray(x, dtype=np.float32)
                        data_y = np.asanyarray(y, dtype=np.float32)
                    else:
                        data_x = np.concatenate((data_x, np.asanyarray(x, dtype=np.float32)))
                        data_y = np.concatenate((data_y, np.asanyarray(y, dtype=np.float32)))
                    x = []
                    y = []
                if count == self.batch_size / 2:
                    break
            for patch, target in self.sample:
                x.append(patch.ravel())
                y.append(target)
                count += 1
                if len(x) == 1000:
                    if data_x is None:
                        data_x = np.asarray(x, dtype=np.float32)
                        data_y = np.asarray(y, dtype=np.float32)
                    else:
                        data_x = np.concatenate((data_x, np.asarray(x, dtype=np.float32)))
                        data_y = np.concatenate((data_y, np.asarray(y, dtype=np.float32)))
                    x = []
                    y = []
                if count == self.batch_size:
                    break
            if len(x):
                if data_x is None:
                    data_x = np.asarray(x, dtype=np.float32)
                    data_y = np.asarray(y, dtype=np.float32)
                else:
                    data_x = np.concatenate((data_x, np.asarray(x, dtype=np.float32)))
                    data_y = np.concatenate((data_y, np.asarray(y, dtype=np.float32)))
            return data_x.reshape((self.batch_size, 3, 101, 101)), data_y
        else:
            raise StopIteration()


class JsonIterator(object):
    def __init__(self, filename, size=(101, 101)):
        self.size = size
        if isinstance(filename, str):
            self.raw = json.load(open(filename))
        elif isinstance(filename, file):
            self.raw = json.load(filename)
        elif isinstance(filename, dict):
            self.raw = filename
        else:
            raise AssertionError('JSON Iterator :: ' + type(filename).__name__ + ' is not iterable.')
        self.files = self.raw.keys()
        self.cur_file = 0
        self.index = 0
        self.orig = cv2.imread(self.files[0])
        self.image = cv2.copyMakeBorder(self.orig, top=self.size[1], bottom=self.size[1], left=self.size[0],
                                        right=self.size[0],
                                        borderType=cv2.BORDER_DEFAULT).transpose(2, 0, 1) / 255.

    def __iter__(self):
        self.index = 0
        self.cur_file = 0
        return self

    def next(self):
        if self.cur_file >= len(self.files):
            raise StopIteration()
        old_filename = filename = self.files[self.cur_file]

        self.index += 1
        while self.cur_file < len(self.files) and self.index >= len(self.raw[filename]):
            self.cur_file += 1
            self.index = 0
            filename = self.files[self.cur_file]

        if self.cur_file >= len(self.files):
            raise StopIteration()

        if old_filename != filename:
            self.orig = cv2.imread(self.files[self.cur_file])
            self.image = cv2.copyMakeBorder(self.orig, top=self.size[1], bottom=self.size[1], left=self.size[0],
                                            right=self.size[0],
                                            borderType=cv2.BORDER_DEFAULT).transpose(2, 0, 1) / 255.

        x, y, p = self.raw[filename][self.index]
        if x < 0 or y < 0 or x > self.orig.shape[0] or y > self.orig.shape[1]:
            return self.next()
        return patch_at(self.image, x, y, self.size), p


def patch_at(image, x, y, size=(101, 101)):
    x_start = x + size[0] / 2
    y_start = y + size[1] / 2
    x_end = x_start + size[0]
    y_end = y_start + size[1]
    p = image[:, x_start:x_end, y_start:y_end]
    return p


def prepare_negative(path):
    sampler = RandomSampler(path=path)
    json.dump(sampler.set_batch_size(65000).sample(), open(os.path.join(path, 'negatives.json'), 'w'))


def prepare_positive(path):
    sampler = RandomSampler(path=path)
    sampler.positive()


def csv2np(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            targets = np.genfromtxt(path, delimiter=',')
            if len(targets.shape) is 1 and len(targets) is 3:
                targets = np.asarray([targets])
            elif len(targets.shape) is 2:
                pass
            else:
                targets = np.asarray([])
    except IOError:
        targets = np.array([])
    return targets
