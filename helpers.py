import json
import os
import random
import warnings

import cv2
import numpy as np
from keras.utils.generic_utils import Progbar


class TT:
    """
    Utility class to pretty print text.
    """

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
    """
    A data sampler interface to curate data points from given directory.
    Directory structure:
    > root
        > DataSet
            > frames
                > x40
                    > .tiff data images
            > mitosis
                > .csv target annotations
    """

    def __init__(self, path, image_size=(1539, 1376), patch_size=(101, 101), verbose=True):
        """
        Samples all the positive pixels or given no. of random pixels from all the images in a path.
        """
        self.path = path  # Root directory
        self.files = None  # List of all dataset files
        self.image_size = image_size  # Size of image. TODO: It can be auto detected.
        self.patch_size = patch_size  # Size of patch. Input size of DNN
        self.pixels_per_image = np.prod(image_size)
        self.i = 0
        self.batch = 1
        self.positives_sorted = None
        self.positives = None  # Iterable list of positive data samples.
        self.sampled = None  # Iterable list of random data samples.
        self.positives_count = 0
        self.verbose = verbose
        self.radius = 10

    def __len__(self):
        if self.files is None:
            self.files = read_all_files(self.path)
        return self.pixels_per_image * len(self.files)

    def set_batch_size(self, batch_size=100):
        self.batch = batch_size
        return self

    def sample(self, batch_size=100):
        if self.sampled is not None and self.batch == batch_size:
            return self.sampled, batch_size

        if os.path.exists(os.path.join(self.path, 'negatives.json')):
            data = json.load(open(os.path.join(self.path, 'negatives.json')))
            if int(data['count']) == batch_size:
                return data['data'], batch_size

        self.batch = batch_size

        if self.verbose:
            TT.info("> Creating a random dataset...")

        if self.files is None:
            self.files = read_all_files(self.path)

        sampled = {}
        bar = Progbar(self.batch)
        count = 0
        positives = 0
        if self.verbose:
            bar.update(count)
        for index in random.sample(range(len(self)), self.batch):
            file_id = index / self.pixels_per_image
            image, csv = self.files[file_id]
            if image not in sampled:
                sampled[image] = []
            pixel = index % self.pixels_per_image
            if image in self.positives_sorted and pixel in self.positives_sorted[image]:
                p = self.positives_sorted[image][pixel]
                positives += 1
            else:
                p = 0.
            (x, y) = self.pixel_to_xy(pixel)
            sampled[image].append([x, y, (p, 1. - p)])
            count += 1
            if self.verbose:
                bar.update(count)
        self.sampled = sampled
        json.dump({'data': sampled, 'count': count}, open(os.path.join(self.path, 'negatives.json')))

        return sampled, count

    def pixel_to_xy(self, pixel):
        return pixel / self.image_size[0], pixel % self.image_size[0]  # TODO: Verify this. `(p / width, p % width)`

    def get_sorted_positives(self):
        if self.positives_sorted is not None:
            return self.positives_sorted

        if os.path.exists(os.path.join(self.path, 'positives_sorted.json')):
            return json.load(open(os.path.join(self.path, 'positives_sorted.json')))

        self.positive()

        if not os.path.exists(os.path.join(self.path, 'positives_sorted.json')):
            json.dump(self.positives_sorted, open(os.path.join(self.path, 'positives_sorted.json'), 'w'))

        return self.positives_sorted

    def positive(self):
        """
        Create a list of positive data points, expand them to square disk of radius `self.radius` pixels.
        :return: Dictionary(key: filename, value: [x, y, (p, 1-p)])
        """
        if self.positives is not None:  # If already calculated then return it.
            return self.positives, self.positives_count

        if os.path.exists(os.path.join(self.path, 'positives.json')):
            data = json.load(open(os.path.join(self.path, 'positives.json')))
            return data['data'], data['count']

        if self.files is None:  # Curate list of files if not done already.
            self.files = read_all_files(self.path)

        bar = Progbar(len(self.files))  # Create instance of progress bar.
        if self.verbose:  # Verbose output for debugging.
            bar.update(0)
            print TT.info("> Collecting positive samples from dataset...")
            print "%d files" % len(self.files)

        index = 0  # File index - to update state of progress bar.
        count = 0  # Holds total number of positive samples.
        expanded = {}  # Holds list of files and positive pixels in flattened image with mitosis probability.
        normal = {}  # Holds list of files and positive pixel (y, x) along with class probabilities.
        #              (0: Mitotic, 1: Non-Mitotic)
        for data_image, target_csv in self.files:
            labels = csv2np(os.path.join(self.path, target_csv))  # Load CSV annotations into numpy array.
            expanded[data_image] = {}  # Initialize list for file
            normal[data_image] = []
            for (x, y, p) in labels:  # Iterate over annotated pixel values.
                x = int(x)
                y = int(y)
                p = float(p)
                # Image position, horizontal -> y, vertical -> x
                # Image size, (y, x)
                # @see http://www.scipy-lectures.org/advanced/image_processing/#basic-manipulations
                range_x = xrange(max(0, x - self.radius), min(x + self.radius, self.image_size[1]))
                range_y = xrange(max(0, y - self.radius), min(y + self.radius, self.image_size[0]))
                for i in range_x:
                    for j in range_y:
                        expanded[data_image][i * self.image_size[0] + j] = p  # TODO: Verify this. `x * width + y`
                        normal[data_image].append([i, j, (p, 1. - p)])  # (x, y) => (row, column)
                        count += 1
            index += 1
            if self.verbose is 1:
                bar.update(index)
        self.positives = normal
        self.positives_sorted = expanded
        self.positives_count = count
        json.dump({'data': self.positives, 'count': self.positives_count},
                  open(os.path.join(self.path, 'positives.json'), 'w'))
        return normal, count


def read_all_files(path):
    files = []
    for directory in os.listdir(path):  # Get everything in path.
        if not os.path.isdir(os.path.join(path, directory)):  # Check if it is directory.
            continue
        # Get everything in directory.
        for name in os.listdir(os.path.join(os.path.join(path, directory), 'frames/x40/')):
            # Append image and csv relative paths.
            files.append(
                (
                    os.path.join(directory, 'frames/x40/' + name),
                    os.path.join(directory, 'mitosis/' + name.replace('.tiff', '_mitosis.csv'))
                )
            )
    files.sort()  # Sort list of files.
    return files


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
                        data_x = np.asanyarray(x, dtype=np.float64)
                        data_y = np.asanyarray(y, dtype=np.float64)
                    else:
                        data_x = np.concatenate((data_x, np.asanyarray(x, dtype=np.float64)))
                        data_y = np.concatenate((data_y, np.asanyarray(y, dtype=np.float64)))
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
                        data_x = np.asarray(x, dtype=np.float64)
                        data_y = np.asarray(y, dtype=np.float64)
                    else:
                        data_x = np.concatenate((data_x, np.asarray(x, dtype=np.float64)))
                        data_y = np.concatenate((data_y, np.asarray(y, dtype=np.float64)))
                    x = []
                    y = []
                if count == self.batch_size:
                    break
            if len(x):
                if data_x is None:
                    data_x = np.asarray(x, dtype=np.float64)
                    data_y = np.asarray(y, dtype=np.float64)
                else:
                    data_x = np.concatenate((data_x, np.asarray(x, dtype=np.float64)))
                    data_y = np.concatenate((data_y, np.asarray(y, dtype=np.float64)))
            return data_x.reshape((self.batch_size, 3, 101, 101)), data_y
        else:
            raise StopIteration()


class ImageIterator(object):
    def __init__(self, input_file, output=None, batch=1, size=(101, 101)):
        orig = normalize(cv2.imread(input_file))
        self.size = size
        self.image_size = orig.shape[:2]
        self.image = cv2.copyMakeBorder(orig, top=self.size[1], bottom=self.size[1], left=self.size[0],
                                        right=self.size[0], borderType=cv2.BORDER_DEFAULT).transpose(2, 0, 1)
        self.output = np.zeros(self.image_size)
        if output is not None:
            v = csv2np(output)
            for pos in v:
                (x, y, p) = pos
                x = int(x)
                y = int(y)
                p = float(p)
                for i in xrange(-10, 10):
                    for j in xrange(-10, 10):
                        _x = x + i
                        _y = y + j
                        self.output[_x, _y] = p
        self.i = 0
        self.len = np.prod(self.image_size)
        self.batch = batch

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        j = 0
        batch = []
        target = []
        while j < self.batch:
            if self.i < self.len:
                x = self.i % self.image_size[1]
                y = self.i / self.image_size[1]
                target.append((self.output[y, x], 1. - self.output[y, x]))
                batch.append(patch_at(self.image, y, x, self.size))
            else:
                raise StopIteration()
            j += 1
            self.i += 1
        return np.asarray(batch), np.asarray(target)


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
    """
    Extract patch from image with extra border of size `size`
    :param image: numpy image matrix
    :param x: row number of center pixel
    :param y: column number of center pixel
    :param size: dimensions of patch
    :return:
    """
    x += (size[1]) / 2
    y += (size[0]) / 2
    return image[:, x:x + size[1], y:y + size[0]]


def csv2np(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            targets = np.genfromtxt(path, delimiter=',')
            if len(targets.shape) is 1 and len(targets) is 3:
                targets = np.asarray([targets], dtype=np.float64)
            elif len(targets.shape) is 2:
                pass
            else:
                targets = np.asarray([])
    except IOError:
        targets = np.array([])
    return targets


def normalize(arr):
    arr = arr.astype('float32')
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def _test_image_iterator():
    import sys

    itr = ImageIterator(sys.argv[1], sys.argv[2], batch=1)
    import matplotlib.pyplot as plt

    for i_itr in itr:
        print i_itr[0].shape
        plt.imshow(i_itr[0].transpose(1, 2, 0))
        plt.show()


def _test_csv_np():
    files = ['empty.csv', 'one.csv', 'two.csv', 'three.csv']
    outputs = [
        [],
        ([1, 1, .5],),
        ([1, 1, .5], [2, 2, .5]),
        ([1, 1, .5], [2, 2, .5], [3, 3, .5])
    ]
    for filename, output in zip(files, outputs):
        path = os.path.abspath(os.path.join('tests/csv2np', filename))
        assert np.array_equiv(output, csv2np(path))


def _test_patch_at():
    size = (101, 101)
    orig = cv2.imread(os.path.abspath('tests/patch_at/test.tiff'))
    image = cv2.copyMakeBorder(orig, top=size[1], bottom=size[1], left=size[0], right=size[0],
                               borderType=cv2.BORDER_DEFAULT).transpose(2, 0, 1)
    patch_0_0 = cv2.imread(os.path.abspath('tests/patch_at/patch_0_0.tiff')).transpose(2, 0, 1)
    patch_300_500 = cv2.imread(os.path.abspath('tests/patch_at/patch_300_500.tiff')).transpose(2, 0, 1)
    patch_500_300 = cv2.imread(os.path.abspath('tests/patch_at/patch_500_300.tiff')).transpose(2, 0, 1)
    pixels = [(51, 51), (351, 551), (551, 351)]
    outputs = (patch_0_0, patch_300_500, patch_500_300)
    for (x, y), expected in zip(pixels, outputs):
        actual = patch_at(image=image, x=x, y=y, size=size)
        actual = normalize(actual)
        expected = normalize(expected)
        try:
            assert np.array_equiv(expected, actual)
        except AssertionError:
            print "Failed: ", (x, y), expected.shape, actual.shape
            import pylab
            f = pylab.figure()
            f.add_subplot(3, 1, 1)
            pylab.imshow(actual.transpose(1, 2, 0))
            f.add_subplot(3, 1, 2)
            pylab.imshow(expected.transpose(1, 2, 0))
            f.add_subplot(3, 1, 3)
            diff = expected - actual
            pylab.imshow(diff.transpose(1, 2, 0))
            from scipy.linalg import norm
            print "Norms: ", np.sum(np.abs(diff)), norm(diff.ravel(), 0)
            pylab.show()

if __name__ == '__main__':
    _test_csv_np()
    _test_patch_at()
