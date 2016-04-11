# coding=utf-8
import json
import os
import random
import warnings

import cv2
import numpy as np
import time
from keras.utils.generic_utils import Progbar
import random


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

    def __init__(self, path, image_size=(2084, 2084), patch_size=(101, 101), verbose=True, ratio=1., filename='dataset'):
        """
        Samples all the positive pixels or given no. of random pixels from all the images in a path.
        """
        self.path = path  # Root directory
        self.files = None  # List of all dataset files
        self.given_image_size = image_size  # Size of image. (width, height) TODO: It can be auto detected.
        self.patch_size = patch_size  # Size of patch. Input size of DNN
        self.pixels_per_image = np.prod(image_size)
        self.i = 0
        self.batch = 1
        self.positives_sorted = None
        self.positives = None  # Iterable list of positive data samples.
        self.sampled = None  # Iterable list of random data samples.
        self.sampled_dataset = None
        self.dataset_size = 0
        self.positives_count = 0
        self.verbose = verbose
        self.radius = 10
        self.ratio = ratio
	self.filename = filename

    def image_size(self):
        if self.given_image_size is not None:
	    return self.given_image_size

    def __len__(self):
        if self.files is None:
            self.files = read_all_files(self.path)
        return self.pixels_per_image * len(self.files)

    def set_batch_size(self, batch_size=100):
        self.batch = batch_size
        return self

    def dataset(self, ratio=None):
        if self.sampled_dataset is not None:
            return self.sampled_dataset, self.dataset_size

        if os.path.exists(os.path.join(self.path, self.filename + '.json')):
            dataset = json.load(open(os.path.join(self.path, self.filename + '.json')))  #filename is dataset
            return dataset['data'], int(dataset['size'])
        if ratio is None:
            ratio = self.ratio
        pos, pos_c = self.positive()
        neg, neg_c = self.sample(int(pos_c * ratio))
        for index in pos:  # Append positive data to negative data.
            if index not in neg:
                neg[index] = pos[index]
            else:
                neg[index] += pos[index]
        self.sampled_dataset = neg
        TT.info("> %d positive and %d negative." % (pos_c, neg_c))
        self.dataset_size = pos_c + neg_c
        json.dump({'data': self.sampled_dataset, 'size': self.dataset_size, 'positive': pos_c, 'negative': neg_c},
                  open(os.path.join(self.path, self.filename + '.json'), 'w'))
        return self.sampled_dataset, self.dataset_size

    def sample(self, batch_size=100):
        if self.sampled is not None and self.batch == batch_size:
            return self.sampled, batch_size

        self.batch = batch_size

        if self.verbose:
            TT.info("> Creating a random dataset...")

        if self.files is None:
            self.files = read_all_files(self.path)

        TT.info("> Sampling from", len(self), "pixels.")
        indices = xrange(len(self))
        sampled = {}
        bar = Progbar(self.batch)
        count = 0
        positives = 0
        if self.verbose:
            bar.update(count)
        for index in random.sample(indices, self.batch):
            file_id = index / self.pixels_per_image
            image, csv = self.files[file_id]
            if image not in sampled:
                sampled[image] = []
            pixel = index % self.pixels_per_image
            if image in self.positives_sorted and pixel in self.positives_sorted[image]:
                p = 1.
                positives += 1
            else:
                p = 0.
            (x, y) = self.pixel_to_xy(pixel)
            sampled[image].append([x, y, p])
            count += 1
            if self.verbose:
                bar.update(count)
        self.sampled = sampled
        if positives > 0:
            TT.warn("> Out of", batch_size, "sampled pixels,", positives, "pixels are positive.")

        return sampled, count

    def pixel_to_xy(self, pixel):
        return pixel / self.image_size[1], pixel % self.image_size[1]  # TODO: Verify this. `(p / width, p % width)`

    def get_sorted_positives(self):
        if self.positives_sorted is not None:
            return self.positives_sorted

        self.positive()

        return self.positives_sorted

    def positive(self, set_positive=1.):
        """
        Create a list of positive data points, expand them to square disk of radius `self.radius` pixels.
        :param set_positive: Set positive to max(1., p) where p is given probability
        :return: Dictionary(key: filename, value: [x, y, (p, 1-p)])
        """
        if self.positives is not None:  # If already calculated then return it.
            return self.positives, self.positives_count

        if self.files is None:  # Curate list of files if not done already.
            self.files = read_all_files(self.path)

        bar = Progbar(len(self.files))  # Create instance of progress bar.
        if self.verbose:  # Verbose output for debugging.
            print TT.info("> Collecting positive samples from dataset...")
            bar.update(0)

        index = 0  # File index - to update state of progress bar.
        count = 0  # Holds total number of positive samples.
        expanded = {}  # Holds list of files and positive pixels in flattened image with mitosis probability.
        normal = {}  # Holds list of files and positive pixel (y, x) along with class probabilities.
        #              (0: Mitotic, 1: Non-Mitotic)
        total = 0
        for data_image, target_csv in self.files:
            labels = csv2np(os.path.join(self.path, target_csv))  # Load CSV annotations into numpy array.
            expanded[data_image] = {}  # Initialize list for file
            normal[data_image] = []
            total += len(labels)
            for (y, x, p) in labels:  # Iterate over annotated pixel values. CSV format: width,height,probability
                x = int(x)
                y = int(y)
                p = max(set_positive, float(p))
                # Image position, horizontal -> y, vertical -> x
                # Image size, (y, x)
                # @see http://www.scipy-lectures.org/advanced/image_processing/#basic-manipulations
                range_x = xrange(max(0, x - self.radius), min(x + self.radius, self.image_size[0]))
                range_y = xrange(max(0, y - self.radius), min(y + self.radius, self.image_size[1]))
                for i in range_x:
                    for j in range_y:
                        expanded[data_image][i * self.image_size[0] + j] = p  # TODO: Verify this. `x * width + y`
                        normal[data_image].append([i, j, p])  # (x, y) => (row, column)
                        count += 1
            index += 1
            if self.verbose:
                bar.update(index)
        self.positives = normal
        self.positives_sorted = expanded
        self.positives_count = count
        TT.success("> Total", count, "positive pixels from", total, "annotations.")
        return normal, count


def read_all_files(path):
    files = []
    for directory in os.listdir(path):  # Get everything in path.
        if not os.path.isdir(os.path.join(path, directory)):  # Check if it is directory.
            continue
        # Get everything in directory.
        for name in os.listdir(os.path.join(path, directory)):  # Changed a line here, removed frames/x40
            # Append image and csv relative paths.
            files.append(
                (
                    os.path.join(directory, name),  
                    os.path.join(directory, name.replace('.bmp', '.csv'))
                )
            )
    files.sort()  # Sort list of files.
    return files


class BatchGenerator(object):
    def __init__(self, dataset, batch_size, verbose=True, pool_size=2000):
        """
        :type dataset: JsonIterator
        :type batch_size: int
        """
        self.verbose = verbose
        self.dataset = dataset
        self.pool_size = pool_size
        TT.info("> %d images in current dataset." % len(dataset))
        self.n = int(len(dataset) / batch_size)
        if abs(float(self.n) - (float(len(dataset)) / batch_size)) > 0.00001:
            TT.warn("> Batch size is not exact multiple. Some images might be truncated.")
        self.i = 1
        self.batch_size = batch_size

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return self.n

    def generator(self):
        def append(dst, pool, item):
            if item is not None:
                pool.append(item)
                if len(pool) < min(self.pool_size, self.batch_size):
                    return dst, pool
            if dst is None:
                return np.asarray(pool, dtype=np.float64), []
            if len(pool):
                return np.concatenate((dst, pool)), []
            return dst, []
        if self.verbose:
            TT.warn("> Creating batch %d of %d" % (self.i, self.n))
        data_x = None
        data_y = None
        pool_x = []
        pool_y = []
        count = 0
        self.i = 0
        start = time.clock()
        for x, y in self.dataset:
            data_x, pool_x = append(data_x, pool_x, x)
            data_y, pool_y = append(data_y, pool_y, (y, 1-y))
            count += 1
            if count >= self.batch_size:
                data_x, pool_x = append(data_x, pool_x, None)
                data_y, pool_y = append(data_y, pool_y, None)
                if self.verbose:
                    TT.info("> Completed in", time.clock() - start, "seconds. This batch has", int(np.sum(data_y[:, 0])),
                            "positive pixels and", int(np.sum(data_y[:, 1])), "negative pixels.")
                yield data_x, data_y
                if self.verbose:
                    self.i += 1
                    TT.warn("> Creating batch %d of %d" % (self.i, self.n))
                count = 0
                data_x = None
                data_y = None
                start = time.clock()


class ImageIterator(object):
    def __init__(self, input_file, output=None, batch=1, size=(101, 101)):
        self.input_file = input_file
        orig = normalize(cv2.imread(input_file))
        self.size = size
        self.radius = 10
        self.image_size = orig.shape[:2]
        self.image = img2np(cv2.copyMakeBorder(orig, top=self.size[1], bottom=self.size[1], left=self.size[0],
                                               right=self.size[0], borderType=cv2.BORDER_DEFAULT))
        self.output = np.zeros(self.image_size)
        if output is not None:
            labels = csv2np(output)
            for (y, x, p) in labels:
                x = int(x)
                y = int(y)
                p = float(p)
                range_x = xrange(max(0, x - self.radius), min(x + self.radius, self.image_size[0]))
                range_y = xrange(max(0, y - self.radius), min(y + self.radius, self.image_size[1]))
                for i in range_x:
                    for j in range_y:
                        self.output[i, j] = 1
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
                batch.append(patch_at(self.image, y, x, self.size))
                target.append((self.output[y, x], 1. - self.output[y, x]))
            else:
                raise StopIteration()
            j += 1
            self.i += 1
        return np.asarray(batch), np.asarray(target)


class JsonIterator(object):
    def __init__(self, raw, path=None, size=(101, 101)):
        self.size = size
        self.path = path
        self.len = raw[1]
        self.raw = raw[0]
        TT.info("> %d images to be iterated." % self.len)
        self.files = self.raw.keys()
        self.x = self.y = self.p = None  # Store state of current pixel.
        self.cur_file = 0
        self.index = 0
        self.orig = None
        self.image = None

    def __len__(self):
        return self.len

    def __iter__(self):
        self.index = 0
        self.cur_file = 0
        return self.generator()

    def generator(self):
        # Shuffle Files.
        random.shuffle(self.files)  # Shuffle files in every epoch
        for cur_file in xrange(len(self.files)):
            filename = self.files[cur_file]
            self.orig = normalize(cv2.imread(os.path.join(self.path, self.files[self.cur_file])))
            self.image = img2np(cv2.copyMakeBorder(self.orig, top=self.size[1], bottom=self.size[1], left=self.size[0],
                                                   right=self.size[0], borderType=cv2.BORDER_DEFAULT))
            for index in xrange(len(self.raw[filename])):
                x, y, p = self.raw[filename][index]
                yield patch_at(self.image, x, y, self.size), p


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
    result = []
    line_number = 1
    csv_type = None
    for line in open(path).readlines():
        # Ignore empty lines.
        if len(line.strip()) == 0:
            continue
        # Parse line into list of numbers.
        points = map(lambda x: float(x), line.strip().split(','))
        # Detect format of csv. csv_type ∈ {1, 2}
        # 1: Format (y, x, p)
        # 2: Format (y, x), (y, x) ...
        if csv_type is None:
            csv_type = len(points) % 2
        # Check data is correct.
        assert len(points) % 2 == csv_type
        if csv_type == 0:
            # Convert (x, y) -> (x, y, line_number)
            result += [points[i:i+1] + [line_number] for i in xrange(len(points), step=2)]
        else:
            result.append(points)
        line_number += 1
    return np.asarray(result, dtype=np.float64)


def normalize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def np2img(arr):
    return arr.transpose(1, 2, 0)


def img2np(arr):
    return arr.transpose(2, 0, 1)


def _test_image_iterator():
    import sys

    itr = ImageIterator(sys.argv[1], sys.argv[2], batch=1)
    import matplotlib.pyplot as plt

    for i_itr in itr:
        print i_itr[0].shape
        plt.imshow(np2img(i_itr[0]))
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
    image = img2np(cv2.copyMakeBorder(orig, top=size[1], bottom=size[1], left=size[0], right=size[0],
                                      borderType=cv2.BORDER_DEFAULT))
    patch_0_0 = img2np(cv2.imread(os.path.abspath('tests/patch_at/patch_0_0.tiff')))
    patch_300_500 = img2np(cv2.imread(os.path.abspath('tests/patch_at/patch_300_500.tiff')))
    patch_500_300 = img2np(cv2.imread(os.path.abspath('tests/patch_at/patch_500_300.tiff')))
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
            pylab.imshow(np2img(actual))
            f.add_subplot(3, 1, 2)
            pylab.imshow(np2img(expected))
            f.add_subplot(3, 1, 3)
            diff = expected - actual
            pylab.imshow(np2img(diff))
            from scipy.linalg import norm
            print "Norms: ", np.sum(np.abs(diff)), norm(diff.ravel(), 0)
            pylab.show()


def _test_labels_are_correctly_loaded():
    sampler = RandomSampler(os.path.abspath('training_aperio'))
    iterable_dataset = JsonIterator(sampler.dataset(1.0), path=os.path.abspath('training_aperio'))
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    old = None
    plt.gca().invert_yaxis()
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    for img, _ in iterable_dataset:
        plt.subplot(gs[1])
        if iterable_dataset.p[0] > .5:
            plt.imshow(np2img(img))
        plt.subplot(gs[0])
        cur = iterable_dataset.cur_file
        if cur != old:
            print iterable_dataset.files[cur]
            plt.imshow(iterable_dataset.orig)
            plt.show()
        marker = 'ro'
        if iterable_dataset.p[0] > .5:
            marker = 'yo'
        plt.plot([iterable_dataset.y], [iterable_dataset.x], marker)
        if iterable_dataset.p[0] > .5:
            print "at %d, %d with %f" % (iterable_dataset.x, iterable_dataset.y, iterable_dataset.p[0])
        old = cur


def _create_dataset():
    sampler = RandomSampler(os.path.abspath('training_aperio'))
    iterable_dataset = JsonIterator(sampler.dataset(2.0), path=os.path.abspath('training_aperio'))
    count = 0
    path = os.path.abspath('training_aperio/dataset/')
    gen = BatchGenerator(iterable_dataset, 1000)
    for batch, target in gen:
        filename = os.path.join(path, str(count))
        TT.info("Saving data to", filename + '_data.npy')
        np.save(filename + '_data.npy', batch)
        TT.info("Saving labels to", filename + '_label.npy')
        np.save(filename + '_label.npy', target)
        count += 1
        TT.info("Batch", count, "of", len(gen), "created.")
    TT.success("Finished.")


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 2:
        _test_csv_np()
        _test_patch_at()
        TT.success('All tests completed.')
    elif len(sys.argv) == 3:
        TT.info('Interactive tests:')
        _test_labels_are_correctly_loaded()
    else:
        _create_dataset()
