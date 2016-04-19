import json
import os
import random
import time
from Queue import Queue
from math import ceil
from thread import start_new_thread

import numpy as np

from utilities import prepared_dataset_image, patch_centered_at, image_size, list_all_files, index_at_pixel, \
    pixel_at_index, TT, load_csv, random_rotation


class BatchGenerator(object):
    def __init__(self, dataset, batch_size, pool_size=4000):
        """
        :type dataset:Dataset|ImageIterator
        :type batch_size:int
        :type pool_size:int
        """
        self.verbose = TT.verbose
        self.dataset = dataset
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.n = int(ceil(len(dataset) * 1.0 / batch_size))
        self.MAX_NUM = 2

    def __len__(self):
        return self.n

    def __iter__(self):
        data = Queue(self.MAX_NUM)

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

        def produce():
            i = 1
            count = 0
            data_x = data_y = None
            pool_x = []
            pool_y = []
            for x, y in self.dataset:
                data_x, pool_x = append(data_x, pool_x, x)
                data_y, pool_y = append(data_y, pool_y, (y, 1 - y))
                count += 1
                if count >= self.batch_size:
                    data_x, pool_x = append(data_x, pool_x, None)
                    data_y, pool_y = append(data_y, pool_y, None)
                    data.put([data_x, data_y])
                    i += 1
                    count = 0
                    data_x = data_y = None
            if count > 0:
                data_x, pool_x = append(data_x, pool_x, None)
                data_y, pool_y = append(data_y, pool_y, None)
                data.put([data_x, data_y])

        start_new_thread(produce, ())
        i = 1
        while i <= self.n:
            start = time.clock()
            X, Y = data.get()
            if self.verbose:
                TT.debug("batch", i, "of", self.n, "completed in", time.clock() - start, "seconds. This batch has",
                         int(np.sum(Y[:, 0])), "positive pixels and", int(np.sum(Y[:, 1])), "negative pixels.")
            yield X, Y
            i += 1


class Dataset(object):
    def __init__(self, root_path, patch_size=(101, 101), verbose=False, ratio=1.0, name='dataset', mapper=None,
                 filename_filter=None, rotation=True):
        TT.debug("Dataset root path set to:", root_path)
        self.name = name
        self.patch_size = patch_size
        self.ratio = ratio
        self.root_path = os.path.abspath(root_path)
        self.verbose = verbose
        self.label_mapper = mapper
        self.filename_filter = filename_filter
        self.rotation = rotation

    def __iter__(self):
        return DatasetIterator(self).generator()

    def __len__(self):
        _, s = self.data
        return s

    @property
    def files(self):
        if not hasattr(self, '_files'):
            self._files = list_all_files(self.root_path, filename_filter=self.filename_filter, mapper=self.label_mapper)
            TT.debug("Found", len(self._files), "matching files in", self.root_path)
        return self._files

    @property
    def image_size(self):
        if not hasattr(self, '_image_size'):
            self._image_size = image_size(prepared_dataset_image(os.path.join(self.root_path, self.files[0][0])))
        return self._image_size

    @property
    def dataset_store_path(self):
        return os.path.join(self.root_path, self.name+'.dataset.json')

    @property
    def data(self):
        if hasattr(self, '_dataset'):
            return self._dataset, self._dataset_size

        if self.load():
            return self.data

        TT.debug("Creating new dataset.")
        pos, pos_c = self.positive
        sam, sam_c = self.sample
        for filename in pos:
            if filename not in sam:
                sam[filename] = pos[filename]
            else:
                sam[filename] += pos[filename]
        self._dataset = sam
        self._dataset_size = sam_c + pos_c
        self.dump()
        return self.data

    def load(self):
        if os.path.exists(self.dataset_store_path):
            TT.debug("Loading dataset from", self.dataset_store_path)
            data = json.load(open(self.dataset_store_path))
            self._dataset = data['data']
            self._dataset_size = data['size']
            self._positive = {}
            self._positive_size = data['positive_size']
            self._sample = {}
            self._sample_size = data['sample_size']
            self.positive_in_sample = 0
            TT.debug("Current dataset has", self._dataset_size, "images.",
                     self._positive_size, "positive and", self._sample_size, "negative.")
            return True
        return False

    def dump(self):
        _ = self.data  # Create data if not already created.
        TT.debug("Current dataset has", self._dataset_size, "images.",
                 self._positive_size, "positive and", self._sample_size, "negative.")
        json.dump({'data': self._dataset, 'size': self._dataset_size,
                   'positive_size': self._positive_size + self.positive_in_sample,
                   'sample_size': self._sample_size - self.positive_in_sample},
                  open(self.dataset_store_path, 'w'))

    @property
    def positive(self):
        if hasattr(self, '_positive'):
            return self._positive, self._positive_size

        TT.debug("Collecting positive samples.")
        self._positive = {}
        self._positive_size = 0
        self._positive_expanded = {}
        for data_file, label_file in self.files:
            labels = load_csv(os.path.join(self.root_path, label_file))
            self._positive[data_file] = labels
            self._positive_size += len(labels)
            self._positive_expanded[data_file] = {}
            for col, row, p in labels:
                self._positive_expanded[data_file][index_at_pixel(col=col, row=row, size=self.image_size)] = p
        TT.debug("Found", self._positive_size, "positive samples.")
        return self.positive

    @property
    def sample(self):
        if hasattr(self, '_sample'):
            return self._sample, self._sample_size

        self._sample = {}
        self._sample_size = 0
        _, n = self.positive
        positives = self._positive_expanded
        n = int(n * self.ratio)
        TT.debug("Collecting", n, "random samples.")
        pixels_per_image = int(np.prod(self.image_size))
        indices = xrange(len(self.files) * pixels_per_image)
        ignored = 0
        for index in random.sample(indices, n):
            data_file, label_file = self.files[index / pixels_per_image]
            if data_file not in self._sample:
                self._sample[data_file] = []
            pixel = index % pixels_per_image
            p = 0.0
            if data_file in positives and pixel in positives[data_file]:
                p = 1.0
                ignored += 1
            col, row = pixel_at_index(pixel, self.image_size)
            self._sample[data_file].append([col, row, p])
            self._sample_size += 1
        TT.debug(ignored, "samples out of", self._sample_size, "random samples are positive.")
        self.positive_in_sample = ignored
        return self.sample


class DatasetIterator(object):
    """
    DatasetIterator can iterate a dataset sampled with Dataset.
    """

    def __init__(self, dataset):
        """
        :type dataset:Dataset
        :return:
        """
        self.patch_size = dataset.patch_size
        self.root_path = dataset.root_path
        self.dataset, self.dataset_size = dataset.data
        self.verbose = dataset.verbose
        self.rotation = dataset.rotation

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        return self.generator()

    def generator(self):
        files = self.dataset.keys()
        random.shuffle(files)
        for filename in files:
            image = prepared_dataset_image(os.path.join(self.root_path, filename), border=self.patch_size)
            random.shuffle(self.dataset[filename])
            for (col, row, p) in self.dataset[filename]:
                patch = patch_centered_at(image, row=row, col=col, size=self.patch_size)
                if self.rotation is False:
                    yield patch, p
                else:
                    yield random_rotation(patch), p


class ImageIterator(object):
    def __init__(self, image_file, label_file, patch_size=(101, 101)):
        self.input = prepared_dataset_image(image_file, border=patch_size)
        self.image_size = image_size(prepared_dataset_image(image_file))
        self.patch_size = patch_size
        width, height = self.image_size
        self.output = np.zeros((height, width))
        self.verbose = TT.verbose
        for (col, row, p) in load_csv(label_file):
            self.output[row, col] = 1.0

    def __len__(self):
        return int(np.prod(self.image_size))

    def __iter__(self):
        for i in xrange(len(self)):
            col, row = pixel_at_index(i, self.image_size)
            yield patch_centered_at(self.input, row=row, col=col, size=self.patch_size), self.output[row, col]
