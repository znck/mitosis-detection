#!/usr/bin/env python
import json
import os
import random
import warnings

import numpy as np
import sys
import PIL.Image as Image
from keras.utils.generic_utils import Progbar
from math import ceil, floor


class Frame(object):
    def __init__(self, filename, targets):
        self._i = 0
        self._n = 0
        self._size = 1
        self._eff_y = self._eff_x = 0
        self._patch = (1, 1)
        self.filename = filename
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.targets = np.genfromtxt(targets, delimiter=',')
                if len(self.targets.shape) is 1 and len(self.targets) is 3:
                    self.targets = np.asarray([self.targets])
                elif len(self.targets.shape) is 2:
                    pass
                else:
                    self.targets = np.asarray([])
        except IOError:
            self.targets = np.array([])

        self.image = np.asarray(Image.open(filename), dtype=np.float32).transpose(2, 0, 1)

    def batches(self, size=500, patch=(101, 101)):
        self._eff_x = self.image.shape[1] - patch[0] + 1
        self._eff_y = self.image.shape[2] - patch[1] + 1
        self._n = ceil(self._eff_x * self._eff_y / size)
        self._i = 0
        self._size = size
        self._patch = patch
        return self

    def __iter__(self):
        return self

    def next(self):
        if self._i < self._n:
            i = self._i
            pad_x = self._patch[0] / 2
            pad_y = self._patch[1] / 2

            batch_x = []
            batch_y = []

            for j in xrange(self._size):
                pixel = (j + i * self._size)
                x = pixel / self._eff_x + pad_x
                y = pixel % self._eff_x + pad_y
                p = 0.0
                if pixel >= self._eff_x * self._eff_y:
                    break
                for target in self.targets:
                    if int(target[0]) == x and int(target[1]) == y:
                        p = float(target[2])
                        break
                batch_x.append(patch_at(self.image, x, y, self._patch))
                batch_y.append(p)

            self._i += 1
            return np.asarray(batch_x), np.asarray(batch_y)
        else:
            raise StopIteration()


class RandomSampler(object):
    def __init__(self, path, size=(1539, 1376), patch=(101, 101), verbose=1):
        self.verbose = verbose
        self._path = path
        self._files = read_all_files(path)
        self._positives = self.positive()
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

    def batch(self, batch_size=100):
        self._batch = batch_size
        return self

    def sample(self):
        if self.verbose is 1:
            print "Generating random sequence..."
        indices = [int(random.uniform(0, self._n)) for i in xrange(0, self._batch)]
        sampled = {}

        if self.verbose is 1:
            print "Generating pixel array..."
        bar = Progbar(len(indices))
        i = 0
        if self.verbose is 1:
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
            sampled[filename].append((x, y, p))
            if self.verbose is 1:
                i += 1
                bar.update(i)

        return sampled

    def positive(self):
        if os.path.exists(os.path.join(self._path, 'positives_sorted.json')):
            return json.load(open(os.path.join(self._path, 'positives_sorted.json')))

        files = read_all_files(self._path)

        if self.verbose is 1:
            print "%d files" % len(files)

        bar = Progbar(len(files))
        index = 0
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
                        normal[f].append((_x, _y, p))
            index += 1
            if self.verbose is 1:
                bar.update(index)

        json.dump(normal, open(os.path.join(self._path, 'positives.json'), 'w'))
        json.dump(expanded, open(os.path.join(self._path, 'positives_sorted.json'), 'w'))
        return expanded


def read_all_files(path):
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


def patch_at(image, x, y, size=(101, 101)):
    x -= size[0] / 2
    y -= size[1] / 2
    return image[:, x:x + size[0], y:y + size[1]]


def prepare_negative(path):
    sampler = RandomSampler(path=path)
    json.dump(sampler.batch(65000).sample(), open(os.path.join(path, 'negatives.json'), 'w'))


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


if __name__ == '__main__':
    if len(sys.argv) == 2:
        print "Preparing positive dataset..."
        prepare_positive(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 3:
        print "Preparing negative dataset..."
        prepare_negative(os.path.abspath(sys.argv[1]))
    else:
        print "Usage: %d <path>" % sys.argv[0]
