# coding=utf-8
# This file contains utility functions.
import random

import numpy as np
import os
import re


def index_at_pixel(x, y, size):
    assert len(size) is 2
    return x * size[0] + y


def pixel_at_index(i, size):
    return i / size[0], i % size[0]


def list_all_files(path, filename_filter=None, mapper=None):
    matched = []

    def file_is_not_csv(name):
        return re.compile(r'\.csv$').search(name) is None

    def default_mapper(name):
        return re.compile(r'\.[a-z]+$').sub('.csv', name)

    if not hasattr(filename_filter, '__call__'):
        filename_filter = file_is_not_csv
    if not hasattr(mapper, '__call__'):
        mapper = default_mapper
    for (dir_name, _, files) in os.walk(path):
        dir_name = dir_name.replace(path, '')
        for filename in files:
            filename = os.path.join(dir_name, filename).strip('/')
            if filename_filter(filename):
                TT.debug(filename)
                matched.append([filename, mapper(filename)])
    return matched


def prepared_dataset_image(filename, border=None):
    import cv2
    assert os.path.exists(filename)
    image = image_normalize(cv2.imread(filename))
    if border is not None:
        assert isinstance(border, (list, tuple)) and len(border) is 2
        image = cv2.copyMakeBorder(image, top=border[1], bottom=border[1], left=border[0], right=border[0],
                                   borderType=cv2.BORDER_DEFAULT)
    return img2np(image)


def patch_centered_at(image, x, y, size=(101, 101)):
    x += (size[1]) / 2
    y += (size[0]) / 2
    assert image_check_point(x + size[1], y + size[0], image_size(image))
    return image[:, x:x + size[1], y:y + size[0]]


def image_rotate(image, k=1):
    if len(image.shape) is 3:
        image[0, :, :] = np.rot90(image[0, :, :], k)
        image[1, :, :] = np.rot90(image[1, :, :], k)
        image[2, :, :] = np.rot90(image[2, :, :], k)
    else:
        return np.rot90(image, k)
    return image


def random_rotation(image):
    ch = random.random()
    if ch <= .5:
        return image
    if ch <= .6:
        return image_rotate(image)
    if ch <= .7:
        return image_rotate(image, 2)
    if ch <= .8:
        return image_rotate(image, 3)
    if ch <= 0.9:
        return np.fliplr(image)
    return np.flipud(image)


def image_check_point(x, y, size):
    if 0 <= x <= size[1] and 0 <= y <= size[0]:
        return True
    TT.danger(x, y, size)
    return False


def image_size(img):
    shape = img.shape
    assert len(shape) is 3
    if shape[0] is 3:
        return shape[1:]
    return shape[:2]


def image_normalize(img):
    img = np.asarray(img, dtype=np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return img


def np2img(np_array):
    np_array = np.asarray(np_array)
    if len(np_array.shape) is not 3:
        raise Exception("np2img function works with 3 dimensional numpy arrays than can be converted to RGB image.")
    return np_array.transpose(1, 2, 0)


def np_append(src, dst):
    dst = np.asarray(dst)
    if src is None:
        return dst
    return np.concatenate((src, dst))


def img2np(img):
    img = np.asarray(img)
    if len(img.shape) is not 3:
        raise Exception("img2np function works with RGB images.")
    return img.transpose(2, 0, 1)


def load_csv(path):
    assert re.compile(r'.*\.csv').match(path) is not None
    result = []
    csv_type = None
    ln = 0
    for line in open(path).readlines():
        ln += 1
        # Ignore empty lines.
        if len(line.strip()) == 0:
            continue
        # Parse line into list of numbers.
        points = map(lambda x: x, line.strip().split(','))
        # Detect format of csv. csv_type ∈ {1, 2}
        #   1: Format (y, x, p)
        #   2: Format (y, x), (y, x) ...
        if csv_type is None:
            csv_type = len(points) % 2
        # Check data is correct.
        assert len(points) % 2 == csv_type
        if csv_type == 0:
            # Convert (x, y) -> (x, y, 1.0)
            result += [[int(points[i + 1]), int(points[i]), 1.0] for i in xrange(0, len(points), 2)]
        elif len(points) == 3:
            result.append([int(points[1]), int(points[0]), float(points[2])])
        else:
            raise Warning("Line %d in %s has invalid value." % (ln, path))
    return result


def csv2np(path):
    return np.asarray(load_csv(path))


def change_ext(path, new_ext):
    name, ext = os.path.splitext(path)
    return name + '.' + new_ext


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
    verbose = False

    @staticmethod
    def debug(*args):
        if TT.verbose:
            print TT.SUCCESS + '> ' + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def info(*args):
        print TT.INFO + '> ' + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def success(*args):
        print TT.SUCCESS + '> ' + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def danger(*args):
        print TT.DANGER + '> ' + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def warn(*args):
        print TT.WARNING + '> ' + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def imp(*args):
        print TT.HEADER + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def ul(*args):
        print TT.UNDERLINE + ' '.join(map(str, args)) + TT.END

    @staticmethod
    def b(*args):
        print TT.BOLD + ' '.join(map(str, args)) + TT.END
