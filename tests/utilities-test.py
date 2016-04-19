# Fix Import
import numpy

from test import root_path
# Actual Imports
from utilities import pixel_at_index, prepared_dataset_image, image_size, index_at_pixel, patch_centered_at, np2img


root = root_path()
image = prepared_dataset_image(root+'/tests/patch_at/utilities.tiff')


def test_image_size():
    global image
    # 1. Image size
    size = image_size(image)
    assert (600, 300) == size


def test_pixel_at():
    global image
    size = image_size(image)
    # 2. pixel index to row, col
    assert (500, 2) == pixel_at_index(1700, size)
    # 3. row, col to pixel index
    assert 1700 == index_at_pixel(500, 2, size)
    assert 120400 == index_at_pixel(400, 200, size)


def test_patch_centered_at():
    patch_size = (101, 101)
    image = prepared_dataset_image(root+'/tests/patch_at/source.tiff', patch_size)
    test1 = prepared_dataset_image(root+'/tests/patch_at/patch_0_0.tiff')
    test2 = prepared_dataset_image(root+'/tests/patch_at/patch_300_500.tiff')
    test3 = prepared_dataset_image(root+'/tests/patch_at/patch_500_300.tiff')
    test4 = prepared_dataset_image(root+'/tests/patch_at/patch_500_500.tiff')
    assert numpy.array_equal(test1, patch_centered_at(image, row=50, col=50, size=patch_size))
    assert numpy.array_equal(test2, patch_centered_at(image, row=350, col=550, size=patch_size))
    assert numpy.array_equal(test3, patch_centered_at(image, row=550, col=350, size=patch_size))
    assert numpy.array_equal(test4, patch_centered_at(image, row=550, col=550, size=patch_size))
