# $> python npy2tiff.py <filename.npy> [show]
import numpy as np
import sys
import scipy.misc as s
args = sys.argv[1:]
for arg in args:
    a = np.load(arg)
    s.imsave(arg.replace('npy', 'tiff'), a)
    s.imsave(arg.replace('npy', 'inverted.tiff'), 1. - a)
