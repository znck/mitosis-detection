# $> python npy2tiff.py <filename.npy> [show]
import numpy as np
import sys
import scipy.misc as s
a = np.load(sys.argv[1])
if len(sys.argv) > 2:
    s.imshow(a)
else:
    s.imsave(sys.argv[1].replace('npy', 'tiff'), a)