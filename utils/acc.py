# $> python npy2tiff.py <expected.npy> <predicted.npy>
import numpy as np
import sys

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
a[a < 0.6] = 0
b[b < 0.6] = 0
a[a >= 0.6] = 1
b[b >= 0.6] = 1

c = a - b
neg_a = 1. - a
n = np.prod(a.shape) * 1.0
p = np.sum(a) / n * 100
acc = (1 - np.count_nonzero(np.abs(c)) / n) * 100
fp, fn, tp, tn = (np.count_nonzero(c[c < 0]) / n * 100, np.count_nonzero(c[c > 0]) / n * 100, np.count_nonzero(np.bitwise_and(a == b, a == 1)) / n * 100, np.count_nonzero(np.bitwise_and(a == b, a == 0)) / n * 100)
precision = (tp / (tp + fp))
recall = (tp / (tp + fn))
print "Positive Data %.2f" % p
print "Accuracy %.2f" % (acc,)
print "False Positive %.2f, False Negative %.2f, True Positive %.2f, True Negative %.2f" % (fp, fn, tp, tn)
print "Precision %.2f" % precision
print "Recall %.2f" % recall
print "F-Score %.2f" % (2 * precision * recall / (precision + recall))