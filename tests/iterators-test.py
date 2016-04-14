# Fix Import
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root+'/src')

# Actual Imports
from utilities import TT
from iterators import BatchGenerator, Dataset
from dataset import icpr2012

# 1. Test BatchGenerator
flt, mapper = icpr2012()
TT.verbose = True
batches = BatchGenerator(Dataset(root+'/datasets/ICPR 2012/testing/set1', mapper=mapper, filename_filter=flt), 1000, 500)
for batch in batches:
    continue
