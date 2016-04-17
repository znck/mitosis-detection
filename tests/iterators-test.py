# Fix Import
from test import root_path
# Actual Imports
from utilities import TT
from iterators import BatchGenerator, Dataset
from dataset import icpr2012

root = root_path()

# 1. Test BatchGenerator
flt, mapper = icpr2012()
TT.verbose = True
batches = BatchGenerator(Dataset(root+'/datasets/ICPR 2012/testing/set1', mapper=mapper, filename_filter=flt), 1000, 500)
for batch in batches:
    continue
