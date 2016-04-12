import re

from utilities import TT


def icpr2012():
    TT.debug("Loading configurations for ICPR 2012.")

    def filename_filter(name):
        return re.compile(r'.+\.bmp').search(name)

    def mapper(name):
        return re.compile(r'\.[a-z]+$').sub('.csv', name)

    return filename_filter, mapper
