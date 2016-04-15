import re

from utilities import TT


def icpr2012():
    """
    Filter dataset and labels.
    :return:
    """
    TT.debug("Loading configurations for ICPR 2012.")

    def filename_filter(name):
        """
        Filter dataset files.
        Return True if file is in dataset.
        :type name: str
        :rtype: bool
        """
        return re.compile(r'.+\.bmp').search(name)

    def mapper(name):
        """
        Map dataset with its labels.
        :type name: str
        :rtype: str
        """
        return re.compile(r'\.[a-z]+$').sub('.csv', name)

    return filename_filter, mapper
