import logging
import time

import numpy as np
import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def timer(func):
    """Decorater that can be used to time function execution.

    :param func: An arbitrary function.
    """
    LOGGER = logging.getLogger("Profiling")

    def func_wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        LOGGER.debug("%s took %f s", func.__name__, time.time() - start)
        return res

    return func_wrapper


# Adapted from https://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
def get_mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr, axis=0)
    return np.median(np.abs(arr - med))
