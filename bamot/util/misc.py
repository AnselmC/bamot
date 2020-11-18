import logging
import time
from typing import List

import numpy as np
import tqdm

RNG = np.random.default_rng()
Color = np.ndarray
COLORS: List[Color] = RNG.random((42, 3))


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


def get_color(
    normalized: bool = True, only_bright: bool = True, as_tuple: bool = False
):
    color = COLORS[RNG.choice(len(COLORS))]
    if only_bright:
        # only bright colors
        color[np.argmin(color)] = 0
        color[np.argmax(color)] = 1.0
    if not normalized:
        color = (255 * color).astype(int)
    return color if not as_tuple else tuple(color.tolist())
