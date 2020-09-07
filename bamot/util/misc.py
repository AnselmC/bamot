import logging
import time

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
