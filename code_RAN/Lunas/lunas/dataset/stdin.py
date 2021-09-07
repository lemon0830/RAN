import os
import sys

import lunas.dataset.core as core

__all__ = ['Stdin']


class Stdin(core.Dataset):
    """Stdin dataset.

    This is a wrapper for `sys.stdin`.

    Warning: Never use this dataset in multi-processing context in order not to observe unexpected behaviours since
    the correctness is not guaranteed.
    """

    def __init__(self, sentinel=None, name: str = None):
        super().__init__(name)
        if sentinel is None:
            sentinel = ''
        self._sentinel = sentinel + os.linesep
        self._resumable = False

    def __len__(self):
        return sys.maxsize

    def generator(self):
        for x in sys.stdin:
            if self._sentinel == x:
                break
            yield x
