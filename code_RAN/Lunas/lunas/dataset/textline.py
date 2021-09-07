import itertools
from pathlib import Path

import lunas.dataset.core as core

__all__ = ['TextLine']


def count_line(filename):
    f = open(filename, 'rb')
    buf_gen = itertools.takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in itertools.repeat(None)))
    n = 0
    end_is_newline = True
    for buf in buf_gen:
        m = buf.count(b'\n')
        n += m
        if m > 0:
            end_is_newline = buf.rindex(b'\n') == len(buf) - 1
    n += int(not end_is_newline)
    return n


class TextLine(core.Dataset):
    """TextLine dataset

    Wraps a text file.
    """

    def __init__(self, filename: str, encoding: str = 'utf-8', name: str = None):
        super().__init__(name)
        filename = Path(filename)
        self._filename: Path = filename
        self._encoding = encoding
        self._size = count_line(filename)

    def __len__(self):
        return self._size

    def generator(self):
        with self._filename.open('r', encoding=self._encoding) as r:
            for line in r:
                yield line
