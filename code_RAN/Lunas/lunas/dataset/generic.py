from __future__ import annotations

import collections.abc
import glob
import itertools
import math
import pathlib
from typing import Union, Tuple, List

import lunas.dataset.core as core

__all__ = ['Array', 'Range', 'Enumerate', 'Zip', 'Concat', 'Glob']


class Array(core.Dataset):
    """A dataset that wraps around any sequential data.

    Array dataset accepts any iterable data. Note that the iterable data will be stored into a list immediately during
    initialisation.
    """

    def __init__(self, data: collections.abc.Iterable, name: str = None):
        super().__init__(name)
        self._data = list(data)

    def __len__(self):
        return len(self._data)

    def generator(self):
        for x in self._data:
            yield x


class Range(core.Dataset):
    """Range dataset.

    Simulates the builtin range function.
    """

    def __init__(self, start: int, stop: int = None, step: int = None, name: str = None):
        super().__init__(name)
        if stop is None:
            stop = start
            start = 0

        step = step or 1

        self._start = start
        self._stop = stop
        self._step = step

    def __len__(self):
        start, stop, step = self._start, self._stop, self._step
        return abs(int(math.ceil((stop - start) / step)))

    def generator(self):
        for x in range(self._start, self._stop, self._step):
            yield x


class Enumerate(core.Nested):
    """Enumerate a dataset.

    Simulates the builtin enumerate function and attach an index to each element in the given dataset.
    """

    def __init__(self, dataset: core.Dataset, start: int = 0, name: str = None):
        super().__init__(dataset, name)
        self._start = start

    def __len__(self):
        return len(self._dataset)

    def generator(self):
        for x in enumerate(self._dataset, self._start):
            yield x


class Chunk(core.Nested):

    def __init__(self, dataset: core.Dataset, chunk_size: int, name: str = None):
        if not (chunk_size >= 1):
            raise ValueError(f'Invalid chunk_size: {chunk_size}.')
        super().__init__(dataset, name)
        self._chunk_size = chunk_size

    def generator(self):
        chunk = []
        for x in self._dataset:
            chunk.append(x)
            if len(chunk) == self._chunk_size:
                yield chunk
        if chunk:
            yield chunk


class Zip(core.NestedN):
    """Zips multiple dataset.

    Zips multiple datasets, potentially with different sizes.
    """

    def __init__(self, datasets: Union[Tuple[core.Dataset], List[core.Dataset]], mode: str = '=', padding: bool = False,
                 name: str = None):
        """Initialises the dataset.

        Args:
            datasets: The dataset objects to zip.
            mode: A character, available options include '=' '<' and '>'.
                '=' requires the datasets to have the same sizes;
                '<' behaves similarly to the builtin `zip`, which truncate the bigger datasets to align with the
                smallest one;
                '>' is similar to `itertools.zip_longest`, fill the smaller datasets with strategy specified by
                `padding`.
            padding: A boolean value that determines how to pad the small datasets when they are exhausted.
                A `False` will produce `None` as padding, while `True` will continue producing elements from smaller
                datasets. Only works when mode is '>'.
            name: Name of the dataset.
        """
        super().__init__(datasets, name)
        sizes = tuple(map(len, datasets))
        if mode == '=':
            if len(set(sizes)) > 1:
                raise RuntimeError(f'Datasets must have exactly the same sizes. Got: {tuple(sizes)}')
            size = sizes[0]
        elif mode == '<':
            size = min(sizes)
        elif mode == '>':
            size = max(sizes)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        if not isinstance(padding, bool):
            raise ValueError(f'Expected padding as a bool value, got {padding}')
        self._size = size
        self._sizes = sizes

        self._mode = mode
        self._padding = padding

    def __len__(self):
        return self._size

    def generator(self):
        if self._mode in ['=', '<']:
            for x in zip(*self._datasets):
                yield x
        else:
            if not self._padding:
                it = itertools.zip_longest(*self._datasets)
                for x in it:
                    yield x
            else:
                # DO NOT USE itertools.cycle EVER!
                # Since itertools.cycle actually stores all elements after one-time iteration,
                # this is super-memory-consuming and breaks the internal state maintenance of a `Dataset`.
                datasets = [itertools.chain.from_iterable(itertools.repeat(d))
                            if len(d) < len(self) else d
                            for d in self._datasets]
                # Additionally islice is used here. zip will stop when one iterator raises StopIteration,
                # so any datasets before it will unexpectedly advance one element.
                for x in itertools.islice(zip(*datasets), len(self)):
                    yield x


class Concat(core.NestedN):
    """Concat dataset.

    Concatenates two datasets.
    """

    def __init__(self, a: core.Dataset, b: core.Dataset, name: str = None):
        super().__init__([a, b], name)

    def __len__(self):
        return sum(map(len, self._datasets))

    def generator(self):
        for x in itertools.chain(*self._datasets):
            yield x


class Glob(core.Dataset):
    """Glob dataset.

    Use standard glob module to wrap matched directories/files for given pattern into a dataset.
    """

    def __init__(self, pattern: str, recursive: bool = False, expand_user: bool = True, name: str = None):
        """Initialises the dataset.

        Args:
            pattern: A glob patter.
            recursive: Whether matches recursively.
            expand_user: Whether expands the user home path.
            name: Name of the dataset.
        """
        super().__init__(name)
        self._pattern = str(pathlib.Path(pattern).expanduser() if expand_user else pathlib.Path(pattern))
        self._files = sorted(glob.glob(self._pattern, recursive=recursive))

    @property
    def pattern(self):
        return self._pattern

    def __len__(self):
        return len(self._files)

    def generator(self):
        for x in self._files:
            yield x
