from typing import Union, Tuple, List

import lunas.dataset.core as core

__all__ = ['Sampling']

import random


def normalise_weights(weights):
    sum_weight = sum(weights)
    return [w / sum_weight for w in weights]


class Sampling(core.NestedN):
    """Sampling dataset

    Samples examples from multiple datasets by given weights.
    """

    def __init__(self, datasets: Union[Tuple[core.Dataset], List[core.Dataset]],
                 weights: Union[List[float], Tuple[float]] = None,
                 replacement: bool = True, name: str = None):
        if weights and sum(weights) != 1:
            raise ValueError(f'Expected the sum of weights to be 1.0, got {sum(weights)} instead.')
        super().__init__(datasets, name)
        self._weights = weights if weights else [1.0 / len(datasets)] * len(datasets)
        self._replacement = replacement
        self._size = None

    def __len__(self):
        if self._size is None:
            self._size = sum(len(dataset) for dataset in self._datasets)
        return self._size

    def generator(self):
        indices = [i for i in range(len(self._datasets))]
        weights = [w for w in self._weights]
        datasets = [iter(dataset) for dataset in self._datasets]
        mask = [True] * len(datasets)
        while True:
            if len(indices) > 1:
                i = random.choices(indices, weights)[0]
            else:
                i = indices[0]
            try:
                yield next(datasets[i])
            except StopIteration:
                if not self._replacement:
                    j = indices.index(i)
                    indices.pop(j)
                    weights.pop(j)
                    weights = normalise_weights(weights)
                    if not indices:
                        break
                else:
                    mask[i] = False
                    if any(mask[i]):
                        datasets[i] = iter(self._datasets[i])
                        yield next(datasets[i])
                    else:
                        break
