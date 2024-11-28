"""Simple visit table implementation.
TODO: 3d dok sparse matrix.
Author: cielmercy@gmail.com

"""
from _operator import mul
from functools import reduce
from typing import Tuple

import numpy as np
from .base import BaseSeed
from flow2mask.utils import trailing_dim_pad
import warnings

class VisitedTable:
    def __init__(self, shape: Tuple[int, ...], default_value: bool = False):
        self.shape = shape
        self.total_size = reduce(mul, shape)
        self.visited_table = np.ones(shape, dtype=bool) * default_value
        # defaultdict(lambda: False) # dok_matrix((self.total_size, 1), dtype=bool)
        # may use other sparse matrices instead of a hashtable in future for scalability

    @classmethod
    def _validate_coord_key(cls, coord: np.ndarray):
        empty = coord.size ==0
        if empty:
            warnings.warn(f"Empty seed detected")
            return []
        return tuple(coord)

    def _assign(self, seed: BaseSeed, value):
        key = self._validate_coord_key(seed.coord)
        self.visited_table[key] = value

    def register(self, seed: BaseSeed):
        self._assign(seed, True)

    def unregister(self, seed: BaseSeed):
        self._assign(seed, False)

    def visited(self, seed: BaseSeed):
        key = self._validate_coord_key(seed.coord)
        return self.visited_table[key]

    def register_coord(self, coord: np.ndarray):
        coord = trailing_dim_pad(coord, 2)
        key = self._validate_coord_key(coord)
        self.visited_table[key] = True

    def coord_visited(self, coord: np.ndarray):
        coord = trailing_dim_pad(coord, 2)
        key = self._validate_coord_key(coord)
        return self.visited_table[key]