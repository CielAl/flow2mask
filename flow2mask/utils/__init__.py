"""Mostly derived from zzha962's implementation but only refactored for readability and code reuse.

"""
from typing import Union, Tuple
import numpy as np
from scipy.ndimage import maximum_filter1d


def trailing_dim_pad(array: Union[np.ndarray, Tuple, int], target_ndim: int):
    """Adding trailing dimension to an array, numeric tuple, or scalar int.
    Args:
        array: input int, np.ndarray, or array-alike
        target_ndim: target number of dimension. Only expand the dimensions if the target_ndim is larger than
            current dim

    Returns:
        np.ndarray with expanded dimensions
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    diff_dims = target_ndim - array.ndim  # Number of trailing dims to add
    if diff_dims > 0:
        return np.expand_dims(array, axis=tuple(range(array.ndim, target_ndim)))
    return array


def apply_cellmask(p: np.ndarray, iscell: np.ndarray):
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims == 3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               np.arange(shape0[2]), indexing="ij")
        elif dims == 2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               indexing="ij")
        else:
            raise ValueError(f'dims not supported: {dims}')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]
    return p


def init_histogram_params(p: np.ndarray, rpad: int):
    pflows = []
    edges = []
    for i in range(len(p)):
        pflows.append(p[i].flatten().astype("int32"))
        edges.append(np.arange(-.5 - rpad, p.shape[1:][i] + .5 + rpad, 1))
    return pflows, edges


def get_flow_hist(pflows, p, edges):
    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(len(p)):
        hmax = maximum_filter1d(hmax, 5, axis=i)
    return h, hmax


def hist_query(h: np.ndarray, coords: np.ndarray, thresh: float = 2):
    """Query whether the coordinate in the flow field fits the convergence criteria

    Args:
        h: histogram array
        coords: coordinate array
        thresh: threshold. Criteria: density > thresh

    Returns:

    """
    ind = tuple(coords.astype(int))
    density = h[ind]
    return density > thresh


def init_seeds(h, hmax):
    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s[:] = s[isort]
    return seeds


def init_neighborhood(num_planes, size: int = 3, offset: int = 0):
    if num_planes == 3:
        expand = np.nonzero(np.ones((size, size, size)))
    else:
        expand = np.nonzero(np.ones((size, size)))
    return np.array(expand) + offset
