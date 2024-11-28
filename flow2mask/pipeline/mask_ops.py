# -*- coding: utf-8 -*-
"""
Derived from zzha962's implementation. Improve the space and time efficiency.
"""
from typing import Optional
import numpy as np
import fastremap
from flow2mask.core_impl.seed import Seed
from flow2mask.utils import apply_cellmask, init_histogram_params, get_flow_hist, init_seeds, \
    init_neighborhood



def final_proc_mask(p: np.ndarray, seed: Seed, h: np.ndarray, pflows: np.ndarray, rpad: int):
    # note we use uint32 as it is unlikely for the number of instances to exceed that.
    new_mask = np.zeros(h.shape, np.uint32)
    new_mask[tuple(seed.coord)] = seed.inst_idx

    for i in range(len(pflows)):
        pflows[i] = pflows[i] + rpad
    out_mask = new_mask[tuple(pflows)]

    # remove big masks
    print("get mask")
    uniq, counts = fastremap.unique(out_mask, return_counts=True)
    big = np.prod(p.shape[1:]) * 0.4
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        out_mask = fastremap.mask(out_mask, bigc)
    print("renumber")
    fastremap.renumber(out_mask, in_place=True)  # convenient to guarantee non-skipped labels
    out_mask = np.reshape(out_mask, p.shape[1:])
    return out_mask


def get_masks(p: np.ndarray, iscell: Optional[np.ndarray] = None, rpad: int = 20,
              thresh: float = 2.,
              neighbor_degree: int = 2,
              final_unique_check: bool = True,
              ):
    """Create masks using pixel convergence after running dynamics.

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.

    Note the mask generation code itself does not keep overlapping instances as it is only for visualization, i.e.,
    one voxel one color.

    Parameters:
        p (float32, 3D or 4D array): Final locations of each pixel after dynamics,
            size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
        iscell (bool, 2D or 3D array): If iscell is not None, set pixels that are
            iscell False to stay in their original location.
        rpad (int, optional): Histogram edge padding. Default is 20.
        thresh (float) : threshold for flow histogram query. Voxel with density > thresh will be considered
            as from an obj. the script default uses 2.
        neighbor_degree (int, optional): degree of seed growth iteration. 0 - return an empty seed.
        final_unique_check: whether to perform validate_unique for the final results
    Returns:
        M0 (int, 2D or 3D array): Masks with inconsistent flow masks removed,
            0=NO masks; 1,2,...=mask labels, size [Ly x Lx] or [Lz x Ly x Lx].
    """
    num_planes = len(p)
    p = apply_cellmask(p, iscell)
    pflows, edges = init_histogram_params(p, rpad)

    # 552 x 552 x 552. padded.
    h, hmax = get_flow_hist(pflows, p, edges)

    # init seeds of instances from histogram. seems to be local maxima?
    seeds = init_seeds(h, hmax)
    assert len(seeds) == num_planes, (f"# of planes of seeds doesn't match #"
                                      f" of planes of voxels: {len(seeds), num_planes}")



    expand = init_neighborhood(num_planes, size=3, offset=-1)
    # todo the original mask rendering itself doesn't even consider instance overlapping scenario.
    # todo therefore we set keep_overlapping as False. Change to True if it is implemented in future.
    result_seeds = Seed.seed_growth(seeds, expand, h,
                                    thresh=thresh,
                                    neighbor_degree=neighbor_degree,
                                    final_unique_check=final_unique_check, keep_overlapping=False)
    print("seeds grown")
    #
    # seed_map = result_seeds.to_seed_map(unique_check=False)

    final_mask = final_proc_mask(p, result_seeds, h, pflows, rpad)
    return final_mask

