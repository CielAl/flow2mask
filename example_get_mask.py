# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:56:51 2024
@author: zzha962
"""

from flow2mask.pipeline import get_masks
import pickle
import matplotlib.pyplot as plt

# use the precomputed flow/cp_mask to accelerate the testing procedure
with open("interm.pkl", 'rb') as f:
    data = pickle.load(f)
[p, cp_mask] = data

mask = get_masks(p, iscell=cp_mask, rpad=20, thresh=2., neighbor_degree=5)

plt.imshow(mask[:, :, 256], 'gray'); plt.show()