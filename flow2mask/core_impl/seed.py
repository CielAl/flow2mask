"""Abstraction of seed growth.

Author: cielmercy@gmail.com

"""
from typing import Union, Tuple, List, Optional, Dict
from .base import BaseSeed
from .visit_table import VisitedTable
from flow2mask.utils import trailing_dim_pad, hist_query
import fastremap
from collections import defaultdict
from functools import partial

import numpy as np


class Seed(BaseSeed):
    """Abstraction of seeds in instance segmentation.

    """
    _coord: np.ndarray
    _inst_idx: np.ndarray

    @property
    def coord(self) -> np.ndarray:
        """Coordinates of voxels

        Returns:
            N_Plane (usually 3) x N_points
        """
        return self._coord

    @property
    def inst_idx(self) -> np.ndarray:
        """Instance IDs for each voxel.

        Returns:
            1D np.ndarray of shape: N_points
        """
        return self._inst_idx

    @staticmethod
    def _center_mask(expand: np.ndarray, center: Optional[Union[int, np.ndarray]]) -> np.ndarray:
        """Given an expand matrix, and an optional designated center of the expand, get the index mask of the expand
            which points to the center of the corresponding neighborhood cube.

        Args:
            expand: expand matrix. N_dim x N_neighborhood. Center is included. For a 3x3x3 cube it is 3 x 27.
            center: designated center of the cube. if None - use the origin. If it is int C, the center will be
                (C, C, ...C). In actual neighbor expansion procedure the center is computed by mean (round to int).

        Returns:
            np.ndarray: the index mask for future array selection.
        """
        if center is None:
            center = 0
        else:
            center = np.array(center)
            center = trailing_dim_pad(center, expand.ndim)
            assert expand.ndim == center.ndim
        match = np.all(expand == center, axis=0)
        assert np.sum(match) == 1, f"Not unique center or no match."
        return match

    @staticmethod
    def _in_boundary_mask(coord: np.ndarray, ub: Union[np.ndarray, Tuple], lb: int = 0) -> np.ndarray:
        """Index mask for whether the coordinates are within the boundary.

        The convention follows the practice of
        common python imaging libraries: half close at lb and half open at ub: lb <= coord < ub.

        Args:
            coord: coordinates: N_plane x N_points
            ub: the upperbound. Each dimension can be customized with a different ub.
                it will be converted to a column vector aligned to the coord if it is not already such one
                (e.g., int or tuple).
            lb: the unified lower bound for all planes (for simplicity).

        Returns:
            np.ndarray: the index mask for future array selection.
        """
        ub = trailing_dim_pad(ub, coord.ndim)
        valid_mask = np.all((coord >= lb) & (coord < ub), axis=0)
        return valid_mask

    @staticmethod
    def _neighbor_coords(seeds: np.ndarray, expand: np.ndarray) -> np.ndarray:
        """Get the coordinates in a cubic neighborhood.

        Args:
            seeds: N_Plane x N_seed
            expand: N_Plane x N_neighbor

        Returns:
            N_Plane x N_neighbor x N_seed
        """
        assert seeds.ndim == 2
        assert expand.ndim == 2

        assert isinstance(seeds, np.ndarray)
        assert isinstance(expand, np.ndarray)
        return seeds[:, None, :] + expand[..., None]

    @staticmethod
    def _neighbor_expansion(seeds: np.ndarray, expand: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Perform neighbor expansion to seeds.

        No boundary check in this step. Only compute the shifted coordinates. Center and the surrounding non-center
        coordinates in the whole neighborhood cube will be separately reported.

        Args:
            seeds: the coordinate of the seeds in shape of N_Plane x N_seed
            expand: expand matrix. N_Plane x N_neighbor

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: center_seeds in shape of N_Plane x N_seed,
                neighbor_coord_list as a list of np.ndarray, each in shape of 3 x N_seed. The length of the list is
                N_neighbor - 1, assuming the N_neighbor is the total number of voxels in the expansion cube (center
                included).
        """
        # NPlane x N_Neighbor x N_instance
        expanded_seeds = Seed._neighbor_coords(seeds, expand)

        center_offset = expand.mean(axis=1).astype(int)
        center_masking = Seed._center_mask(expand, center_offset) # np.all(centered_expand == 0, axis=0)

        center_seeds = expanded_seeds[:, center_masking, :]
        neighbor_seeds = expanded_seeds[:, ~center_masking, :]
        assert expand.shape[1] == center_seeds.shape[1] + neighbor_seeds.shape[1]
        # squeeze the neighbor column
        neighbor_coord_list = [neighbor_seeds[:, k, :] for k in range(neighbor_seeds.shape[1])]

        # squeeze the neighbor column
        center_seeds = center_seeds[:, 0, :]
        return center_seeds, neighbor_coord_list

    def expansion(self, expand: np.ndarray, ub: Union[np.ndarray, Tuple],
                  check_boundary: bool = True,
                  lb: int = 0) -> "SeedList":
        """Perform neighborhood expansion to the current seeds.

        Will check the boundary if check_boundary is True.
        Coordinates out of the boundary will be discarded, along with the corresponding
        instance indices in the inst_idx

        Args:
            expand: expansion matrix. N_dim x N_neighbor
            ub: the upperbound. Each dimension can be customized with a different ub.
                it will be converted to a column vector aligned to the coord if it is not already such one
                (e.g., int or tuple).
            check_boundary: If True, will perform validate_boundary.
            lb: lower bound integer. default is 0.

        Returns:
            SeedList: SeedList which encapsules all neighbors (exclude the center voxel) that are within the
            boundary (if check_boundary is True).
        """
        _, neighbor_coord_list = self._neighbor_expansion(self.coord, expand)
        neighbor_seed_list: List["Seed"] = [Seed(self.inst_idx, k) for k in neighbor_coord_list]
        seed_list = SeedList(neighbor_seed_list)
        if check_boundary:
            seed_list.validate_boundary(ub, lb)
        return seed_list

        # boundary check
    def _masking(self, valid_mask: np.ndarray):
        """Helper function.

        Update the content of _coord and _inst_idx to what are included in the valid_mask.


        Args:
            valid_mask: indexing mask array

        Returns:

        """
        assert valid_mask.ndim == 1, (f"index mask's dim must be 1 (along num_points),"
                                      f" got: {valid_mask.ndim, valid_mask.shape}")
        self._coord = self._coord[:, valid_mask]
        self._inst_idx = self._inst_idx[valid_mask]

    @staticmethod
    def _validate_ub_value(coord: np.ndarray, ub: Optional[Union[np.ndarray, Tuple]]):
        """Helper function to sanitize the ub for boundary check.

        If ub is None then simply using the max shape of each dimension.

        Args:
            coord: N_Plane x N_seed
            ub: Optional. If none it will be simply the maximum of each dimension of coord.

        Returns:

        """
        if ub is not None:
            return ub
        return coord.max(axis=1)

    def validate_boundary(self, ub: Union[np.ndarray, Tuple], lb: int = 0):
        """Boundary check of coordinates. Remove out-of-boundary coordinates and the corresponding instance ids.

        Args:
            ub: upperbound. See _in_boundary_mask
            lb: lowerbound. See _in_boundary_mask

        Returns:

        """
        ub = Seed._validate_ub_value(self.coord, ub)
        valid_mask = self._in_boundary_mask(self.coord, ub, lb)
        self._masking(valid_mask)

    def validate_density(self, h: np.ndarray, thresh: float):
        """Only keep the coord and corresponding instance ids if the density in the flow histogram is > thresh.

        Detail see hist_query.

        Args:
            h: histogram of flow.
            thresh: threshold. ">"

        Returns:
        """
        # default density 2.0
        valid_mask = hist_query(h, self.coord, thresh=thresh)
        self._masking(valid_mask)

    def validate_visited(self, table: VisitedTable):
        """Validate whether the coordinates are already visited by a visit table. Only keep unvisited coord/inst_idx.

        Args:
            table: see VisitedTable.

        Returns:

        """
        visited_mask = table.visited(self)
        unique_mask = ~visited_mask
        self._masking(unique_mask)

    def validate_unique(self, keep_overlapping: bool ):
        """Perform unique check of coordinates. Discard the duplicated instance. It will not consider overlapping.

        Note: how do we address overlapping objects?

        Returns:

        """
        # todo - instance overlapping? but seems like the mask generation code doesn't really care about that.
        coord_entity = self.coord
        if not keep_overlapping:
            # each column vector for unique_check is now [inst, a1, a2, ..., an]^T
            coord_entity = np.vstack([self.inst_idx, self.coord])
        _, index = fastremap.unique(coord_entity, axis=1, return_index=True)
        self._masking(index)

    def register_to_table(self, table: VisitedTable):
        """register the presence of the coordinates to the visit table.

        Args:
            table: See VisitedTable.

        Returns:

        """
        table.register(self)

    def __init__(self, inst_idx: np.ndarray, coord: np.ndarray):
        self._inst_idx = np.array(inst_idx)
        self._coord = trailing_dim_pad(np.array(coord), target_ndim=2)
        assert self._coord.ndim == 2

    @classmethod
    def new(cls, *, coord: np.ndarray) -> "Seed":
        """Factory builder to create a new Seed object.

        Each seed location (N_planesx1 vector) in the coord (N_planes x N_seed)  will be considered as the initial
        location of an instance and will be sequentially assigned with an instance id.

        Args:
            coord:

        Returns:

        """
        assert coord.ndim == 2, f"2D coord array is expected - N_plane x N_seeds. Got: {coord.ndim}"
        num_seeds = coord.shape[1]
        inst_idx = np.arange(num_seeds)
        return cls(inst_idx, coord)

    def __len__(self) -> int:
        assert self.coord.shape[1] == self.inst_idx.shape[0]
        return self.coord.shape[1]

    def grow(self, expand: np.ndarray, h: np.ndarray,
             thresh: float,
             neighbor_degree: int,
             final_unique_check: bool,
             keep_overlapping: bool
             ):
        """Grow the seed based on histogram of the flow. See seed_growth.

        Args:
            expand: expand matrices - N_planes x N_neighbors (center included) shift on each plane of a neighboring cube
            h: histogram of flow
            thresh: threshold for flow histogram query. Voxel with density > thresh will be considered as from an obj.
                the script default uses 2.
            neighbor_degree: degree of expansion. 0 will return an empty seed.
            final_unique_check: whether to perform validate_unique for the final results
            keep_overlapping: in validate_unique procedure, If True, duplicated coordinates with different instance id
                will be kept. If False, only keep the instance of first occurrence in Seed.coord.
                Will also affect validate_unique in the neighborhood expansion.
        Returns:

        """
        return self.seed_growth(self, expand, h, thresh, neighbor_degree, final_unique_check, keep_overlapping)

    @staticmethod
    def seed_growth(seeds: "Seed", expand: np.ndarray, h: np.ndarray,
                    thresh: float,
                    neighbor_degree: int,
                    final_unique_check: bool,
                    keep_overlapping: bool):
        """Grow the seed based on histogram of the flow.

        Args:
            seeds: initial seeds.
            expand: expand matrices - shift on each plane of a neighboring cube
            h: histogram of flow
            thresh: threshold for flow histogram query. Voxel with density > thresh will be considered as from an obj.
                the script default uses 2.
            neighbor_degree: degree of expansion. 0 will return an empty seed.
            final_unique_check: whether to perform validate_unique for the final results
            keep_overlapping: in validate_unique procedure, If True, duplicated coordinates with different instance id
                will be kept. If False, only keep the instance of first occurrence in Seed.coord.
                Will also affect validate_unique in the neighborhood expansion.
        Returns:
            A new seed object containing all instance indices and expanded coordinates after growing.
        """
        if not isinstance(seeds, Seed):
            seeds = np.array(seeds)
            seeds = Seed.new(coord=seeds)
        shape = h.shape

        # num_planes = len(seeds)


        result_stack: SeedList = SeedList([])
        table = VisitedTable(shape, False)  # np.zeros(shape, dtype=bool)  # VisitedTable(shape) # dok_matrix(shape, dtype=bool)
        for step in range(neighbor_degree):
            print(f"iter {step}")
            print(f"Seed size: {seeds.coord.shape}")
            # register the center of cube
            seeds.register_to_table(table)
            neighbors = seeds.expansion(expand, check_boundary=True, ub=shape, lb=0)
            # remove potential seeds in padded regions after expansion.
            seeds.validate_boundary(ub=shape, lb=0)

            # density check
            seeds.validate_density(h, thresh=thresh)

            # validate here before further neighbor operations as neighbors may already have centers
            # depending on seeds' spatial allocation
            neighbors.validate_visited(table)
            neighbors.validate_density(h, thresh=thresh)

            # register

            if step == 0:
                # only apply the first round as all future center are already collected in the previous round
                result_stack.append(seeds)
            # neighbors.validate_visited(table)
            neigbhor_merged = neighbors.merge(unique_check=True, keep_overlapping=keep_overlapping)
            result_stack.append(neigbhor_merged)
            seeds = neigbhor_merged

        final_seeds = result_stack.merge(unique_check=final_unique_check, keep_overlapping=keep_overlapping)
        return final_seeds

    def to_seed_map(self, unique_check: bool, keep_overlapping: bool) -> Dict[int, np.ndarray]:
        """Can be extremely slow. Avoid to use if # of instances is huge.

        Export to a map of instance id --> coordinates

        Args:
            unique_check: if perform validate_unique beforehand. Will discard duplicates inplace.
            keep_overlapping: in validate_unique procedure, If True, duplicated coordinates with different instance id
                will be kept. If False, only keep the instance of first occurrence in Seed.coord
        Returns:

        """
        # partial(np.ndarray, shape=(self.coord.shape[0], 0), dtype=self.coord.dtype)
        seed_map = defaultdict(partial(np.ndarray, shape=(self.coord.shape[0], 0), dtype=self.coord.dtype))
        if unique_check:
            self.validate_unique(keep_overlapping)
        # get unique instance indices
        unique_inst = fastremap.unique(self.inst_idx)
        for inst in unique_inst:
            target_coord = self.coord[:, self.inst_idx == inst]
            seed_map[int(inst)] = target_coord
        return seed_map


class SeedList(BaseSeed):
    """Vectorization helper class for a list of Seed objects. Simulate relevant interfaces of Seed class.

    Can be merged to a single Seed

    """
    _seeds_list: List[Seed]

    def __init__(self, seeds: List[Seed]):
        self._seeds_list = seeds

    def validate_boundary(self, ub: Union[np.ndarray, Tuple], lb: int = 0):
        for seed in self._seeds_list:
            seed.validate_boundary(ub=ub, lb=lb)

    def validate_density(self, h: np.ndarray, thresh: float):
        for seed in self._seeds_list:
            seed.validate_density(h=h, thresh=thresh)

    def validate_visited(self, table: VisitedTable):
        for seed in self._seeds_list:
            seed.validate_visited(table)

    def register_to_table(self, table: VisitedTable):
        for seed in self._seeds_list:
            seed.register_to_table(table)

    def merge(self, unique_check: bool, keep_overlapping: bool) -> Seed:
        """Merge the coord and inst_idx of every Seed object in the list to a new single Seed object.

        Could induce heavy memory operations --> do it as infrequent as possible

        Args:
            unique_check: whether perform validate_unique
            keep_overlapping: in validate_unique procedure, If True, duplicated coordinates with different instance id
                will be kept. If False, only keep the instance of first occurrence in Seed.coord

        Returns:

        """

        coord_list_all = [n.coord for n in self._seeds_list]
        inst_idx_all = [n.inst_idx for n in self._seeds_list]

        coords = np.hstack(coord_list_all)
        inst_idx = np.hstack(inst_idx_all)
        new_seed = Seed(inst_idx, coords)
        if unique_check:
            new_seed.validate_unique(keep_overlapping)
        return new_seed

    def append(self, seeds: Seed):
        """Append the new seed to the internal list.

        Args:
            seeds: Seed object to append.

        Returns:

        """
        self._seeds_list.append(seeds)

    def __len__(self):
        return len(self._seeds_list)
