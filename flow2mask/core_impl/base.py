from typing import Protocol, runtime_checkable, Union, Tuple, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from .visit_table import VisitedTable


@runtime_checkable
class BaseSeed(Protocol):

    @property
    def coord(self):
        ...

    @property
    def inst_idx(self):
        ...

    def validate_boundary(self, ub: Union[np.ndarray, Tuple], lb: int = 0):
        ...

    def validate_density(self, h: np.ndarray, thresh: float):
        ...

    def validate_visited(self, table: "VisitedTable"):
        ...

    def register_to_table(self, table: "VisitedTable"):
        ...