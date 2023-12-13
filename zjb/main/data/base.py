from typing import TYPE_CHECKING

import numpy as np

from zjb.dos.data import Data

from ..trait_types import ArrayLike

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


class ArrayData(Data):
    data = ArrayLike()

    def __array__(self, dtype: "DTypeLike" = None):
        return np.asarray(self.data, dtype=dtype)
