import pickle

import numpy as np
from traits.api import Array, Bool, Float, Property, Str

from zjb._traits.types import Instance
from zjb.dos.data import Data

from ..dtb.atlas import RegionSpace
from .space import Space


class SpaceCorrelation(Data):
    space = Instance(Space)

    data = Array()


class Connectivity(SpaceCorrelation):
    space = Instance(RegionSpace, required=True)

    region_labels = Array()

    undirected = Bool()

    average_weights = Property()

    def _get_average_weights(self):
        return self.data / self.space.atlas.areas[:, np.newaxis]

    def save_file(self, file_path, save_average_weights=False):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        if save_average_weights:
            average_weights_file_path = file_path.replace(
                ".npy", "_average_weights.npy"
            )
            with open(average_weights_file_path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "rb") as f:
            cls = pickle.load(f)
        return cls


class StructuralConnectivity(Connectivity):
    tract_lengths = Array()

    delays = Property()

    def _get_delays(self, speed=3.0):
        return self.tract_lengths / speed

    def save_file(self, file_path, save_average_weights=False, save_delays=False):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        if save_average_weights:
            average_weights_file_path = file_path.replace(
                ".npy", "_average_weights.npy"
            )
            with open(average_weights_file_path, "wb") as f:
                pickle.dump(self, f)

        if save_delays:
            delays_file_path = file_path.replace(".npy", "_delays.npy")
            with open(delays_file_path, "wb") as f:
                pickle.dump(self, f)


class FunctionalConnectivity(Connectivity):
    state = Str()
