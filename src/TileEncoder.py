import matplotlib.pyplot as plt
import numpy as np

import parameters


class TileEncoder:

    def __init__(self):
        # Number of tiles along each corresponding dimension --> (number along x, number along y).
        self.size = (parameters.SIZE_OF_TILING_X, parameters.SIZE_OF_TILING_Y)
        self.num_of_tilings = parameters.NUM_OF_TILINGS
        # Lower bounds for each dimension of the continuous space --> (x, y).
        self.low = [parameters.X_RANGE[0], parameters.Y_RANGE[0]]
        # Upper bounds for each dimension of the continuous space --> (x, y).
        self.high = [parameters.X_RANGE[1], parameters.Y_RANGE[1]]
        self.tilings = self.create_tilings()

    def create_tiling_grid(self, low, high, num_tiles=(10, 10), offsets=(0.0, 0.0)):
        """Define a uniformly-spaced grid that can be used for tile-coding a space.
        Returns
        -------
        grid :      A list of arrays containing split points for each dimension.
        """
        grid = [np.linspace(low[dim], high[dim], num_tiles[dim] + 1)[1:-1] + offsets[dim]
                for dim in range(len(num_tiles))]
        return grid

    def create_tilings(self):
        low = self.low
        high = self.high
        num_tiles = self.size
        num_tilings = self.num_of_tilings

        tiling_specs = []
        offset_x = (low[0] - high[0]) / (num_tilings * num_tiles[0])
        offset_y = (low[1] - high[1]) / (num_tilings * num_tiles[1])

        for _ in range(num_tilings):
            tiling_specs.append(((num_tiles), (offset_x, offset_y)))
            offset_x -= (low[0] - high[0]) / (num_tilings * num_tiles[0])
            offset_y -= (low[1] - high[1]) / (num_tilings * num_tiles[1])
        return [self.create_tiling_grid(low, high, tiles, offsets) for tiles, offsets in tiling_specs]

    def discretize(self, sample, grid):
        """
        Returns
        -------
        discretized_sample : A sequence of integers with the same number of dimensions as sample.
        """
        return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension

    def coordinate_to_index(self, x, y):
        return x + y * self.size[0]

    def tile_encode(self, state, tilings=None, flatten=False):
        """Encode given sample using tile-coding.

        Parameters
        ----------
        sample :    A single sample from the (original) continuous space.
        tilings :   A list of tilings (grids), each produced by create_tiling_grid().
        flatten :   If true, flatten the resulting binary arrays into a single long vector.

        Returns
        -------
        encoded_sample : A list of binary vectors, one for each tiling, or flattened into one.
        """
        if tilings is None:
            tilings = self.tilings

        tiling_features = [[0] * (self.size[0] * self.size[1])] * self.num_of_tilings
        encoded_state = [self.discretize(state, grid) for grid in tilings]

        for i in range(len(encoded_state[0])):
            index = self.coordinate_to_index(encoded_state[i][0], encoded_state[i][1])
            tiling_features[i][index] = 1
        return np.array(tiling_features).flatten()
