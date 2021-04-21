import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import parameters

class TileEncoder:

    def __init__(self):
        self.size = (parameters.SIZE_OF_TILING_X, parameters.SIZE_OF_TILING_Y)   # Number of tiles along each corresponding dimension --> (number along x, number along y).
        self.num_of_tilings = parameters.NUM_OF_TILINGS + 1
        self.low = [parameters.X_RANGE[0], parameters.Y_RANGE[0]]            # Lower bounds for each dimension of the continuous space --> (x, y).
        self.high = [parameters.X_RANGE[1], parameters.Y_RANGE[1]]           # Upper bounds for each dimension of the continuous space --> (x, y).


    def create_tiling_grid(self, low, high, num_tiles=(10,10), offsets=(0.0, 0.0)):
        """Define a uniformly-spaced grid that can be used for tile-coding a space.
        Returns
        -------
        grid :      A list of arrays containing split points for each dimension.
        """
        grid = [np.linspace(low[dim], high[dim], num_tiles[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(num_tiles))]
        print("Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>")
        for l, h, b, o, splits in zip(low, high, num_tiles, offsets, grid):
            print("    [{}, {}] / {} + ({}) => {}".format(l, h, b, o, splits))
        return grid


    def create_tilings(self):
        low = self.low
        high = self.high
        num_tiles = self.size
        num_tilings = self.num_of_tilings

        tiling_specs = []
        offset_x =  (low[0] - high[0]) / (num_tilings * (num_tiles[0] - 1))
        offset_y =  (low[1] - high[1]) / (num_tilings * (num_tiles[1] - 1))

        for i in range(num_tilings - 1):
            tiling_specs.append(((num_tiles), (offset_x, offset_y)))
            offset_x -= (low[0] - high[0]) / (num_tilings * (num_tiles[0] - 1))
            offset_y -= (low[1] - high[1]) / (num_tilings * (num_tiles[1] - 1))
            print(tiling_specs)
            print(offset_x)
            print(offset_y)
        return [create_tiling_grid(low, high, tiles, offsets) for tiles, offsets in tiling_specs]


    def discretize(self, sample, grid):
        """
        Returns
        -------
        discretized_sample : A sequence of integers with the same number of dimensions as sample.
        """
        return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension

    def coordinate_to_index(self, x, y):
        return (x + (y * self.size[0]))

    def tile_encode(state, tilings, flatten=False):
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
        # TODO: Implement this
        tiling_features= [[0]*(size**2)]*self.size[0]
        encoded_state = [discretize(state, grid) for grid in tilings]

        for i in range(len(encoded_state[0])):
            index = coordinate_to_index(encoded_state[i][0], encoded_state[i][1])
            tiling_features[i][index] = 1
        feature_vector = (np.array(tiling_features)).flatten()

        return feature_vector


#    def visualize_tilings(self, tilings):
#        """Plot each tiling as a grid."""
#        prop_cycle = plt.rcParams['axes.prop_cycle']
#        colors = prop_cycle.by_key()['color']
#        linestyles = ['-', '--', ':']
#        legend_lines = []
#
#        fig, ax = plt.subplots(figsize=(10, 10))
#        for i, grid in enumerate(tilings):
#            for x in grid[0]:
#                l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
#            for y in grid[1]:
#                l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
#            legend_lines.append(l)
#        ax.grid('off')
#        ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
#        ax.set_title("Tilings")
#        plt.xlim([-1.2, 0.6])
#        plt.ylim([-0.07, 0.07])
#        plt.show()
#        return ax  # return Axis object to draw on later, if needed
