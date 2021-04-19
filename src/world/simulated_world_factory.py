from typing import Optional, Tuple

import parameters
from game import Game

from .hex import Hex
from .ledge import Ledge
from .simulated_world import SimulatedWorld


class SimulatedWorldFactory:

    @staticmethod
    def get_simulated_world(state: Optional[Tuple[int, ...]] = None) -> SimulatedWorld:
        if parameters.GAME_TYPE == Game.Ledge:
            return Ledge(state)
        else:
            return Hex(state)
