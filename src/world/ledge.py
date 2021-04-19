import math
from typing import List, Optional, Tuple

import parameters
from world.simulated_world import SimulatedWorld


class Ledge(SimulatedWorld):
    opposite_player = {
        1: 2,
        2: 1
    }

    def __init__(self, state: Optional[Tuple[int, ...]] = None):
        self.__size = parameters.SIZE
        self.__action_space = parameters.NUMBER_OF_ACTIONS
        self.reset(state)

    def reset(self, state: Optional[Tuple[int, ...]] = None) -> Tuple[int, ...]:
        if state is None:
            self.__player_id = 1
            self.__board = list(parameters.LEDGE_BOARD)
        else:
            self.__player_id, *self.__board = list(state)
        return self.__get_state()

    def is_final_state(self) -> bool:
        return 2 not in self.__board

    def get_winner_id(self) -> int:
        if self.is_final_state():
            return Ledge.opposite_player[self.__player_id]
        else:
            return 0

    def step(self, action: int) -> Tuple[Tuple[int, ...], int]:
        coin_position, landing_position = self.index_to_tuple(action)
        if landing_position >= 0:
            self.__board[landing_position], self.__board[coin_position] = self.__board[coin_position], 0
        else:
            self.__board[coin_position] = 0
        self.__player_id = Ledge.opposite_player[self.__player_id]
        return self.__get_state(), self.get_winner_id()

    def get_legal_actions(self) -> Tuple[int, ...]:
        legal_actions = []
        for action in range(self.__action_space):
            legal_actions.append(int(self.__is_legal_action(self.__board, self.index_to_tuple(action))))
        return tuple(legal_actions)

    def __get_state(self) -> Tuple[int, ...]:
        return (self.__player_id, *self.__board)

    def generate_state(self, action: int) -> Tuple[int, ...]:
        coin_position, landing_position = self.index_to_tuple(action)
        board = list(self.__board)
        if landing_position >= 0:
            board[landing_position], board[coin_position] = board[coin_position], 0
        else:
            board[coin_position] = 0
        return (self.__player_id, *board)

    def __is_legal_action(self, board: List[int], action: Tuple[int, int]) -> bool:
        coin_position, landing_position = action
        if board[coin_position] == 0:
            return False
        if coin_position < landing_position:
            return False
        if coin_position == 0:
            return True
        if sum(board[landing_position:coin_position]) > 0:
            return False
        return True

    def index_to_tuple(self, index: int) -> Tuple[int, int]:
        coin_position = math.ceil((math.sqrt(8 * index + 1) - 1) / 2)
        landing_position = index - int(coin_position * (coin_position - 1) / 2) - 1  # index % coin_position (also works, might be cheaper)
        return (coin_position, landing_position)
