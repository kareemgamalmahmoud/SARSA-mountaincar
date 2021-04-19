from __future__ import annotations

from collections import deque
from typing import Optional, Set, Tuple

import parameters
from world.simulated_world import SimulatedWorld


class Hex(SimulatedWorld):

    opposite_player = {
        1: 2,
        2: 1,
    }

    def __init__(self, state: Optional[Tuple[int, ...]] = None):
        self.__size: int = parameters.SIZE
        self.__length = self.__size ** 2
        self.__ending_indices = {
            1: set([self.__length - (i + 1) for i in range(self.__size)]),
            2: set([self.__size * (i + 1) - 1 for i in range(self.__size)]),
        }
        self.reset(state)

    def reset(self, state: Optional[Tuple[int, ...]] = None) -> Tuple[int, ...]:
        self.__is_final_state = False
        self.__modified_list = {
            1: [False for _ in range(self.__size)],
            2: [False for _ in range(self.__size)]
        }
        if state is None:
            self.__player_id = 1
            self.__board = [0 for _ in range(self.__length)]
        else:
            self.__player_id, self.__board = state[0], list(state[1:])
            for action, player_id in enumerate(self.__board):
                if player_id != 0:
                    self.__modified_list[player_id][self.__player_axis(player_id, action)] = True
        return self.__get_state()

    @staticmethod
    def index_to_coordinates(index: int, size: int) -> Tuple[int, int]:
        return index // size, index % size

    # Used by BasicClientActor
    @staticmethod
    def get_valid_actions(state: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(1 if i == 0 else 0 for i in state[1:])

    def get_legal_actions(self) -> Tuple[int, ...]:
        return tuple(1 if i == 0 else 0 for i in self.__board)

    def generate_state(self, action: int) -> Tuple[int, ...]:
        next_board = list(self.__board)
        next_board[action] = self.__player_id
        return (Hex.opposite_player[self.__player_id], *next_board)

    def is_final_state(self) -> bool:
        return self.__is_final_state

    def __update_final_state(self) -> None:
        """
        Checks whether the opposite player has won the game.
        """
        opposite_player = Hex.opposite_player[self.__player_id]

        if sum(self.__modified_list[opposite_player]) < self.__size:  # Does the player have the sufficient amount of pegs along its axis?
            self.__is_final_state = False
            return

        # Sufficient amount of pegs, check for path using BFS
        visited_cells = set()
        for i in range(self.__size):
            index = i if opposite_player == 1 else i * self.__size
            if self.__board[index] == opposite_player and index not in visited_cells:

                # BFS
                visited_cells.add(index)
                queue = deque()
                queue.append(index)
                while len(queue) > 0:
                    current_cell = queue.popleft()
                    for neighbor in self.__get_filled_neighbors(current_cell, opposite_player):
                        if neighbor not in visited_cells and neighbor not in queue:
                            queue.append(neighbor)
                            if neighbor in self.__ending_indices[opposite_player]:
                                self.__is_final_state = True
                                return
                    visited_cells.add(current_cell)
        self.__is_final_state = False
        return

    def step(self, action: int) -> Tuple[Tuple[int, ...], int]:
        assert 0 <= action < self.__size ** 2, 'Illegal action, index out of range'
        assert self.__board[action] == 0, 'Illegal action, cell is occupied'

        self.__board[action] = self.__player_id
        self.__modified_list[self.__player_id][self.__player_axis(self.__player_id, action)] = True  # Used to speed up winning condition check
        self.__player_id = Hex.opposite_player[self.__player_id]
        self.__update_final_state()
        return self.__get_state(), self.get_winner_id()

    def get_winner_id(self) -> int:
        if self.__is_final_state:
            return Hex.opposite_player[self.__player_id]
        else:
            return 0

    def __get_state(self) -> Tuple[int, ...]:
        return (self.__player_id, *self.__board)

    def __player_axis(self, player_id: int, action: int) -> int:
        if player_id == 1:
            return action // self.__size
        return action % self.__size

    def __get_filled_neighbors(self, index: int, player_id: int) -> Set[int]:
        def is_cell_neighbor(cell: int) -> bool:
            if not (0 <= cell < self.__length):
                return False
            if abs((cell % self.__size) - (index % self.__size)) > 1:
                return False
            if self.__board[cell] != player_id:
                return False
            return True

        return set(filter(is_cell_neighbor, self.__get_neighboring_indices(index)))

    def __get_neighboring_indices(self, index: int) -> Set[int]:
        return {
            index - self.__size,
            index + self.__size,
            index + 1,
            index - 1,
            index - self.__size + 1,
            index + self.__size - 1
        }

    def __str__(self) -> str:
        return str(self.__get_state())
